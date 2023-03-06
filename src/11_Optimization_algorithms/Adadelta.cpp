
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <chrono>

#include "../csvloader.h"
#include "../utils.h"
#include "../utils/ch_11_util.h"

#include <matplot/matplot.h>
using namespace matplot;

using namespace std::chrono;

std::vector<std::pair<torch::Tensor, torch::Tensor>> init_adadelta_states(int64_t feature_dim) {
    torch::Tensor s_w = torch::zeros({feature_dim, 1}), s_b = torch::zeros(1);
    torch::Tensor delta_w = torch::zeros({feature_dim, 1}), delta_b = torch::zeros(1);
    std::vector<std::pair<torch::Tensor, torch::Tensor>> s_d = {std::make_pair(s_w, delta_w), std::make_pair(s_b, delta_b)};
    return s_d;
}


void adadelta(std::vector<torch::Tensor>& params, std::vector<std::pair<torch::Tensor, torch::Tensor>>& states,
			  std::map<std::string, float> hyperparams) {
    auto rho = hyperparams["rho"];
    float eps = 1e-5;
    for( int i = 0; i < params.size(); i++ ) { // p, (s, delta) in zip(params, states) {
    	auto p = params[i];
    	auto s = states[i].first;
    	auto delta = states[i].second;
    	torch::NoGradGuard no_grad;
        // In-place updates via [:]
        s = rho * s + (1 - rho) * torch::square(p.grad());
        auto g = (torch::sqrt(delta + eps) / torch::sqrt(s + eps)) * p.grad();
        p -= g;
        delta = rho * delta + (1 - rho) * g * g;
        p.grad().data().zero_();
    }
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	// ---------------------------------------------------------
	// Implementation from Scratch
	// ---------------------------------------------------------

	// Load train CSV data
	std::ifstream file;
	std::string path = "./data/airfoil_self_noise.csv";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		file.close();
		return -1;
	}

	int num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	std::cout << "records in file = " << num_records << '\n';

	// set file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);

	std::pair<std::vector<float>, std::vector<float>> data_label = process_data(file, true, true);

	int64_t batch_size = 10;

	std::vector<float> datas;
	std::vector<float> labels;

	datas  = data_label.first;
	labels = data_label.second;
	std::cout << datas.size() << std::endl;

	// Convert vectors to a tensor
	auto t_label = torch::from_blob(labels.data(), {int(labels.size()), 1}).clone();
	auto t_data = torch::from_blob(datas.data(), {int(labels.size()), int(datas.size()/labels.size())}).clone();

	int64_t num_epochs = 10;
	std::map<std::string, float> hyperparams = {{"rho", 0.9}};

	std::vector<std::pair<torch::Tensor, torch::Tensor>> states = init_adadelta_states(t_data.size(1));

	// Initialization
	torch::Tensor w = torch::normal(0.0, 0.01, {t_data.size(1), 1}).requires_grad_(true);
	torch::Tensor b = torch::zeros({1}, torch::requires_grad(true));
	std::vector<torch::Tensor> params = {w, b};

    torch::Tensor loss, t;

    auto start = high_resolution_clock::now();
    std::vector<float> epochs, losses;

	for( int64_t  epoch = 0; epoch < num_epochs; epoch++ ) {
		std::list<std::pair<torch::Tensor, torch::Tensor>> data_iter = get_data_ch11(t_data, t_label, batch_size);

		float t_loss = 0.0;
		int64_t b_cnt = 0;
		for (auto &batch : data_iter) {

			auto X = batch.first;
			auto y = batch.second;

	    	t = linreg(X, params[0], params[1]); // X, w, b
	        loss = squared_loss(t, y).mean();

	        // Compute gradient on `l` with respect to [`w`, `b`]
	        loss.backward();

	        // Update parameters using their gradient
	        adadelta(params, states, hyperparams);

	        t_loss += loss.item<float>();
	        b_cnt++;
	    }

		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		std::cout << "Epoch: " << (epoch + 1) << ", loss: " << (t_loss/b_cnt)  << ", duration: " << (duration.count() / 1e6) << " sec.\n";
		epochs.push_back(epoch*1.0);
		losses.push_back((t_loss/b_cnt));
	}

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::legend();
	matplot::plot(ax1, epochs, losses, "b")->line_width(2)
		.display_name("Train loss");
    matplot::xlabel(ax1, "epoch");
    matplot::ylabel(ax1, "loss");
    matplot::title("Adadelta scratch");
    matplot::show();

	std::cout << "Done!\n";
	return 0;
}




