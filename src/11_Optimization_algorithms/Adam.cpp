
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <cmath>
#include <tuple>
#include <chrono>

#include "../csvloader.h"
#include "../utils.h"
#include "../utils/ch_11_util.h"

#include <matplot/matplot.h>
using namespace matplot;

using namespace std::chrono;

std::vector<std::pair<torch::Tensor, torch::Tensor>>  init_adam_states( int64_t feature_dim) {
	torch::Tensor v_w = torch::zeros({feature_dim, 1}), v_b = torch::zeros(1);
	torch::Tensor s_w = torch::zeros({feature_dim, 1}), s_b = torch::zeros(1);
	std::vector<std::pair<torch::Tensor, torch::Tensor>> v_s = {std::make_pair(v_w, s_w), std::make_pair(v_b, s_b)};
    return v_s;
}

void adam(std::vector<torch::Tensor>& params, std::vector<std::pair<torch::Tensor, torch::Tensor>>& states,
		  std::map<std::string, float> hyperparams ) {

    float beta1 = 0.9, beta2 = 0.999, eps = 1e-6;

    for( int i = 0; i < params.size(); i++ ) { //p, (v, s) in zip(params, states):
    	auto p = params[i];
    	auto v = states[i].first;
    	auto s = states[i].second;

    	torch::NoGradGuard no_grad;
    		v = beta1 * v + (1 - beta1) * p.grad();
            s = beta2 * s + (1 - beta2) * torch::square(p.grad());

            auto v_bias_corr = v / (1 - std::pow(beta1, hyperparams["t"]));
            auto s_bias_corr = s / (1 - std::pow(beta2, hyperparams["t"]));
            p -= hyperparams["lr"] * v_bias_corr / (torch::sqrt(s_bias_corr) + eps);
        p.grad().data().zero_();
    }
    hyperparams["t"] += 1;
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
	std::map<std::string, float> hyperparams = {{"lr", 0.01}, {"t", 1}};

	std::vector<std::pair<torch::Tensor, torch::Tensor>> states = init_adam_states(t_data.size(1));

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
	        adam(params, states, hyperparams);

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
	F->size(1200, 500);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	subplot(1, 2, 0);
	matplot::plot(epochs, losses, "b")->line_width(2).display_name("Train loss");
    matplot::xlabel("epoch");
    matplot::ylabel("loss");
    matplot::title("Adam scratch");
    matplot::legend({});

	// ------------------------------------
	// A more concise implementation
	// ------------------------------------
	auto net = torch::nn::Sequential( torch::nn::Linear(5, 1) );

	// init weight
	for(auto& module : net->modules(false)) { 			// include_self= false
			if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
				torch::nn::init::normal_(M->weight, 0.01);  // torch.nn.init.normal_(m.weight, std=0.01)
		}
	}

	auto optimizer = torch::optim::Adam(net->parameters(), 0.01);
	auto loss_f = torch::nn::MSELoss(torch::nn::MSELossOptions(torch::kNone));

	epochs.clear();
	losses.clear();

	start = high_resolution_clock::now();

	for( int64_t  epoch = 0; epoch < num_epochs; epoch++ ) {
		std::list<std::pair<torch::Tensor, torch::Tensor>> data_iter = get_data_ch11(t_data, t_label, batch_size);

		float t_loss = 0.0;
		int64_t b_cnt = 0;
		for (auto &batch : data_iter) {
			auto X = batch.first;
			auto y = batch.second;

			optimizer.zero_grad();

			auto out = net->forward(X);

			y = y.reshape(out.sizes());
			auto l = loss_f(out, y).mean();
			l.backward();

			optimizer.step();

		    t_loss += l.item<float>();
		    b_cnt++;
		}

		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		std::cout << "Epoch: " << (epoch + 1) << ", loss: " << (t_loss/b_cnt)  << ", duration: " << (duration.count() / 1e6) << " sec.\n";
		epochs.push_back(epoch*1.0);
		losses.push_back((t_loss/b_cnt));
	}

	subplot(1, 2, 1);

	matplot::plot(epochs, losses, "m")->line_width(2).display_name("Train loss");
	matplot::xlabel("epoch");
	matplot::ylabel("loss");
    matplot::title("Adam concise");
    matplot::legend({});
    F->draw();
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}

