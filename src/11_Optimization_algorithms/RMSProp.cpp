
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <atomic>
#include <string>
#include <algorithm>
#include <iostream>
#include <chrono>

#include "../csvloader.h"
#include "../utils.h"
#include "../utils/ch_11_util.h"

using namespace std::chrono;

std::tuple<double, double, double, double> rmsprop_2d(double x1, double x2,
											double s1, double s2, double eta, double gamma) {
    double g1 = 0.2 * x1,  g2 = 4 * x2, eps = 1e-6;
    s1 = gamma * s1 + (1 - gamma) * g1 * g1;
    s2 = gamma * s2 + (1 - gamma) * g2 * g2;
    x1 -= eta / std::sqrt(s1 + eps) * g1;
    x2 -= eta / std::sqrt(s2 + eps) * g2;
    return std::make_tuple(x1, x2, s1, s2);
}

std::pair<std::vector<double>, std::vector<double>> train_2d(int steps, double eta, double gamma) {
	double x1 = -5, x2 = -2, s1 = 0, s2 = 0;
    std::vector<double> x, xx; // = [(x1, x2)]
    x.push_back(x1);
    xx.push_back(x2);
    for(int  i = 0; i < steps; i++ ) {
    	std::tie(x1, x2, s1, s2) = rmsprop_2d(x1, x2, s1, s2, eta, gamma);
        x.push_back(x1);
        xx.push_back(x2);
    }

    std::cout << "epoch: " << steps << " , x1: " << x1 << " , x2: " << x2 << '\n';

    return std::make_pair(x, xx);
}

// Implementation from Scratch
std::vector<torch::Tensor> init_rmsprop_states(int64_t feature_dim) {
	auto s_w = torch::zeros({feature_dim, 1});
    auto s_b = torch::zeros(1);
    return {s_w, s_b};
}

void rmsprop(std::vector<torch::Tensor>& params, std::vector<torch::Tensor>& states,
													 std::map<std::string, float> hyperparams) {
	double gamma = hyperparams["gamma"], eps = 1e-6;

	for( int i = 0; i < states.size(); i++ ) {
		torch::NoGradGuard no_grad;
		states[i] = gamma * states[i] + (1 - gamma) * torch::square(params[i].grad());
        params[i] -= hyperparams["lr"] * params[i].grad() / torch::sqrt(states[i] + eps);
        params[i].grad().data().zero_();
	}
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	std::vector<double> gmmas = {0.95, 0.9, 0.8, 0.7};
	std::vector<std::string> strs = {"b-", "y-", "g-", "r-"};

	// RMSProp
	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	std::vector<std::string> lgd;
	matplot::hold(ax1, true);

	int i = 0;
	for( auto& b : gmmas ) {
		std::vector<double> x, y;
		for( double t = 0.0; t < 40.0; t += 1.0) {
			x.push_back(t);
			y.push_back((1-b) * std::pow(b, t));
		}
		matplot::plot(ax1, x, y, strs[i].c_str() )->line_width(2);
		lgd.push_back(("gamma = " + std::to_string(b)).c_str());
		i++;
	}
	matplot::hold(ax1, false);
	matplot::ylim(ax1, {0.0, 0.3});
	matplot::xlabel(ax1, "time");
	matplot::legend(ax1, lgd);
	matplot::show();

	// ----------------------------------------------------
	// Implementation from Scratch
	// ----------------------------------------------------
	double eta = 0.4, gamma = 0.9;

	show_trace_2d( train_2d(20, eta, gamma) );

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

	int64_t num_epochs = 5;
	std::map<std::string, float> hyperparams = {{"lr", 0.01}, {"gamma", 0.9}};

	std::vector<torch::Tensor> states = init_rmsprop_states(t_data.size(1));

	// Initialization
	torch::Tensor w = torch::normal(0.0, 0.01, {t_data.size(1), 1}).requires_grad_(true);
	torch::Tensor b = torch::zeros({1}, torch::requires_grad(true));
	std::vector<torch::Tensor> params = {w, b};

    torch::Tensor loss, t;

    auto start = high_resolution_clock::now();
    std::vector<double> epochs, losses;

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
	        rmsprop(params, states, hyperparams);

	        t_loss += loss.item<float>();
	        b_cnt++;
	    }

		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		std::cout << "Epoch: " << (epoch + 1) << ", loss: " << (t_loss/b_cnt)  << ", duration: " << (duration.count() / 1e6) << " sec.\n";
		epochs.push_back(epoch*1.0);
		losses.push_back((t_loss/b_cnt)*1.0);
	}

	F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	ax1 = F->nexttile();
	matplot::plot(ax1, epochs, losses, "b")->line_width(2);
	matplot::xlabel(ax1, "epoch");
	matplot::ylabel(ax1, "loss");
	matplot::title("RMSProp scratch");
	matplot::show();

	// ------------------------------------------
	// Concise Implementation
	// ------------------------------------------

	auto net = torch::nn::Sequential( torch::nn::Linear(5, 1) );

	// init weight
	for(auto& module : net->modules(false)) { 			// include_self= false
			if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
				torch::nn::init::normal_(M->weight, 0.01);  // torch.nn.init.normal_(m.weight, std=0.01)
		}
	}

	auto optimizer = torch::optim::RMSprop(net->parameters(), torch::optim::RMSpropOptions(0.01).alpha(0.9));
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
		losses.push_back((t_loss/b_cnt)*1.0);
	}

	F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	ax1 = F->nexttile();
	matplot::plot(ax1, epochs, losses, "b")->line_width(2);
	matplot::xlabel(ax1, "epoch");
	matplot::ylabel(ax1, "loss");
	matplot::title(ax1, "RMSprop concise");
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}




