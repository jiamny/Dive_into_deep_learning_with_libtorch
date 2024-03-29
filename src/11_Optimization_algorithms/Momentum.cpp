
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
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

#include <matplot/matplot.h>
using namespace matplot;

using namespace std::chrono;

std::tuple<double, double> gd_2d(double x1, double x2, double eta) {
    return std::make_tuple(x1 - eta * 0.2 * x1, x2 - eta * 4 * x2);
}

std::pair<std::vector<double>, std::vector<double>> train_2d( std::function<std::tuple<double, double>(double, double, double)> func,
																int steps, double et) {
    double x1 = -5.0, x2 = -2.0;
    std::vector<double> x, xx; // = [(x1, x2)]
    x.push_back(x1);
    xx.push_back(x2);
    for(int  i = 0; i < steps; i++ ) {
    	std::tie(x1, x2) = func(x1, x2, et);
    	x.push_back(x1);
    	xx.push_back(x2);
    }

    std::cout << "epoch: " << steps << " , x1: " << x1 << " , x2: " << x2 << '\n';

    return std::make_pair(x, xx);
}


std::tuple<double, double, double, double> momentum_2d(double x1, double x2, double v1, double v2, double eta, double beta) {
    v1 = beta * v1 + 0.2 * x1;
    v2 = beta * v2 + 4 * x2;
    return std::make_tuple(x1 - eta * v1, x2 - eta * v2, v1, v2);
}

std::pair<std::vector<double>, std::vector<double>> m_train_2d(int steps, double eta, double beta) {
    double x1 = -5.0, x2 = -2.0, v1 = 0.0, v2 = 0.0;
    std::vector<double> x, xx; // = [(x1, x2)]
    x.push_back(x1);
    xx.push_back(x2);
    for(int  i = 0; i < steps; i++ ) {
    	std::tie(x1, x2, v1, v2) = momentum_2d(x1, x2, v1, v2, eta, beta);
    	x.push_back(x1);
    	xx.push_back(x2);
    }

    std::cout << "epoch: " << steps << " , x1: " << x1 << " , x2: " << x2 << '\n';

    return std::make_pair(x, xx);
}

// Implementation from Scratch
std::vector<torch::Tensor> init_momentum_states(int64_t feature_dim) {
    auto v_w = torch::zeros({feature_dim, 1});
    auto v_b = torch::zeros(1);
    return {v_w, v_b};
}

void sgd_momentum(std::vector<torch::Tensor>& params, std::vector<torch::Tensor>& states,
													 std::map<std::string, float> hyperparams) {

	for( int i = 0; i < states.size(); i++ ) {
        auto p = params[i];
        auto v = states[i];
		torch::NoGradGuard no_grad;
            v = hyperparams["momentum"] * v + p.grad();
            p -= hyperparams["lr"] * v;
        p.grad().data().zero_();
	}
}

void train_momentum_sgd(float lr, float momentum, int64_t num_epochs,
									torch::Tensor t_data, torch::Tensor t_label, int64_t batch_size ) {

	std::map<std::string, float> hyperparams = {{"momentum", momentum},{"lr", lr}};

	std::vector<torch::Tensor> states = init_momentum_states(t_data.size(1));

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
	        sgd_momentum(params, states, hyperparams);

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
	matplot::plot(ax1, epochs, losses, "b")->line_width(2);
    matplot::xlabel(ax1, "epoch");
    matplot::ylabel(ax1, "loss");
    matplot::title(ax1, "Momentum scratch");
    matplot::show();
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);
	double eta = 0.4;

	// -----------------------------------------------------
	// Leaky Averages
	// An Ill-conditioned Problem
	// -----------------------------------------------------
	show_trace_2d( train_2d(&gd_2d, 20, eta) );

	// increase in learning rate from 0.4 to 0.6. Convergence in the x1 direction improves
	// but the overall solution quality is much worse.
	eta = 0.6;
	show_trace_2d( train_2d(&gd_2d, 20, eta) );

	// ------------------------------------------------------
	// The Momentum Method
	// Note that for β=0 we recover regular gradient descent.
	// ------------------------------------------------------
	eta = 0.6;
	double beta = 0.5;
	show_trace_2d( m_train_2d(20, eta, beta) );

	// Halving it to β=0.25 leads to a trajectory that barely converges at all.
	beta = 0.25;
	show_trace_2d( m_train_2d(20, eta, beta) );

	// Effective Sample Weight
	std::vector<double> betas = {0.95, 0.9, 0.6, 0};
	std::vector<std::string> strs = {"b-", "y-", "g-", "r-"};

//	plt::figure_size(700, 500);
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
	for( auto& b : betas ) {
		std::vector<double> x, y;
		for( double t = 0.0; t < 40.0; t += 1.0) {
			x.push_back(t);
			y.push_back(std::pow(b, t));
		}
		matplot::plot(ax1, x, y, strs[i].c_str() )->line_width(2);
		lgd.push_back(("beta = " + std::to_string(b)).c_str());
		i++;
	}
	matplot::hold(ax1, false);
	matplot::xlabel(ax1, "time");
	matplot::legend(ax1, lgd);
	matplot::show();

	// --------------------------------------------------
	// Practical Experiments
	// Implementation from Scratch
	// --------------------------------------------------
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
	float lr = 0.02, momentum = 0.5;
	train_momentum_sgd(lr, momentum, num_epochs, t_data, t_label, batch_size);

	lr = 0.01;
	momentum = 0.9;
	train_momentum_sgd(lr, momentum, num_epochs, t_data, t_label, batch_size);

	lr = 0.005;
	momentum = 0.9;
	train_momentum_sgd(lr, momentum, num_epochs, t_data, t_label, batch_size);

	// ------------------------------------------------
	// Concise Implementation
	// ------------------------------------------------
	auto net = torch::nn::Sequential( torch::nn::Linear(5, 1) );

	// init weight
	for(auto& module : net->modules(false)) { 			// include_self= false
		if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
			torch::nn::init::normal_(M->weight, 0.01);  // torch.nn.init.normal_(m.weight, std=0.01)
		}
	}

	auto optimizer = torch::optim::SGD(net->parameters(), torch::optim::SGDOptions(lr).momentum(momentum));
	auto loss = torch::nn::MSELoss(torch::nn::MSELossOptions(torch::kNone));

	auto start = high_resolution_clock::now();
	std::vector<float> epochs, losses;

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
			auto l = loss(out, y).mean();
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
	matplot::title("Momentum concise");
	matplot::show();

	// Theoretical Analysis
	std::vector<double> lambdas = {0.1, 1.0, 10.0, 19.0};
	double leta = 0.1;
	std::vector<std::string> lstrs = {"b-", "y-", "g-", "r-"};

	F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	lgd.clear();
	ax1 = F->nexttile();
	matplot::hold(ax1, true);

	i = 0;
	for( auto& l : lambdas ) {
		std::vector<double> x, y;
		for( double t = 0.0; t < 20.0; t += 1.0) {
			x.push_back(t);
			y.push_back(std::pow(1 - leta * l, t));
		}
		matplot::plot(ax1, x, y, lstrs[i].c_str() )->line_width(2);
		lgd.push_back(("lambda = " + std::to_string(l)).c_str());
		i++;
	}
	matplot::hold(ax1, false);
	matplot::xlabel(ax1, "time");
	matplot::legend(ax1, lgd);
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}








