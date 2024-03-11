#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../utils.h"

#include <matplot/matplot.h>
using namespace matplot;

void train_test(std::pair<torch::Tensor, torch::Tensor> train_data,
		std::pair<torch::Tensor, torch::Tensor> test_data,
		int64_t num_inputs,
		int64_t batch_size,
		double lambd,
		std::vector<double>& train_loss,
		std::vector<double>& test_loss,
		std::vector<double>& xx,
		bool scratch, torch::Device device) {

	// -----------------------------------------------------------------------------------------
	// init_params
	// -----------------------------------------------------------------------------------------
	torch::Tensor w = torch::empty({num_inputs, 1},
						torch::TensorOptions().requires_grad(true)).to(device);
	torch::nn::init::normal_(w, 0.0, 1.0);
	torch::Tensor b = torch::zeros(1,
						torch::TensorOptions().requires_grad(true)).to(device);

	int64_t num_epochs = 100;
	float lr = 0.003;

	if( scratch ) {

		auto dataset = LRdataset(train_data)
						   .map(torch::data::transforms::Stack<>());

		auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
			                   std::move(dataset), batch_size);

		auto tst_dataset = LRdataset(test_data)
						   .map(torch::data::transforms::Stack<>());

		auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
			                   std::move(tst_dataset), batch_size);

		for( size_t  epoch = 0; epoch < num_epochs; epoch++ ) {
			torch::AutoGradMode enable_grad(true);

			double epoch_train_loss = 0.0;
			double epoch_test_loss = 0.0;
			int64_t num_train_samples = 0;
			int64_t num_test_samples = 0;

			for (auto &batch : *train_loader) {

				auto X = batch.data.to(device);
				auto y = batch.target.to(device);

			    auto t = linreg(X, w, b);
			    w.retain_grad();
			    b.retain_grad();

			    auto loss = squared_loss(t, y) + lambd * l2_penalty(w);

			    epoch_train_loss += loss.sum().item<float>() * X.size(0);

			    loss.sum().backward();

			    sgd(w, b, lr, X.size(0));  // Update parameters using their gradient

			    num_train_samples += X.size(0);
			}

			//std::cout << (epoch_train_loss/num_train_samples) << std::endl;

			torch::NoGradGuard no_grad;

			for (auto &batch : *test_loader) {

				auto X = batch.data.to(device);
				auto y = batch.target.to(device);

				auto out = linreg(X, w, b);
				auto loss = squared_loss(out, y) + lambd * l2_penalty(w);

				epoch_test_loss += loss.sum().item<float>() * X.size(0);

				num_test_samples += X.size(0);
			}

			train_loss.push_back(epoch_train_loss*1.0/num_train_samples);
			test_loss.push_back(epoch_test_loss*1.0/num_test_samples);
			xx.push_back((epoch + 1)*1.0);
		}

		std::cout << "Scratch implementation, " << "lambd=" << lambd << ": L2 norm of w: " << torch::norm(w).item<float>() << std::endl;
	} else {
		// load data
		auto dataset = LRdataset(train_data)
						   .map(torch::data::transforms::Stack<>());

		auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
			                   std::move(dataset), batch_size);

		auto tst_dataset = LRdataset(test_data)
						   .map(torch::data::transforms::Stack<>());

		auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
			                   std::move(tst_dataset), batch_size);


		auto net = torch::nn::Sequential(torch::nn::Linear(num_inputs, 1));
		net->to(device);

		// initialize the weights at random with zero mean and standard deviation 0.01
		if (auto M = dynamic_cast<torch::nn::LinearImpl*>(net.get())) {
			torch::nn::init::normal_(M->weight, 0.0, 0.01);
		}

		auto criterion = torch::nn::MSELoss();
		auto trainer = torch::optim::SGD(net->parameters(), lr);

		for( size_t  epoch = 0; epoch < num_epochs; epoch++ ) {
			//torch::AutoGradMode enable_grad(true);
			net->train();

			double epoch_train_loss = 0.0;
			double epoch_test_loss = 0.0;
			int64_t num_train_samples = 0;
			int64_t num_test_samples = 0;

			for (auto &batch : *train_loader) {

				auto X = batch.data.to(device);
				auto y = batch.target.to(device);

			    auto t = net->forward(X);
			    auto loss = criterion(t, y) + lambd * l2_penalty(w);

			    epoch_train_loss += loss.item<float>() * X.size(0);

			    trainer.zero_grad();
			    loss.sum().backward();
			    trainer.step();

			    num_train_samples += X.size(0);
			}

			//torch::AutoGradMode enable_grad2(false);
			torch::NoGradGuard no_grad;
			net->eval();
			for (auto &batch : *test_loader) {

				auto X = batch.data.to(device);
				auto y = batch.target.to(device);

				auto out = net->forward(X);
				auto loss = criterion(out, y) + lambd * l2_penalty(w);

				epoch_test_loss += loss.item<float>() * X.size(0);

				num_test_samples += X.size(0);
			}

			train_loss.push_back(epoch_train_loss/num_train_samples);
			test_loss.push_back(epoch_test_loss/num_test_samples);
			xx.push_back((epoch + 1)*1.0);
		}

		std::cout << "Consice implementation, " << "lambd=" << lambd << ": L2 norm of w: " << torch::norm(w).item<float>() << std::endl;
	}
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	// Norms and Weight Decay
	/*
	 * We can illustrate the benefits of weight decay through a simple synthetic example.
	 */
	int64_t n_train=20, n_test=100, num_inputs=200, batch_size = 5;
	auto true_w = torch::ones({num_inputs, 1}) * 0.01;
	auto true_b = 0.05;

	std::cout << true_w.sizes() << std::endl;

	auto train_data = synthetic_data(true_w, true_b, n_train);
	auto test_data  = synthetic_data(true_w, true_b, n_test);


	// Implementation from Scratch
	/*
	 * First, we will define a function to randomly initialize our model parameters.
	 */
	// init_params
//	auto weights = init_params(num_inputs);
//	auto w = std::move(weights.first);
//	auto b = std::move(weights.second);


	// create optimizer parameters
//	std::vector<torch::Tensor> params = {w, b};
//	std::vector<torch::optim::OptimizerParamGroup> parameters;
//	parameters.push_back(torch::optim::OptimizerParamGroup(params));

//	auto trainer = torch::optim::SGD(parameters, lr);


	std::vector<double> train_loss;
	std::vector<double> test_loss;
	std::vector<double> xx;
	double lambd = 0;

	train_test( train_data, test_data, num_inputs, batch_size,
			lambd, train_loss, test_loss, xx, true, device);

	std::vector<double> train_loss2;
	std::vector<double> test_loss2;
	std::vector<double> xx2;

	lambd = 3;
	train_test( train_data, test_data, num_inputs, batch_size,
				lambd, train_loss2, test_loss2, xx2, true, device);

	/* ---------------------------------------------------------
	 * Concise Implementation
	 * ---------------------------------------------------------
	 */
	std::vector<double> train_loss3;
	std::vector<double> test_loss3;
	std::vector<double> xx3;
	lambd = 0;

	train_test( train_data, test_data, num_inputs, batch_size,
				lambd, train_loss3, test_loss3, xx3, false, device);

	std::vector<double> train_loss4;
	std::vector<double> test_loss4;
	std::vector<double> xx4;
	lambd = 3;

	train_test( train_data, test_data, num_inputs, batch_size,
				lambd, train_loss4, test_loss4, xx4, false, device);

	auto F = figure(true);
	F->size(1200, 1000);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(2, 2);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::plot(ax1, xx, train_loss, "b")->line_width(2);
	matplot::plot(ax1, xx, test_loss, "r:")->line_width(2);
	matplot::hold(ax1, false);
	matplot::xlabel(ax1, "epoch");
	matplot::ylabel(ax1, "loss");
	matplot::title(ax1, "Scratch implementation: lambd=0");
	matplot::legend(ax1, {"Train loss", "Test loss"});

	auto ax2 = F->nexttile();
	matplot::hold(ax2, true);
	matplot::plot(ax2, xx2, train_loss2, "b")->line_width(2);
	matplot::plot(ax2, xx2, test_loss2, "r:")->line_width(2);
	matplot::hold(ax2, false);
	matplot::xlabel(ax2, "epoch");
	matplot::ylabel(ax2, "loss");
	matplot::title(ax2, "Scratch implementation: lambd=3");
	matplot::legend(ax2, {"Train loss", "Test loss"});

	auto ax3 = F->nexttile();
	matplot::hold(ax3, true);
	matplot::plot(ax3, xx3, train_loss3, "b")->line_width(2);
	matplot::plot(ax3, xx3, test_loss3, "r:")->line_width(2);
	matplot::hold(ax3, false);
	matplot::xlabel(ax3, "epoch");
	matplot::ylabel(ax3, "loss");
	matplot::title(ax3, "Concise implementation: lambd=0");
	matplot::legend(ax3, {"Train loss", "Test loss"});

	auto ax4 = F->nexttile();
	matplot::hold(ax4, true);
	matplot::plot(ax4, xx4, train_loss4, "b")->line_width(2);
	matplot::plot(ax4, xx4, test_loss4, "r:")->line_width(2);
	matplot::hold(ax4, false);
	matplot::xlabel(ax4, "epoch");
	matplot::ylabel(ax4, "loss");
	matplot::title(ax4, "Concise implementation: lambd=3");
	matplot::legend(ax4, {"Train loss", "Test loss"});
	F->draw();
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}



