#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../utils.h"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

void train_test(std::pair<torch::Tensor, torch::Tensor> train_data,
		std::pair<torch::Tensor, torch::Tensor> test_data,
		int64_t num_inputs,
		int64_t batch_size,
		float lambd,
		std::vector<double>& train_loss,
		std::vector<double>& test_loss,
		std::vector<double>& xx,
		bool scratch) {
	// -----------------------------------------------------------------------------------------
	// init_params
	// -----------------------------------------------------------------------------------------
	torch::Tensor w = torch::empty({num_inputs, 1}, torch::TensorOptions().requires_grad(true));
	torch::nn::init::normal_(w, 0.0, 1.0);
	torch::Tensor b = torch::zeros(1, torch::TensorOptions().requires_grad(true));

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

				auto X = batch.data;
				auto y = batch.target;

			    auto t = linreg(X, w, b);
			    auto loss = squared_loss(t, y) + lambd * l2_penalty(w);

			    epoch_train_loss += loss.sum().item<float>() * X.size(0);

			    loss.sum().backward();

			    sgd(w, b, lr, X.size(0));  // Update parameters using their gradient

			    num_train_samples += X.size(0);
			}

			//std::cout << (epoch_train_loss/num_train_samples) << std::endl;

			torch::NoGradGuard no_grad;

			for (auto &batch : *test_loader) {

				auto X = batch.data;
				auto y = batch.target;

				auto out = linreg(X, w, b);
				auto loss = squared_loss(out, y) + lambd * l2_penalty(w);

				epoch_test_loss += loss.sum().item<float>() * X.size(0);

				num_test_samples += X.size(0);
			}

			train_loss.push_back(epoch_train_loss/num_train_samples);
			test_loss.push_back(epoch_test_loss/num_test_samples);
			xx.push_back((epoch + 1));
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

		// initialize the weights at random with zero mean and standard deviation 0.01
		if (auto M = dynamic_cast<torch::nn::LinearImpl*>(net.get())) {
			torch::nn::init::normal_(M->weight, 0.0, 0.01);
		}

		auto criterion = torch::nn::MSELoss();
		auto trainer = torch::optim::SGD(net->parameters(), lr);

		for( size_t  epoch = 0; epoch < num_epochs; epoch++ ) {
			torch::AutoGradMode enable_grad(true);

			double epoch_train_loss = 0.0;
			double epoch_test_loss = 0.0;
			int64_t num_train_samples = 0;
			int64_t num_test_samples = 0;

			for (auto &batch : *train_loader) {

				auto X = batch.data;
				auto y = batch.target;

			    auto t = net->forward(X);
			    auto loss = criterion(t, y) + lambd * l2_penalty(w);

			    epoch_train_loss += loss.item<float>() * X.size(0);

			    trainer.zero_grad();
			    loss.sum().backward();
			    trainer.step();

			    num_train_samples += X.size(0);
			}

			torch::NoGradGuard no_grad;

			for (auto &batch : *test_loader) {

				auto X = batch.data;
				auto y = batch.target;

				auto out = net->forward(X);
				auto loss = criterion(out, y) + lambd * l2_penalty(w);

				epoch_test_loss += loss.item<float>() * X.size(0);

				num_test_samples += X.size(0);
			}

			train_loss.push_back(epoch_train_loss/num_train_samples);
			test_loss.push_back(epoch_test_loss/num_test_samples);
			xx.push_back((epoch + 1));
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
	auto test_data = synthetic_data(true_w, true_b, n_test);


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
	float lambd = 0;

	train_test( train_data, test_data, num_inputs, batch_size,
			lambd, train_loss, test_loss, xx, true);

	std::vector<double> train_loss2;
	std::vector<double> test_loss2;
	std::vector<double> xx2;

	lambd = 3;
	train_test( train_data, test_data, num_inputs, batch_size,
				lambd, train_loss2, test_loss2, xx2, true);

	/* ---------------------------------------------------------
	 * Concise Implementation
	 * ---------------------------------------------------------
	 */
	std::vector<double> train_loss3;
	std::vector<double> test_loss3;
	std::vector<double> xx3;
	lambd = 0;

	train_test( train_data, test_data, num_inputs, batch_size,
				lambd, train_loss3, test_loss3, xx3, false);

	std::vector<double> train_loss4;
	std::vector<double> test_loss4;
	std::vector<double> xx4;
	lambd = 3;

	train_test( train_data, test_data, num_inputs, batch_size,
				lambd, train_loss4, test_loss4, xx4, false);

	plt::figure_size(1200, 1000);
//	plt::subplot(2, 2, 1);
	plt::subplot2grid(2, 2, 0, 0, 1, 1);
	plt::named_plot("Train loss", xx, train_loss, "b");
	plt::named_plot("Test loss", xx, test_loss, "c:");
	plt::ylabel("loss");
	plt::xlabel("epoch");
	plt::legend();
	plt::title("Scratch implementation: lambd = 0");

//	plt::subplot(2, 2, 2);
	plt::subplot2grid(2, 2, 0, 1, 1, 1);
	plt::named_plot("Train loss", xx2, train_loss2, "b");
	plt::named_plot("Test loss", xx2, test_loss2, "c:");
	plt::ylabel("loss");
	plt::xlabel("epoch");
	plt::legend();
	plt::title("Scratch implementation: lambd = 3");

//	plt::subplot(2, 2, 3);
	plt::subplot2grid(2, 2, 1, 0, 1, 1);
	plt::named_plot("Train loss", xx3, train_loss3, "b");
	plt::named_plot("Test loss", xx3, test_loss3, "c:");
	plt::ylabel("loss");
	plt::xlabel("epoch");
	plt::legend();
	plt::title("Concise implementation: lambd = 0");

//	plt::subplot(2, 2, 4);
	plt::subplot2grid(2, 2, 1, 1, 1, 1);
	plt::named_plot("Train loss", xx4, train_loss4, "b");
	plt::named_plot("Test loss", xx4, test_loss4, "c:");
	plt::ylabel("loss");
	plt::xlabel("epoch");
	plt::legend();
	plt::title("Concise implementation: lambd = 3");
	plt::show();

	std::cout << "Done!\n";
	return 0;
}



