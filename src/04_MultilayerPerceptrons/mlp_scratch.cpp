#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../utils.h"
#include "../fashion.h"

#include <matplot/matplot.h>
using namespace matplot;

using torch::indexing::Slice;
using torch::indexing::None;

// Model
/*
 * Because we are disregarding spatial structure, we reshape each two-dimensional image into a flat vector of length num_inputs.
 * Finally, we (implement our model) with just a few lines of code.
 */

torch::Tensor net(torch::Tensor X, torch::Tensor& W1, torch::Tensor& b1, torch::Tensor& W2, torch::Tensor& b2) {
	//auto output = torch::relu(torch::add(torch::mm(X, W1), b1));  // Here '@' stands for matrix multiplication
	auto output = d2l_relu(torch::add(torch::mm(X, W1), b1));  // Here '@' stands for matrix multiplication
	output = torch::add(torch::mm(output, W2), b2);
	return output;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

//	auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	auto options = torch::TensorOptions().dtype(torch::kDouble).device(device).requires_grad(true);

	// Initializing Model Parameters
	/*
	 * we can think of this as simply a classification dataset with 784 input features and 10 classes. To begin, we will [implement an MLP with one
	 * hidden layer and 256 hidden units.] Note that we can regard both of these quantities as hyperparameters. Typically, we choose layer widths
	 * in powers of 2, which tend to be computationally efficient because of how memory is allocated and addressed in hardware.

	 * Again, we will represent our parameters with several tensors. Note that for every layer, we must keep track of one weight matrix and one bias vector.
	 * As always, we allocate memory for the gradients of the loss with respect to these parameters.
	 */
	int64_t num_inputs = 784, num_outputs=10, num_hiddens = 256;

	torch::Tensor W1 = torch::randn({num_inputs, num_hiddens}, options);
	torch::Tensor b1 = torch::zeros(num_hiddens, options);
	torch::Tensor W2 = torch::randn({num_hiddens, num_outputs}, options);
	torch::Tensor b2 = torch::zeros(num_outputs, options);

	std::cout << "W1:\n" << W1.data().index({Slice(None, 2), Slice(None, 10)}) << std::endl;

	// create optimizer parameters
	std::vector<torch::Tensor> params = {W1, b1, W2, b2};
	std::vector<torch::optim::OptimizerParamGroup> parameters;
	parameters.push_back(torch::optim::OptimizerParamGroup(params));

	int64_t num_epochs = 10;
	float lr = 0.1;
	int64_t batch_size = 256;

	// Activation Function in utils d2l_relu()

	// Loss Function
	auto criterion = torch::nn::CrossEntropyLoss();

	auto trainer = torch::optim::SGD(parameters, lr);


	const std::string FASHION_data_path("./data/fashion/");

	// fashion custom dataset
	auto train_dataset = FASHION(FASHION_data_path, FASHION::Mode::kTrain)
				    		.map(torch::data::transforms::Stack<>());

	auto test_dataset = FASHION(FASHION_data_path, FASHION::Mode::kTest)
			                .map(torch::data::transforms::Stack<>());

	// Number of samples in the training set
	auto num_train_samples = train_dataset.size().value();
	std::cout << "num_train_samples: " << num_train_samples << std::endl;

	// Reading a Minibatch
	// Data loader
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
					         std::move(train_dataset), batch_size);

	// Number of samples in the testset
	auto num_test_samples = test_dataset.size().value();
	std::cout << "num_test_samples: " << num_test_samples << std::endl;

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
											         std::move(test_dataset), batch_size);

/*
//	torch::AutoGradMode enable_grad(true);

	auto batch = *train_loader->begin();
	auto data  = batch.data.view({batch.data.size(0), -1}).to(device);
	auto target = batch.target.to(device);

	//auto data_hat = net(data, W1, b1, W2, b2);

	//X = X.reshape({-1, W1.size(0)});
	auto data_hat = torch::relu(torch::add(torch::mm(data, W1), b1));  // Here '@' stands for matrix multiplication
	data_hat = torch::add(torch::mm(data_hat, W2), b2);
	std::cout << "data_hat.grad_fn:\n" << data_hat.grad_fn() << '\n';

	// auto z = data_hat.mean(); // grad only available for scalar variable
	// z.backward();
	auto loss = criterion(data_hat, target);
	std::cout << loss.item<float>() << std::endl;

	//loss.backward(torch::ones_like(data_hat), true);
	loss.backward();

//	W1.detach();
	std::cout << W1.grad().sizes() << std::endl;
	std::cout << b1.grad().sizes() << std::endl;

	W1 = torch::sub(W1, torch::div(torch::mul(W1.grad() , lr), data.size(0)));
//	b1 = torch::sub(b1, torch::div(torch::mul(b1.grad() , lr), data.size(0)));
//	W2 = torch::sub(W2, torch::div(torch::mul(W2.grad() , lr), data.size(0)));
//	b2 = torch::sub(b2, torch::div(torch::mul(b2.grad() , lr), data.size(0)));

	std::cout << W1.grad().sizes() << std::endl;
//	std::cout << b1.grad().sizes() << std::endl;
//	std::cout << W2.grad().sizes() << std::endl;
//	std::cout << b2.grad().sizes() << std::endl;
//	b1.detach() = b1.detach() - (lr * b1.detach().grad() / data.size(0));
//	W2.detach() = W2.detach() - (lr * W2.detach().grad() / data.size(0) );
//	b2.detach() = b2.detach() - (lr * b2.detach().grad() / data.size(0));
	std::cout << "W1:\n" << W1.data().index({Slice(None, 2), Slice(None, 10)}) << std::endl;
*/

	/*
	 * Training
	 */
	std::vector<double> train_loss;
	std::vector<double> train_acc;
	std::vector<double> test_loss;
	std::vector<double> test_acc;
	std::vector<double> xx;

	for( size_t epoch = 0; epoch < num_epochs; epoch++ ) {

		torch::AutoGradMode enable_grad(true);

		// Initialize running metrics
		double epoch_loss = 0.0;
		int64_t epoch_correct = 0;
		int64_t num_train_samples = 0;

		for(auto &batch : *train_loader) {
			//auto x = batch.data.view({batch.data.size(0), -1}).to(device);
			auto x = batch.data.view({batch.data.size(0), -1}).to(device);
			auto y = batch.target.to(device);

			// ------------------------------------------------------
			// model net
			// ------------------------------------------------------
			//auto y_hat = torch::relu(torch::add(torch::mm(x, W1), b1));  // Here '@' stands for matrix multiplication
			//y_hat = torch::add(torch::mm(y_hat, W2), b2);

			auto y_hat = net(x.to(torch::kDouble), W1, b1, W2, b2);

			auto loss = criterion(y_hat, y); // cross_entropy(y_hat, y);

			// Update running loss
			epoch_loss += loss.item<float>() * x.size(0);

			// Update number of correctly classified samples
			epoch_correct += accuracy( y_hat, y);

			trainer.zero_grad();
			loss.backward();
			trainer.step();

			num_train_samples += x.size(0);
		}

		auto sample_mean_loss = epoch_loss / num_train_samples;
		auto tr_acc = static_cast<double>(epoch_correct) / num_train_samples;

		std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
					            << sample_mean_loss << ", Accuracy: " << tr_acc << '\n';

		//std::cout << w.index({0, Slice()}) << std::endl;

		train_loss.push_back((sample_mean_loss/4.0));
		train_acc.push_back(tr_acc);

		std::cout << "Training finished!\n\n";
		std::cout << "Testing...\n";

		torch::NoGradGuard no_grad;

		double tst_loss = 0.0;
		epoch_correct = 0;
		int64_t num_test_samples = 0;

		for(auto& batch : *test_loader) {
			//auto data = batch.data.view({batch.data.size(0), -1}).to(device);
			auto data = batch.data.view({batch.data.size(0), -1}).to(device);
			auto target = batch.target.to(device);

			// ------------------------------------------------------
			// model net
			// ------------------------------------------------------
			//auto output = torch::relu(torch::add(torch::mm(data, W1), b1));  // Here '@' stands for matrix multiplication
			//output = torch::add(torch::mm(output, W2), b2);
			auto output = net(data.to(torch::kDouble), W1, b1, W2, b2);

			auto loss = criterion(output, target);

			tst_loss += loss.item<float>() * data.size(0);

			epoch_correct += accuracy( output, target );

			num_test_samples += data.size(0);
		}

		std::cout << "Testing finished!\n";

		auto test_accuracy = static_cast<double>(epoch_correct) / num_test_samples;
		auto test_sample_mean_loss = tst_loss / num_test_samples;

		test_loss.push_back((test_sample_mean_loss/4.0));
		test_acc.push_back(test_accuracy);

		std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
		xx.push_back((epoch + 1));
	}

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::ylim(ax1, {0.2, 0.99});
	matplot::plot(ax1, xx, train_loss, "b")->line_width(2);
	matplot::plot(ax1, xx, test_loss, "m-:")->line_width(2);
	matplot::plot(ax1, xx, train_acc, "g--")->line_width(2);
	matplot::plot(ax1, xx, test_acc, "r-.")->line_width(2);
    matplot::hold(ax1, false);
    matplot::xlabel(ax1, "epoch");
    matplot::title(ax1, "Concise implementation");
    matplot::legend(ax1, {"Train loss", "Test loss", "Train acc", "Test acc"});
    matplot::show();

	std::cout << "Done!\n";
	return 0;
}





