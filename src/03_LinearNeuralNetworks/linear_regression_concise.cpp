#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include "../utils.h"

using torch::indexing::Slice;
using torch::indexing::None;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);

	// Generating the Dataset

	torch::Tensor true_w = torch::tensor({2.0, -3.4}, options);
	std::cout << true_w.size(0) << std::endl;
	float true_b = 4.2;
	int64_t num_samples = 3000;

	std::pair<torch::Tensor, torch::Tensor> data_and_label = synthetic_data(true_w, true_b, num_samples);


	// ----------------------------------------------------------------------------
	// Reading the Dataset
	// ----------------------------------------------------------------------------

	int64_t batch_size = 10;

	auto dataset = LRdataset(data_and_label)
				.map(torch::data::transforms::Stack<>());

	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
	        std::move(dataset), batch_size);

	// ---------------------------------------------------------------------------
	// Defining the Model
	// ---------------------------------------------------------------------------

	torch::nn::Linear net(2, 1);

	// --------------------------------------------------------------------------
	// Initializing Model Parameters
	// --------------------------------------------------------------------------
	/*
	 * Here we specify that each weight parameter should be randomly sampled from a normal distribution with mean 0 and standard deviation 0.01.
	 * The bias parameter will be initialized to zero.
	 */
	net->weight.data().normal_(0.0, 0.01);
	net->bias.data().fill_(0);

	// Defining the Loss Function
	/*
	 * The MSELoss class computes the mean squared error, also known as squared 𝐿2 norm
	 */
	auto loss = torch::nn::MSELoss();

	// Defining the Optimization Algorithm
	/*
	 * Minibatch stochastic gradient descent just requires that we set the value lr, which is set to 0.03 here.
	 */
	auto trainer = torch::optim::SGD(net->parameters(), 0.03);

//	auto features = data_and_label.data().index({Slice(None, 2)});
//	auto labels = data_and_label.data().index({Slice(), 2});
	auto features = std::move(data_and_label.first);
    auto labels = std::move(data_and_label.second);

	// Training
	size_t num_epochs = 3;
	for( size_t i =0 ; i < num_epochs; i++ ) {

		for (auto &batch : *data_loader) {
			//std::cout << batch.data()->data.sizes() << '\n';
			//auto x = batch[0].data.index({Slice(None, 2)}); // x
			//auto y = torch::tensor(batch[0].data[2].item<float>());  // y
			auto x = batch.data;
			auto y = batch.target;

			std::cout << x.sizes() << std::endl;
			//std::cout << y.sizes() << std::endl;

			auto output = net->forward(x);
			auto loss = torch::mse_loss(output, y);
			//std::cout << loss << std::endl;

			trainer.zero_grad();
			loss.backward();
			trainer.step();
		}
		auto epoch_output = net->forward(features);
		auto epoch_loss = torch::mse_loss(epoch_output, labels);
		std::cout << "Epoch: " << i << " loss: " << epoch_loss << std::endl;
	}

	// -------------------------------------------------------------------------------------------
	// ompare the model parameters learned by training on finite data and the actual parameters
	// ------------------------------------------------------------------------------------------
	auto w = net->weight.data();
	std::cout << w.sizes() << "\n";
	std::cout << true_w.sizes() << "\n";
	auto w_dif = true_w - w.reshape(true_w.sizes());
	std::cout << "error in estimating w:\n" << w_dif << std::endl;
	auto b = net->bias.data().item<float>();
	std::cout << "error in estimating b: " << (true_b - b) << std::endl;

	std::cout << "Done!\n";
	return 0;
}



