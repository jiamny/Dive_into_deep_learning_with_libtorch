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

using torch::indexing::Slice;
using torch::indexing::None;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);


	auto true_w = torch::tensor({2.0, -3.4}, options);
	std::cout << "true_w.len = " << true_w.size(0) << std::endl;
	float true_b = 4.2;
	int64_t num_samples = 1000;

	std::pair<torch::Tensor, torch::Tensor> data_and_label = synthetic_data(true_w, true_b, num_samples);

	auto features = data_and_label.first.clone();
    auto labels = data_and_label.second.clone();

	/*
	 * each row in features consists of a 2-dimensional data example and that each row in labels consists of a 1-dimensional label value (a scalar).]
	 */
    std::cout << "x0 = " << features[0] << std::endl;
    std::cout << "y0 = " << labels[0] << std::endl;

	/*
	 * By generating a scatter plot using the second feature features[:, 1] and labels, we can clearly observe the linear correlation between the two.
	 */

	plt::figure_size(800, 600);
	plt::subplot(1, 1, 1);

	auto x = features.data().index({Slice(), 1});

	std::vector<float> xx(x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
	std::vector<float> yy(labels.data_ptr<float>(), labels.data_ptr<float>() + labels.numel());
	plt::scatter(xx, yy);
	plt::xlabel("x2");
	plt::ylabel("y");
	plt::show();

	/*
	 * Initializing Model Parameters
	 */
    torch::Tensor w = torch::empty({2, 1}, torch::requires_grad(true));
    torch::nn::init::normal_(w, 0, 0.01);
	//auto w = torch::normal(0, 0.01, {2, 1}, torch::requires_grad(true)); // requires_grad=
    torch::Tensor b = torch::zeros(1, torch::requires_grad(true));                // requires_grad=
	std::cout << "w = " << w << std::endl;
	std::cout << "b = " << b << std::endl;

	// implemented in utils.cpp
	/*
	 * Defining the Model
	 *
	 * Defining the Loss Function
	 *
	 * Defining the Optimization Algorithm
	 */
	// --------------------------------------------------------------
	// Training
	// --------------------------------------------------------------
	/*
	 * In each epoch, we will iterate through the entire dataset (using the data_iter function) once passing through every example in the
	 * training dataset (assuming that the number of examples is divisible by the batch size). The number of epochs num_epochs and the learning
	 * rate lr are both hyperparameters, which we set here to 3 and 0.03, respectively. Unfortunately, setting hyperparameters is tricky and
	 * requires some adjustment by trial and error. We elide these details for now but revise them later in :numref:chap_optimization.
	 */

	float lr = 0.05;
	size_t num_epochs = 5;

	// ---- params - hyperparameters
	//std::vector<torch::Tensor> params = {w, b};

	int64_t batch_size = 32;

//	w -= lr * w / batch_size;
//	std::cout << "w2 = " << w << std::endl;

	auto dataset = LRdataset(data_and_label)
				   .map(torch::data::transforms::Stack<>());

	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
	                   std::move(dataset), batch_size);

	for( size_t  epoch = 0; epoch < num_epochs; epoch++ ) {

		for (auto &batch : *data_loader) {

			auto X = batch.data;
			auto y = batch.target;

//			std::cout << X.sizes() << std::endl;
//			std::cout << y.sizes() << std::endl;

	    	auto t = linreg(X, w, b);
	        auto loss = squared_loss(t, y);

	        // Compute gradient on `l` with respect to [`w`, `b`]
	        //std::cout << loss << std::endl;

	        loss.sum(0).backward();

	        sgd(w, b, lr, X.size(0));  // Update parameters using their gradient
	    }

		//with torch.no_grad():
		torch::NoGradGuard no_grad;

		auto train_l = squared_loss(linreg(features, w, b), labels);
	    //std::cout << train_l.mean(0) << std::endl;

	    std::cout << "==========> Epoch " << (epoch + 1) << " loss " << train_l.mean().data().item<float>() << std::endl;
		std::cout << "w = " << w << std::endl;
		std::cout << "b = " << b << std::endl;
	}

	std::cout << "------------------------------------------------------- \n";
	std::cout << "error in estimating w: \n" << (true_w - w.reshape(true_w.sizes())) << std::endl;
	std::cout << "error in estimating b: \n" <<  (true_b - b) << std::endl;

	std::cout << "Done!\n";
	return 0;
}




