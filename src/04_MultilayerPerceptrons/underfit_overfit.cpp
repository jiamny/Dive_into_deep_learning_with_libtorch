#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <cmath>

#include "../utils.h"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;


void train_ana_test(torch::Tensor train_features, torch::Tensor train_labels,
		torch::Tensor test_features, torch::Tensor test_labels,
		std::vector<float> &train_loss, std::vector<float> &test_loss, std::vector<float> &xx) {

	int64_t num_epochs=400;
	auto loss = torch::nn::MSELoss();

	auto input_shape = train_features.size(-1);
	//std::cout << input_shape << std::endl;

	//# Switch off the bias since we already catered for it in the polynomial
	//# features
	//auto net = torch::nn::Sequential(torch::nn::Linear(torch::nn::LinearOptions(input_shape, 1).bias(false)));
	auto net = torch::nn::Linear(torch::nn::LinearOptions(input_shape, 1).bias(false));

	auto batch_size = std::min(10, static_cast<int>(train_labels.size(0)));

	auto train_data = LRdataset(train_features, train_labels)
						   .map(torch::data::transforms::Stack<>());
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		                   std::move(train_data), batch_size);

	auto test_data = LRdataset(test_features, test_labels)
							   .map(torch::data::transforms::Stack<>());
	auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
			                   std::move(test_data), batch_size);


	auto trainer = torch::optim::SGD(net->parameters(), 0.01);

	for( int64_t epoch = 0; epoch < num_epochs; epoch++ ) {
		//Evaluate the loss of a model on the given dataset.
		//Sum of losses, no. of examples

		net->train(true);
		torch::AutoGradMode enable_grad(true);

		float tot_loss = 0.0;
		int64_t tot_samples = 0;

		for( auto& batch : *data_loader ) {
			auto X = batch.data;
			auto y = batch.target;

			auto out = net->forward(X);
			y = y.reshape(out.sizes());
			auto l = loss(out, y);

			tot_loss += l.sum().item<float>();
			tot_samples += l.numel();

			trainer.zero_grad();
			l.sum().backward();
			trainer.step();
		}

		train_loss.push_back(tot_loss / tot_samples);

		// Test the model
		net->eval();
		torch::NoGradGuard no_grad;

		tot_loss = 0.0;
		tot_samples = 0;

		for( auto& batch : *test_loader ) {
			auto X = batch.data;
			auto y = batch.target;

			auto out = net->forward(X);
			y = y.reshape(out.sizes());
			auto l = loss(out, y);

			tot_loss += l.sum().item<float>();
			tot_samples += l.numel();
		}

		test_loss.push_back(tot_loss / tot_samples);

		if( epoch == 0 || (epoch + 1) % 50 == 0 ) {
			std::cout << "weight: " << net->weight.data() << std::endl;
		}

		xx.push_back(epoch);
	}
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	// Underfitting or Overfitting
	/*
	 * Generating the Dataset
	 */

	int64_t max_degree = 20;    		// Maximum degree of the polynomial
	int64_t n_train=100, n_test = 100;  // Training and test dataset sizes
	auto true_w = torch::zeros(max_degree);       //  Allocate lots of empty space // , torch::TensorOptions().requires_grad(true)
	true_w[0] = 5;
	true_w[1] = 1.2;
	true_w[2] = -3.4;
	true_w[3] = 5.6;

	//auto features = torch::random.normal(size=(n_train + n_test, 1))

	torch::Tensor features = torch::empty({n_train + n_test, 1}); // , torch::TensorOptions().requires_grad(true)
	torch::nn::init::normal_(features, 0.0, 1.0);

	std::cout << features[1] << std::endl;
	std::vector<float> ff;
	for( int i = 0; i < features.size(0); i++ ) ff.push_back(features[i].item<float>());
	std::random_shuffle(ff.begin(), ff.end());
	for( int i = 0; i < features.size(0); i++ ) features[i] = ff[i];

	std::cout << features.sizes() << std::endl;

	auto poly_features = torch::pow(features, torch::arange(max_degree).reshape({1, -1}));
	std::cout << poly_features.sizes() << std::endl;

	for(int i = 0; i < max_degree; i++ ) {
		//std::cout << std::tgamma(i + 1) << std::endl;
		poly_features.index({Slice(), i}) /= std::tgamma(i + 1);  // `gamma(n)` = (n-1)!
		//std::cout << poly_features.index({Slice(), i}) << std::endl;
	}
	std::cout << poly_features.sizes() << " " << true_w.sizes() << std::endl;

	auto labels = torch::mm(poly_features, true_w.reshape({-1, 1}));
	labels += torch::normal(0, 0.1, labels.sizes());

	std::cout << "features[:2]: \n" << features.index({Slice(None, 2)}) << std::endl;
	std::cout << "poly_features[:2, :]: \n" << poly_features.index({Slice(None, 2), Slice()}) << std::endl;
	std::cout << "labels[:2]: \n" << labels.index({Slice(None, 2)}) << std::endl;

	// -------------------------------------------------------------------------
	// Training and Testing the Model
	std::vector<float> train_loss1;
	std::vector<float> test_loss1;
	std::vector<float> xx1;

	auto train_features =  poly_features.index({Slice(None, n_train), Slice(None, 4)});
	auto train_labels = labels.index({Slice(None, n_train)});

	auto test_features =  poly_features.index({Slice(n_train, None), Slice(None, 4)});
	auto test_labels = labels.index({Slice(n_train, None)});

	train_ana_test(train_features, train_labels, test_features, test_labels, train_loss1, test_loss1, xx1);

	// Linear Function Fitting (Underfitting)
	std::vector<float> train_loss2;
	std::vector<float> test_loss2;
	std::vector<float> xx2;

	train_features =  poly_features.index({Slice(None, n_train), Slice(None, 2)});
	train_labels = labels.index({Slice(None, n_train)});

	test_features =  poly_features.index({Slice(n_train, None), Slice(None, 2)});
	test_labels = labels.index({Slice(n_train, None)});

	train_ana_test(train_features, train_labels, test_features, test_labels, train_loss2, test_loss2, xx2);

	// Higher-Order Polynomial Function Fitting (Overfitting)
	std::vector<float> train_loss3;
	std::vector<float> test_loss3;
	std::vector<float> xx3;

	train_features =  poly_features.index({Slice(None, n_train), Slice()});
	train_labels = labels.index({Slice(None, n_train)});

	test_features =  poly_features.index({Slice(n_train, None), Slice()});
	test_labels = labels.index({Slice(n_train, None)});

	train_ana_test(train_features, train_labels, test_features, test_labels, train_loss3, test_loss3, xx3);

	plt::figure_size(1500, 500);
//	plt::subplot(1, 3, 1);
	plt::subplot2grid(1, 3, 0, 0, 1, 1);
	plt::named_plot("Train loss", xx1, train_loss1, "b");
	plt::named_plot("Test loss", xx1, test_loss1, "c:");
	plt::ylabel("loss");
	plt::xlabel("epoch");
	plt::title("Third-Order Polynomial Fitting");
	plt::legend();

//	plt::subplot(1, 3, 2);
	plt::subplot2grid(1, 3, 0, 1, 1, 1);
	plt::named_plot("Train loss", xx2, train_loss2, "b");
	plt::named_plot("Test loss", xx2, test_loss2, "c:");
	plt::ylabel("loss");
	plt::xlabel("epoch");
	plt::title("Linear Function Fitting (Underfitting)");
	plt::legend();

//	plt::subplot(1, 3, 3);
	plt::subplot2grid(1, 3, 0, 2, 1, 1);
	plt::named_plot("Train loss", xx3, train_loss3, "b");
	plt::named_plot("Test loss", xx3, test_loss3, "c:");
	plt::ylabel("loss");
	plt::xlabel("epoch");
	plt::title("Higher-Order Fitting (Overfitting)");
	plt::legend();
	plt::show();
	plt::close();

	std::cout << "Done!\n";
	return 0;
}



