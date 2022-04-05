#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../fashion.h"
#include "../utils.h"

#include "../matplotlibcpp.h"

namespace plt = matplotlibcpp;

using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;


/*
 * implement the softmax regression model
 */
torch::Tensor net(torch::Tensor X, torch::Tensor& W, torch::Tensor& b) {
	return softmax(torch::mm(X.view({X.size(0), -1}), W) + b); // torch::matmul vs torch::mm .reshape({-1, W.size(0)})
}

torch::Tensor cross_entropy(torch::Tensor y_hat, torch::Tensor y){
    	    //return -torch.log(y_hat[range(len(y_hat)), y])
	std::vector<int64_t> index1;
	std::vector<int64_t> index2;

	for( int64_t i = 0; i < y_hat.size(0); i++ )
		index1.push_back(i);

	for( int64_t i = 0; i < y.size(0); i++ )
		index2.push_back(y.index({i}).item<int64_t>());

	std::vector<double> slt_data;
	for( int64_t r = 0; r < index1.size(); r++ ) {
		slt_data.push_back(y_hat.index({index1[r], index2[r]}).item<double>());
	}

	torch::Tensor out =  torch::from_blob(slt_data.data(), { static_cast<int64_t>(slt_data.size()) }, dtype(torch::kDouble));
	out.requires_grad_(true);

	return -torch::log(out);
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	/*
	* Let us stick with the Fashion-MNIST dataset and keep the batch size at 256 as in :numref:sec_softmax_scratch.
	*/
	const std::string FASHION_data_path = "./data/fashion/";
	int64_t batch_size = 32;
/*
	   auto train_dataset = torch::data::datasets::MNIST(FASHION_data_path, torch::data::datasets::MNIST::Mode::kTrain)
	        .map(torch::data::transforms::Stack<>());

	    // Number of samples in the training set
	    auto num_train_samples = train_dataset.size().value();

	    auto test_dataset = torch::data::datasets::MNIST(FASHION_data_path, torch::data::datasets::MNIST::Mode::kTest)
	        .map(torch::data::transforms::Stack<>());

	    // Number of samples in the testset
	    auto num_test_samples = test_dataset.size().value();

	    // Data loaders
	    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
	        std::move(train_dataset), batch_size);

	    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
	        std::move(test_dataset), batch_size);
*/



	// fashion custom dataset
	auto train_dataset = FASHION(FASHION_data_path, FASHION::Mode::kTrain)
					    			.map(torch::data::transforms::Stack<>());

	// Number of samples in the training set
//	auto num_train_samples = train_dataset.size().value();

//	std::cout << "num_train_samples: " << num_train_samples << std::endl;

	// Reading a Minibatch
	// Data loader
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
						         std::move(train_dataset), batch_size);

	auto test_dataset = FASHION(FASHION_data_path, FASHION::Mode::kTest)
				                .map(torch::data::transforms::Stack<>());

	// Number of samples in the testset
//	auto num_test_samples = test_dataset.size().value();

//	std::cout << "num_test_samples: " << num_test_samples << std::endl;

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
										         std::move(test_dataset), batch_size);

	/*
	 * Initializing Model Parameters
	 */
	int64_t num_inputs = 784;
	int64_t num_outputs = 10;

	torch::Tensor w = torch::empty({num_inputs, num_outputs}, torch::requires_grad(true));
	torch::nn::init::normal_(w, 0, 0.01);

	torch::Tensor b = torch::zeros(num_outputs, torch::requires_grad(true));
	std::cout << "w = " << w.index({0, Slice()}) << std::endl;
	std::cout << "b = " << b[0] << std::endl;

	/*
	 * Defining the Softmax Operation
	 */
	auto X = torch::tensor({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}, dtype(torch::kDouble));
	std::cout << "X.sum(0) = " << X.sum(0, true) << std::endl;
	std::cout << "X.sum(1) = " << X.sum(1, true) << std::endl;

	/*
	 * As you can see, for any random input, [we turn each element into a non-negative number. Moreover, each row sums up to 1,] as is required for a probability.
	 */
	X = torch::normal(0, 1, {2, 5});
	auto X_prob = softmax(X);
	std::cout << "X_prob = " <<	X_prob << std::endl;
	std::cout << "X_prob.sum(1) = \n" <<	X_prob.sum(1) << std::endl;

	/*
	 * Defining the Mode in net() function
	 */

	// Defining the Loss Function
	auto y = torch::tensor({0, 2}).to(torch::kLong);
	std::cout << y.data() << std::endl;
	auto y_hat = torch::tensor({{0.1, 0.3, 0.6}, {0.3, 0.2, 0.5}}, torch::requires_grad(true)).to(torch::kDouble);
	int64_t array1[2] = {0,1};
	int64_t array2[2] = {y.index({0}).item<int64_t>(), y.index({1}).item<int64_t>()};
	//torch::Tensor idx = torch::tensor({{array1[0], array2[0]}, {array1[1], array2[1]}});
	//std::cout << idx << std::endl;

	std::cout << torch::tensor({y_hat.index({array1[0], array2[0]}).item<double>(), y_hat.index({array1[1], array2[1]}).item<double>()}) << std::endl;

	/*
	 * Now we can (implement the cross-entropy loss function) efficiently with just one line of code.
	 */
	printf("-----------------------------1\n");
	auto out = cross_entropy(y_hat, y);
	std::cout << out << std::endl;
	std::cout << out.sum(0).item<double>() << std::endl;
	out.sum().backward();
	std::cout << y_hat.grad_fn() << std::endl;
	printf("-----------------------------2\n");

	torch::nn::CrossEntropyLoss criterion;

	/*
	 * Classification Accuracy
	 */
	int64_t cmp = accuracy( y_hat,  y );

	std::cout << static_cast<float>((cmp * 1.0) / y.size(0)) << std::endl;

	/*
	 * Training
	 */
	size_t num_epochs = 10;
	std::vector<double> train_loss;
	std::vector<double> train_acc;
	std::vector<double> test_loss;
	std::vector<double> test_acc;
	std::vector<double> xx;
	float lr = 0.2;
/*
	auto batch = *train_loader->begin();
	auto data  = batch.data.to(device);
	auto target = batch.target.to(device);

	auto data_hat = net(data, w, b);

	auto loss = criterion(data_hat, target);

	loss.backward();

	sgd(w, b,  lr, batch_size);

	std::cout << loss << std::endl;
	std::cout << loss.item<double>() << std::endl;
	std::cout << accuracy(data_hat, target) << std::endl;
	std::cout << target.numel() << std::endl;
*/

	for( size_t epoch = 0; epoch < num_epochs; epoch++ ) {

		torch::AutoGradMode enable_grad(true);

		// Initialize running metrics
		double epoch_loss = 0.0;
		int64_t epoch_correct = 0;
		int64_t num_train_samples = 0;

		for(auto &batch : *train_loader) {
			//auto x = batch.data.view({batch.data.size(0), -1}).to(device);
			auto x = batch.data.to(device);
			auto y = batch.target.to(device);

			auto y_hat = net(x, w, b).to(device);
			//std::cout << y_hat << std::endl;
			//std::cout << y << std::endl;
			auto loss = criterion(y_hat, y); // cross_entropy(y_hat, y);

			//std::cout << loss.item<double>() << std::endl;
			// Update running loss
			epoch_loss += loss.item<double>() * x.size(0);

			// Update number of correctly classified samples
			epoch_correct += accuracy( y_hat, y);

			loss.backward();

			sgd(w, b, lr, x.size(0));  // Update parameters using their gradient

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
			auto data = batch.data.to(device);
			auto target = batch.target.to(device);

			auto output = net(data, w, b);

			auto loss = criterion(output, target);
			tst_loss += loss.item<double>() * data.size(0);

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

	plt::figure_size(800, 600);
	plt::ylim(0.2, 0.9);
	plt::named_plot("Train loss", xx, train_loss, "b");
	plt::named_plot("Test loss", xx, test_loss, "c:");
	plt::named_plot("Train acc", xx, train_acc, "g--");
	plt::named_plot("Test acc", xx, test_acc, "r-.");
	plt::xlabel("epoch");
	plt::legend();
	plt::show();

	std::cout << "Done!\n";
	return 0;
}




