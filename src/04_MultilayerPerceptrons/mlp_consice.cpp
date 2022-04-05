#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <torch/nn.h>

#include "../utils.h"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';


	// Concise Implementation of Multilayer Perceptrons
	/*
	 * As compared with our concise implementation of softmax regression implementation (:numref:sec_softmax_concise), the only difference is that
	 * we add two fully-connected layers (previously, we added one). The first is [our hidden layer], which (contains 256 hidden units and applies
	 * the ReLU activation function). The second is our output layer.
	 */

	auto net = torch::nn::Sequential(torch::nn::Flatten(), torch::nn::Linear(784, 256), torch::nn::ReLU(),
	                    torch::nn::Linear(256, 10));

	/*
	 * init_weights
	 */
	// initialize the weights at random with zero mean and standard deviation 0.01
	if (auto M = dynamic_cast<torch::nn::LinearImpl*>(net.get())) {
		torch::nn::init::normal_(M->weight, 0, 0.01);
		//torch::nn::init::zeros_(M->bias);
	}

	/*
	 * [The training loop] is exactly the same as when we implemented softmax regression. This modularity enables us to separate matters
	 * concerning the model architecture from orthogonal considerations.
	 */

	int64_t num_epochs = 10;
	float lr = 0.1;
	int64_t batch_size = 256;

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

	auto criterion = torch::nn::CrossEntropyLoss();
	auto trainer = torch::optim::SGD(net->parameters(), lr);

	std::vector<double> train_loss;
	std::vector<double> train_acc;
	std::vector<double> test_loss;
	std::vector<double> test_acc;
	std::vector<double> xx;

	for( size_t epoch = 0; epoch < num_epochs; epoch++ ) {
		net->train(true);
		torch::AutoGradMode enable_grad(true);

		 // Initialize running metrics
		double epoch_loss = 0.0;
		int64_t epoch_correct = 0;
        int64_t num_train_samples = 0;

		for(auto &batch : *train_loader) {
			auto x = batch.data.view({batch.data.size(0), -1}).to(device);
			auto y = batch.target.to(device);

			auto y_hat = net->forward(x);
			auto loss = criterion(y_hat, y); //torch::cross_entropy_loss(y_hat, y);
			//std::cout << loss.item<double>() << std::endl;

			// Update running loss
			epoch_loss += loss.item<double>() * x.size(0);

			// Calculate prediction
			auto prediction = y_hat.argmax(1);

			// Update number of correctly classified samples
			epoch_correct += (prediction == y).sum().item<int>(); //prediction.eq(y).sum().item<int64_t>();
			//std::cout << epoch_correct << std::endl;
			trainer.zero_grad();
			loss.backward();
			trainer.step();
			num_train_samples += x.size(0);
		}
		auto sample_mean_loss = (epoch_loss / num_train_samples);
		auto accuracy = static_cast<double>(epoch_correct *1.0 / num_train_samples);

		std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
			            << sample_mean_loss << ", Accuracy: " << accuracy << '\n';

		train_loss.push_back((sample_mean_loss));
		train_acc.push_back(accuracy);

		std::cout << "Training finished!\n\n";
		std::cout << "Testing...\n";

		// Test the model
		net->eval();
		torch::NoGradGuard no_grad;

		double tst_loss = 0.0;
		epoch_correct = 0;
		int64_t num_test_samples = 0;

		for(auto& batch : *test_loader) {
			auto data = batch.data.view({batch.data.size(0), -1}).to(device);
			auto target = batch.target.to(device);

			//std::cout << data.sizes() << std::endl;

			auto output = net->forward(data);

			auto loss = criterion(output, target); //torch::nn::functional::cross_entropy(output, target);
			tst_loss += loss.item<double>() * data.size(0);

			auto prediction = output.argmax(1);
			epoch_correct += prediction.eq(target).sum().item<int64_t>();

			num_test_samples += data.size(0);
		}

		std::cout << "Testing finished!\n";

		auto test_accuracy = static_cast<double>(epoch_correct * 1.0 / num_test_samples);
		auto test_sample_mean_loss = tst_loss / num_test_samples;

		test_loss.push_back((test_sample_mean_loss));
		test_acc.push_back(test_accuracy);

		std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
		xx.push_back((epoch + 1));
	}

	plt::figure_size(800, 600);
	plt::ylim(0.3, 0.9);
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


