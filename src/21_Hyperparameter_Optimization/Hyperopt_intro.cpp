#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include <random>
#include <cmath>
#include "../fashion.h"

#include <matplot/matplot.h>
using namespace matplot;



int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	torch::manual_seed(1000);
	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Load data\n";
	std::cout << "// --------------------------------------------------\n";

	std::string data_path = "./data/fashion/";
	int64_t batch_size = 32;

	// fashion custom dataset
	auto train_dataset = FASHION(data_path, FASHION::Mode::kTrain)
			    			.map(torch::data::transforms::Stack<>());

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
									         std::move(train_dataset), batch_size);

	auto validation_dataset = FASHION(data_path, FASHION::Mode::kTest)
					                .map(torch::data::transforms::Stack<>());

	auto validation_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
						         std::move(validation_dataset), batch_size);

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Create softmax classification model and training model\n";
	std::cout << "// --------------------------------------------------\n";


	int num_iterations = 5;
	size_t num_epochs = 10;
	std::vector<float> validation_errors;
	std::vector<float> learning_rates;

	for(int ite = 0; ite < num_iterations; ite++ ) {
		/*
		* Train a model
		*/
		torch::nn::Linear net(784, 10);
		net->to(device);

		// initialize the weights at random with zero mean and standard deviation 0.01
		if (auto M = dynamic_cast<torch::nn::LinearImpl*>(net.get())) {
		   torch::nn::init::normal_(M->weight, 0, 0.01);
		}

		torch::nn::CrossEntropyLoss criterion;

		float lr = torch::rand({1}).data().item<float>() * 1e-2;

		auto trainer = torch::optim::SGD(net->parameters(), lr);

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

				// Update running loss
				epoch_loss += loss.item<double>() * x.size(0);

				// Calculate prediction
				auto prediction = y_hat.argmax(1);

				// Update number of correctly classified samples
				epoch_correct += (prediction == y).sum().item<int>(); //prediction.eq(y).sum().item<int64_t>();
				trainer.zero_grad();
				loss.backward();
				trainer.step();
				num_train_samples += x.size(0);
			}

			auto sample_mean_loss = (epoch_loss / num_train_samples);
			auto accuracy = static_cast<double>(epoch_correct *1.0 / num_train_samples);

			std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
				            << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
		}

		std::cout << "Validation...\n";
		// Test the model
		net->eval();
		torch::NoGradGuard no_grad;

		double tst_loss = 0.0;
		int64_t epoch_correct = 0;
		int64_t num_test_samples = 0;

		for(auto& batch : *validation_loader) {
			auto data = batch.data.view({batch.data.size(0), -1}).to(device);
			auto target = batch.target.to(device);

			auto output = net->forward(data);

			auto loss = criterion(output, target); //torch::nn::functional::cross_entropy(output, target);

			auto prediction = output.argmax(1);
			epoch_correct += prediction.eq(target).sum().item<int64_t>();

			num_test_samples += data.size(0);
		}

		std::cout << "Validation finished!\n";

		auto error = 1.0 - static_cast<double>(epoch_correct * 1.0 / num_test_samples);

		validation_errors.push_back(error);
		learning_rates.push_back(lr);

		std::cout << "Validation error: " << error << '\n';

	}

	int best_idx = 0;
	float min_er = 10000.0;
	for(int i = 0; i < validation_errors.size(); i++) {
		if( validation_errors[i] < min_er ) {
			min_er = validation_errors[i];
			best_idx = i;
		}
	}

	std::cout << "Optimal learning rate = " << learning_rates[best_idx] << '\n';

	std::cout << "Done!\n";
}

