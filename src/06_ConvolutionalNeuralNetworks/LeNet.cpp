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

using Options = torch::nn::Conv2dOptions;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	auto net = torch::nn::Sequential(torch::nn::Conv2d(Options(1, 6, 5).padding(2)),
									 torch::nn::Sigmoid(),
									 torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2).stride(2)),
									 torch::nn::Conv2d(Options(6, 16, 5)),
									 torch::nn::Sigmoid(),
									 torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2).stride(2)),
									 torch::nn::Flatten(),
									 torch::nn::Linear(16 * 5 * 5, 120),
									 torch::nn::Sigmoid(),
									 torch::nn::Linear(120, 84),
									 torch::nn::Sigmoid(),
									 torch::nn::Linear(84, 10));

	// make sure that its operations line up with what we expect from
	auto X = torch::randn({256, 1, 28, 28}).to(torch::kFloat32);
	std::cout << X.sizes() << std::endl;

	std::cout << net << std::endl;

	for(auto& layer : *net.ptr()) {
		//auto layer = net.get()->modules().;
		X = layer.forward(X);
		std::cout << "name: " << layer.ptr()->name() << "output shape: \t" <<  X.sizes() << std::endl;
	}

	/*
	* Now that we have implemented the model, let us [run an experiment to see how LeNet fares on Fashion-MNIST].
	*/
	std::string data_path = "./data/fashion/";
	int64_t batch_size = 256;

	// fashion custom dataset
	auto train_dataset = FASHION(data_path, FASHION::Mode::kTrain)
			    			.map(torch::data::transforms::Stack<>());

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
									         std::move(train_dataset), batch_size);

	auto test_dataset = FASHION(data_path, FASHION::Mode::kTest)
					                .map(torch::data::transforms::Stack<>());

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
						         std::move(test_dataset), batch_size);
	float lr = 0.9;
	int64_t num_epochs = 50;

	// initialize_weights
	for (auto& module : net->modules(false) ) { //modules(include_self=false))

	    if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
	    	std::cout << module->name() << std::endl;
	        torch::nn::init::xavier_uniform_( M->weight, 1.0);
	      //torch::nn::init::constant_(M->bias, 0);
	    }
	    /*
	     else if (
	        auto M = dynamic_cast<torch::nn::BatchNorm2dImpl*>(module.get())) {
	      torch::nn::init::constant_(M->weight, 1);
	      torch::nn::init::constant_(M->bias, 0);
	    } else if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
	      torch::nn::init::normal_(M->weight, 0, 0.01);
	      torch::nn::init::constant_(M->bias, 0);
	    }
	    */
	}

	std::cout << "training on: " <<  device << std::endl;
	net->to(device);
	auto optimizer = torch::optim::SGD(net->parameters(), lr);
	auto loss = torch::nn::CrossEntropyLoss();

	std::vector<float> train_loss;
	std::vector<float> train_acc;
	std::vector<float> test_acc;
	std::vector<float> xx;
/*
	auto batch = *train_loader->begin();
	auto data  = batch.data.to(device);
	auto y  = batch.target.to(device);
	std::cout << "y: " << y << std::endl;

	auto y_hat = net->forward(data);
	std::cout << "y_hat: " << y_hat << std::endl;
	auto l = loss(y_hat, y);
	std::cout << l.item<float>() * data.size(0) << std::endl;
	std::cout << accuracy( y_hat, y) << std::endl;
	optimizer.zero_grad();
	l.backward();
	optimizer.step();
*/

	//    timer, num_batches = d2l.Timer(), len(train_iter)
	for( int64_t epoch = 0; epoch < num_epochs; epoch++ ) {

		double epoch_loss = 0.0;
		int64_t epoch_correct = 0;
		int64_t num_train_samples = 0;
		int64_t num_batch = 0;

		//torch::AutoGradMode enable_grad(true);
		// Sum of training loss, sum of training accuracy, no. of examples
	    net->train(true);
	    for( auto& batch : *train_loader ) {
	    	auto X = batch.data.to(device);
	    	auto y = batch.target.to(device);

	    	auto y_hat = net->forward(X);
	    	auto l = loss(y_hat, y);

	    	epoch_loss += l.item<float>();
	    	epoch_correct += accuracy( y_hat, y);

	    	optimizer.zero_grad();
	    	l.backward();
	    	optimizer.step();

	    	num_train_samples += X.size(0);
	    	num_batch++;
	    }

	    auto sample_mean_loss = epoch_loss / num_batch;
	    auto tr_acc = static_cast<double>(epoch_correct) / num_train_samples;

	    std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
	    					            << sample_mean_loss << ", Accuracy: " << tr_acc << '\n';

	    train_loss.push_back((sample_mean_loss));
	    train_acc.push_back(tr_acc);

		std::cout << "Training finished!\n\n";
		std::cout << "Testing...\n";

		net->eval();
		torch::NoGradGuard no_grad;

		epoch_correct = 0;
		int64_t num_test_samples = 0;

		for(auto& batch : *test_loader) {
			auto data = batch.data.to(device);
			auto target = batch.target.to(device);

			auto output = net->forward(data);

			epoch_correct += accuracy( output, target );
			num_test_samples += data.size(0);
		}

		std::cout << "Testing finished!\n";

		auto test_accuracy = static_cast<double>(epoch_correct) / num_test_samples;

		test_acc.push_back(test_accuracy);

		std::cout << "Testset - Accuracy: " << test_accuracy << '\n';
		xx.push_back((epoch + 1));
	}

	plt::figure_size(800, 600);
	plt::ylim(0.0, 1.0);
	plt::named_plot("Train loss", xx, train_loss, "b");
	plt::named_plot("Train acc", xx, train_acc, "g--");
	plt::named_plot("Test acc", xx, test_acc, "r-.");
	plt::ylabel("loss");
	plt::xlabel("epoch");
	plt::legend();
	plt::show();
	plt::close();

	std::cout << "Done!\n";
	return 0;
}


