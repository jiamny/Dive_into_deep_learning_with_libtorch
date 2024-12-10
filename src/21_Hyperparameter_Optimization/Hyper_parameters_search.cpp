#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include <random>
#include <cmath>
#include "../utils.h"
#include "../fashion.h"
#include "../TempHelpFunctions.hpp"

#include <matplot/matplot.h>
using namespace matplot;

using Options = torch::nn::Conv2dOptions;

std::vector<float> loguniform_sample(float a, float b, int t = 1) {

	std::vector<float> v;
    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> distrib(a, b);

    float q = std::log(b) - std::log(a);

    for(auto& m : range(t)) {
        v.push_back(std::exp(std::log(a) + distrib(gen) * (q)));
    }

    return v;
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	torch::manual_seed(1000);
	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Create The LeNet-5 model and training model\n";
	std::cout << "// --------------------------------------------------\n";

	float a = 0.01, b = 1.0;
	int num_iterations = 6;
	int64_t num_epochs = 20;
	std::vector<float> validation_errors;
	std::vector<float> learning_rates;
	std::vector<long> batch_sizes;

	auto F = figure(true);
	F->size(1200, 800);
	F->add_axes(false);
	F->reactive_mode(false);
	F->position(0, 0);

	for(auto& ite : range(num_iterations, 0)) {

		std::string data_path = "./data/fashion/";
		int64_t batch_size = torch::randint(32, 256, {1}).data().item<long>();
		float lr = loguniform_sample(a, b)[0];

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


		// Now that we have implemented the model, let us [run an experiment to see how LeNet fares on Fashion-MNIST].

		// fashion custom dataset
		auto train_dataset = FASHION(data_path, FASHION::Mode::kTrain)
				    			.map(torch::data::transforms::Stack<>());

		auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
										         std::move(train_dataset), batch_size);

		auto test_dataset = FASHION(data_path, FASHION::Mode::kTest)
						                .map(torch::data::transforms::Stack<>());

		auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
							         std::move(test_dataset), batch_size);

		// initialize_weights
		for (auto& module : net->modules(false) ) {

		    if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
		    	//std::cout << module->name() << std::endl;
		        torch::nn::init::xavier_uniform_( M->weight, 1.0);
		    }
		}


		net->to(device);

		auto optimizer = torch::optim::SGD(net->parameters(), lr);
		auto loss = torch::nn::CrossEntropyLoss();

		std::vector<double> train_loss;
		std::vector<double> train_acc;
		std::vector<double> valid_acc;
		std::vector<double> valid_loss;
		std::vector<double> xx;

		double valid_err = 0.;
		double tr_ls = 0.;
		double val_ls = 0.;
		double val_acc = 0.;

		for( int64_t epoch = 0; epoch < num_epochs; epoch++ ) {

			double epoch_loss = 0.0;
			int64_t epoch_correct = 0;
			int64_t num_train_samples = 0;
			int64_t num_batch = 0;

			// Sum of training loss, sum of training accuracy, no. of examples
		    net->train(true);
		    torch::AutoGradMode enable_grad(true);

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
		    tr_ls += sample_mean_loss;

		    train_loss.push_back((sample_mean_loss*1.0));
		    train_acc.push_back(tr_acc);

			net->eval();
			torch::NoGradGuard no_grad;

			epoch_correct = 0;
			int64_t num_test_samples = 0;
			double tst_loss = 0.;
			num_batch = 0;
			for(auto& batch : *test_loader) {
				auto data = batch.data.to(device);
				auto target = batch.target.to(device);

				auto output = net->forward(data);
				auto l = loss(output, target);
				tst_loss += l.item<float>();

				epoch_correct += accuracy( output, target );
				num_test_samples += data.size(0);
				num_batch++;
			}

			auto test_accuracy = static_cast<double>(epoch_correct) / num_test_samples;
			valid_loss.push_back(tst_loss / num_batch);
			valid_acc.push_back(test_accuracy);
			val_ls += (tst_loss / num_batch);
			val_acc += test_accuracy;
			valid_err += (1.0 - test_accuracy);

			xx.push_back((epoch + 1));
		}

		validation_errors.push_back( valid_err / num_epochs);
		learning_rates.push_back(lr);
		batch_sizes.push_back(batch_size);

		auto ax = subplot(2, 3, ite);
		ax->hold(true);
		matplot::plot(ax, xx, train_loss, "-")->line_width(2).display_name("train loss");
		matplot::plot(ax, xx, valid_loss, "--")->line_width(2).display_name("valid loss");
		matplot::plot(ax, xx, train_acc, "-")->line_width(2).display_name("train acc");
		matplot::plot(ax, xx, valid_acc, "--")->line_width(2).display_name("valid acc");
		matplot::xlabel(ax, "epoch");
		matplot::legend(ax, {});
		matplot::title(ax, "lr: " + std::to_string(lr) + ", batchSize: " + std::to_string(batch_size));

		std::cout << "lr = " << lr << ", batch_size = " << batch_size << '\n';
		std::cout << "Avg. train_loss: " << (tr_ls / num_epochs) << ", Avg. valid_loss: "
				  << (val_ls / num_epochs) << ", Avg. val_accuracy: " << (val_acc / num_epochs) << '\n';
	}
	matplot::show();

	printf("%10s %20s %10s\n", "batch_size", "learning_rate", "error");
	for(auto& i : range( num_iterations, 0 )) {
		printf("%10ld %20.4f %10.4f\n", batch_sizes[i], learning_rates[i], validation_errors[i]);
	}

	std::cout << "Done!\n";
}




