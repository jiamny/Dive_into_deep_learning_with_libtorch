
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <atomic>
#include <string>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <tuple>
#include <map>
#include <vector>
#include <functional>
#include <utility> 		// make_pair etc.

#include "../TempHelpFunctions.hpp" // range()
#include "../fashion.h"
#include "../utils.h"

#include <matplot/matplot.h>
using namespace matplot;

using Options = torch::nn::Conv2dOptions;

torch::nn::Sequential net_fn() {
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

    return net;
}

struct SquareRootScheduler {
	float lr;
	SquareRootScheduler(float l) {
		lr = l;
	}
	SquareRootScheduler(){}

	double get_lr(int64_t num_update) {
		return lr * std::pow(num_update + 1.0, -0.5);
	}
};

struct FactorScheduler {
	float base_lr, stop_factor_lr, factor;
	FactorScheduler(float f, float stop_factor_l, float base_l) {
		factor = f;
		stop_factor_lr = stop_factor_l;
		base_lr = base_l;
	}
	FactorScheduler() {}

	double get_lr(int64_t epoch) {
		base_lr = std::max(stop_factor_lr, base_lr * factor);
		return base_lr;
	}
};

struct  CosineScheduler {
	float base_lr, base_lr_orig, final_lr, warmup_begin_lr;
	int64_t max_update, warmup_steps, max_steps;

	CosineScheduler(int64_t max_upd, float base_l, float final_l,
					int64_t warmup_stp=0, float warmup_begin_l=0.0) {
		base_lr_orig = base_l;
		base_lr = base_l;
        max_update = max_upd;
        final_lr = final_l;
        warmup_steps = warmup_stp;
        warmup_begin_lr = warmup_begin_l;
        max_steps = max_upd - warmup_stp;
	}

    double get_warmup_lr(int64_t epoch) {
        auto increase = (base_lr_orig - warmup_begin_lr)
                       * static_cast<float>(max_steps - epoch) / static_cast<float>(warmup_steps);
        return warmup_begin_lr + increase;
    }

    double get_lr( int64_t epoch ) {
        if( epoch < warmup_steps )
            return get_warmup_lr(epoch);
        if( epoch <= max_update )
            base_lr = final_lr + (base_lr_orig - final_lr) * (1 + std::cos(
                M_PI * (epoch - warmup_steps) / max_steps)) / 2.0;
        return base_lr;
    }
};


template<typename Scheduler>
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> train(
		float lr, int64_t num_epochs, torch::Device device, bool lr_sh, Scheduler& SL) {

	auto net = net_fn();

	//
	// Now that we have implemented the model, let us [run an experiment to see how LeNet fares on Fashion-MNIST].
	//
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

	// initialize_weights
	for (auto& module : net->modules(false) ) { //modules(include_self=false))

	    if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
	    //	std::cout << module->name() << std::endl;
	        torch::nn::init::xavier_uniform_( M->weight, 1.0);
	    //  torch::nn::init::constant_(M->bias, 0);
	    }
	}

	net->to(device);
	torch::optim::SGD optimizer = torch::optim::SGD(net->parameters(), lr);
	auto loss = torch::nn::CrossEntropyLoss();

	std::vector<double> train_loss;
	std::vector<double> train_acc;
	std::vector<double> test_acc;
	std::vector<double> xx;

	for( int64_t epoch = 0; epoch < num_epochs; epoch++ ) {

		double epoch_loss = 0.0;
		int64_t epoch_correct = 0;
		int64_t num_train_samples = 0;
		int64_t num_batch = 0;

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

		epoch_correct = 0;
		int64_t num_test_samples = 0;

		net->eval();
		torch::NoGradGuard no_grad;

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
		xx.push_back((epoch + 1)*1.0);

		{
			torch::AutoGradMode enable_grad(true);
			if( lr_sh && epoch % 2 == 0 ) {
				//Using custom defined scheduler
				for(auto& param_group : optimizer.param_groups() ) {
					param_group.options().set_lr(SL.get_lr(epoch));
					std::cout << "New lr: " << param_group.options().get_lr() << "\n";
				}
			}
		}
	}

	return std::make_tuple(train_loss, train_acc, test_acc, xx);
}

void plot_scheduler(std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> dt,
		int64_t num_epochs, std::vector<double> x, std::vector<double> y) {

	std::vector<double> train_loss = std::get<0>(dt);
	std::vector<double> train_acc = std::get<1>(dt);
	std::vector<double> test_acc = std::get<2>(dt);
	std::vector<double> xx = std::get<3>(dt);

	double xL = 1.0;
	if( num_epochs > 10 ) xL = 5.0;

	auto F = figure(true);
	F->size(1400, 400);
	F->add_axes(false);
	F->reactive_mode(false);
	F->position(0, 0);

	subplot(1, 3, 0);
	matplot::plot(x, y, "b")->line_width(2).display_name("scheduler");
	matplot::xlabel("epoch");
	matplot::ylabel("lr");
	matplot::legend({});

	subplot(1, 3, 1);
	matplot::ylim({0.0, 1.5});
	matplot::xlim({xL, num_epochs*1.0});
	matplot::plot(xx, train_loss, "b")->line_width(2).display_name("Train loss");
	matplot::xlabel("epoch");
	matplot::ylabel("loss");
	matplot::legend({});

	subplot(1, 3, 2);
	matplot::plot(xx, train_acc, "g--")->line_width(2).display_name("Train acc");
	matplot::hold(on);
	matplot::plot(xx, test_acc, "r-.")->line_width(2).display_name("Test acc");
	matplot::hold(off);
    matplot::xlabel("epoch");
    matplot::ylabel("acc");
    matplot::legend({});
	F->draw();
    show();
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	int64_t num_epochs = 50;
	float lr = 0.9;

	SquareRootScheduler nl;
	std::tuple<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> dt =
			train(lr, num_epochs, device, false, nl);

	// -------------------------------------
	// SquareRootScheduler
	SquareRootScheduler SL = SquareRootScheduler( lr );

	std::vector<double> x, y;
	for(auto& i : range(num_epochs)) {
		x.push_back(i*1.0);
		y.push_back(SL.get_lr(i)*1.0);
	}

	dt = train(lr, num_epochs, device, true, SL);
	plot_scheduler(dt, num_epochs, x, y);

	// -------------------------------------
	// Factor Scheduler
	lr = 0.3;
	FactorScheduler FL = FactorScheduler(0.9, 1e-2, 2.0);
	x.clear();
	y.clear();
	for(auto& i : range(num_epochs)) {
		x.push_back(i*1.0);
		y.push_back(FL.get_lr(i));
	}

	FL = FactorScheduler(0.9, 1e-2, 2.0);
	dt = train(lr, num_epochs, device, true, FL);

	plot_scheduler(dt, num_epochs, x, y);

	// -------------------------------------
	// Cosine Scheduler

	lr = 0.9;
	CosineScheduler CL = CosineScheduler(14, 0.9, 0.01, 5);
	x.clear();
	y.clear();
	for(auto& i : range(num_epochs)) {
		x.push_back(i*1.0);
		y.push_back(CL.get_lr(i));
	}

	CL = CosineScheduler(14, 0.9, 0.01, 5);
	dt = train(lr, num_epochs, device, true, CL);

	plot_scheduler(dt, num_epochs, x, y);

	std::cout << "Done!\n";
	return 0;
}


