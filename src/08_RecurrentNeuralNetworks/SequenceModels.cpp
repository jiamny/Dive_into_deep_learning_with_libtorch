#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../utils.h"

#include <matplot/matplot.h>
using namespace matplot;

using torch::indexing::Slice;
using torch::indexing::None;

// Function for initializing the weights of the network
void init_weights(torch::nn::Sequential& module) {
	if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get()))
        torch::nn::init::xavier_uniform_(M->weight);
}

// A simple MLP
torch::nn::Sequential get_net(){
    auto net = torch::nn::Sequential(torch::nn::Linear(4, 10), torch::nn::ReLU(), torch::nn::Linear(10, 1));
    init_weights(net);
    return net;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	//===============================================
	// Training
	//===============================================

	/*
	 * To keep things simple we (generate our sequence data by using a sine function with some
	 * additive noise for time steps 1,2,…,1000.)
	 */

	int64_t T = 1000; // # Generate a total of 1000 points
	auto time = torch::arange(1, T + 1).to(torch::kFloat32);
//	std::cout << time.sizes() << "\n" << time << std::endl;
	auto x = torch::sin(0.01 * time) + torch::normal(0, 0.2, {T,}); //(T,));
	std::cout << x.sizes() << "\n" << x << std::endl;

	std::vector<float> yy(x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
	std::vector<float> xx(time.data_ptr<float>(), time.data_ptr<float>() + time.numel());

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::plot(ax1, xx, yy, "b")->line_width(2);
    matplot::xlabel(ax1, "time");
    matplot::ylabel(ax1, "x");
    matplot::show();

	//Next, we need to turn such a sequence into features and labels that our model can train on.
	int tau = 4;

	auto features = torch::zeros({T - tau, tau});
	for(int i = 0; i < tau; i++ ) {
	    features.index({Slice(), i}) = x.index({Slice(i, (T - tau + i))}).clone();
	}

	auto labels = x.index({Slice(tau, T)}).reshape({-1, 1}).clone();
	std::cout << features.sizes() << "\n" << labels.sizes() << "\n";

	int64_t batch_size = 16, n_train = 600;
	// Only the first `n_train` examples are used for training
	auto X = features.index({Slice(0, n_train), Slice()});
	auto y = labels.index({Slice(0, n_train), Slice()});

	std::pair<torch::Tensor, torch::Tensor> data_and_label = {X, y};

	auto dataset = LRdataset(data_and_label)
					   .map(torch::data::transforms::Stack<>());

	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		                   std::move(dataset), batch_size);

	auto net = get_net();
	net->to(device);

	// Square loss
	auto loss = torch::nn::MSELoss();

	/*
	 *  The code below is essentially identical to the training loop in previous sections,
	 *  such as :numref:sec_linear_concise. Thus, we will not delve into much detail.
	 */
	float lr = 0.01;
	int epochs = 10;
	auto trainer = torch::optim::Adam(net->parameters(), lr);

	for( int epoch = 0; epoch < epochs; epoch++ ) {
		float ls = 0.0;

		for(auto & batch : *data_loader ) {
			auto X = batch.data;
			auto y = batch.target;
			X = X.to(device);
			y = y.to(device);

			trainer.zero_grad();
	        auto l = loss(net->forward(X), y);

	        l.backward();
	        trainer.step();

	        ls += l.to(torch::kCPU).item<float>();
		}
		printf("epoch %2i, loss: %.2f\n", (epoch + 1), ls/epochs);
	}

	//===============================================
	// Prediction
	//===============================================

	// predict what happens just in the next time step
	auto onestep_preds = net->forward(features.to(device)).reshape(-1);

	// transfer data from device to CPU
	onestep_preds = onestep_preds.to(torch::kCPU);

	std::cout << onestep_preds << std::endl;
	std::cout << onestep_preds.sizes() << std::endl;

	auto tt = time.index({Slice(tau, T)});
	auto yx = x.index({Slice(tau, T)});

	std::vector<float> x1(tt.data_ptr<float>(), tt.data_ptr<float>() + tt.numel());
	std::vector<float> y1(yx.data_ptr<float>(), yx.data_ptr<float>() + yx.numel());
	std::vector<float> y2(onestep_preds.data_ptr<float>(), onestep_preds.data_ptr<float>() + onestep_preds.numel());

	F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::plot(ax1, x1, y1, "b")->line_width(2);
	matplot::plot(ax1, x1, y2, "r--")->line_width(2);
	matplot::hold(ax1, false);
	matplot::xlabel(ax1, "time");
	matplot::ylabel(ax1, "data");
	matplot::legend(ax1, {"data", "1-step preds"});
	matplot::show();

	// Let us [take a closer look at the difficulties in 𝑘-step-ahead predictions] by computing predictions on
	// the entire sequence for 𝑘=1,4,16,64.
	int max_steps = 64;

	features = torch::zeros({T - tau - max_steps + 1, tau + max_steps});

	// Column `i` (`i` < `tau`) are observations from `x` for time steps from
	// `i + 1` to `i + T - tau - max_steps + 1`
	for( int i = 0;  i < tau; i++ ) {
//		std::cout << features.sizes() << " " << i << " " << (i + T - tau - max_steps + 1) << std::endl;
	    features.index({Slice(), i}) = x.index({Slice(i, (i + T - tau - max_steps + 1))});
	}
	std::cout << features.index({Slice(0,2), Slice(0,6)}) << std::endl;

	std::vector<int> steps = {1, 4, 16, 64};
	// Column `i` (`i` >= `tau`) are the (`i - tau + 1`)-step-ahead predictions for
	// time steps from `i + 1` to `i + T - tau - max_steps + 1`

	std::vector<torch::Tensor> preds; // save n - step preds !!!
	int j = 0;
	for( int i = tau; i < (tau + max_steps); i++ ) {
		auto pt = net->forward(features.index({Slice(), Slice((i - tau), i)}).to(device));

		// transfer data from device to CPU
		pt = pt.to(torch::kCPU);
		if( j < 4 && i == (tau + steps[j] - 1)) {
			preds.push_back(pt.clone());
			j++;
		}
	    features.index({Slice(), i}) = pt.reshape({-1});
//	    std::cout << " -----\n" << features.index({Slice(0,2), Slice(i - tau, i)}) << std::endl;
	}

//	std::cout << features.index({Slice(0, 10), Slice(60,64)}) << std::endl;
//	std::cout << features.sizes() << std::endl;

	std::vector<std::string> colors = {"b", "m", "g", "r."};

	F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::xlim(ax1, {0, 1000});

	std::vector<std::string> lgd;

	for( int i = 0; i < steps.size(); i++ ) {
		auto tt2 = time.index({Slice(tau + steps[i] - 1, T - max_steps + steps[i])});

		std::vector<float> x2(tt2.data_ptr<float>(), tt2.data_ptr<float>() + tt2.numel());

		std::string tlt = std::to_string(steps[i]) + "-step preds";

		// have to reshape() and clone()!!!
		auto yp = features.index({Slice(), (tau + steps[i] - 1)}).reshape({-1,1}).clone(); //preds[i]; // features.index({Slice(), (tau + steps[i] - 1)}).reshape({-1,1});

		//std::cout << "------\n" << preds[i].sizes() << std::endl;
		//std::cout << features.index({Slice(), (tau + steps[i] - 1)}).reshape({-1,1}).sizes() << std::endl;

		std::vector<float> y3(yp.data_ptr<float>(), yp.data_ptr<float>() + yp.numel());

		matplot::plot(ax1, x2, y3, colors[i].c_str());
		lgd.push_back(tlt.c_str());
	}
	matplot::xlabel("time");
	matplot::ylabel("data");
	matplot::legend(lgd);
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}





