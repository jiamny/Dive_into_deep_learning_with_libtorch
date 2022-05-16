#include <unistd.h>
#include <iomanip>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	// ---------------------------------------
	// The Maximum Likelihood Principle
	// ---------------------------------------

	auto theta = torch::arange(0, 1, 0.001);
	auto p = torch::pow(theta, 9) * torch::pow((1 - theta), 4.);

	std::vector<float> x(theta.data_ptr<float>(), theta.data_ptr<float>() + theta.numel());
	std::vector<float> y(p.data_ptr<float>(), p.data_ptr<float>() + p.numel());

	plt::figure_size(500, 400);
	plt::plot(x, y, "b-");
	plt::xlabel("theta");
	plt::ylabel("likelihood");
	plt::show();
	plt::close();

	// ---------------------------------------
	// Numerical Optimization and the Negative Log-Likelihood
	// ---------------------------------------
	// Set up our data
	int n_H = 8675309;
	int n_T = 25624;

	// Initialize our paramteres
	theta = torch::tensor({0.5}).requires_grad_(true);

	// Perform gradient descent
	double lr = 0.00000000001;
	for(int iter = 0; iter < 10; iter++ ) {
	    auto loss = -1*(n_H * torch::log(theta) + n_T * torch::log(1 - theta));
	    loss.backward();

	    torch::NoGradGuard nograd;
	    theta -= lr * theta.grad();

	    theta.grad().zero_();
	}

	// Check output
	std::cout << "theta: " << theta << '\n';
	std::cout << "n_H / (n_H + n_T): " << n_H*1.0 / (n_H + n_T) << '\n';

	std::cout << "Done!\n";
}

