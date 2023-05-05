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

#include <matplot/matplot.h>
using namespace matplot;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	torch::manual_seed(123);

	// ---------------------------------------
	// The Maximum Likelihood Principle
	// ---------------------------------------

	auto theta = torch::arange(0, 1, 0.001).to(torch::kDouble);
	auto p = torch::pow(theta, 9) * torch::pow((1 - theta), 4.).to(torch::kDouble);

	std::vector<double> x(theta.data_ptr<double>(), theta.data_ptr<double>() + theta.numel());
	std::vector<double> y(p.data_ptr<double>(), p.data_ptr<double>() + p.numel());

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::plot(ax1, x, y, "b")->line_width(2);
    matplot::xlabel(ax1, "theta");
    matplot::ylabel(ax1, "likelihood");
    matplot::show();

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

