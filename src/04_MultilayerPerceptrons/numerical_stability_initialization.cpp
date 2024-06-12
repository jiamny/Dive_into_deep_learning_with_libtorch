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

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Vanishing and Exploding Gradients

	auto x = torch::arange(-8.0, 8.0, 0.1, torch::requires_grad(true)).to(torch::kDouble);
	auto y = torch::sigmoid(x);
	x.retain_grad();

	y.backward(torch::ones_like(x));

	std::cout << "x.grad: " << x.grad().sizes() << std::endl;

	std::vector<double> xx(x.detach().data_ptr<double>(),
						x.detach().data_ptr<double>() + x.detach().numel());
	std::vector<double> yy(y.detach().data_ptr<double>(),
						y.detach().data_ptr<double>() + y.detach().numel());
	std::vector<double> yy2(x.grad().data().data_ptr<double>(),
						x.grad().data().data_ptr<double>() + x.grad().data().numel());

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::plot(ax1, xx, yy, "b")->line_width(2);
	matplot::plot(ax1, xx, yy2, "r:")->line_width(2);
	matplot::hold(ax1, false);
	matplot::xlabel(ax1, "x");
	matplot::ylabel(ax1, "sigmoid(x)");
	matplot::title(ax1, "Sigmoid");
	matplot::legend(ax1, {"sigmoid", "gradient"});
	matplot::show();

	//Exploding Gradients
	/*
	 * The opposite problem, when gradients explode, can be similarly vexing. To illustrate this a bit better, we draw 100 Gaussian random matrices and
	 * multiply them with some initial matrix. For the scale that we picked (the choice of the variance 𝜎2=1), the matrix product explodes. When this
	 * happens due to the initialization of a deep network, we have no chance of getting a gradient descent optimizer to converge.
	 */

	auto M = torch::normal(0, 1, {4, 4});
	std::cout << "a single matrix \n" << M << std::endl;
	for( int i = 0; i < 100; i++ ){
	    M = torch::mm(M, torch::normal(0, 1, {4, 4}));
	}

	std::cout << "after multiplying 100 matrices \n" << M << std::endl;

	std::cout << "Done!\n";
	return 0;
}




