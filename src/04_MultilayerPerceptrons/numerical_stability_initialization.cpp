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


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Vanishing and Exploding Gradients

	auto x = torch::arange(-8.0, 8.0, 0.1, torch::requires_grad(true));
	auto y = torch::sigmoid(x);
	y.backward(torch::ones_like(x));

	std::cout << "x.grad: " << x.grad().sizes() << std::endl;
	plt::figure_size(800, 600);
	plt::tight_layout();
	plt::subplot(1, 1, 1);
	std::vector<float> xx(x.detach().data_ptr<float>(), x.detach().data_ptr<float>() + x.detach().numel());
	std::vector<float> yy(y.detach().data_ptr<float>(), y.detach().data_ptr<float>() + y.detach().numel());
	std::vector<float> yy2(x.grad().data().data_ptr<float>(), x.grad().data().data_ptr<float>() + x.grad().data().numel());
	plt::named_plot("sigmoid", xx, yy, "b");
	plt::named_plot("gradient", xx, yy2, "g:");
	plt::xlabel("x");
	plt::ylabel("sigmoid(x)");
	plt::title("Sigmoid");
	plt::legend();
	plt::show();

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




