
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <cstdio>
#include <cmath>
#include <map>

#include "../matplotlibcpp.h"

using namespace torch::autograd;
namespace plt = matplotlibcpp;

/*
 * To illustrate derivatives, let us experiment with an example. (Define 𝑢=𝑓(𝑥)=3𝑥2−4𝑥.)
 */
double f( double x ) {
    return( 3 * std::pow(x, 2) - 4 * x);
}

/*
 * [By setting 𝑥=1 and letting ℎ approach 0, the numerical result of 𝑓(𝑥+ℎ)−𝑓(𝑥)ℎ] in :eqref:eq_derivative (approaches 2.) Though this experiment is not a
 * mathematical proof, we will see later that the derivative 𝑢′ is 2 when 𝑥=1.
 */

double numerical_lim(double x, double h) {
    return ((f(x + h) - f(x)) / h);
}


int main() {

	auto options = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU);

	double h = 0.1;
	for( int i = 0; i < 5; i++ ) {
		printf("h=%.5f, numerical limit=%.5f\n", h, numerical_lim(1, h));
		h *= 0.1;
	}

	// Now we can [plot the function 𝑢=𝑓(𝑥) and its tangent line 𝑦=2𝑥−3 at 𝑥=1], where the coefficient 2 is the slope of the tangent line.

	auto w_target = torch::tensor({4.0, 3.0}, options);
	auto b_target = torch::tensor({0.0}, options);

	std::printf("f(x):  y = %.2f - %.2f * x + %.2f * x^2 \n", b_target.item<double>(),
	    		w_target[0].item<double>(), w_target[1].item<double>());

	plt::figure_size(700, 500);
	plt::subplot(1, 1, 1);

	auto x_sample = torch::arange(0, 3, 0.1, options);

    auto f1 = torch::mul(x_sample, w_target[0].item<double>());
    auto f2 = torch::mul(x_sample.pow(2), w_target[1].item<double>());

    torch::Tensor y_sample = torch::add( torch::sub(f2, f1), b_target.item<double>());

    std::vector<double> xx(x_sample.data_ptr<double>(), x_sample.data_ptr<double>() + x_sample.numel());
    std::vector<double> yy(y_sample.data_ptr<double>(), y_sample.data_ptr<double>() + y_sample.numel());

    // tangent line
    auto w_ttarget = torch::tensor({2.0}, options);
    auto b_ttarget = torch::tensor({3}, options);

    auto tf1 = torch::mul(x_sample, w_ttarget.item<double>());
    torch::Tensor ty_sample = torch::sub(tf1, b_ttarget.item<double>());

    std::vector<double> txx(x_sample.data_ptr<double>(), x_sample.data_ptr<double>() + x_sample.numel());
    std::vector<double> tyy(ty_sample.data_ptr<double>(), ty_sample.data_ptr<double>() + ty_sample.numel());

	plt::named_plot("f(x)", xx, yy, "r");
	plt::named_plot("Tangent line(x=1)", txx, tyy, "b");
	plt::legend();
    plt::title("function u=f(x) and its tangent line y=2x−3 at x=1");
	plt::xlabel("x");
	plt::ylabel("f(x)");
    plt::show();

	std::cout << "Done!\n";
	return 0;
}




