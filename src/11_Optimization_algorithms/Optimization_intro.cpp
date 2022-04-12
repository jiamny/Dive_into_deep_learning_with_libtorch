#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <cmath>

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

torch::Tensor f(torch::Tensor x) {
    return (x * torch::cos(M_PI * x)).to(torch::kDouble);
}

torch::Tensor g(torch::Tensor x) {
    return (f(x) + 0.2 * torch::cos(5 * M_PI * x)).to(torch::kDouble);
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	auto x = torch::arange(0.5, 1.5, 0.01).to(torch::kDouble);
	auto xs = f(x);
	auto ys = g(x);

	std::vector<double> xx(x.data_ptr<double>(), x.data_ptr<double>() + x.numel());
	std::vector<double> y1(xs.data_ptr<double>(), xs.data_ptr<double>() + xs.numel());
	std::vector<double> y2(ys.data_ptr<double>(), ys.data_ptr<double>() + ys.numel());

	plt::figure_size(800, 600);
	plt::named_plot("f(x)", xx, y1, "b");
	plt::named_plot("g(x)", xx, y2, "m--");
	plt::annotate("min of\nempirical risk", 0.6, -1.0);
	plt::arrow(0.70, -1.03, 0.22, -0.14, "k", "k", 0.05, 0.02);
	plt::annotate("min of risk", 1.05, -0.5);
	plt::arrow(1.10, -0.56, 0.0, -0.42, "k", "k", 0.05, 0.02);
	plt::xlabel("x");
	plt::ylabel("risk");
	plt::legend();
	plt::show();
	plt::close();

	// Local Minima
	x = torch::arange(-1.0, 2.0, 0.01).to(torch::kDouble);
	plt::figure_size(800, 600);
	xs = f(x);
	std::vector<double> y3(xs.data_ptr<double>(), xs.data_ptr<double>() + xs.numel());
	std::vector<double> x2(x.data_ptr<double>(), x.data_ptr<double>() + x.numel());
	plt::plot(x2, y3, "b");
	plt::annotate("local minimum", -0.30, -0.8);
	plt::arrow(-0.1, -0.72, -0.17, 0.4, "k", "k", 0.05, 0.02);
	plt::annotate("global minimum", 0.95, 0.8);
	plt::arrow(1.35, 0.75, -0.25, -1.72, "k", "k", 0.05, 0.02);
	plt::xlabel("x");
	plt::ylabel("f(x)");
	plt::show();
	plt::close();

	// --------------------------------------
	// Saddle Points
	// A saddle point is any location where all gradients of a function vanish but which is neither a global nor a local minimum.
	// --------------------------------------
	x = torch::arange(-2.0, 2.0, 0.01).to(torch::kDouble);
	plt::figure_size(800, 600);
	xs = torch::pow(x, 3);
	std::vector<double> y4(xs.data_ptr<double>(), xs.data_ptr<double>() + xs.numel());
	std::vector<double> x3(x.data_ptr<double>(), x.data_ptr<double>() + x.numel());
	plt::plot(x3, y4, "b");
	plt::annotate("saddle point", 0.0, -5.0);
	plt::arrow(0.4, -4.5, -0.3, 4.0, "k", "k", 0.1, 0.03);
	plt::xlabel("x");
	plt::ylabel("x^3");
	plt::show();
	plt::close();

	// Saddle points in higher dimensions are even more insidious, as the example below shows.
	// Consider the function 𝑓(𝑥,𝑦)=𝑥2−𝑦2. It has its saddle point at (0,0). This is a maximum with
	// respect to 𝑦 and a minimum with respect to 𝑥. Moreover, it looks like a saddle, which is where
	// this mathematical property got its name.

	std::vector<std::vector<double>> x_, y_, z_;

	for( double i = -1.0; i <= 1.0;  i += 0.1 ) {
		std::vector<double> x_row, y_row, z_row;
		for( double j = -1.0; j <= 1.0; j += 0.1 ) {
	            x_row.push_back(i);
	            y_row.push_back(j);
	            z_row.push_back(std::pow(i, 2) - std::pow(j, 2));
		}
		x_.push_back(x_row);
		y_.push_back(y_row);
		z_.push_back(z_row);
	}

	plt::plot_surface(x_, y_, z_);
	plt::xlabel("x");
	plt::ylabel("y");
	plt::show();
	plt::close();

	std::cout << "Done!\n";
	return 0;
}










