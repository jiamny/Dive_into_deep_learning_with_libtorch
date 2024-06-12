#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <cmath>

#include <matplot/matplot.h>
using namespace matplot;

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

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::plot(ax1, xx, y1, "b")->line_width(2).display_name("f(x)");
	hold(ax1, on);
	F->draw();
	matplot::plot(ax1, xx, y2, "m--")->line_width(2).display_name("g(x)");
	F->draw();
	auto [t, a] = matplot::textarrow(ax1, 0.9, -1.0, 1.01, -1.2, "min of empirical risk");
	t->color("red").font_size(14).display_name("");
	a->color("black").display_name("");
	auto [tt, aa] = matplot::textarrow(ax1, 1.05, -0.5, 1.1, -1.05, "min of risk");
	tt->color("red").font_size(14).display_name("");
	aa->color("black").display_name("");
	matplot::xlabel(ax1, "x");
	matplot::ylabel(ax1, "risk");
	matplot::legend(ax1, {"", "", "", "", "", ""});
	matplot::show();

	// Local Minima
	x = torch::arange(-1.0, 2.0, 0.01).to(torch::kDouble);
	xs = f(x);
	std::vector<double> y3(xs.data_ptr<double>(), xs.data_ptr<double>() + xs.numel());
	std::vector<double> x2(x.data_ptr<double>(), x.data_ptr<double>() + x.numel());

	F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::plot(ax1, x2, y3, "b")->line_width(2);
	auto [t1, a1] = matplot::textarrow(ax1, -0.30, -0.8, -0.35, -0.16, "local minimum");
	t1->color("red").font_size(14);
	a1->color("black");
	auto [tt1, aa1] = matplot::textarrow(ax1, 0.95, 0.8, 1.11, -1.05, "global minimum");
	tt1->color("red").font_size(14);
	aa1->color("black");
	matplot::xlabel(ax1, "x");
	matplot::ylabel(ax1, "f(x)");
	matplot::show();

	// --------------------------------------
	// Saddle Points
	// A saddle point is any location where all gradients of a function vanish but which is neither a global nor a local minimum.
	// --------------------------------------
	x = torch::arange(-2.0, 2.0, 0.01).to(torch::kDouble);

	xs = torch::pow(x, 3);
	std::vector<double> y4(xs.data_ptr<double>(), xs.data_ptr<double>() + xs.numel());
	std::vector<double> x3(x.data_ptr<double>(), x.data_ptr<double>() + x.numel());

	F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::plot(ax1, x3, y4, "b")->line_width(2);
	auto [t2, a2] = matplot::textarrow(ax1, 0.0, -5.0, 0., 0., "saddle point");
	t2->color("red").font_size(14);
	a2->color("black");
	matplot::xlabel(ax1, "x");
	matplot::ylabel(ax1, "x^3");
	matplot::show();

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

	F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	ax1 = F->nexttile();
	matplot::surf(ax1, x_, y_, z_);
	matplot::xlabel(ax1, "x");
	matplot::ylabel(ax1, "y");
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}










