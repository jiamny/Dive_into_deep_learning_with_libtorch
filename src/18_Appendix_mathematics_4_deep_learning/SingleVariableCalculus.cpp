#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/ch_18_util.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <matplot/matplot.h>
using namespace matplot;

// Define our function
double L(double x) {
    return std::pow(x, 2) + 1701*std::pow((x-4), 3);
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	// ----------------------------------------
	// Differential Calculus
	// ----------------------------------------
	// Plot a function in a normal range
	auto x_big = torch::arange(0.01, 3.01, 0.01);
	auto ys = torch::sin(torch::pow(x_big, x_big));

	std::vector<float> xx(x_big.data_ptr<float>(), x_big.data_ptr<float>() + x_big.numel());
	std::vector<float> yy(ys.data_ptr<float>(), ys.data_ptr<float>() + ys.numel());

	// Plot a the same function in a tiny range
	auto x_med = torch::arange(1.75, 2.25, 0.001);
	ys = torch::sin(torch::pow(x_med, x_med));

	std::vector<float> xx2(x_med.data_ptr<float>(), x_med.data_ptr<float>() + x_med.numel());
	std::vector<float> yy2(ys.data_ptr<float>(), ys.data_ptr<float>() + ys.numel());

	// Taking this to an extreme, if we zoom into a tiny segment, the behavior becomes far simpler: it is just a straight line.
	// Plot a the same function in a tiny range
	auto x_small = torch::arange(2.0, 2.01, 0.0001);
	ys = torch::sin(torch::pow(x_small, x_small));

	std::vector<float> xx3(x_small.data_ptr<float>(), x_small.data_ptr<float>() + x_small.numel());
	std::vector<float> yy3(ys.data_ptr<float>(), ys.data_ptr<float>() + ys.numel());

	auto f = figure(true);
	f->width(f->width() * 2);
	f->height(f->height() * 2);
	f->x_position(0);
	f->y_position(0);

	matplot::subplot(3, 1, 0);
	matplot::plot(xx, yy, "b-")->line_width(2);
	matplot::xlabel("x");
	matplot::ylabel("f(x)");
	matplot::title("x_big");

	matplot::subplot(3, 1, 1);
	matplot::plot(xx2, yy2, "b-")->line_width(2);
	matplot::xlabel("x");
	matplot::ylabel("f(x)");
	matplot::title("x_med");

	matplot::subplot(3, 1, 2);
	matplot::plot(xx3, yy3, "b-")->line_width(2);
	matplot::xlabel("x");
	matplot::ylabel("f(x)");
	matplot::title("x_small");
	matplot::show();

	// Print the difference divided by epsilon for several epsilon
	std::vector<double> rr = {0.1, 0.001, 0.0001, 0.00001};
	for(auto& epsilon : rr )
	    std::cout << "epsilon = " << epsilon << " -> " << (L(4+epsilon) - L(4)) / epsilon << '\n';

	// ----------------------------------
	// Linear Approximation
	// ----------------------------------
	// # Compute sin
	auto xs = torch::arange(-1 * M_PI, M_PI, 0.01);
	std::vector<std::vector<float>> yys;
	auto yt = torch::sin(xs);
	std::vector<float> xst(xs.data_ptr<float>(), xs.data_ptr<float>() + xs.numel());
	std::vector<float> ytt(yt.data_ptr<float>(), yt.data_ptr<float>() + yt.numel());
	yys.push_back(ytt);

	// Compute some linear approximations. Use d(sin(x))/dx = cos(x)
	for(auto& x0 : {-1.5, 0.0, 2.0} ) {
		yt = torch::sin(torch::tensor({x0})) + (xs - x0) *  torch::cos(torch::tensor({x0}));
		std::vector<float> ytt0(yt.data_ptr<float>(), yt.data_ptr<float>() + yt.numel());
		yys.push_back(ytt0);
	}

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::plot(ax1, xst, yys[0], "b-")->line_width(2);
	matplot::plot(ax1, xst, yys[1], "m--")->line_width(2);
	matplot::plot(ax1, xst, yys[2], "g-.")->line_width(2);
	matplot::plot(ax1, xst, yys[3], "r:")->line_width(2);
	matplot::hold(ax1, false);
    matplot::xlabel(ax1, "x");
    matplot::ylabel(ax1, "f(x)");
    matplot::ylim(ax1, {-1.5, 1.5});
    matplot::title("Linear Approximation");
    matplot::show();

	// ---------------------------------------
	// Higher Order Derivatives
	// ---------------------------------------
	// # Compute sin

	yys.clear();
	yt = torch::sin(xs);
	std::vector<float> yth(yt.data_ptr<float>(), yt.data_ptr<float>() + yt.numel());
	yys.push_back(yth);

	// Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
	for(auto& x0 : {-1.5, 0.0, 2.0} ) {
		yt = torch::sin(torch::tensor({x0})) + (xs - x0) *
        	 torch::cos(torch::tensor({x0})) - torch::pow((xs - x0), 2) *
			 torch::sin(torch::tensor({x0})) / 2;

		std::vector<float> yth0(yt.data_ptr<float>(), yt.data_ptr<float>() + yt.numel());
		yys.push_back(yth0);
	}

	F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::plot(ax1, xst, yys[0], "b-")->line_width(2);
	matplot::plot(ax1, xst, yys[1], "m--")->line_width(2);
	matplot::plot(ax1, xst, yys[2], "g-.")->line_width(2);
	matplot::plot(ax1, xst, yys[3], "r:")->line_width(2);
	matplot::hold(ax1, false);
    matplot::xlabel(ax1, "x");
    matplot::ylabel(ax1, "f(x)");
    matplot::ylim(ax1, {-1.5, 1.5});
    matplot::title("Higher Order Derivatives");
    matplot::show();
	// ---------------------------------------
	// Taylor Series
	// ---------------------------------------
	// # Compute the exponential function
	xs = torch::arange(0, 3, 0.01);
	ys = torch::exp(xs);

	// Compute a few Taylor series approximations
	auto P1 = 1 + xs;
	auto P2 = 1 + xs + torch::pow(xs, 2) / 2;
	auto P5 = 1 + xs + torch::pow(xs, 2) / 2 + torch::pow(xs, 3) / 6 + torch::pow(xs, 4) / 24 + torch::pow(xs, 5) / 120;

	std::vector<float> xss(xs.data_ptr<float>(), xs.data_ptr<float>() + xs.numel());
	std::vector<float> yxp(ys.data_ptr<float>(), ys.data_ptr<float>() + ys.numel());
	std::vector<float> yxp1(P1.data_ptr<float>(), P1.data_ptr<float>() + P1.numel());
	std::vector<float> yxp2(P2.data_ptr<float>(), P2.data_ptr<float>() + P2.numel());
	std::vector<float> yxp5(P5.data_ptr<float>(), P5.data_ptr<float>() + P5.numel());

	F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::plot(ax1, xss, yxp, "b-")->line_width(2);
	matplot::plot(ax1, xss, yxp1, "m--")->line_width(2);
	matplot::plot(ax1, xss, yxp2, "g-.")->line_width(2);
	matplot::plot(ax1, xss, yxp5, "r:")->line_width(2);
	matplot::hold(ax1, false);
    matplot::xlabel(ax1, "x");
    matplot::ylabel(ax1, "f(x)");
    matplot::title(ax1, "Taylor Series");
    matplot::legend(ax1, {"Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series", "Degree 5 Taylor Series"});
    matplot::show();

	std::cout << "Done!\n";
}


