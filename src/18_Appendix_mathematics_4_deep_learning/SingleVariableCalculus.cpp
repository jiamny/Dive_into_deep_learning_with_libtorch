#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/Ch_18_util.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

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

	plt::figure_size(1400, 300);
	plt::subplot2grid(1,3,0,0,1,1);
	plt::plot(xx, yy, "b-");
	plt::xlabel("x");
	plt::ylabel("f(x)");
	plt::title("x_big");

	plt::subplot2grid(1,3,0,1,1,1);
	plt::plot(xx2, yy2, "b-");
	plt::xlabel("x");
	plt::ylabel("f(x)");
	plt::title("x_med");

	plt::subplot2grid(1,3,0,2,1,1);
	plt::plot(xx3, yy3, "b-");
	plt::xlabel("x");
	plt::ylabel("f(x)");
	plt::title("x_small");
	plt::show();
	plt::close();

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

	plt::figure_size(700, 500);
	plt::plot(xst, yys[0], "b-");
	plt::plot(xst, yys[1], "m--");
	plt::plot(xst, yys[2], "g-.");
	plt::plot(xst, yys[3], "r:");
	plt::xlabel("x");
	plt::ylabel("f(x)");
	plt::ylim(-1.5, 1.5);
	plt::title("Linear Approximation");
	plt::show();
	plt::close();

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

	plt::figure_size(700, 500);
	plt::plot(xst, yys[0], "b-");
	plt::plot(xst, yys[1], "m--");
	plt::plot(xst, yys[2], "g-.");
	plt::plot(xst, yys[3], "r:");
	plt::xlabel("x");
	plt::ylabel("f(x)");
	plt::ylim(-1.5, 1.5);
	plt::title("Higher Order Derivatives");
	plt::show();
	plt::close();

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

	plt::figure_size(700, 500);
	plt::named_plot("Exponential", xss, yxp, "b-");
	plt::named_plot("Degree 1 Taylor Series", xss, yxp1, "m--");
	plt::named_plot("Degree 2 Taylor Series", xss, yxp2, "g-.");
	plt::named_plot("Degree 5 Taylor Series", xss, yxp5, "r:");
	plt::xlabel("x");
	plt::ylabel("f(x)");
	plt::title("Taylor Series");
	plt::legend();
	plt::show();
	plt::close();

	std::cout << "Done!\n";
}


