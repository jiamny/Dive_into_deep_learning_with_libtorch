#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/ch_18_util.h"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

using torch::indexing::Slice;
using torch::indexing::None;

// Define a helper to plot these figures
void plot_chebyshev(float a, float yp) {

	std::vector<float> xa, ya;
	xa.push_back(a - 2);
	xa.push_back(a);
	xa.push_back(a + 2);

	ya.push_back(yp);
	ya.push_back(1 - 2*yp);
	ya.push_back(yp);

	plt::figure_size(500, 400);
	plt::stem(xa, ya);
	plt::xlim(-4, 4);
	plt::xlabel("x");
	plt::ylabel("p.m.f.");

	plt::plot({a - 4 * std::sqrt(2 * yp),
	            a + 4 * std::sqrt(2 * yp)}, {0.5, 0.5}, {{"color", "black"}, {"lw", "4"}});
	plt::plot({a - 4 * std::sqrt(2 * yp), a - 4 * std::sqrt(2 * yp)}, {0.53, 0.47}, {{"color", "black"}, {"lw", "1"}});
	plt::plot({a + 4 * std::sqrt(2 * yp), a + 4 * std::sqrt(2 * yp)}, {0.53, 0.47}, {{"color", "black"}, {"lw", "1"}});
	plt::title("p = " + std::to_string(yp));
	plt::show();
	plt::close();
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	// --------------------------------------
	// Continuous Random Variables
	// --------------------------------------

	torch::Tensor pi = torch::acos(torch::zeros(1)) * 2;  // Define pi in torch

	// Plot the probability density function for some random variable
	auto x = torch::arange(-5, 5, 0.01);
	auto p = 0.2*torch::exp(-1 * torch::pow((x - 3), 2) / 2)/torch::sqrt(2 * pi) +
	    0.8*torch::exp(-1 * torch::pow((x + 1), 2) / 2) / torch::sqrt(2 * pi);

	std::vector<float> xx(x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
	std::vector<float> yy(p.data_ptr<float>(), p.data_ptr<float>() + p.numel());

	plt::figure_size(500, 400);
	plt::plot(xx, yy, "b-");
	plt::xlabel("x");
	plt::ylabel("Density");
	plt::show();
	plt::close();

	// --------------------------------------
	// Probability Density Functions
	// --------------------------------------
	//# Approximate probability using numerical integration
	float epsilon = 0.01;

	std::vector<float> y0;
	for( size_t i = 0; i < xx.size(); i++ ) {
		y0.push_back(0.0);
	}

	std::map<std::string, std::string> fill_parameters;
	fill_parameters["color"] = "blue";
	fill_parameters["alpha"] = "0.5";

	plt::figure_size(500, 400);
	plt::plot(xx, yy, "k-");
	plt::fill_between(vector_slice(xx, 300, 800), vector_slice(y0, 300, 800), vector_slice(yy, 300, 800), fill_parameters);
	plt::show();
	plt::close();

	std::cout << "approximate Probability: " << torch::sum(epsilon*p.index({Slice(300,800)})) << '\n';

	// # Define a helper to plot these figures
	float a = 0.0;
	float yp = 0.2;
	// Plot interval when p > 1/8
	plot_chebyshev(a, yp);

	// Plot interval when p = 1/8
	plot_chebyshev(0.0, 0.125);

	// Plot interval when p < 1/8
	plot_chebyshev(0.0, 0.05);

	// -----------------------------------------
	// Means and Variances in the Continuum
	// -----------------------------------------
	//  Plot the Cauchy distribution p.d.f.
	x = torch::arange(-5, 5, 0.01);
	p = 1 / (1 + torch::pow(x,2));

	std::vector<float> xb(x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
	std::vector<float> yb(p.data_ptr<float>(), p.data_ptr<float>() + p.numel());

	plt::figure_size(500, 400);
	plt::plot(xb, yb, "b-");
	plt::xlabel("x");
	plt::ylabel("p.d.f.");
	plt::show();
	plt::close();

	// Plot the integrand needed to compute the variance
	x = torch::arange(-20, 20, 0.01);
	p = torch::pow(x, 2) / (1 + torch::pow(x, 2));

	std::vector<float> xc(x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
	std::vector<float> yc(p.data_ptr<float>(), p.data_ptr<float>() + p.numel());

	plt::figure_size(500, 400);
	plt::plot(xc, yc, "b-");
	plt::xlabel("x");
	plt::ylabel("integrand");
	plt::show();
	plt::close();

	// -----------------------------------
	// Covariance
	// -----------------------------------
	// Plot a few random variables adjustable covariance
	std::vector<float> covs = {-0.9, 0.0, 1.2};

	plt::figure_size(1200, 300);
	for( int i = 0; i < 3; i++ ) {
	    auto X = torch::randn(500);
	    auto Y = covs[i]*X + torch::randn(500);

	    std::vector<float> xv(X.data_ptr<float>(), X.data_ptr<float>() + X.numel());
	    std::vector<float> yv(Y.data_ptr<float>(), Y.data_ptr<float>() + Y.numel());

	    plt::subplot2grid(1,3,0,i,1,1);
	    plt::scatter(xv, yv);
	    plt::xlabel("X");
	    plt::ylabel("Y");
	    plt::title("cov = " + std::to_string(covs[i]));
	}
	plt::show();
	plt::close();

	// --------------------------------------
	// Correlation
	// --------------------------------------
	std::vector<float> cors = {-0.9, 0.0, 1.0};

	plt::figure_size(1200, 300);
	for( int i = 0; i < 3; i++ ) {
		    auto X = torch::randn(500);
		    auto Y = cors[i] * X + torch::sqrt(torch::tensor({1}) -
                     std::pow(cors[i], 2)) * torch::randn(500);

		    std::vector<float> xv(X.data_ptr<float>(), X.data_ptr<float>() + X.numel());
		    std::vector<float> yv(Y.data_ptr<float>(), Y.data_ptr<float>() + Y.numel());

		    plt::subplot2grid(1,3,0,i,1,1);
		    plt::scatter(xv, yv);
		    plt::xlabel("X");
		    plt::ylabel("Y");
		    plt::title("cor = " + std::to_string(cors[i]));
	}
	plt::show();
	plt::close();

	std::cout << "Done!\n";
}

