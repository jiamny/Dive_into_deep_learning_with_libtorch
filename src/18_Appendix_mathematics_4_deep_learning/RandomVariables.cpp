#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/ch_18_util.h"

#include <matplot/matplot.h>
using namespace matplot;

using torch::indexing::Slice;
using torch::indexing::None;

// Define a helper to plot these figures
void plot_chebyshev(double a, double yp) {

	std::vector<double> xa, ya;
	xa.push_back(a - 2);
	xa.push_back(a);
	xa.push_back(a + 2);

	ya.push_back(yp);
	ya.push_back(1 - 2*yp);
	ya.push_back(yp);

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::stem(ax1, xa, ya)->line_width(2);
	matplot::line(ax1, a - 4 * std::sqrt(2 * yp), 0.5, a + 4 * std::sqrt(2 * yp), 0.5)->line_width(4).color("k");
	matplot::line(ax1, a - 4 * std::sqrt(2 * yp), 0.53, a - 4 * std::sqrt(2 * yp), 0.47)->line_width(1).color("k");
	matplot::line(ax1, a + 4 * std::sqrt(2 * yp), 0.53, a + 4 * std::sqrt(2 * yp), 0.47)->line_width(1).color("k");
	matplot::hold(ax1, false);
	matplot::xlim(ax1, {-4, 4});
	matplot::title("p = " + std::to_string(yp));
	matplot::show();
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	torch::manual_seed(123);

	// --------------------------------------
	// Continuous Random Variables
	// --------------------------------------

	torch::Tensor pi = (torch::acos(torch::zeros(1)) * 2).to(torch::kDouble);  // Define pi in torch

	// Plot the probability density function for some random variable
	auto x = torch::arange(-5, 5, 0.01).to(torch::kDouble);
	auto p = (0.2*torch::exp(-1 * torch::pow((x - 3), 2) / 2)/torch::sqrt(2 * pi) +
	    0.8*torch::exp(-1 * torch::pow((x + 1), 2) / 2) / torch::sqrt(2 * pi)).to(torch::kDouble);

	std::vector<double> xx(x.data_ptr<double>(), x.data_ptr<double>() + x.numel());
	std::vector<double> yy(p.data_ptr<double>(), p.data_ptr<double>() + p.numel());

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::plot(ax1, xx, yy, "b-")->line_width(2);
    matplot::xlabel(ax1, "x");
    matplot::ylabel(ax1, "Density");
    matplot::show();

	// --------------------------------------
	// Probability Density Functions
	// --------------------------------------
	//# Approximate probability using numerical integration
	double epsilon = 0.01;

	std::vector<double> y0;
	for( size_t i = 0; i < xx.size(); i++ ) {
		if( i >= 300 && i < 800 )
			y0.push_back(yy[i]);
		else
			y0.push_back(0.0);
	}

	F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	ax1 = F->nexttile();
	matplot::plot(ax1, xx, yy, "k-")->line_width(2);
    matplot::hold( true);
    matplot::area(ax1, xx, y0);
    matplot::hold( false);
    matplot::show();

	std::cout << "approximate Probability: " << torch::sum(epsilon*p.index({Slice(300,800)})) << '\n';

	// # Define a helper to plot these figures
	double a = 0.0;
	double yp = 0.2;
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
	x = torch::arange(-5, 5, 0.01).to(torch::kDouble);
	p = 1 / (1 + torch::pow(x,2)).to(torch::kDouble);

	std::vector<double> xb(x.data_ptr<double>(), x.data_ptr<double>() + x.numel());
	std::vector<double> yb(p.data_ptr<double>(), p.data_ptr<double>() + p.numel());

	F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	ax1 = F->nexttile();
	matplot::plot(ax1, xb, yb, "b-")->line_width(2);
    matplot::xlabel(ax1, "x");
    matplot::ylabel(ax1, "p.d.f.");
    matplot::show();


	// Plot the integrand needed to compute the variance
	x = torch::arange(-20, 20, 0.01).to(torch::kDouble);
	p = torch::pow(x, 2).to(torch::kDouble) / (1 + torch::pow(x, 2).to(torch::kDouble));

	std::vector<double> xc(x.data_ptr<double>(), x.data_ptr<double>() + x.numel());
	std::vector<double> yc(p.data_ptr<double>(), p.data_ptr<double>() + p.numel());

	F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	ax1 = F->nexttile();
	matplot::plot(ax1, xc, yc, "b-")->line_width(2);
    matplot::xlabel(ax1, "x");
    matplot::ylabel(ax1, "integrand");
    matplot::show();
	// -----------------------------------
	// Covariance
	// -----------------------------------
	// Plot a few random variables adjustable covariance
	std::vector<double> covs = {-0.9, 0.0, 1.2};

	auto f = figure(true);
	f->width(f->width() * 2);
	f->height(f->height() * 2);
	f->x_position(0);
	f->y_position(0);

	for( int i = 0; i < 3; i++ ) {
	    auto X = torch::randn(500).to(torch::kDouble);
	    auto Y = covs[i]*X + torch::randn(500).to(torch::kDouble);

	    std::vector<double> xv(X.data_ptr<double>(), X.data_ptr<double>() + X.numel());
	    std::vector<double> yv(Y.data_ptr<double>(), Y.data_ptr<double>() + Y.numel());

		matplot::subplot(1, 3, i);
		matplot::scatter(xv, yv);
		matplot::xlabel("X");
		matplot::ylabel("Y");
		matplot::title("cov = " + std::to_string(covs[i]));
	}
	matplot::show();

	// --------------------------------------
	// Correlation
	// --------------------------------------
	std::vector<double> cors = {-0.9, 0.0, 1.0};

//	plt::figure_size(1200, 300);
	f = figure(true);
	f->width(f->width() * 2);
	f->height(f->height() * 2);
	f->x_position(0);
	f->y_position(0);

	for( int i = 0; i < 3; i++ ) {
		    auto X = torch::randn(500).to(torch::kDouble);
		    auto Y = cors[i] * X + torch::sqrt(torch::tensor({1}).to(torch::kDouble) -
                     std::pow(cors[i], 2)) * torch::randn(500).to(torch::kDouble);

		    std::vector<double> xv(X.data_ptr<double>(), X.data_ptr<double>() + X.numel());
		    std::vector<double> yv(Y.data_ptr<double>(), Y.data_ptr<double>() + Y.numel());

			matplot::subplot(1, 3, i);
			matplot::scatter(xv, yv);
			matplot::xlabel("X");
			matplot::ylabel("Y");
			matplot::title("cor = " + std::to_string(cors[i]));
	}
	matplot::show();

	std::cout << "Done!\n";
}

