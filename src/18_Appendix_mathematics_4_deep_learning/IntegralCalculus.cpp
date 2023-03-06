#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/ch_18_util.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <matplot/matplot.h>
using namespace matplot;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	// --------------------------------------------------
	// Geometric Interpretation
	// --------------------------------------------------

	auto xT = torch::arange(-2, 2, 0.01).to(torch::kDouble);
	auto fT = torch::exp(-1*torch::pow(xT, 2)).to(torch::kDouble);

	std::vector<double> x(xT.data_ptr<double>(), xT.data_ptr<double>() + xT.numel());
	std::vector<double> f(fT.data_ptr<double>(), fT.data_ptr<double>() + fT.numel());

	std::vector<double> y0;
	for( size_t i = 0; i < x.size(); i++ ) {
		y0.push_back(0.0);
	}

    auto F = figure(true);
    F->size(800, 600);
    F->add_axes(false);
    F->reactive_mode(false);
    F->tiledlayout(1, 1);
    F->position(0, 0);

    auto ax1 = F->nexttile();
    matplot::plot(ax1, x, f, "b-")->line_width(2);
    matplot::hold(ax1, true);
    matplot::area(ax1, x, f);
    matplot::hold(ax1, false);
    matplot::show();


    for( int i = 50; i < 250; i++ )
    	y0[i] = f[i];

    F = figure(true);
    F->size(800, 600);
    F->add_axes(false);
    F->reactive_mode(false);
    F->tiledlayout(1, 1);
    F->position(0, 0);

    ax1 = F->nexttile();
    matplot::plot( x, f, "b-")->line_width(2);
    matplot::hold( true);
    matplot::area(ax1, x, y0);
    matplot::hold( false);
    matplot::show();

	//  Let us take a look at an example doing this in code. We will see how to get the true value in a later section.
	float epsilon = 0.05;
	int a = 0;
	int b = 2;

	xT = torch::arange(a, b, epsilon);
	fT = xT / (1 + torch::pow(xT, 2));

	std::vector<float> xx(xT.data_ptr<float>(), xT.data_ptr<float>() + xT.numel());
	std::vector<float> ff(fT.data_ptr<float>(), fT.data_ptr<float>() + fT.numel());

	auto approx = torch::sum(epsilon*fT);
	auto tru = torch::log(torch::tensor({5.})) / 2;


	F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	ax1 = F->nexttile();
	matplot::ylim(ax1, {0.0, 1.0});
	matplot::plot(ax1, xx, ff, "k-")->line_width(2);
	matplot::hold(ax1, true);
	matplot::bar(ax1, xx, ff);
	matplot::hold(ax1, false);
	matplot::show();

	std::cout << "approximation: " << approx << "\ntruth: " << tru << '\n';

	// ----------------------------------------------
	// Multiple Integrals
	// ----------------------------------------------
	// Construct grid and compute function

	std::vector<std::vector<double>> x_, y_, z_;

	for( double i = -2.0; i < 2.0; i += 0.04 ) {
		std::vector<double> x_row, y_row, z_row;
		for( double j = -2.0; j < 2.0; j += 0.04 ) {
	            x_row.push_back(i);
	            y_row.push_back(j);
	            z_row.push_back(std::exp(-1 * std::pow(i, 2) - std::pow(j, 2)));
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
	matplot::xlim(ax1, {-2, 2});
	matplot::ylim(ax1, {-2, 2});
	matplot::surf(ax1, x_, y_, z_);
	matplot::xlabel(ax1, "x");
	matplot::ylabel(ax1, "y");
	matplot::show();

	std::cout << "Done!\n";
}




