#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>    // transform
#include <functional>
#include <utility> 		// make_pair etc.

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	// ------------------------------------
	// Convex Functions
	// ------------------------------------
	auto f = [](const double x) { return 0.5*x*x; };

	auto g = [](const double x) { return std::cos(M_PI*x); };

	auto h = [](const double x) { return std::exp(0.5*x); };

	std::vector<double> x, yf, yg, yh;
	for( double i = -2.0; i <= 2; i += 0.01 ) {
		x.push_back(i);
		yf.push_back(i);
		yg.push_back(i);
		yh.push_back(i);
	}
	std::vector<double> sx = {-1.5, 1.0};
	std::vector<double> sf = {-1.5, 1.0};
	std::vector<double> sg = {-1.5, 1.0};
	std::vector<double> sh = {-1.5, 1.0};

	std::transform(yf.begin(), yf.end(), yf.begin(), f);
	std::transform(yg.begin(), yg.end(), yg.begin(), g);
	std::transform(yh.begin(), yh.end(), yh.begin(), h);

	std::transform(sf.begin(), sf.end(), sf.begin(), f);
	std::transform(sg.begin(), sg.end(), sg.begin(), g);
	std::transform(sh.begin(), sh.end(), sh.begin(), h);

    std::for_each( yg.begin(), yg.end(), [](const auto & elem ) {std::cout << elem << " "; });
    printf("\n");

    plt::figure_size(1500, 450);
    plt::subplot2grid(1, 3, 0, 0, 1, 1);
    plt::plot(x, yf, "b-");
    plt::plot(sx, sf, "m--");
    plt::xlabel("x");
    plt::title("f(x)");

    plt::subplot2grid(1, 3, 0, 1, 1, 1);
    plt::plot(x, yg, "b-");
    plt::plot(sx, sg, "m--");
    plt::xlabel("x");
    plt::title("g(x)");

    plt::subplot2grid(1, 3, 0, 2, 1, 1);
    plt::plot(x, yh, "b-");
    plt::plot(sx, sh, "m--");
    plt::xlabel("x");
    plt::title("h(x)");
    plt::show();
    plt::close();

    // -----------------------------------------------------------------------------------------
    // Properties
    // Convex functions have many useful properties. We describe a few commonly-used ones below.
    // Local Minima Are Global Minima
    // -----------------------------------------------------------------------------------------
    //f = lambda x: (x - 1) ** 2
    auto f2 = [](const double x) { return std::pow(x-1, 2); };
    std::vector<double> x2, yf2;
    for( double i = -2.0; i <= 2; i += 0.01 ) {
    	x2.push_back(i);
    	yf2.push_back(i);
    }

    std::vector<double> sx2 = {-1.5, 1.0};
    std::vector<double> sf2 = {-1.5, 1.0};

    std::transform(yf2.begin(), yf2.end(), yf2.begin(), f2);
    std::transform(sf2.begin(), sf2.end(), sf2.begin(), f2);

    plt::figure_size(500, 450);
    plt::plot(x2, yf2, "b-");
    plt::plot(sx2, sf2, "m--");
    plt::xlabel("x");
    plt::ylabel("f(x)");
    plt::show();
    plt::close();

	std::cout << "Done!\n";
	return 0;
}





