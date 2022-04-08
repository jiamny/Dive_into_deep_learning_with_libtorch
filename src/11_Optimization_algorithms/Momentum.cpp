
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <unistd.h>
#include <iomanip>
#include <string>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <tuple>
#include <map>
#include <vector>
#include <functional>
#include <utility> 		// make_pair etc.

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

double f_2d(double x1, double x2) {
    return 0.1 * x1 * x1 + 2 * x2 * x2;
}

std::tuple<double, double> gd_2d(double x1, double x2, double eta) {
    return std::make_tuple(x1 - eta * 0.2 * x1, x2 - eta * 4 * x2);
}

std::pair<std::vector<double>, std::vector<double>> train_2d( std::function<std::tuple<double, double>(double, double, double)> func,
																int steps, double et) {
    double x1 = -5.0, x2 = -2.0;
    std::vector<double> x, xx; // = [(x1, x2)]
    x.push_back(x1);
    xx.push_back(x2);
    for(int  i = 0; i < steps; i++ ) {
    	std::tie(x1, x2) = func(x1, x2, et);
    	x.push_back(x1);
    	xx.push_back(x2);
    }

    std::cout << "epoch: " << steps << " , x1: " << x1 << " , x2: " << x2 << '\n';

    return std::make_pair(x, xx);
}

void show_trace_2d(std::function<double(double, double)> func, std::pair<std::vector<double>, std::vector<double>> rlt) {

//	std::for_each( rlt.first.begin(), rlt.first.end(), [](const auto & elem ) {std::cout << elem << " "; });
//	printf("\n");

	plt::figure_size(700, 500);
	plt::plot(rlt.first, rlt.second, "oy-"); // {{"marker": "o"}, {"color": "yellow"}, {"linestyle": "-"}}

	std::vector<std::vector<double>> x, y, z;
	for (double i = -5.5; i <= 1.0;  i += 0.1) {
	    std::vector<double> x_row, y_row, z_row;
	    for (double j = -3.0; j <= 1.0; j += 0.1) {
	            x_row.push_back(i);
	            y_row.push_back(j);
	            z_row.push_back(func(i, j));
	    }
	    x.push_back(x_row);
	    y.push_back(y_row);
	    z.push_back(z_row);
	}

	plt::contour(x, y, z);
	plt::xlabel("x1");
	plt::ylabel("x2");
	plt::show();
	plt::close();
}

std::tuple<double, double, double, double> momentum_2d(double x1, double x2, double v1, double v2, double eta, double beta) {
    v1 = beta * v1 + 0.2 * x1;
    v2 = beta * v2 + 4 * x2;
    return std::make_tuple(x1 - eta * v1, x2 - eta * v2, v1, v2);
}

std::pair<std::vector<double>, std::vector<double>> m_train_2d(int steps, double eta, double beta) {
    double x1 = -5.0, x2 = -2.0, v1 = 0.0, v2 = 0.0;
    std::vector<double> x, xx; // = [(x1, x2)]
    x.push_back(x1);
    xx.push_back(x2);
    for(int  i = 0; i < steps; i++ ) {
    	std::tie(x1, x2, v1, v2) = momentum_2d(x1, x2, v1, v2, eta, beta);
    	x.push_back(x1);
    	xx.push_back(x2);
    }

    std::cout << "epoch: " << steps << " , x1: " << x1 << " , x2: " << x2 << '\n';

    return std::make_pair(x, xx);
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);
	double eta = 0.4;

	// -----------------------------------------------------
	// Leaky Averages
	// An Ill-conditioned Problem
	// -----------------------------------------------------
	show_trace_2d(&f_2d, train_2d(&gd_2d, 20, eta));

	// increase in learning rate from 0.4 to 0.6. Convergence in the x1 direction improves
	// but the overall solution quality is much worse.
	eta = 0.6;
	show_trace_2d(&f_2d, train_2d(&gd_2d, 20, eta));

	// ------------------------------------------------------
	// The Momentum Method
	// Note that for β=0 we recover regular gradient descent.
	// ------------------------------------------------------
	eta = 0.6;
	double beta = 0.5;
	show_trace_2d(&f_2d, m_train_2d(20, eta, beta));

	// Halving it to β=0.25 leads to a trajectory that barely converges at all.
	beta = 0.25;
	show_trace_2d(&f_2d, m_train_2d(20, eta, beta));

	// Effective Sample Weight
	std::vector<double> betas = {0.95, 0.9, 0.6, 0};
	std::vector<std::string> strs = {"b-", "y-", "g-", "r-"};

	plt::figure_size(700, 500);
	int i = 0;
	for( auto& b : betas ) {
		std::vector<double> x, y;
		for( double i = 0.0; i < 40.0; i += 1.0) {
			x.push_back(i);
			y.push_back(std::pow(b, i));
		}
		plt::named_plot(("beta = " + std::to_string(b)).c_str(), x, y, strs[i].c_str() );
		i++;
	}
	plt::xlabel("time(x)");
	plt::legend();
	plt::show();
	plt::close();

	std::cout << "Done!\n";
	return 0;
}








