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

double f( double x ) { return x*x; }		// Objective function

double f_grad( double x ) { return 2*x; }	// Gradient (derivative) of the objective function

std::vector<double> gd(const double eta, std::function<double(double)> func) {
    double x = 10.0;
    std::vector<double> results;
    results.push_back(x);

    for( int i = 0; i < 10; i++ ) {
        x -= eta * func(x);
        results.push_back(x);
    }
    std::cout << "epoch " << 10 << ", x: " << x << "\n";
    return results;
}

void show_trace(std::vector<double> results, std::function<double(double)> fc) {

	std::vector<double> fresults;
	for(auto& a : results)
		fresults.push_back(fc(a));

	double n = (std::max( std::abs(*min_element(results.begin(), results.end())),
					     std::abs(*max_element(results.begin(), results.end())))) * 1.0;
	std::cout << "n = " << n << "\n";

	std::vector<double> f_line, fx;
	for( double i = (-1*n); i <= n; i += 0.01 ) {
		f_line.push_back(i);
		fx.push_back(fc(i));
	}

//	std::for_each( f_line.begin(), f_line.end(), [](const auto & elem ) {std::cout << elem << " "; });
//	printf("\n");

    plt::figure_size(700, 500);
    plt::plot(f_line, fx, "b-");
    plt::plot(results, fresults, "y-o");
    plt::xlabel("x");
    plt::ylabel("f(x)");
    plt::show();
    plt::close();
}

double ftri(double x) {  			// Objective function
	const double c = 0.15 * M_PI;
    return x * std::cos(c * x);
}

double ftri_grad(double x) {  		// Gradient of the objective function
	const double c = 0.15 * M_PI;
    return std::cos(c * x) - c * x * std::sin(c * x);
}

// -------------------------------------------------
// Multivariate Gradient Descent
// -------------------------------------------------
double f_2d(double x1, double x2) {			   				// Objective function
    return x1 * x1 + 2 * x2 * x2;
}

std::pair<double, double> f_2d_grad(double x1, double x2) {	// Gradient of the objective function
    return std::make_pair(2 * x1, 4 * x2);
}

std::tuple<double, double> gd_2d(double x1, double x2, const double et) {
	std::pair<double, double> gs = f_2d_grad(x1, x2);
    return std::make_tuple(x1 - et * gs.first, x2 - et * gs.second);
}

std::pair<std::vector<double>, std::vector<double>> train_2d(int steps, const double et) {
    double x1 = -5.0, x2 = -2.0;
    std::vector<double> x, xx; // = [(x1, x2)]
    x.push_back(x1);
    xx.push_back(x2);
    for(int  i = 0; i < steps; i++ ) {
    	std::tie(x1, x2) = gd_2d(x1, x2, et);
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

// ----------------------------------------------
// Newton's Method
const double c = 0.5;

double fn(double x) { return std::cosh(c * x); }

double fn_grad(double x) { return c * std::sinh(c * x); }

double fn_hess(double x) {  // Hessian of the objective function
    return c*c * std::cosh(c * x);
}

std::vector<double> newton( std::function<double(double)> fg, std::function<double(double)> fh, double eta=1) {
    double x = 10.0;
    std::vector<double> results;
    results.push_back(x);
    for( int i = 0; i < 10; i++ ) {
        x -= eta * fg(x) / fh(x);
        results.push_back(x);
    };
    std::cout << "epoch " << 10 << " , x:" <<  x << "\n";
    return results;
}

const double c2 = 0.15 * M_PI;

double fnoncov(double x) { return x * std::cos(c2 * x); }

double fnoncov_grad(double x) {
    return std::cos(c2 * x) - c2 * x * std::sin(c2 * x);
}

double fnoncov_hess(double x) {
    return - 2 * c2 * std::sin(c2 * x) - x * c2*c2 * std::cos(c2 * x);
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	// ------------------------------------------------
	// we use x=10 as the initial value and assume η=0.2. Using gradient descent to iterate x for 10 times
	// we can see that, eventually, the value of x approaches the optimal solution.
	std::vector<double> results = gd(0.3, &f_grad);

	show_trace(results, &f);

	// -----------------------------------------------
	// Learning Rate
	// consider the progress in the same optimization problem for η=0.05

	show_trace(gd(0.05, &f_grad), &f);

	// when we set the learning rate to η=1.1, x overshoots the optimal solution x=0 and gradually diverges.

	show_trace(gd(1.1, &f_grad), &f);

	// ----------------------------------------------
	// Local Minima
	// the case of f(x)=x⋅cos(cx) for some constant c.

	show_trace(gd(2.0, &ftri_grad), &ftri);

	// ----------------------------------------------
	// Multivariate Gradient Descent
	// we observe the trajectory of the optimization variable x for learning rate η=0.1

	const double et = 0.2;
	show_trace_2d(&f_2d, train_2d(20, et));

	// ----------------------------------------------
	// Adaptive Methods

	std::cout << "Newton's Method - dividing by the Hessian, convex function\n";
	show_trace(newton(&fn_grad, &fn_hess), &fn);

	std::cout << "Newton's Method - dividing by the Hessian, nonconvex function\n";
	show_trace(newton(&fnoncov_grad, &fnoncov_hess), &fnoncov);

	std::cout << "Newton's Method - let us see how this works with a slightly smaller learning rate, say η=0.5.\n";
	show_trace(newton(&fnoncov_grad, &fnoncov_hess, 0.5), &fnoncov);

	std::cout << "Done!\n";
	return 0;
}



