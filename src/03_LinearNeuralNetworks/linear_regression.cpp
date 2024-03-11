#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <cmath>

#include "../TempHelpFunctions.hpp"
#include "../utils.h"

#include <matplot/matplot.h>
using namespace matplot;

std::vector<double> normal(std::vector<double> x, double mu, double sigma) {
	std::cout << "mu = " << mu << " sigma = " << sigma << std::endl;
	std::vector<double> result;
    double p = 1 / std::sqrt(2 * M_PI * sigma*sigma);

    for( int i = 0; i < x.size(); i ++ ) {
    	double tmp = std::exp( (-0.5 / sigma*sigma) * (x[i] - mu)*(x[i] - mu) );
    	result.push_back( (p * tmp) );
    }
    return result;
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	std::setprecision(5);

	/*
	 * Dating back to the dawn of the 19th century, linear regression flows from a few simple assumptions. First, we assume that the relationship between
	 * the independent variables 𝐱 and the dependent variable 𝑦 is linear, i.e., that 𝑦 can be expressed as a weighted sum of the elements in 𝐱, given some
	 * noise on the observations. Second, we assume that any noise is well-behaved (following a Gaussian distribution).
	 */

	// test Timer class

	using namespace std::literals;
	precise_timer stopwatch;
	for (auto wait_time = 100ms; wait_time <= 1s; wait_time += 100ms) {
	        std::this_thread::sleep_for(wait_time);
	        auto actual_wait_time = stopwatch.stop<unsigned int, std::chrono::microseconds>();
	        std::cout << actual_wait_time << std::endl;
	        std::cout << "Scheduler overhead is roughly " << actual_wait_time - (wait_time + 0us).count() << " microseconds"
	                  << " for " << wait_time.count() << " milliseconds of requested sleep time\n";
	}

	auto avg_time = stopwatch.avg<unsigned int, std::chrono::microseconds>();
	auto sum_time = stopwatch.sum<unsigned int, std::chrono::microseconds>();
	std::cout << "avg = " << avg_time <<  " microseconds" << std::endl;
	std::cout << "sum = " << sum_time <<  " microseconds" << std::endl;

	std::vector<std::chrono::high_resolution_clock::duration::rep> cumtimes = stopwatch.cumsum();
	for( auto& c : cumtimes ) {
		std::cout << (c/1000000.0) << " seconds" << std::endl;
	}

	// ------------------------------------------------------------------
	// Basic Elements of Linear Regression
	// ------------------------------------------------------------------
	int64_t n = 10000;
	auto a = torch::ones(n);
	auto b = torch::ones(n);
	/*
	 * Now we can benchmark the workloads. First, [we add them, one coordinate at a time, using a for-loop.]
	 */
	auto c = torch::zeros(n);
	precise_timer timer = Timer<>();
	for( int i = 0; i < n; i++ ) {
		c[i] = a[i] + b[i];
	}
	unsigned int dul = timer.stop<unsigned int, std::chrono::microseconds>();
	std::cout << "Roughly takes " << (dul/1000000.0) << " seconds\n";

	/*
	 * (Alternatively, we rely on the reloaded + operator to compute the elementwise sum.)
	 */
	timer.restartTimer();
	auto d = a + b;
	dul = timer.stop<unsigned int, std::chrono::microseconds>();
	std::cout << "Roughly takes " << (dul/1000000.0) << " seconds\n";

	// -----------------------------------------------------------------------------------
	// The Normal Distribution and Squared Loss
	// ----------------------------------------------------------------------------------
	auto options = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU);
	torch::Tensor x_range = torch::arange(-7, 7, 0.01, options);

	std::vector<double> xx(x_range.data_ptr<double>(), x_range.data_ptr<double>() + x_range.numel());

	//Mean and standard deviation pairs
	double params[3][2] = {{0, 1}, {0, 2}, {3, 1}};

	//std:: cout << params[0][0] << " " << params[0][1] << std::endl;
	//normal(xx, params[0][0], params[0][1]);
	auto F = figure(true);
	F->size(1200, 500);
	F->add_axes(false);
	F->reactive_mode(false);


	auto ax1 = subplot(1, 2, 0); //F->nexttile();

	std::vector<double> yy0 = normal(xx, params[0][0], params[0][1]);
	std::string legend_label0 = "mean " +
				to_string_with_precision(params[0][0], 0) +
				", std " +
				to_string_with_precision(params[0][1], 0);
	plot(xx, yy0, "b")->line_width(2).display_name(legend_label0);
	ax1->xlabel("x");
	ax1->ylabel("p(x)");
	ax1->title("Linear regression");
	legend({});

	auto ax2 = subplot(1, 2, 1);
	std::vector<double> yy1 = normal(xx, params[1][0], params[1][1]);
	std::string legend_label1 = "mean " +
						to_string_with_precision(params[1][0], 0) +
						", std " +
						to_string_with_precision(params[1][1], 0);
	plot(xx, yy1, "g--")->line_width(2).display_name(legend_label1);
	hold(on);

	std::vector<double> yy2 = normal(xx, params[2][0], params[2][1]);
	std::string legend_label2 = "mean " +
						to_string_with_precision(params[2][0], 0) +
						", std " +
						to_string_with_precision(params[2][1], 0);
	plot(xx, yy2, "r-.")->line_width(2).display_name(legend_label2);

    ax2->xlabel("x");
    ax2->ylabel("p(x)");
    ax2->title("Linear regression");
    legend({});
    hold(off);
    F->draw();
    show();

    std::cout << "Done!\n";
    return 0;
}




