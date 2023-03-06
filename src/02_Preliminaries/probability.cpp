
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <string>

#include <matplot/matplot.h>
using namespace matplot;

int main() {

	auto options = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU);

	// -----------------------------------------------------------
	// Basic Probability Theory
	// -----------------------------------------------------------
	/*
	 * To draw a single sample, we simply pass in a vector of probabilities. The output is another vector of the same length: its value at index 𝑖 is
	 * the number of times the sampling outcome corresponds to 𝑖.
	 */
	auto fair_probs = torch::ones({6}) / 6;
	auto sample = torch::multinomial(fair_probs, 1, true); // Multinomial(1, fair_probs).sample();
	std::cout << "sample = \n" << sample << std::endl;

	/*
	 * drawing multiple samples at once, returning an array of independent samples in any shape we might desire.
	 */
	sample = torch::multinomial(fair_probs, 20, true);
	std::cout << "sample = \n" << sample << std::endl;

	/*
	 * Now that we know how to sample rolls of a die, we can simulate 1000 rolls. We can then go through and count, after each of the 1000 rolls,
	 * how many times each number was rolled. Specifically, we calculate the relative frequency as the estimate of the true probability.
	 */
	sample = torch::multinomial(fair_probs, 100, true);
	std::cout << "sample.shape = \n" << sample.sizes() << std::endl;
	std::cout << sample.size(0) << std::endl;

	auto counts = torch::zeros({6});
	for( int i = 0; i < 100; i++ ) {
		int die_num = sample[i].item<int>();
		counts[die_num] += 1;
	}

	std::cout << "probability = \n" << counts / 100 << std::endl;

	/*
	 * Because we generated the data from a fair die, we know that each outcome has true probability 16, roughly 0.167, so the above output estimates look good.

	 * We can also visualize how these probabilities converge over time towards the true probability. Let us conduct 500 groups of experiments where each group
	 * draws 10 samples.
	 */

	int rep = 150;
	int smp = 50;
	counts = torch::zeros({rep, 6});
	for( int r = 0; r < rep; r++ ) {
		sample = torch::multinomial(fair_probs, smp, true);
		for( int i = 0; i < smp; i++ ) {
				int die_num = sample[i].item<int>();
				counts[r][die_num] += 1;
		}
	}

	auto cum_counts = counts.cumsum(0);
	auto estimates = cum_counts / cum_counts.sum(1, true);

	std::cout << "cum_counts = \n" << cum_counts.sizes() << std::endl;
	std::cout << "estimates = \n" << estimates.sizes() << std::endl;

	std::cout << "cum_counts.shape = \n" << cum_counts.sizes() << std::endl;
	std::cout << "estimates.shape = \n" << estimates.sizes() << std::endl;

	auto x_sample = torch::arange(0, rep, 1, options);
	auto h_sample = torch::zeros({rep}, options);
	std::cout << "x_sample.options(): \n" << x_sample.options() << std::endl;
	std::cout << "h_sample.options(): \n" << h_sample.options() << std::endl;

	using torch::indexing::Slice;
	using torch::indexing::None;

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::hold(ax1, true);

	std::vector<double> xx(x_sample.data_ptr<double>(), x_sample.data_ptr<double>() + x_sample.numel());
	std::vector<double> w(rep, 0.167);
	std::vector<std::string> lgd;

	estimates = estimates.to(torch::kDouble);

	for(int i = 0; i < 6; i++ ) {
		auto y = estimates.index({Slice(), i});
		std::vector<double> yy(y.data_ptr<double>(), y.data_ptr<double>() + y.numel());
		std::string die_label = "P(die=" + std::to_string(i+1) + ")";

		switch( i ) {
		case 0:
			matplot::plot(ax1, xx, yy, "b")->line_width(2);
			break;
		case 1:
			matplot::plot(ax1, xx, yy, "g")->line_width(2);
			break;
		case 2:
			matplot::plot(ax1, xx, yy, "r")->line_width(2);
			break;
		case 3:
			matplot::plot(ax1, xx, yy, "c")->line_width(2);
			break;
		case 4:
			matplot::plot(ax1, xx, yy, "m")->line_width(2);
			break;
		case 5:
			matplot::plot(ax1, xx, yy, "k")->line_width(2);
			break;
		default:
			break;
		}
		lgd.push_back(die_label);
	}
    matplot::hold(ax1, false);
    matplot::xlabel(ax1, "Groups of experiments");
    matplot::ylabel(ax1, "Estimated probability");
    matplot::legend(ax1, lgd);
    matplot::title(ax1, "Basic Probability Theory");
    matplot::show();

	std::cout << "Done!\n";
	return 0;
}



