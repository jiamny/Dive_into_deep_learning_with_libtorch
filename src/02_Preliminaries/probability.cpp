
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main() {

	auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);

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

	using torch::indexing::Slice;
	using torch::indexing::None;

	plt::figure_size(800, 600);

	std::vector<float> xx(x_sample.data_ptr<float>(), x_sample.data_ptr<float>() + x_sample.numel());
	std::vector<float> w(rep, 0.167);

	for(int i = 0; i < 6; i++ ) {
		auto y = estimates.index({Slice(), i});
		std::vector<float> yy(y.data_ptr<float>(), y.data_ptr<float>() + y.numel());
		std::string die_label = "P(die=" + std::to_string(i+1) + ")";
		switch( i ) {
		case 0:
			plt::named_plot(die_label, xx, yy, "b");
			break;
		case 1:
			plt::named_plot(die_label.c_str(), xx, yy, "g");
			break;
		case 2:
			plt::named_plot(die_label.c_str(), xx, yy, "r");
			break;
		case 3:
			plt::named_plot(die_label.c_str(), xx, yy, "c");
			break;
		case 4:
			plt::named_plot(die_label.c_str(), xx, yy, "m");
			break;
		case 5:
			plt::named_plot(die_label.c_str(), xx, yy, "y");
			break;
		default:
			break;
		}
	}
	plt::xlabel("Groups of experiments");
	plt::ylabel("Estimated probability");
	plt::plot(xx, w, "k--");
	plt::legend();
	plt::show();

	std::cout << "Done!\n";
	return 0;
}



