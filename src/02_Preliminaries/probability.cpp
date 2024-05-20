
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

	torch::Tensor counts = torch::zeros({6});
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

	torch::Tensor cum_counts = counts.cumsum(0);
	torch::Tensor estimates = cum_counts / cum_counts.sum(1, true);

	std::cout << "cum_counts = \n" << cum_counts.sizes() << std::endl;
	std::cout << "estimates = \n" << estimates.sizes() << std::endl;

	std::cout << "cum_counts.shape = \n" << cum_counts.sizes() << std::endl;
	std::cout << "estimates.shape = \n" << estimates.sizes() << std::endl;

	torch::Tensor x_sample = torch::arange(0, rep, 1, options);
	torch::Tensor h_sample = torch::zeros({rep}, options);
	std::cout << "x_sample.options(): \n" << x_sample.options() << std::endl;
	std::cout << "h_sample.options(): \n" << h_sample.options() << std::endl;

	using torch::indexing::Slice;
	using torch::indexing::None;

	std::vector<double> xx(x_sample.data_ptr<double>(), x_sample.data_ptr<double>() + x_sample.numel());
	std::vector<double> w(rep, 0.167);
	std::vector<std::string> lgd0;

	estimates = estimates.to(torch::kDouble);

	auto F = figure(true);
	F->size(2000, 500);
	auto ax0 = subplot(1, 3, 0);
	auto y0 = estimates.index({Slice(), 0});
	std::vector<double> yy0(y0.data_ptr<double>(), y0.data_ptr<double>() + y0.numel());
	std::string die_label = "P(die=" + std::to_string(1) + ")";
	lgd0.push_back(die_label);
	plot(xx, yy0, "b")->line_width(2);
	hold(on);
	auto y1 = estimates.index({Slice(), 1});
	std::vector<double> yy1(y1.data_ptr<double>(), y1.data_ptr<double>() + y1.numel());
	die_label = "P(die=" + std::to_string(2) + ")";
	lgd0.push_back(die_label);
	plot(xx, yy1, "g")->line_width(2);
    ax0->ylabel("Estimated probability");
    legend(lgd0);
    hold(off);

    auto ax1 = subplot(1, 3, 1);
    auto y2 = estimates.index({Slice(), 2});
    std::vector<std::string> lgd1;
    std::vector<double> yy2(y2.data_ptr<double>(), y2.data_ptr<double>() + y2.numel());
    die_label = "P(die=" + std::to_string(3) + ")";
    lgd1.push_back(die_label);
    plot(xx, yy2, "r")->line_width(2);
    hold(on);
    auto y3 = estimates.index({Slice(), 3});
    std::vector<double> yy3(y3.data_ptr<double>(), y3.data_ptr<double>() + y3.numel());
    die_label = "P(die=" + std::to_string(4) + ")";
    lgd1.push_back(die_label);
    plot(xx, yy3, "c")->line_width(2);
    ax1->xlabel("Groups of experiments");
    ax1->title("Basic Probability Theory");
    legend(lgd1);
    hold(off);

    auto ax2 = subplot(1, 3, 2);
    auto y4 = estimates.index({Slice(), 4});
    std::vector<std::string> lgd2;
    std::vector<double> yy4(y4.data_ptr<double>(), y4.data_ptr<double>() + y4.numel());
    die_label = "P(die=" + std::to_string(5) + ")";
    lgd2.push_back(die_label);
    plot(xx, yy4, "m")->line_width(2);
    hold(on);
    auto y5 = estimates.index({Slice(), 5});
    std::vector<double> yy5(y5.data_ptr<double>(), y5.data_ptr<double>() + y5.numel());
    die_label = "P(die=" + std::to_string(6) + ")";
    lgd2.push_back(die_label);
    plot(xx, yy5, "k")->line_width(2);
    legend(lgd2);
    hold(off);
    F->draw();
    show();

	std::cout << "Done!\n";
	return 0;
}



