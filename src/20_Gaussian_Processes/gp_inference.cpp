#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include <random>
#include <cmath>
#include "../TempHelpFunctions.hpp"
#include "../utils/ch_20_util.h"

#include <matplot/matplot.h>
using namespace matplot;

torch::Tensor data_maker1(torch::Tensor x, float sig) {
    return torch::sin(x) + 0.5 * torch::sin(4 * x) + torch::randn(x.size(0)) * sig;
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	torch::manual_seed(345);
	std::cout << "// --------------------------------------------------\n";
	std::cout << "// Interpreting Equations for Learning and Predictions\n";
	std::cout << "// --------------------------------------------------\n";
	float sig = 0.25;
	torch::Tensor train_x = torch::linspace(0, 5, 50);
	torch::Tensor test_x = torch::linspace(0, 5, 500);
	torch::Tensor train_y = data_maker1(train_x, sig);
	torch::Tensor test_y = data_maker1(test_x, 0.);

	auto F = figure(true);
	F->size(800, 640);
	F->x_position(0);
	F->y_position(0);

	auto ax = F->nexttile();
	matplot::hold(ax, true);
	matplot::scatter(ax, tensorTovec(train_x), tensorTovec(train_y), 20);
	matplot::plot(ax, tensorTovec(test_x), tensorTovec(test_y))->line_width(2);
	matplot::xlabel(ax, "x");
	matplot::ylabel(ax, "Observations y");
	matplot::show();

	torch::Tensor meanvec = torch::zeros(test_x.size(0));
	torch::Tensor covmat = rbfkernel(test_x, test_x, 0.2);

	MultivariateNormalx mvn = MultivariateNormalx(meanvec, covmat);
	torch::Tensor prior_samples = mvn.rsample(5).t();
	std::cout << prior_samples.sizes() << '\n';


	auto F2 = figure(true);
	F2->size(800, 640);
	F2->x_position(0);
	F2->y_position(0);

	auto ax2 = F2->nexttile();
	matplot::hold(ax2, true);

	matplot::plot(ax2, tensorTovec(test_x), tensorTovec(meanvec), "k--")->line_width(4);
    for(int i = 0; i < prior_samples.size(0); i++) {
    	std::vector<double> y_;
    	for(int j = 1; j < prior_samples.size(1); j++)
    		y_.push_back(prior_samples[i][j].data().item<double>());
    	matplot::plot(ax2, tensorTovec(test_x), y_, "b-")->line_width(2);
    }
    matplot::plot(ax2, tensorTovec(test_x), tensorTovec(meanvec - 2 * torch::diag(covmat, 0)), "m:")->line_width(2);
    matplot::plot(ax2, tensorTovec(test_x), tensorTovec(meanvec + 2 * torch::diag(covmat, 0)), "m:")->line_width(2);

	matplot::show();

	std::cout << "Done!\n";
}



