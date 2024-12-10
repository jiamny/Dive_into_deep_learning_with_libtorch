
#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include <random>
#include <cmath>
#include "../utils/ch_20_util.h"

#include <matplot/matplot.h>
using namespace matplot;


torch::Tensor lin_func(torch::Tensor x, int n_sample) {
	int m = x.size(0);
	torch::Tensor preds = torch::zeros({n_sample, m});
    for(auto& ii : range(n_sample, 0)) {
    	torch::Tensor w = torch::normal(0, 1, {2});
    	torch::Tensor y = w[0] + w[1] * x;
        preds.index_put_({ii},  y);
    }
    return preds;
}



int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	torch::manual_seed(1000);
	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  Definition A Simple Gaussian Process\n";
	std::cout << "// --------------------------------------------------\n";

	torch::Tensor x_points = torch::linspace(-5, 5, 50);
	torch::Tensor outs = lin_func(x_points, 10);

	torch::Tensor lw_bd = -2 * torch::sqrt((1 + torch::pow(x_points, 2)));
	torch::Tensor up_bd = 2 * torch::sqrt((1 + torch::pow(x_points, 2)));

	std::vector<double> x_, y_, lw_b, up_b, y_h;
	std::vector<double> color;
	for(int i = 1; i < x_points.size(0); i++) {
		x_.push_back(x_points[i].data().item<double>());
		up_b.push_back(up_bd[i].data().item<double>());
		lw_b.push_back(lw_bd[i].data().item<double>());
		color.push_back(i * 1.);
		y_h.push_back(0.);
	}

	auto f = figure(true);
    f->size(800, 640);
    f->x_position(0);
    f->y_position(0);

    matplot::hold(true);
    matplot::plot(x_, y_h, "k--")->line_width(4);
    for(int i = 0; i < outs.size(0); i++) {
    	std::vector<double> y_;
    	for(int j = 1; j < outs.size(1); j++)
    		y_.push_back(outs[i][j].data().item<double>());
    	matplot::plot(x_, y_)->line_width(2);
    }
    matplot::plot(x_, lw_b, "m:")->line_width(2);
    matplot::plot(x_, up_b, "m:")->line_width(2);
    matplot::show();

	std::cout << "// --------------------------------------------------\n";
	std::cout << "//  The Radial Basis Function (RBF) Kernel\n";
	std::cout << "// --------------------------------------------------\n";
	torch::Tensor meanvec = torch::zeros(x_points.size(0));
	torch::Tensor covmat = rbfkernel(x_points, x_points, 1.);

	MultivariateNormalx mvn = MultivariateNormalx(meanvec, covmat);
	torch::Tensor prior_samples = mvn.rsample(5).t();
	std::cout << prior_samples.sizes() << '\n';


	auto F = figure(true);
	F->size(800, 640);
	F->x_position(0);
	F->y_position(0);

	auto ax = F->nexttile();
	matplot::hold(ax, true);

    for(int i = 0; i < prior_samples.size(0); i++) {
    	std::vector<double> y_;
    	for(int j = 1; j < prior_samples.size(1); j++)
    		y_.push_back(prior_samples[i][j].data().item<double>());
    	matplot::plot(ax, x_, y_)->line_width(2);
    }
	matplot::show();

	std::cout << "Done!\n";
}



