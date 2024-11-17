
#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include <random>
#include <cmath>
#include "../TempHelpFunctions.hpp"
#include "../utils.h"

#include <matplot/matplot.h>
using namespace matplot;

class MultivariateNormalx{
    torch::Tensor mean, stddev, var, L;
    int d = 0;
    // Define epsilon.
    double epsilon = 0.0001;
public:
    MultivariateNormalx(const torch::Tensor &mean, const torch::Tensor &std) : mean(mean), stddev(std), var(std * std) {
      	d = mean.size(0);
    	// Add small pertturbation.
    	torch::Tensor K = stddev + epsilon*torch::eye(d).to(mean.dtype());
    	// Cholesky decomposition.
    	L = torch::linalg::cholesky(K);
    }

    torch::Tensor rsample(int n = 1) {
    	torch::Tensor u = torch::normal(0., 1., d*n).reshape({d, n}).to(mean.dtype());
    	torch::Tensor x = mean.reshape({d, 1}) + torch::mm(L, u);
    	return x;
    }
};



torch::Tensor distance_matrix(torch::Tensor x, torch::Tensor y) {
	assert(x.size(1) == y.size(1));
    int m = x.size(0);
    int n = y.size(0);
	torch::Tensor z = torch::zeros({m, n});
    for(auto& i : range(m, 0) ) {
        for(auto& j : range(n, 0)) {
            z[i][j] = std::sqrt(torch::sum(torch::pow((x[i] - y[j]),2)).data().item<float>());
        }
    }
    return z;
}

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

torch::Tensor rbfkernel(torch::Tensor x1, torch::Tensor x2, float ls=4.) {
	torch::Tensor dist = distance_matrix(x1.unsqueeze(1), x2.unsqueeze(1));
    return torch::exp(-(1. / ls / 2) * (torch::pow(dist, 2)));
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
		//up_b.push_back(up_bd[i].data().item<double>());
		//lw_b.push_back(lw_bd[i].data().item<double>());
		color.push_back(i * 1.);
		y_h.push_back(0.);
	}

	auto f = figure(true);
    f->size(800, 640);
    f->x_position(0);
    f->y_position(0);

    matplot::hold(true);
    matplot::plot(x_, y_h, "k--")->line_width(3);
    for(int i = 0; i < outs.size(0); i++) {
    	std::vector<double> y_;
    	for(int j = 1; j < outs.size(1); j++)
    		y_.push_back(outs[i][j].data().item<double>());
    	matplot::plot(x_, y_)->line_width(2);
    }

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



