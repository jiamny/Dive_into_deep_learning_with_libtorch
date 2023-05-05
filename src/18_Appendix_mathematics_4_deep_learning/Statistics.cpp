#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/ch_18_util.h"

#include <matplot/matplot.h>
using namespace matplot;

using torch::indexing::Slice;
using torch::indexing::None;

// Statistical bias
torch::Tensor stat_bias(double true_theta, torch::Tensor est_theta) {
    return(torch::mean(est_theta) - true_theta);
}

// Mean squared error
torch::Tensor mse(torch::Tensor data, double true_theta) {
    return(torch::mean(torch::square(data - true_theta)));
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	torch::manual_seed(8675309);

	auto torch_pi = torch::acos(torch::zeros(1)) * 2.0; // define pi in torch

	// Sample datapoints and create y coordinate
	double epsilon = 0.1;

	auto xs = torch::randn({300}).to(torch::kDouble);

	std::cout << xs.size(0) << '\n';

	std::vector<double> ys;

	for(int i = 0; i < xs.size(0); i++) {
		torch::Tensor yi = (torch::sum(torch::exp(-1*torch::pow((xs.index({Slice(None, i)}) - xs[i]), 2) / (2 * epsilon*epsilon)))
				/ torch::sqrt(2*torch_pi*epsilon*epsilon) / xs.size(0)).to(torch::kDouble);
		ys.push_back(yi.data().item<double>());
	}

	// Compute true density
	auto xd = torch::arange(torch::min(xs).data().item<double>(), torch::max(xs).data().item<double>(), 0.01).to(torch::kDouble);
	auto yd = (torch::exp(-1*torch::pow(xd, 2)/2) / torch::sqrt(2 * torch_pi)).to(torch::kDouble);

	std::vector<double> xx(xs.data_ptr<double>(), xs.data_ptr<double>() + xs.numel());
	std::vector<double> xxd(xd.data_ptr<double>(), xd.data_ptr<double>() + xd.numel());
	std::vector<double> yyd(yd.data_ptr<double>(), yd.data_ptr<double>() + yd.numel());

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::plot(ax1, xxd, yyd, "m")->line_width(2);
	matplot::scatter(ax1, xx, ys);
	matplot::line(ax1, 0.0, 0.0, 0.0, 0.5)->line_width(2.5).color("c");
    matplot::xlabel(ax1, "x");
    matplot::ylabel(ax1, "density");
    matplot::title("sample mean: " + std::to_string(xs.mean().data().item<float>()));
    matplot::show();

	double theta_true = 1.0;
	double sigma = 4.0;
	int sample_len = 10000;
	auto samples = torch::normal(theta_true, sigma, {sample_len, 1}).to(torch::kDouble);
	auto theta_est = torch::mean(samples);
	std::cout << "theta_est: " << theta_est << '\n';

	std::cout << "mse(samples, theta_true): " << mse(samples, theta_true) << '\n';

	// Next, we calculate Var(θ^n)+[bias(θ^n)]2 as below.
	auto bias = stat_bias(theta_true, theta_est);
	std::cout << "torch::square(samples.std(false)) + torch::square(bias): " <<
			torch::square(samples.std(false)) + torch::square(bias) <<'\n';

	std::cout << "Done!\n";
}

