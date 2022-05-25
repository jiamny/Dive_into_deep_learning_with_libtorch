#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/Ch_18_util.h"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

using torch::indexing::Slice;
using torch::indexing::None;

// Statistical bias
torch::Tensor stat_bias(float true_theta, torch::Tensor est_theta) {
    return(torch::mean(est_theta) - true_theta);
}

// Mean squared error
torch::Tensor mse(torch::Tensor data, float true_theta) {
    return(torch::mean(torch::square(data - true_theta)));
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(8675309);

	auto torch_pi = torch::acos(torch::zeros(1)) * 2; // define pi in torch

	// Sample datapoints and create y coordinate
	float epsilon = 0.1;

	auto xs = torch::randn({300});

	std::cout << xs.size(0) << '\n';

	std::vector<float> ys;

	for(int i = 0; i < xs.size(0); i++) {
		auto yi = torch::sum(torch::exp(-1*torch::pow((xs.index({Slice(None, i)}) - xs[i]), 2) / (2 * epsilon*epsilon)))
				/ torch::sqrt(2*torch_pi*epsilon*epsilon) / xs.size(0);
		ys.push_back(yi.data().item<float>());
	}

	// Compute true density
	auto xd = torch::arange(torch::min(xs).data().item<float>(), torch::max(xs).data().item<float>(), 0.01);
	auto yd = torch::exp(-1*torch::pow(xd, 2)/2) / torch::sqrt(2 * torch_pi);

	std::vector<float> xx(xs.data_ptr<float>(), xs.data_ptr<float>() + xs.numel());
	std::vector<float> xxd(xd.data_ptr<float>(), xd.data_ptr<float>() + xd.numel());
	std::vector<float> yyd(yd.data_ptr<float>(), yd.data_ptr<float>() + yd.numel());

	plt::figure_size(700, 500);
	plt::plot(xxd, yyd, "b-");
	plt::scatter(xx, ys);
	plt::plot({0.0,0.0}, {0.0, 0.5}, {{"color", "purple"}, {"linestyle", "--"}, {"lw", "2"}});
	plt::xlabel("x");
	plt::ylabel("density");
	plt::title("sample mean: " + std::to_string(xs.mean().data().item<float>()));
	plt::show();
	plt::close();

	float theta_true = 1.0;
	float sigma = 4.0;
	int sample_len = 10000;
	auto samples = torch::normal(theta_true, sigma, {sample_len, 1});
	auto theta_est = torch::mean(samples);
	std::cout << "theta_est: " << theta_est << '\n';

	std::cout << "mse(samples, theta_true): " << mse(samples, theta_true) << '\n';

	// Next, we calculate Var(θ^n)+[bias(θ^n)]2 as below.
	auto bias = stat_bias(theta_true, theta_est);
	std::cout << "torch::square(samples.std(false)) + torch::square(bias): " <<
			torch::square(samples.std(false)) + torch::square(bias) <<'\n';

	std::cout << "Done!\n";
}

