#include <unistd.h>
#include <iomanip>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include "../utils.h"	// RangeToensorIndex()

using torch::indexing::Slice;
using torch::indexing::None;


torch::Tensor nansum_c( torch::Tensor x ) {
    // Define nansum, as pytorch doesn't offer it inbuilt.
    return x.masked_select(~torch::isnan(x)).sum();
}

double self_information(double p) {
	std::vector<double> tp;
	tp.push_back(p);
	auto tt = torch::from_blob(tp.data(), {(int64_t)tp.size()}, at::TensorOptions(torch::kDouble)).clone();

    return -1 * torch::log2(tt).data().item<double>();
}

torch::Tensor entropy(torch::Tensor x) {
	x = -1 * x * torch::log2(x);
	// Operator `nansum` will sum up the non-nan number
	auto out = nansum_c(x);
	return out;
}

torch::Tensor joint_entropy(torch::Tensor p_xy) {
    auto joint_ent = -1 * p_xy * torch::log2(p_xy);
    // Operator `nansum` will sum up the non-nan number
    auto out = nansum_c(joint_ent);
    return out;
}

torch::Tensor conditional_entropy(torch::Tensor p_xy, torch::Tensor p_x) {
    auto p_y_given_x = p_xy/p_x;
    auto cond_ent = -1 * p_xy * torch::log2(p_y_given_x);
    // Operator `nansum` will sum up the non-nan number
    auto out = nansum_c(cond_ent);
    return out;
}

torch::Tensor mutual_information(torch::Tensor p_xy, torch::Tensor p_x, torch::Tensor p_y) {
    auto p = p_xy / (p_x * p_y);
    auto mutual = p_xy * torch::log2(p);
    // Operator `nansum` will sum up the non-nan number
    auto out = nansum_c(mutual);
    return out;
}

double kl_divergence(torch::Tensor p, torch::Tensor q) {
    auto kl = p * torch::log2(p / q);
    auto out = nansum_c(kl);
    return out.abs().item<double>();
}

torch::Tensor cross_entropy(torch::Tensor y_hat, torch::Tensor y) {
	//  ce = -torch.log(y_hat[range(len(y_hat)), y])
	torch::Tensor ridx = RangeTensorIndex(y_hat.size(0));;
    auto ce = -1 * torch::log(y_hat.index({ridx, y}));
    return ce.mean();
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	// Self-information
	std::cout << self_information(1.0 / 64) << '\n';

	// Entropy
	auto x = torch::tensor({0.1, 0.5, 0.1, 0.3});
	std::cout << "Entropy: " << entropy(x) << '\n';

	// Joint Entropy
	x = torch::tensor({{0.1, 0.5}, {0.1, 0.3}});
	std::cout << "Joint Entropy: " << joint_entropy(x) << '\n';

	// Conditional Entropy
	x = torch::tensor({{0.1, 0.5}, {0.2, 0.3}});
	auto y = torch::tensor({0.2, 0.8});
	//std::cout << x/y << '\n';
	std::cout << "Conditional Entropy: " << conditional_entropy(x, y) << '\n';

	// Mutual Information
	x = torch::tensor({{0.1, 0.5}, {0.1, 0.3}});
	y = torch::tensor({0.2, 0.8});
	auto z = torch::tensor({{0.75, 0.25}});
	puts("1\n");
	std::cout << (y * z) << '\n';
	auto d = x / (y * z);
	puts("2\n");
	auto mutual = x * torch::log2(d);
	    // Operator `nansum` will sum up the non-nan number
	auto out = nansum_c(mutual);
	//std::cout << "Mutual Information: " << mutual_information(x, y, z) << '\n';

	// Kullback–Leibler Divergence
	torch::manual_seed(1);

	int64_t tensor_len = 10000;
	auto p = torch::normal(0, 1, {tensor_len});
	auto q1 = torch::normal(-1, 1, {tensor_len});
	auto q2 = torch::normal(1, 1, {tensor_len});
	std::cout << "p.sizes: " << p.sizes() << '\n';

	p = std::get<0>(torch::sort(p));
	q1 = std::get<0>(torch::sort(q1));
	q2 = std::get<0>(torch::sort(q2));

	std::cout << "p: " << p[0] << "\nq1: " << q1[0] << "\nq2: " << q2[0] << '\n';

	auto kl_pq1 = kl_divergence(p, q1);
	auto kl_pq2 = kl_divergence(p, q2);
	auto similar_percentage = abs(kl_pq1 - kl_pq2) / ((kl_pq1 + kl_pq2) / 2) * 100;

	std::cout << "kl_pq1: " << kl_pq1 << ", kl_pq2: " << kl_pq2 << ", similar_percentage: " << similar_percentage << '\n';

	// In contrast, you may find that DKL(q2∥p) and DKL(p∥q2) are off a lot, with around 40% off as shown below.
	auto kl_q2p = kl_divergence(q2, p);
	auto differ_percentage = abs(kl_q2p - kl_pq2) / ((kl_q2p + kl_pq2) / 2) * 100;

	std::cout << "kl_q2p: " << kl_q2p << ", differ_percentage: " << differ_percentage << '\n';

	// Cross-Entropy
	auto labels = torch::tensor({0, 2});
	auto preds = torch::tensor({{0.3, 0.6, 0.1}, {0.2, 0.3, 0.5}});

	auto ce = cross_entropy(preds, labels);
	std::cout << "Cross-Entropy: " << ce << '\n';

	//Implementation of cross-entropy loss in PyTorch combines `nn.LogSoftmax()` and `nn.NLLLoss()`
	auto nll_loss = torch::nn::NLLLoss();
	auto loss = nll_loss(torch::log(preds), labels);
	std::cout << "loss: " << loss << '\n';

	std::cout << "Done!\n";
}



