#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../utils/ch_10_util.h"

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Use GPU." : "Use CPU.") << '\n';

	torch::manual_seed(1000);

	// To [demonstrate how masked_softmax function works], consider a minibatch of two 2×4 matrix examples,
	// where the valid lengths for these two examples are two and three, respectively. As a result of the masked
	// softmax operation, values beyond the valid lengths are all masked as zero.
	torch::Tensor X = torch::rand({2, 2, 4}).to(device);
	torch::Tensor y = torch::tensor({2, 3}).to(device);
	std::cout << masked_softmax(X, y) << "\n";

/*
	X = torch::tensor({-0.1325, -0.1325, -0.1325, -0.1325, -0.1325, -0.1325, -0.1325, -0.1325, -0.1325, -0.1325,
						0.3546,  0.3546,  0.3546, 0.3546,  0.3546,  0.3546,  0.3546,  0.3546,  0.3546,  0.3546});
	X = X.reshape({2,1,10});
	auto shape = X.sizes();
	std::cout << "shape:\n" << shape << "\n";
	auto Z = X.reshape({-1, shape[shape.size() - 1]});
	std::cout << "Z:\n" << Z << "\n";

	if( y.dim() == 1) {
		y = torch::repeat_interleave(y, shape[shape.size() - 2]);
	} else {
	   y = y.reshape(-1);
	}
	std::cout << "y:\n" << y << "\n";
//	auto W = sequence_mask(Z, y, -1e6);

    int64_t maxlen = Z.size(1);
    auto mask = torch::arange((maxlen),
    torch::TensorOptions().dtype(torch::kFloat32).device(Z.device())).index({None, Slice()}) < y.index({Slice(), None});
    std::cout << "mask:\n" << mask << std::endl;
    auto W = Z.index_put_({torch::ones_like(mask) ^ mask}, -1e6);

	std::cout << "W:\n" << W << "\n";

*/
	// demonstrate the above AdditiveAttention class
	auto queries = torch::normal(0, 1, {2, 1, 20});
	auto keys = torch::ones({2, 10, 2});

	// The two value matrices in the `values` minibatch are identical
	auto values = torch::arange(40, torch::TensorOptions(torch::kFloat32)).reshape({1, 10, 4}).repeat({2, 1, 1});

	auto valid_lens = torch::tensor({2, 6});

	auto attention = AdditiveAttention(2, 20, 8, 0.1);

	attention->eval();
	auto AA = attention->forward(queries, keys, values, valid_lens);
	std::cout << "demonstrate AdditiveAttention class:\n" << AA << std::endl;

	// demonstrate the above DotProductAttention class
	queries = torch::normal(0, 1, {2, 1, 2});
	auto dattention = DotProductAttention(0.5);
	dattention->eval();

    std::cout << "queries:\n" << queries.sizes() << std::endl;
    std::cout << "keys:\n" << keys.sizes() << std::endl;
    std::cout << "values:\n" << values.sizes() << std::endl;
    std::cout << "valid_lens:\n" << valid_lens.sizes() << std::endl;

	auto DA = dattention->forward(queries, keys, values, valid_lens);
	std::cout << "demonstrate DotProductAttention class:\n" << DA << std::endl;

	auto tsr = dattention->attention_weights.reshape({1, 1, 2, 10});
	std::cout << tsr.squeeze() << "\n";

	plot_heatmap(tsr.squeeze(), "keys", "Queries");

	std::cout << "Done!\n";
	return 0;
}




