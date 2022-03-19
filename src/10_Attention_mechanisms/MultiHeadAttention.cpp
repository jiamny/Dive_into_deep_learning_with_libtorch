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
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	// Test MultiHeadAttention
	int64_t num_hiddens = 100, num_heads = 5;

	/*
	auto queries = torch::normal(0, 1, {10, 4, 20});
	auto keys = torch::normal(0, 1, {10, 6, 20});
	auto values = torch::normal(0, 1, {10, 6, 20});
	auto valid_lens = torch::tensor({3,2});
	std::cout << valid_lens.defined() << std::endl;

	if( valid_lens.defined() ) {
		valid_lens = torch::repeat_interleave(valid_lens, num_heads, 0);
	}

	auto dattention = DotProductAttention(0.5);
	dattention.eval();

	std::cout << "queries:\n" << queries.sizes() << std::endl;
	std::cout << "keys:\n" << keys.sizes() << std::endl;
	std::cout << "values:\n" << values.sizes() << std::endl;
	std::cout << "valid_lens:\n" << valid_lens.sizes() << std::endl;

	auto DA = dattention.forward(queries, keys, values, valid_lens);
	std::cout << "demonstrate DotProductAttention class:\n" << DA.sizes() << std::endl;
	*/

	auto attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
	                               num_hiddens, num_heads, 0.5);
	attention.eval();

	std::cout << attention << std::endl;

	int64_t batch_size = 2, num_queries = 4, num_kvpairs = 6;
	auto valid_lens = torch::tensor({3, 2});

	auto X = torch::ones({batch_size, num_queries, num_hiddens});
	auto Y = torch::ones({batch_size, num_kvpairs, num_hiddens});
	std::cout << attention.forward(X, Y, Y, valid_lens).sizes() << std::endl;

	std::cout << "Done!\n";
	return 0;
}





