#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../utils/ch_10_util.h"

struct PositionWiseFFN : public torch::nn::Module {
	torch::nn::Linear dense1{nullptr}, dense2{nullptr};
	torch::nn::ReLU relu{nullptr};
    //Positionwise feed-forward network.
	PositionWiseFFN( int64_t ffn_num_input, int64_t ffn_num_hiddens, int64_t ffn_num_outputs ) {
        //super(PositionWiseFFN, self).__init__(**kwargs)
        dense1 = torch::nn::Linear(ffn_num_input, ffn_num_hiddens);
        relu = torch::nn::ReLU();
        dense2 = torch::nn::Linear(ffn_num_hiddens, ffn_num_outputs);
        register_module("dense1", dense1);
        register_module("relu", relu);
        register_module("dense2", dense2);
	}

	torch::Tensor forward(torch::Tensor X) {
        return dense2->forward(relu->forward(dense1->forward(X)));
	}
};

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	// The following example shows that [the innermost dimension of a tensor changes] to the number of outputs
	// in the positionwise feed-forward network. Since the same MLP transforms at all the positions, when the
	// inputs at all these positions are the same, their outputs are also identical.

	auto ffn = PositionWiseFFN(4, 4, 8);
	ffn.to(device);
	ffn.eval();

	std::cout << ffn.forward(torch::ones({2, 3, 4}))[0] << "\n";

	// Residual Connection and Layer Normalization
	std::cout << "Residual Connection and Layer Normalization\n";
	auto ln = torch::nn::LayerNorm(torch::nn::LayerNormOptions({2,2}));
	auto bn = torch::nn::BatchNorm1d(2);
	auto X = torch::tensor({{1, 2}, {2, 3}}, torch::kFloat);
	// Compute mean and variance from `X` in the training mode
	std::cout << "layer norm:" << X.sizes() << "\nbatch norm:" << bn->forward(X) << "\n";

	std::cout << "Done!\n";
	return 0;
}




