#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <vector>
#include <cmath>

#include "../utils/ch_10_util.h"
#include "../utils.h"
#include "../TempHelpFunctions.hpp"

struct PatchEmbeddingImpl : public torch::nn::Module {
	int64_t num_patches;
	torch::nn::Conv2d conv{nullptr};

	PatchEmbeddingImpl(std::vector<int64_t> img_sz, std::vector<int64_t> patch_sz, int64_t num_hd=512, int64_t in_channels=3) {

		num_patches = static_cast<int64_t>(img_sz[0] / patch_sz[0]) *
					  static_cast<int64_t>(img_sz[1] / patch_sz[1]);

		//conv = torch::nn::LazyConv2d(num_hiddens, kernel_size=patch_size,stride=patch_size);
		conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, num_hd,
				{patch_sz[0], patch_sz[1]}).stride({patch_sz[0], patch_sz[1]}));

		register_module("conv", conv);
	}

	torch::Tensor forward(torch::Tensor X) {
		// Output shape: (batch size, no. of patches, no. of channels)
        return conv->forward(X).flatten(2).transpose(1, 2);
	}
};
TORCH_MODULE(PatchEmbedding);


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	std::vector<int64_t> img_size = {96, 96};
	std::vector<int64_t> patch_size = {16, 16};
	int64_t num_hiddens = 512, batch_size = 4, in_channels = 3;

	auto patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens, in_channels);
	patch_emb->to(device);
	torch::Tensor X = torch::zeros({batch_size, in_channels, img_size[0], img_size[1]}).to(device);
	X = patch_emb->forward(X);
	std::cout << "X.shape: " << X.sizes() << '\n';

	int64_t ss = std::pow(static_cast<int64_t>(img_size[0] / patch_size[0]), 2);
	std::vector<int64_t> ref =  {batch_size, ss, num_hiddens};

	std::cout << "check_shape: " << check_shape(X, ref) << '\n';
}

