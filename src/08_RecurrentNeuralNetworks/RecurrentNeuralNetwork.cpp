#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	// Recurrent Neural Networks with Hidden States
	/*
	 * We just mentioned that the calculation of 𝐗𝑡𝐖𝑥ℎ+𝐇𝑡−1𝐖ℎℎ for the hidden state is equivalent to matrix multiplication of
	 * concatenation of 𝐗𝑡 and 𝐇𝑡−1 and concatenation of 𝐖𝑥ℎ and 𝐖ℎℎ. Though this can be proven in mathematics, in the following
	 * we just use a simple code snippet to show this. To begin with, we define matrices X, W_xh, H, and W_hh, whose shapes
	 * are (3, 1), (1, 4), (3, 4), and (4, 4), respectively. Multiplying X by W_xh, and H by W_hh, respectively, and then adding t
	 * hese two multiplications, we obtain a matrix of shape (3, 4).
	 */

	auto X = torch::normal(0, 1, {3, 1});
	auto W_xh = torch::normal(0, 1, {1, 4});

	auto H = torch::normal(0, 1, {3, 4}), W_hh = torch::normal(0, 1, {4, 4});

	std::cout << torch::matmul(X, W_xh) + torch::matmul(H, W_hh) << std::endl;

	/*
	 * Now we concatenate the matrices X and H along columns (axis 1), and the matrices W_xh and W_hh along rows (axis 0).
	 * These two concatenations result in matrices of shape (3, 5) and of shape (5, 4), respectively. Multiplying these two concatenated
	 * matrices, we obtain the same output matrix of shape (3, 4) as above.
	 */
    std::cout << "\ntorch::cat({X, H}, 1): " << torch::cat({X, H}, 1).sizes() << std::endl;
    std::cout << "torch::cat({W_xh, W_hh}, 0): " << torch::cat({W_xh, W_hh}, 0).sizes() << std::endl;
	std::cout << torch::matmul(torch::cat({X, H}, 1), torch::cat({W_xh, W_hh}, 0)) << std::endl;

	std::cout << "Done!\n";
	return 0;
}

