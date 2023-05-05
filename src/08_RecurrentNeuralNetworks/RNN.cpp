#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	//=============================================
	// Recurrent Neural Networks
	//=============================================
	auto X = torch::normal(0, 1, {3, 1});
	auto W_xh = torch::normal(0, 1, {1, 4});
	auto H = torch::normal(0, 1, {3, 4});
	auto W_hh = torch::normal(0, 1, {4, 4});

	auto out1 = torch::matmul(X, W_xh) + torch::matmul(H, W_hh);
	std::cout << out1 << std::endl;

	/*
	 * Now we concatenate the matrices X and H along columns (axis 1), and the matrices W_xh and W_hh along rows (axis 0).
	 * These two concatenations result in matrices of shape (3, 5) and of shape (5, 4), respectively. Multiplying these
	 * two concatenated matrices, we obtain the same output matrix of shape (3, 4) as above.
	 */
	auto out2 = torch::matmul(torch::cat({X, H}, 1), torch::cat({W_xh, W_hh}, 0));
	std::cout << out2 << std::endl;

	std::cout << "Done!\n";
	return 0;
}




