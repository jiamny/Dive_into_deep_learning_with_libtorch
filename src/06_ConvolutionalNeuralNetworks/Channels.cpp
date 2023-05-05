#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

using torch::indexing::Slice;
using torch::indexing::None;

// 1×1 Convolutional Layer
torch::Tensor corr2d_multi_in_out_1x1(torch::Tensor X, torch::Tensor K) {
    int64_t c_i=X.size(0), h=X.size(1), w = X.size(2);
    int64_t c_o = K.size(0);
    X = X.reshape({c_i, h * w});
    K = K.reshape({c_o, c_i});
    //# Matrix multiplication in the fully-connected layer
    auto Y = torch::matmul(K, X);
    return Y.reshape({c_o, h, w});
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Multiple Input Channels
	auto X = torch::tensor({{{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}, {6.0, 7.0, 8.0}},
	                  {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}}});

	auto K = torch::tensor({{{0.0, 1.0}, {2.0, 3.0}}, {{1.0, 2.0}, {3.0, 4.0}}});

	std::cout << "X: " << X << std::endl;
	std::cout << "K: " << K << std::endl;
	auto k1 = K.index({0, Slice(), Slice()});
	auto k2 = K.index({1, Slice(), Slice()});
	std::cout << K.size(2) << std::endl;

	torch::Tensor out = torch::zeros({2,2});

	for( int r = 0; (r < k1.size(0)) && ((r + k1.size(0)) <= X.size(1)); r++ ) {
		for( int c = 0; (c < k1.size(1)) && ((c + k1.size(1)) <= X.size(2)); c++ ) {
			//std:: cout << "r= " << r << ", c= " << c << std::endl;
			auto x1 = X.index({0, Slice(r, (k1.size(0) + r)), Slice(c, (k1.size(1) + c))});
			auto x2 = X.index({1, Slice(r, (k1.size(0) + r)), Slice(c, (k1.size(1) + c))});
			out.index({r,c}) = torch::sum(x1*k1 + x2*k2);
		}
	}

	std::cout << out << std::endl;

	// Multiple Output Channels
	K = torch::stack({K, K + 1, K + 2}, 0);
	std::cout << K.sizes() << std::endl;

	// 1×1 Convolutional Layer
	X = torch::normal(0, 1, {3, 3, 3});
	K = torch::normal(0, 1, {2, 3, 1, 1});

	auto Y1 = corr2d_multi_in_out_1x1(X, K);
	std::cout << "Y1: " << Y1 << std::endl;
	//auto Y2 = corr2d_multi_in_out(X, K);

	std::cout << "Done!\n";
	return 0;
}




