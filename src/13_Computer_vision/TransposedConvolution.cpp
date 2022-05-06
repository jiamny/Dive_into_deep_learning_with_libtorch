
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <torch/utils.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

//#include "../utils/Ch_13_util.h"
#include "../utils.h"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

using torch::indexing::Slice;
using torch::indexing::None;


// implement this basic transposed convolution operation
torch::Tensor trans_conv(torch::Tensor X, torch::Tensor K) {
	int h = K.size(0), w = K.size(1);
    int64_t start = 0;

	auto Y = torch::zeros({X.size(0) + h - 1, X.size(1) + w - 1});
	for( auto& i : range(X.size(0), start) ) {
	    for( auto& j : range(X.size(1), start) ) {
	    	auto tmp = Y.index({Slice(i, i + h), Slice(j, j + w)});
	        Y.index_put_({Slice(i, i + h), Slice(j, j + w)}, tmp + (X.index({i, j}) * K));
	    }
	}
    return Y;
}

torch::Tensor corr2d(torch::Tensor X, torch::Tensor K) {
    //Compute 2D cross-correlation.

	int64_t start = 0;
    int h = K.size(0), w = K.size(1);
    auto Y = torch::zeros({X.size(0) - h + 1, X.size(1) - w + 1});

    for(auto& i : range(Y.size(0), start) ) {
        for(auto& j : range(Y.size(1), start) ) {
        	auto tmp = (X.index({Slice(i, i + h), Slice(j, j + w)}) * K).sum();
            Y.index_put_({i, j}, tmp);
        }
    }
    return Y;
}

torch::Tensor kernel2matrix(torch::Tensor K) {
	auto k = torch::zeros(5);
	auto W = torch::zeros({4, 9});

	k.index_put_({Slice(None, 2)}, K.index({0, Slice()}));
	k.index_put_({Slice(3, 5)}, K.index({1, Slice()}));
	W.index_put_({0, Slice(None, 5)}, k);
	W.index_put_({1, Slice(1, 6)}, k);
	W.index_put_({2, Slice(3, 8)}, k);
	W.index_put_({3, Slice(4, None)}, k);

	return W;
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	// ---------------------------------
	// Basic Operation
	// ---------------------------------

	auto X = torch::tensor({{0.0, 1.0}, {2.0, 3.0}});
	auto K = torch::tensor({{0.0, 1.0}, {2.0, 3.0}});

	auto Y = trans_conv(X, K);

	std::cout << "Y: " << Y << '\n';

	// use high-level APIs to obtain the same results

	X = X.reshape({1, 1, 2, 2});
	K = K.reshape({1, 1, 2, 2});

	auto tconv = torch::nn::ConvTranspose2d(
			torch::nn::ConvTranspose2dOptions(1, 1, 2).bias(false));

	torch::autograd::GradMode::set_enabled(false);  	// make parameters copying possible

	tconv.get()->weight.data().copy_(K);

	torch::autograd::GradMode::set_enabled(true);

	std::cout << "tconv(X): " << tconv->forward(X) << '\n';

	// ---------------------------------------
	// Padding, Strides, and Multiple Channels
	// ---------------------------------------
	tconv = torch::nn::ConvTranspose2d(
			torch::nn::ConvTranspose2dOptions(1, 1, 2).padding(1).bias(false));

	torch::autograd::GradMode::set_enabled(false);  	// make parameters copying possible

	tconv.get()->weight.data().copy_(K);

	torch::autograd::GradMode::set_enabled(true);

	std::cout << "tconv(X) padding = 1: " << tconv->forward(X) << '\n';

	// The following code snippet can validate the transposed convolution output for stride of 2
	tconv = torch::nn::ConvTranspose2d(
			torch::nn::ConvTranspose2dOptions(1, 1, 2).stride(2).bias(false));

	torch::autograd::GradMode::set_enabled(false);  	// make parameters copying possible

	tconv.get()->weight.data().copy_(K);

	torch::autograd::GradMode::set_enabled(true);

	std::cout << "tconv(X) stride = 2: " << tconv->forward(X) << '\n';

	// the number of output channels being the number of channels in X, then g(Y) will have the same shape as X

	X = torch::rand({1, 10, 16, 16});

	auto conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, 5).padding(2).stride(3));
	tconv = torch::nn::ConvTranspose2d(
			torch::nn::ConvTranspose2dOptions(20, 10, 5).padding(2).stride(3));

	std::cout << "tconv(conv(X)).sizes(): " <<
			tconv->forward(conv->forward(X)).sizes()
			<< " X.sizes(): " << X.sizes() << '\n';

	// -----------------------------------
	// Connection to Matrix Transposition
	// ----------------------------------
	X = torch::arange(9.0).reshape({3, 3});
	K = torch::tensor({{1.0, 2.0}, {3.0, 4.0}});
	Y = corr2d(X, K);

	std::cout << "corr2d(X, K):\n" << Y << '\n';

	// we rewrite the convolution kernel K as a sparse weight matrix W containing a lot of zeros.
	// The shape of the weight matrix is (4, 9), where the non-zero elements come from the convolution kernel K.
	auto W = kernel2matrix(K);

	std::cout << "kernel2matrix(K):\n" << W << '\n';

	// Concatenate the input X row by row to get a vector of length 9. Then the matrix multiplication of W
	// and the  vectorized X gives a vector of length 4. After reshaping it, we can obtain the same result Y
	// from the original convolution operation above: we just implemented convolutions using matrix multiplications.
	auto comp = (Y == torch::matmul(W, X.reshape(-1)).reshape({2, 2}));

	std::cout << "Y == torch::matmul(W, X.reshape(-1)).reshape({2, 2}):\n" << comp << '\n';

	// To implement this operation by multiplying matrices, we only need to transpose
	// the weight matrix W with the new shape (9,4).
	std::cout << "Y:\n" << Y << '\n';
	std::cout << "K:\n" << K << '\n';

	auto Z = trans_conv(Y, K);
	std::cout << "Z:\n" << Z << '\n';
	std::cout << "W.t():\n" << W.t() << '\n';
	std::cout << "Y.reshape(-1):\n" << Y.reshape(-1) << '\n';
	std::cout << "torch::matmul(W.t(), Y.reshape(-1)).reshape({3, 3}):\n"
			  << torch::matmul(W.t(), Y.reshape(-1)).reshape({3, 3}) << '\n';

	comp = (Z == torch::matmul(W.t(), Y.reshape(-1)).reshape({3, 3}));

	std::cout << "Z == torch::matmul(W.t(), Y.reshape(-1)).reshape({3, 3}):\n" << comp << '\n';

	std::cout << "Done!\n";
}




