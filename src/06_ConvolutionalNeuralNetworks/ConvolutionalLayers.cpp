#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

using torch::indexing::Slice;
using torch::indexing::None;

// The Cross-Correlation Operation

torch::Tensor corr2d(torch::Tensor X, torch::Tensor K) {
    //Compute 2D cross-correlation.
    int64_t h=K.size(0), w = K.size(1);
    auto Y = torch::zeros({X.size(0) - h + 1, X.size(1) - w + 1});
    for( int64_t i  = 0; i < Y.size(0); i ++ ) {
        for( int64_t j = 0; j < Y.size(1); j++ )
            Y.index({i, j}) = (X.index({Slice(i, i + h), Slice(j, j + w)}) * K).sum();
    }
    return Y;
}

// Convolutional Layers
struct Conv2D : public  torch::nn::Module {
	torch::Tensor weight, bias;

    explicit Conv2D(int64_t kernel_size) {
        weight = torch::randn(kernel_size);
        bias = torch::zeros(1);
    }

    torch::Tensor forward(torch::Tensor x) {
        return corr2d(x, weight) + bias;
    }
};

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// The Cross-Correlation Operation
	auto X = torch::tensor({{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}, {6.0, 7.0, 8.0}});
	auto K = torch::tensor({{0.0, 1.0}, {2.0, 3.0}});
	std::cout << corr2d(X, K) << std::endl;

	// Convolutional Layers

	// Object Edge Detection in Images
	X = torch::ones({6, 8});
	X.index({Slice(), Slice(2,6)}) = 0;
	std::cout << X << std::endl;

	/*
	 * Next, we construct a kernel K with a height of 1 and a width of 2. When we perform the cross-correlation operation
	 * with the input, if the horizontally adjacent elements are the same, the output is 0. Otherwise, the output is non-zero.
	 */
	K = torch::tensor({{1.0, -1.0}});

	// As you can see, [we detect 1 for the edge from white to black and -1 for the edge from black to white.] All other outputs take value 0.

	auto Y = corr2d(X, K);
	std::cout << Y << std::endl;

	// We can now apply the kernel to the transposed image. As expected, it vanishes. [The kernel K only detects vertical edges.]
	std::cout << corr2d(X.transpose(1, 0), K) << std::endl;

	//Construct a two-dimensional convolutional layer with 1 output channel and a
	//kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
	auto conv2d = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 1, {1, 2}).bias(false));

	// The two-dimensional convolutional layer uses four-dimensional input and
	// output in the format of (example, channel, height, width), where the batch
	// size (number of examples in the batch) and the number of channels are both 1
	X = X.reshape({1, 1, 6, 8});
	Y = Y.reshape({1, 1, 6, 7});
	float lr = 3e-2;   // Learning rate

	for(int i = 0; i < 10; i++ ) {
		auto Y_hat = conv2d->forward(X);
		auto l = (Y_hat - Y) * (Y_hat - Y);

		conv2d->zero_grad();
		l.sum().backward();
    	//# Update the kernel
		conv2d->weight.data() -= lr * conv2d->weight.grad();
		if( (i + 1) % 2 == 0 )
			std::cout << "batch " << (i + 1) << ", loss " << l.sum().item<float>() << std::endl;
	}

	std::cout << conv2d->weight.data().reshape({1, 2}) << std::endl;

	std::cout << "Done!\n";
	return 0;
}





