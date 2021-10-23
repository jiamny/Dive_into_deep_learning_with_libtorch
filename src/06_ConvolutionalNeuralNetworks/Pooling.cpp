#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

using torch::indexing::Slice;
using torch::indexing::None;

// Maximum Pooling and Average Pooling
torch::Tensor pool2d(torch::Tensor X, std::vector<int64_t> pool_size, std::string mode) {
    int64_t p_h = pool_size[0], p_w = pool_size[1];
    auto Y = torch::zeros({X.size(0) - p_h + 1, X.size(1) - p_w + 1});
    for(int64_t i = 0; i < Y.size(0); i++ ) {
        for( int64_t j = 0; j < Y.size(0); j++ ) {
            if( mode == "max" )
                Y.index({i, j}) = X.index({Slice(i,i + p_h), Slice(j,j + p_w)}).max();
            if( mode == "avg" )
                Y.index({i, j}) = X.index({Slice(i, i + p_h), Slice(j, j + p_w)}).mean();
        }
    }
    return Y;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Maximum Pooling and Average Pooling
	auto X = torch::tensor({{0.0, 1.0, 2.0}, {3.0, 4.0, 5.0}, {6.0, 7.0, 8.0}});
	std::cout << pool2d(X, std::vector<int64_t>({2, 2}), "max") << std::endl;

	//Also, we experiment with (the average pooling layer).
	std::cout << pool2d(X, std::vector<int64_t>({2, 2}), "avg") << std::endl;

	//Padding and Stride
	/*
	 * We first construct an input tensor X whose shape has four dimensions, where the number of examples (batch size)
	 *  and number of channels are both 1.
	 */
	X = torch::arange(16).to(torch::kFloat32).reshape({1, 1, 4, 4});
	std::cout << X << std::endl;

	/*
	 * By default, (the stride and the pooling window in the instance from the framework's built-in class have the same shape.)
	 * Below, we use a pooling window of shape (3, 3), so we get a stride shape of (3, 3) by default.
	 */
	auto pool2dn = torch::nn::MaxPool2d(3);
	std::cout << "By default:\n" << pool2dn->forward(X) << std::endl;

	// The stride and padding can be manually specified.
	pool2dn = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).padding(1).stride(2));
	std::cout << "Specify stride and padding:\n" << pool2dn->forward(X) << std::endl;

	/*
	 * Of course, we can (specify an arbitrary rectangular pooling window and specify the padding and stride for height and width), respectively.
	 */
	pool2dn = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2, 3}).stride({2, 3}).padding({0, 1}));
	std::cout << "Specify stride and padding for heightn and width:\n" << pool2dn->forward(X) << std::endl;

	//Multiple Channels
	/*
	 * Below, we will concatenate tensors X and X + 1 on the channel dimension to construct an input with 2 channels.
	 */
	X = torch::cat({X, X + 1}, 1);
	std::cout << "cat X and X + 1:\n" << X << std::endl;

	// As we can see, the number of output channels is still 2 after pooling.
	pool2dn = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).padding(1).stride(2));
	std::cout << "output channels is still 2 after pooling:\n" << pool2dn->forward(X) << std::endl;


	std::cout << "Done!\n";
	return 0;
}



