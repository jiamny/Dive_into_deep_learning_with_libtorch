#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

using torch::indexing::Slice;
using torch::indexing::None;

/*
# We define a convenience function to calculate the convolutional layer. This
# function initializes the convolutional layer weights and performs
# corresponding dimensionality elevations and reductions on the input and
# output
*/
torch::Tensor comp_conv2d(torch::nn::Conv2d conv2d, torch::Tensor X){
    //# Here (1, 1) indicates that the batch size and the number of channels
    //# are both 1
    X = X.reshape({1 , 1, X.size(0), X.size(1)});
    auto Y = conv2d->forward(X);
    //# Exclude the first two dimensions that do not interest us: examples and
    //# channels
    return Y.reshape({Y.size(2), -1});
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	//# Note that here 1 row or column is padded on either side, so a total of 2
	//# rows or columns are added
	auto conv2d = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 1, 3).padding(1));
	auto X = torch::randn({8, 8});
	std::cout << comp_conv2d(conv2d, X).sizes() << std::endl;

	/*
	 * When the height and width of the convolution kernel are different, we can make the output and input have the same
	 * height and width by [setting different padding numbers for height and width.]
	 */
	//# Here, we use a convolution kernel with a height of 5 and a width of 3. The
	//# padding numbers on either side of the height and width are 2 and 1,
	//# respectively
	conv2d =  torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 1, {5, 3}).padding({2, 1}));
	std::cout << comp_conv2d(conv2d, X).sizes() << std::endl;

	// Stride
	/*
	 * Below, we [set the strides on both the height and width to 2], thus halving the input height and width.
	 */
	printf("set the strides on both the height and width to 2:\n");
	conv2d = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 1, 3).padding(1).stride(2));
	std::cout << comp_conv2d(conv2d, X).sizes() << std::endl;

	/*
	 * Next, we will look at (a slightly more complicated example).
	 */
	printf("a slightly more complicated example:\n");
	conv2d = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 1, {3, 5}).padding({0, 1}).stride({3, 4}));
	std::cout << comp_conv2d(conv2d, X).sizes() << std::endl;

	std::cout << "Done!\n";
	return 0;
}





