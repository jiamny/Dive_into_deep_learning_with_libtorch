#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../utils.h"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

	// Activation Functions
	/*
	 * Activation functions decide whether a neuron should be activated or not by calculating the weighted sum and further adding bias with it.
	 * They are differentiable operators to transform input signals to outputs, while most of them add non-linearity. Because activation functions
	 * are fundamental to deep learning,
	 */

	// The most popular choice, due to both simplicity of implementation and its good performance on a variety of predictive tasks,
	// is the rectified linear unit (ReLU).

	auto x = torch::arange(-8.0, 8.0, 1.0, torch::requires_grad(true)).to(options);
	auto y = torch::relu(x);

	y.backward(torch::ones_like(x), true);

	auto relu_grad = x.grad();
	std::cout << "torch::ones_like(x):" << relu_grad.data() << std::endl;

	plt::figure_size(800, 1200);
	//plt::subplot(3,2, true); // true - constrained_layout adjusts subplots and decorations automatically to fit them in the figure as best as possible.
/*
	plt::subplots_adjust(0.125, // left
            0.1,    // bottom=
            0.9, 	// right=
            0.9,    // top=
           0.2, 	//  wspace=
            0.35);  // hspace=
*/
	plt::tight_layout();
//	plt::subplot(3, 2, 1);
	plt::subplot2grid(3, 2, 0, 0, 1, 1);

	std::vector<float> xx(x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
	std::vector<float> yy(y.data_ptr<float>(), y.data_ptr<float>() + y.numel());
	plt::plot(xx, yy, "b");
//	plt::xlabel("x");
	plt::ylabel("relu(x)");
	plt::title("ReLU");

	/*
	 * When the input is negative, the derivative of the ReLU function is 0, and when the input is positive, the derivative of the ReLU function is 1.
	 * Note that the ReLU function is not differentiable when the input takes value precisely equal to 0. In these cases, we default to the left-hand-side
	 * derivative and say that the derivative is 0 when the input is 0. We can get away with this because the input may never actually be zero.
	 * There is an old adage that if subtle boundary conditions matter, we are probably doing (real) mathematics, not engineering. That conventional wisdom may
	 * apply here. We plot the derivative of the ReLU function plotted below.
	 */

	std::vector<float> gyy(relu_grad.data().data_ptr<float>(), relu_grad.data().data_ptr<float>() + relu_grad.data().numel());

//	plt::subplot(3, 2, 2);
	plt::subplot2grid(3, 2, 0, 1, 1, 1);
	plt::plot(xx, gyy, "g--");
//	plt::xlabel("x");
	plt::ylabel("grad of relu");
	plt::title("ReLU grad");

	// Sigmoid Function
	/*
	 * Below, we plot the sigmoid function. Note that when the input is close to 0, the sigmoid function approaches a linear transformation.
	 */
	y = torch::sigmoid(x);
	std::vector<float> sx(x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
	std::vector<float> sy(y.data_ptr<float>(), y.data_ptr<float>() + y.numel());
//	plt::subplot(3, 2, 3);
	plt::subplot2grid(3, 2, 1, 0, 1, 1);
	plt::plot(sx, sy, "b");
//	plt::xlabel("x");
	plt::ylabel("sigmoid(x)");
	plt::title("Sigmoid");

	//Clear out previous gradients
	x.grad().data().zero_();
	y.backward(torch::ones_like(x), true);
	std::vector<float> gy(x.grad().data().data_ptr<float>(), x.grad().data().data_ptr<float>() + x.grad().data().numel());
//	plt::subplot(3, 2, 4);
	plt::subplot2grid(3, 2, 1, 1, 1, 1);
	plt::plot(sx, gy, "g--");
//	plt::xlabel("x");
	plt::ylabel("grad of sigmoid");
	plt::title("Sigmoid grad");

	// Tanh Function
	/*
	 * Like the sigmoid function, [the tanh (hyperbolic tangent) function also squashes its inputs], transforming them into
	 * elements on the interval (between -1 and 1):
	 */
	y = torch::tanh(x);
	std::vector<float> tx(x.detach().data_ptr<float>(), x.detach().data_ptr<float>() + x.detach().numel());
	std::vector<float> ty(y.detach().data_ptr<float>(), y.detach().data_ptr<float>() + y.detach().numel());
//	plt::subplot(3, 2, 5);
	plt::subplot2grid(3, 2, 2, 0, 1, 1);
	plt::plot(tx, ty, "b");
	plt::xlabel("x");
	plt::ylabel("tanh(x)");
	plt::title("Tanh");

	//# Clear out previous gradients.
	x.grad().data().zero_();
	y.backward(torch::ones_like(x), true);
	std::vector<float> tgy(x.grad().data().data_ptr<float>(), x.grad().data().data_ptr<float>() + x.grad().data().numel());
//	plt::subplot(3, 2, 6);
	plt::subplot2grid(3, 2, 2, 1, 1, 1);
	plt::plot(tx, tgy, "g--");
	plt::xlabel("x");
	plt::ylabel("grad of tanh");
	plt::title("Tanh grad");
	plt::show();

	std::cout << "Done!\n";
	return 0;
}



