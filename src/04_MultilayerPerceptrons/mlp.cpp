#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../utils.h"

#include <matplot/matplot.h>
using namespace matplot;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';
	auto options = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU);

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
	x.retain_grad();

	y.backward(torch::ones_like(x), true);

	auto relu_grad = x.grad();
	std::cout << "torch::ones_like(x):\n" << relu_grad.data() << std::endl;

	auto F = figure(true);
	F->size(1000, 1500);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(3, 2);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	std::vector<double> xx(x.data_ptr<double>(), x.data_ptr<double>() + x.numel());
	std::vector<double> yy(y.data_ptr<double>(), y.data_ptr<double>() + y.numel());
	matplot::plot(ax1, xx, yy, "b")->line_width(2);
	matplot::ylabel(ax1, "relu(x)");
	matplot::title(ax1, "ReLU");

	/*
	 * When the input is negative, the derivative of the ReLU function is 0, and when the input is positive, the derivative of the ReLU function is 1.
	 * Note that the ReLU function is not differentiable when the input takes value precisely equal to 0. In these cases, we default to the left-hand-side
	 * derivative and say that the derivative is 0 when the input is 0. We can get away with this because the input may never actually be zero.
	 * There is an old adage that if subtle boundary conditions matter, we are probably doing (real) mathematics, not engineering. That conventional wisdom may
	 * apply here. We plot the derivative of the ReLU function plotted below.
	 */

	std::vector<double> gyy(relu_grad.data().data_ptr<double>(),
						relu_grad.data().data_ptr<double>() + relu_grad.data().numel());

	auto ax2 = F->nexttile();
	matplot::plot(ax2, xx, gyy, "g--")->line_width(2);
	matplot::ylabel(ax2, "grad of relu");
	matplot::title(ax2, "ReLU grad");

	// Sigmoid Function
	/*
	 * Below, we plot the sigmoid function. Note that when the input is close to 0, the sigmoid function approaches a linear transformation.
	 */
	y = torch::sigmoid(x);
	std::vector<double> sx(x.data_ptr<double>(), x.data_ptr<double>() + x.numel());
	std::vector<float> sy(y.data_ptr<double>(), y.data_ptr<double>() + y.numel());

	auto ax3 = F->nexttile();
	matplot::plot(ax3, sx, sy, "b")->line_width(2);
	matplot::ylabel(ax3, "sigmoid(x)");
	matplot::title(ax3, "Sigmoid");

	//Clear out previous gradients
	x.grad().data().zero_();
	y.backward(torch::ones_like(x), true);
	std::vector<double> gy(x.grad().data().data_ptr<double>(),
						x.grad().data().data_ptr<double>() + x.grad().data().numel());

	auto ax4 = F->nexttile();
	matplot::plot(ax4, sx, gy, "g--")->line_width(2);
	matplot::ylabel(ax4, "grad of sigmoid");
	matplot::title(ax4, "Sigmoid grad");

	// Tanh Function
	/*
	 * Like the sigmoid function, [the tanh (hyperbolic tangent) function also squashes its inputs], transforming them into
	 * elements on the interval (between -1 and 1):
	 */
	y = torch::tanh(x);
	std::vector<double> tx(x.detach().data_ptr<double>(),
						x.detach().data_ptr<double>() + x.detach().numel());
	std::vector<double> ty(y.detach().data_ptr<double>(),
						y.detach().data_ptr<double>() + y.detach().numel());

	auto ax5 = F->nexttile();
	matplot::plot(ax5, tx, ty, "b")->line_width(2);
	matplot::xlabel(ax5, "x");
	matplot::ylabel(ax5, "tanh(x)");
	matplot::title(ax5, "Tanh");

	//# Clear out previous gradients.
	x.grad().data().zero_();
	y.backward(torch::ones_like(x), true);
	std::vector<double> tgy(x.grad().data().data_ptr<double>(),
						x.grad().data().data_ptr<double>() + x.grad().data().numel());

	auto ax6 = F->nexttile();
	matplot::plot(ax6, tx, tgy, "g--")->line_width(2);
	matplot::xlabel(ax6, "x");
	matplot::ylabel(ax6, "grad of tanh");
	matplot::title(ax6, "Tanh grad");

	matplot::show();

	std::cout << "Done!\n";
	return 0;
}



