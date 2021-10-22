
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <map>

using namespace torch::autograd;


torch::Tensor f( torch::Tensor a ) {
    auto b = a * 2;
    torch::Tensor c;

    while( b.norm().item<double>() < 1000 ) {
    	b = b * 2;
    	if( b.norm().item<double>() > 0 )
    	   c = b;
    	else
    	   c = 100 * b;
    }
    return c;
}

int main() {

	auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);

	// Automatic Differentiation

	/*
	 * As a toy example, say that we are interested in (differentiating the function 𝑦=2𝐱⊤𝐱 with respect to the column vector 𝐱.) To start,
	 * let us create the variable x and assign it an initial value.
	 */
	auto x = torch::arange(4.0, options);
	std::cout << "x=\n" << x << std::endl;

	/*
	 * [Before we even calculate the gradient of 𝑦 with respect to 𝐱, we will need a place to store it.] It is important that we do not allocate new memory
	 * every time we take a derivative with respect to a parameter because we will often update the same parameters thousands or millions of times and could
	 * quickly run out of memory. Note that a gradient of a scalar-valued function with respect to a vector 𝐱 is itself vector-valued and has the same shape as 𝐱.
	 */

	x.requires_grad_(true);								//# Same as `x = torch.arange(4.0, requires_grad=True)`
	std::cout <<"x.grad=\n" << x.grad() << std::endl;	//# The default value is None

	// Now let us calculate 𝑦
	auto y = 2 * torch::dot(x, x);
	std::cout << "y = " << y << std::endl;

	/*
	 * Since x is a vector of length 4, an dot product of x and x is performed, yielding the scalar output that we assign to y. Next, [we can automatically
	 * calculate the gradient of y with respect to each component of x] by calling the function for backpropagation and printing the gradient.
	 */
	y.backward();
	std::cout <<"After y.backward(), x.grad=\n" << x.grad() << std::endl;

	/*
	 * (The gradient of the function 𝑦=2𝐱⊤𝐱 with respect to 𝐱 should be 4𝐱.) Let us quickly verify that our desired gradient was calculated correctly.
	 */
	std::cout << (x.grad() == 4 * x) << std::endl;

	// Now let us calculate another function of x.]
	// PyTorch accumulates the gradient in default, we need to clear the previous values
	x.grad().zero_();
	y = x.sum();
	y.backward();
	std::cout <<"After clear the previous values, x.grad=\n" << x.grad() << std::endl;

	// Backward for Non-Scalar Variables
	/*
	 Invoking `backward` on a non-scalar requires passing in a `gradient` argument
 	 which specifies the gradient of the differentiated function w.r.t `self`.
 	 In our case, we simply want to sum the partial derivatives, so passing
 	 in a gradient of ones is appropriate
 	*/
	x.grad().zero_();
	y = x * x;

	// y.backward(torch.ones(len(x))) equivalent to the below
	y.sum().backward();
	std::cout <<"x.grad=\n" << x.grad() << std::endl;

	// Detaching Computation
	/*
	 * Sometimes, we wish to [move some calculations outside of the recorded computational graph.] For example, say that y was calculated as a function of x,
	 * and that subsequently z was calculated as a function of both y and x. Now, imagine that we wanted to calculate the gradient of z with respect to x,
	 * but wanted for some reason to treat y as a constant, and only take into account the role that x played after y was calculated.
	 *
	 * Here, we can detach y to return a new variable u that has the same value as y but discards any information about how y was computed in the computational graph.
	 * In other words, the gradient will not flow backwards through u to x. Thus, the following backpropagation function computes the partial derivative of z = u * x
	 * with respect to x while treating u as a constant, instead of the partial derivative of z = x * x * x with respect to x.
	 */
	x.grad().zero_();
	y = x * x;
	auto u = y.detach();
	auto z = u * x;

	z.sum().backward();
	std::cout << "Detaching Computation:\n" << (x.grad() == u) << std::endl;

	// Since the computation of y was recorded, we can subsequently invoke backpropagation on y to get the derivative of y = x * x with respect to x, which is 2 * x.
	x.grad().zero_();
	y.sum().backward();
	std::cout << "Computation of y was recorded:\n" <<  (x.grad() == 2 * x) << std::endl;

	/*
	 * One benefit of using automatic differentiation is that [even if] building the computational graph of (a function required passing through a maze of control flow,
	 * conditionals, loops, and arbitrary function calls),(we can still calculate the gradient of the resulting variable.)
	 */
	torch::Tensor a = torch::randn({}, options).requires_grad_(true);
	std::cout <<"a = " << a << std::endl;

	auto d = f(a);
	d.backward();

	/*
	 * We can now analyze the f function defined above. Note that it is piecewise linear in its input a. In other words, for any a there exists some constant
	 * scalar k such that f(a) = k * a, where the value of k depends on the input a. Consequently d / a allows us to verify that the gradient is correct.
	 */

	std::cout << "Verify that the gradient:\n" <<  (a.grad() == d / a) << std::endl;

	std::cout << "Done!\n";
	return 0;
}


