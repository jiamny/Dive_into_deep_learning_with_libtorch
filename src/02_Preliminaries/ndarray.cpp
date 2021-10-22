
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <cstdio>

int main() {
	auto options = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU);

	// --------------------------------------------------------------------------------------------------------------------------------------
	// Data Manipulation
	// --------------------------------------------------------------------------------------------------------------------------------------
	/*
	 * A tensor represents a (possibly multi-dimensional) array of numerical values.] With one axis, a tensor corresponds (in math) to a vector.
	 * With two axes, a tensor corresponds to a matrix. Tensors with more than two axes do not have special mathematical names.

	 * To start, we can use arange to create a row vector x containing the first 12 integers starting with 0, though they are created as floats by default.
	 * Each of the values in a tensor is called an element of the tensor. For instance, there are 12 elements in the tensor x. Unless otherwise specified,
	 * a new tensor will be stored in main memory and designated for CPU-based computation.
	 */

	auto x = torch::arange(12);
	std::cout << "x = \n" << x << std::endl;

	/*
	 * We can access a tensor's shape) (and the total number of elements) (the length along each axis) by inspecting its shape property.
	 */

	std::cout << "x shape = " << x.sizes() << std::endl;

	/*
	 * If we just want to know the total number of elements in a tensor, i.e., the product of all of the shape elements, we can inspect its size.
	 * Because we are dealing with a vector here, the single element of its shape is identical to its size.
	 */
	std::cout << "x elements = " << x.numel() << std::endl;

	/*
	 * To [change the shape of a tensor without altering either the number of elements or their values], we can invoke the reshape function.
	 * For example, we can transform our tensor, x, from a row vector with shape (12,) to a matrix with shape (3, 4). This new tensor contains
	 * the exact same values, but views them as a matrix organized as 3 rows and 4 columns. To reiterate, although the shape has changed,
	 * the elements have not. Note that the size is unaltered by reshaping.
	 */
	auto X = x.reshape({3, 4});
	std::cout << "x.reshpae(3,4) = \n" << X << std::endl;

	/*
	 * Fortunately, tensors can automatically work out one dimension given the rest. We invoke this capability by placing -1 for the dimension that
	 * we would like tensors to automatically infer. In our case, instead of calling x.reshape(3, 4), we could have equivalently
	 * called x.reshape(-1, 4) or x.reshape(3, -1)
	 */
	auto X2 = x.reshape({-1, 4});
	std::cout << "x.reshpae(-1,4) = \n" << X2 << std::endl;

	/*
	 * Typically, we will want our matrices initialized either with zeros, ones, some other constants, or numbers randomly sampled from a
	 * specific distribution. [We can create a tensor representing a tensor with all elements set to 0] (or 1) and a shape of (2, 3, 4) as follows:
	 */

	std::cout << "torch.zeros((2, 3, 4)):\n" << torch::zeros({2, 3, 4}) << std::endl;

	/*
	 * Similarly, we can create tensors with each element set to 1 as follows:
	 */
	std::cout << "torch.ones((2, 3, 4)):\n" << torch::ones({2, 3, 4}) << std::endl;

	/*
	 * Often, we want to [randomly sample the values for each element in a tensor] from some probability distribution. For example, when we construct
	 * arrays to serve as parameters in a neural network, we will typically initialize their values randomly. The following snippet creates a tensor with
	 * shape (3, 4). Each of its elements is randomly sampled from a standard Gaussian (normal) distribution with a mean of 0 and a standard deviation of 1.	 *
	 */
	std::cout << "torch.randn(3, 4):\n" << torch::randn({3, 4}) << std::endl;

	/*
	 * We can also [specify the exact values for each element] in the desired tensor
	 * Here, the outermost list corresponds to axis 0, and the inner list to axis 1.
	 */
	std::cout << "torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]):\n";
	std::cout << torch::tensor({{2, 1, 4, 3}, {1, 2, 3, 4}, {4, 3, 2, 1}}) << std::endl;

	// --------------------------------------------------------------------------------------------------------------------------------------
	// Operations
	// --------------------------------------------------------------------------------------------------------------------------------------
	/*
	 * The common standard arithmetic operators (+, -, *, /, and ``) have all been lifted to elementwise operations.**
	 */
	x = torch::tensor({1.0, 2, 4, 8}, options);
	auto y = torch::tensor({2, 2, 2, 2}, options);
	std::cout << "x + y:\n" << (x + y) << std::endl;
	std::cout << "x - y:\n" << (x - y) << std::endl;
	std::cout << "x * y:\n" << (x * y) << std::endl;
	std::cout << "x / y:\n" << (x / y) << std::endl;
	std::cout << "x ** y:\n" << (x.pow(y))  << std::endl; //The ^ operator is exponentiation

	/*
	 * Many (more operations can be applied elementwise), including unary operators like exponentiation.
	 */
	std::cout << "torch.exp(x):\n" << torch::exp(x) << std::endl;

	/*
	 * We can also [concatenate multiple tensors together,] stacking them end-to-end to form a larger tensor. We just need to provide a list of tensors
	 * and tell the system along which axis to concatenate. The example below shows what happens when we concatenate two matrices along rows (axis 0,
	 * the first element of the shape) vs. columns (axis 1, the second element of the shape). We can see that the first output tensor's axis-0 length (6)
	 * is the sum of the two input tensors' axis-0 lengths (3+3); while the second output tensor's axis-1 length (8) is the sum of the two input tensors'
	 * axis-1 lengths (4+4).
	 */
	X = torch::arange(12, options).reshape({3, 4});
	auto Y = torch::tensor({{2.0, 1.0, 4.0, 3.0}, {1.0, 2.0, 3.0, 4.0}, {4.0, 3.0, 2.0, 1.0}}, options);
	std::cout << "torch.cat((X, Y), dim=0):\n" << torch::cat({X, Y}, 0) << std::endl;
	std::cout << "torch.cat((X, Y), dim=1):\n" << torch::cat({X, Y}, 1) << std::endl;

	/*
	 * Sometimes, we want to [construct a binary tensor via logical statements.] Take X == Y as an example. For each position,
	 * if X and Y are equal at that position, the corresponding entry in the new tensor takes a value of 1, meaning that the logical
	 * statement X == Y is true at that position; otherwise that position takes 0
	 */
	std::cout << "X == Y:\n" << (X == Y) << std::endl;

	/*
	 * Summing all the elements in the tensor] yields a tensor with only one element.
	 */
	std::cout << "X.sum() = " << X.sum() << std::endl;

	// --------------------------------------------------------------------------------------------------------------------------------------
	// Broadcasting Mechanism
	// --------------------------------------------------------------------------------------------------------------------------------------
	/*
	 * Under certain conditions, even when shapes differ, we can still [perform elementwise operations by invoking the broadcasting mechanism.]
	 * This mechanism works in the following way: First, expand one or both arrays by copying elements appropriately so that after this transformation,
	 * the two tensors have the same shape. Second, carry out the elementwise operations on the resulting arrays.

	 * In most cases, we broadcast along an axis where an array initially only has length 1, such as in the following example:
	 */
	auto a = torch::arange(3).reshape({3, 1});
	auto b = torch::arange(2).reshape({1, 2});
	std::cout << "a = \n" << a << std::endl;
	std::cout << "b = \n" << b << std::endl;

	/*
	 * Since a and b are 3×1 and 1×2 matrices respectively, their shapes do not match up if we want to add them. We broadcast the entries of both matrices
	 * into a larger 3×2 matrix as follows: for matrix a it replicates the columns and for matrix b it replicates the rows before adding up both elementwise.
	 */
	std::cout << "a + b = \n" << (a + b) << std::endl;

	// --------------------------------------------------------------------------------------------------------------------------------------
	// Indexing and Slicing
	// --------------------------------------------------------------------------------------------------------------------------------------
	/*
	 * the first element has index 0 and ranges are specified to include the first but before the last element.
	 * we can access elements according to their relative position to the end of the list by using negative indices.

	 * Thus, [[-1] selects the last element and [1:3] selects the second and the third elements] as follows:
	 */
	using torch::indexing::Slice;
	using torch::indexing::None;
	using torch::indexing::Ellipsis;

	std::cout << "X = \n" << X << std::endl;
	std::cout << "X[-1] = \n" << X[-1] << std::endl;
	std::cout << "X[1:3] = \n" << X.index({Slice(1,3), Slice()}) << std::endl;

	/*
	 * Beyond reading, (we can also write elements of a matrix by specifying indices.)
	 */
	X[1][2] = 9;
	std::cout << "X = \n" << X << std::endl;

	/*
	 * If we want [to assign multiple elements the same value, we simply index all of them and then assign them the value.] For instance, [0:2, :] accesses
	 * the first and second rows, where : takes all the elements along axis 1 (column). While we discussed indexing for matrices, this obviously also works
	 * for vectors and for tensors of more than 2 dimensions.
	 */

	X.index({Slice(None, 2), Slice()}) = 12;
	std::cout << "(X[0:2, :] = 12): \n" << X << std::endl;

	// --------------------------------------------------------------------------------------------------------------------------------------
	// Saving Memory
	// --------------------------------------------------------------------------------------------------------------------------------------
	/*
	 * After running Y = Y + X, we will find that id(Y) points to a different location. That is because Python first evaluates Y + X,
	 * allocating new memory for the result and then makes Y point to this new location in memory.
	 */
	auto before = &(Y);
	std::cout << "before = " <<  before << std::endl;
	Y = Y + X;
	std::cout << "&(Y) = " << &(Y) << std::endl;
	std::cout << "(&(Y) == before): \n" << (&(Y) == before) << std::endl;

	/*
	 * Fortunately, (performing in-place operations) is easy. We can assign the result of an operation to a previously allocated array
	 *  with slice notation, e.g., Y[:] = <expression>.
	 */
	auto Z = torch::zeros_like(Y);
	auto bZ = &(Z);
	std::printf("id(Z):%p\n", bZ);
	//	Z({Slice()}) = X + Y;
	Z = X + Y;
	auto aZ = &(Z);
	std::printf("id(Z):%p\n", aZ);

	/*
	 * If the value of X is not reused in subsequent computations, we can also use X[:] = X + Y or X += Y to reduce the memory overhead of the operation.
	 */
	before = &(X);
	X += Y;
	std::cout << "(&(X) == before): \n" << (&(X) == before) << std::endl;

	// Conversion to Other Objects
	a = torch::tensor({3.5});

	std::cout << "a.type = "   <<  a.options() << std::endl;
	std::cout << "a = " << a   << std::endl;
	std::cout << "float(a) = " << a.item<float>() << std::endl;
	std::cout << "int(a) = "   << a.item<int>() << std::endl;


	std::cout << "Done!\n";
	return 0;
}




