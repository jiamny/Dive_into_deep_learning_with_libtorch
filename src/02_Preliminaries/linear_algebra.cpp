
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

int main() {
	// ------------------------------------
	// Scalars
	// ------------------------------------
	/*
	 * (A scalar is represented by a tensor with just one element.) In the next snippet, we instantiate two scalars and perform some familiar arithmetic
	 * operations with them, namely addition, multiplication, division, and exponentiation.
	 */
	auto x = torch::tensor({3.0});
	auto y = torch::tensor({2.0});

	std::cout << "x + y:\n" << (x + y) << std::endl;
	std::cout << "x - y:\n" << (x - y) << std::endl;
	std::cout << "x * y:\n" << (x * y) << std::endl;
	std::cout << "x / y:\n" << (x / y) << std::endl;
	std::cout << "x ** y:\n" << (x.pow(y))  << std::endl;

	// ------------------------------------
	// Vectors
	// ------------------------------------
	/*
	 * You can think of a vector as simply a list of scalar values.] We call these values the elements (entries or components) of the vector.
	 * When our vectors represent examples from our dataset, their values hold some real-world significance
	 */
	x = torch::arange(4);
	std::cout << "x = \n" << x << std::endl;
	/*
	 * access any element by indexing into the tensor
	 */
	std::cout << "x[3] = \n" << x[3] << std::endl;

	// ------------------------------------
	// Length, Dimensionality, and Shape
	// ------------------------------------
	std::cout << "len(x) = \n" << x.size(0) << std::endl;

	/*
	 * For tensors with just one axis, the shape has just one element.
	 */
	std::cout << "x.shape = \n" << x.sizes() << std::endl;

	// ------------------------------------
	// Matrices
	// ------------------------------------
	/*
	 * create an 𝑚×𝑛 matrix]
	 */
	auto A = torch::arange(20).reshape({5, 4});
	std::cout << "A = \n" << A << std::endl;

	/*
	 * matrix's transpose
	 */
	std::cout << "A.T = \n" << A.transpose(1, 0) << std::endl;

	/*
	 * As a special type of the square matrix, [a symmetric matrix 𝐀 is equal to its transpose: 𝐀=𝐀⊤.] Here we define a symmetric matrix B.
	 */
	auto B = torch::tensor({{1, 2, 3}, {2, 0, 4}, {3, 4, 5}});
	std::cout << "B = \n" << B << std::endl;

	/*
	 * Now we compare B with its transpose.
	 */
	std::cout << "B == B.T \n" << (B == B.transpose(1, 0)) << std::endl;

	// ------------------------------------
	// Matrices
	// ------------------------------------
	/*
	 * a generic way of describing 𝑛-dimensional arrays with an arbitrary number of axes
	 */
	auto X = torch::arange(24).reshape({2, 3, 4});
	std::cout << "X = \n" << X << std::endl;

	// ------------------------------------
	// Basic Properties of Tensor Arithmetic
	// ------------------------------------
	/*
	 * given any two tensors with the same shape, the result of any binary elementwise operation will be a tensor of that same shape.
	 */
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
	A = torch::arange(20, options).reshape({5, 4});
	B = A.clone(); // Assign a copy of `A` to `B` by allocating new memory
	std::cout << "A = \n" << A << std::endl;
	std::cout << "A + B = \n" << (A + B) << std::endl;

	/*
	 * Specifically, [elementwise multiplication of two matrices is called their Hadamard product] (math notation ⊙).
	 */
	std::cout << "A * B = \n" << (A * B) << std::endl;

	/*
	 * [Multiplying or adding a tensor by a scalar] also does not change the shape of the tensor, where each element of the operand tensor
	 * will be added or multiplied by the scalar.
	 */
	auto a = 2;
	X = torch::arange(24).reshape({2, 3, 4});

	std::cout << "a + X = \n" << (a + X) << std::endl;
	std::cout << "(a * X).shape = \n" << (a * X).sizes() << std::endl;

	// ------------------------------------
	// Reduction
	// ------------------------------------
	/*
	 * One useful operation that we can perform with arbitrary tensors is to calculate [the sum of their elements.]
	 */
	x = torch::arange(4, options);
	std::cout << "x = \n" << x << std::endl;
	std::cout << "x.sum() = \n" << x.sum() << std::endl;

	/*
	 * We can express [sums over the elements of tensors of arbitrary shape.] For example, the sum of the elements of an 𝑚×𝑛 matrix 𝐀 could be written ∑𝑚𝑖=1∑𝑛𝑗=1𝑎𝑖𝑗.
	 */
	std::cout << "A.shape = \n" << A.sizes() << std::endl;
	std::cout << "A.sum() = \n" << A.sum() << std::endl;

	/*
	 * By default, invoking the function for calculating the sum reduces a tensor along all its axes to a scalar. We can also [specify the axes along which
	 * the tensor is reduced via summation.] Take matrices as an example. To reduce the row dimension (axis 0) by summing up elements of all the rows,
	 * we specify axis=0 when invoking the function. Since the input matrix reduces along axis 0 to generate the output vector, the dimension of axis 0 of
	 * the input is lost in the output shape.
	 */
	std::cout << "A_sum_axis0 = \n" << A.sum(0) << std::endl;
	std::cout << "A_sum_axis0.shape = \n" << A.sum(0).sizes() << std::endl;

	/*
	 * Specifying axis=1 will reduce the column dimension (axis 1) by summing up elements of all the columns. Thus, the dimension of axis 1 of the
	 * input is lost in the output shape.
	 */
	std::cout << "A_sum_axis1 = \n" << A.sum(1) << std::endl;
	std::cout << "A_sum_axis1.shape = \n" << A.sum(1).sizes() << std::endl;

	/*
	 * Reducing a matrix along both rows and columns via summation is equivalent to summing up all the elements of the matrix.
	 */
	std::cout << "A.sum() = \n" << A.sum() << std::endl;
	std::cout << "A dim names = \n" << A.names() << std::endl;
	//std::cout << "A.sum({0,1}) = \n" << A.sum(A.names()) << std::endl;

	/*
	 * [A related quantity is the mean, which is also called the average.] We calculate the mean by dividing the sum by the total number of elements.
	 * In code, we could just call the function for calculating the mean on tensors of arbitrary shape.
	 */
	std::cout << "A.mean() = \n" << A.mean() << std::endl;
	std::cout << "A.sum() / A.numel() = \n" << (A.sum() / A.numel()) << std::endl;

	/*
	 * Likewise, the function for calculating the mean can also reduce a tensor along the specified axes.
	 */
	std::cout << "A.mean(axis=0) = \n" << A.mean(0) << std::endl;
	std::cout << "A.sum(axis=0) / A.shape[0] = \n" << (A.sum(0) / A.sizes()[0]) << std::endl;

	// ------------------------------------
	// Reduction
	// ------------------------------------
	/*
	 * However, sometimes it can be useful to [keep the number of axes unchanged] when invoking the function for calculating the sum or mean.
	 */
	auto sum_A = A.sum(1, true); // sum_A = A.sum(axis=1, keepdims=True)
	std::cout << "sum_A = \n" << sum_A << std::endl;

	/*
	 * For instance, since sum_A still keeps its two axes after summing each row, we can (divide A by sum_A with broadcasting.)
	 */
	std::cout << "A / sum_A = \n" << (A / sum_A) << std::endl;

	/*
	 * If we want to calculate [the cumulative sum of elements of A along some axis], say axis=0 (row by row), we can call the cumsum
	 * function. This function will not reduce the input tensor along any axis.
	 */
	std::cout << "A.cumsum(axis=0) = \n" << A.cumsum(0) << std::endl;

	// ------------------------------------
	// Dot Products
	// ------------------------------------
	/*
	 * one of the most fundamental operations is the dot product. Given two vectors 𝐱,𝐲∈ℝ𝑑, their dot product 𝐱⊤𝐲 (or ⟨𝐱,𝐲⟩) is a
	 * sum over the products of the elements at the same position: 𝐱⊤𝐲=∑𝑑𝑖=1𝑥𝑖𝑦𝑖.
	 */
	y = torch::ones(4, options);
	std::cout << "x = \n" << x << std::endl;
	std::cout << "y = \n" << y << std::endl;
	std::cout << "torch.dot(x, y) = \n" << torch::dot(x, y) << std::endl;

	/*
	 * Note that (we can express the dot product of two vectors equivalently by performing an elementwise multiplication and then a sum:)
	 */
	std::cout << "torch.sum(x * y) = \n" << torch::sum(x * y) << std::endl;

	// ------------------------------------
	// Matrix-Vector Products
	// ------------------------------------
	/*
	 * Expressing matrix-vector products in code with tensors, we use the mv function. When we call torch.mv(A, x) with a matrix A and a vector x,
	 * the matrix-vector product is performed. Note that the column dimension of A (its length along axis 1) must be the same as the dimension of x (its length).
	 */
	std::cout << "A.shape = \n" << A.sizes() << std::endl;
	std::cout << "x.shape = \n" << x.sizes() << std::endl;
	std::cout << "torch.mv(A, x) = \n" << torch::mv(A, x) << std::endl;

	// ------------------------------------
	// Matrix-Matrix Multiplication
	// ------------------------------------
	/*
	 *  Here, A is a matrix with 5 rows and 4 columns, and B is a matrix with 4 rows and 3 columns. After multiplication, we obtain a matrix with 5 rows and 3 columns.
	 */
	B = torch::ones({4, 3});
	std::cout << "torch.mm(A, B) = \n" << torch::mm(A, B) << std::endl;

	// ------------------------------------
	// Norms
	// ------------------------------------
	/*
	 * The 𝐿2 norm of 𝐱 is the square root of the sum of the squares of the vector elements
	 */
	auto u = torch::tensor({3.0, -4.0});
	std::cout << "torch.norm(u) = \n" << torch::norm(u) << std::endl;

	/*
	 * the 𝐿1 norm], which is expressed as the sum of the absolute values of the vector elements:
	 */
	std::cout << "torch.abs(u).sum() = \n" << torch::abs(u).sum() << std::endl;

	/*
	 * he Frobenius norm of a matrix 𝐗∈ℝ𝑚×𝑛] is the square root of the sum of the squares of the matrix elements:
	 */
	std::cout << "torch.norm(torch.ones((4, 9))) = \n" << torch::norm(torch::ones({4, 9})) << std::endl;

	std::cout << "Done!\n";
	return 0;
}



