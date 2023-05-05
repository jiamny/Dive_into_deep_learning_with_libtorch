#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>


// Layers without Parameters
struct CenteredLayer: public torch::nn::Module {

	explicit CenteredLayer(){}

    torch::Tensor forward( torch::Tensor X) {
        return (X - X.mean());
    }
};

// Layers with Parameters

struct MyLinear : public torch::nn::Module {
	torch::Tensor weight, bias;

	explicit MyLinear(int64_t in_units, int64_t units) {
        weight = torch::randn({in_units, units}, torch::requires_grad(true));
        bias = torch::randn(units, torch::requires_grad(true));
    }

    torch::Tensor forward(torch::Tensor X){
        auto linear = torch::matmul(X, weight.data()) + bias.data();
        return torch::relu(linear);
    }
};

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	/*
	 * Let us verify that our layer works as intended by feeding some data through it.
	 */
	 auto layer = CenteredLayer();
	 std::cout << layer.forward(torch::tensor({1, 2, 3, 4, 5}, torch::TensorOptions().dtype(torch::kFloat))) << std::endl;

	 /*
	  * We can now [incorporate our layer as a component in constructing more complex models.]
	  */
	 auto net = torch::nn::Sequential(torch::nn::Linear(8, 128), CenteredLayer());
	 auto Y = net->forward(torch::randn({4, 8}));
	 std::cout << Y.mean() << std::endl;

	 /*
	  * Next, we instantiate the MyLinear class and access its model parameters.
	  */
	 auto linear = MyLinear(5, 3);
	 std::cout << linear.weight << std::endl;

	 /*
	  * We can [directly carry out forward propagation calculations using custom layers.]
	  */
	 std::cout << linear.forward(torch::randn({2, 5})) << std::endl;

	 /*
	  * We can also (construct models using custom layers.) Once we have that we can use it just like the built-in fully-connected layer.
	  */
	 net = torch::nn::Sequential(MyLinear(64, 8), MyLinear(8, 1));
	 std::cout << net->forward(torch::randn({2, 64})) << std::endl;

	 std::cout << "Done!\n";
	 return 0;
}





