#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

/*
 * summarize the basic functionality that each block must provide:

    Ingest input data as arguments to its forward propagation function.
    Generate an output by having the forward propagation function return a value. Note that the output may have a different shape from the input. For example, the first fully-connected layer in our model above ingests an input of dimension 20 but returns an output of dimension 256.
    Calculate the gradient of its output with respect to its input, which can be accessed via its backpropagation function. Typically this happens automatically.
    Store and provide access to those parameters necessary to execute the forward propagation computation.
    Initialize model parameters as needed.
 */
struct MLP : public torch::nn::Module {
	torch::nn::Linear hidden{nullptr}, out{nullptr};
    //# Declare a layer with model parameters. Here, we declare two fully
    //# connected layers
    explicit MLP(void) {
        //# Call the constructor of the `MLP` parent class `Module` to perform
        //# the necessary initialization. In this way, other function arguments
        //# can also be specified during class instantiation, such as the model
        //# parameters, `params` (to be described later)
        hidden = torch::nn::Linear(20, 256);  // Hidden layer
        out = torch::nn::Linear(256, 10);     // Output layer
    }

    //# Define the forward propagation of the model, that is, how to return the
    //# required model output based on the input `X`
    torch::Tensor forward(torch::Tensor X) {
        //# Note here we use the funtional version of ReLU defined in the
        //# nn.functional module.
        return out->forward(torch::relu(hidden->forward(X)));
    }
};

/*
 * The Sequential Block
 * To build our own simplified MySequential, we just need to define two key function:

    A function to append blocks one by one to a list.
    A forward propagation function to pass an input through the chain of blocks, in the same order as they were appended.
 */
struct MySequential : public torch::nn::Module {
	torch::nn::Sequential modules;
    explicit MySequential(torch::nn::Linear l1, torch::nn::ReLU relu, torch::nn::Linear l2) {
        //for idx, module in enumerate(args):
        //# Here, `module` is an instance of a `Module` subclass. We save it
        //# in the member variable `_modules` of the `Module` class, and its
        //# type is OrderedDict
        //self._modules[str(idx)] = module
    	modules = torch::nn::Sequential(l1, relu, l2);
    }

    torch::Tensor forward(torch::Tensor  X) {
        //# OrderedDict guarantees that members will be traversed in the order
        //# they were added
        return modules->forward(X);
    }
};

// Executing Code in the Forward Propagation Function
struct FixedHiddenMLP : public torch::nn::Module {
	torch::nn::Linear linear{nullptr};
	torch::Tensor rand_weight;
    explicit FixedHiddenMLP(void) {
        // # Random weight parameters that will not compute gradients and
        // # therefore keep constant during training
        rand_weight = torch::randn({20, 20}, torch::requires_grad(false));
        linear = torch::nn::Linear(20, 20);
    }

    torch::Tensor forward(torch::Tensor X) {
        X = linear->forward(X);
        //# Use the created constant parameters, as well as the `relu` and `mm`
        //# functions
        X = torch::relu(torch::mm(X, rand_weight) + 1);
        //# Reuse the fully-connected layer. This is equivalent to sharing
        //# parameters with two fully-connected layers
        X = linear->forward(X);
        //# Control flow
        while(X.abs().sum().item<float>() > 1 )
            X /= 2;
        return X.sum();
    }
};

// In the following example, we nest blocks in some creative ways.
struct NestMLP : public torch::nn::Module {
    torch::nn::Sequential net{nullptr};
    torch::nn::Linear linear{nullptr};
	explicit NestMLP(void) {
        net = torch::nn::Sequential(torch::nn::Linear(20, 64), torch::nn::ReLU(),
                                 torch::nn::Linear(64, 32), torch::nn::ReLU());
        linear = torch::nn::Linear(32, 16);
	}

    torch::Tensor forward(torch::Tensor X){
        return linear->forward(net->forward(X));
    }
};

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	/*
	 * Layers and Blocks
	 */
	auto net = torch::nn::Sequential(torch::nn::Linear(20, 256), torch::nn::ReLU(), torch::nn::Linear(256, 10));
	auto X = torch::randn({2, 20});
	auto output = net->forward(X);
	std::cout << output << std::endl;
	std::cout << output.grad_fn() << std::endl;

	// A Custom Block
	auto mlp = MLP();
	std::cout << mlp.forward(X) << std::endl;

	// The Sequential Block
	auto myseq = MySequential(torch::nn::Linear(20, 256), torch::nn::ReLU(), torch::nn::Linear(256, 10));
	std::cout << myseq.forward(X) << std::endl;

	// Executing Code in the Forward Propagation Function
	auto fixnet = FixedHiddenMLP();
	std::cout << fixnet.forward(X) << std::endl;

	// In the following example, we nest blocks in some creative ways.
	auto chimera = torch::nn::Sequential(NestMLP(), torch::nn::Linear(16, 20), FixedHiddenMLP());
	std::cout << chimera->forward(X) << std::endl;

	std::cout << "Done!\n";
	return 0;
}

