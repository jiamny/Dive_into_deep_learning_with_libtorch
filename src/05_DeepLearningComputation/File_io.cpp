#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

struct  MLPImpl : public torch::nn::Module {
	torch::nn::Linear hidden{nullptr}, output{nullptr};
	explicit MLPImpl(void) {
		hidden = torch::nn::Linear(20, 256);
    	output = torch::nn::Linear(256, 10);
	}

	torch::Tensor forward(torch::Tensor x){
		return output->forward(torch::relu(hidden->forward(x)));
	}
};

TORCH_MODULE(MLP);

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Loading and Saving Tensors
	auto x = torch::arange(4);
	std::cout << "x: " << x << std::endl;
	torch::save(x, "x-file");

	/*
	 * We can now read the data from the stored file back into memory.
	 */
	//auto x2 = torch::arange(4);
	torch::Tensor x2;
	torch::load(x2, "x-file");
	std::cout << "x2: " << x2 << std::endl;

	// Loading and Saving Model Parameters
	MLP net = MLP();
	auto X = torch::randn({2, 20});
	auto Y = net->forward(X);

	std::cout << "net: " << net << std::endl;
	std::cout << "net.hidden: " << net.get()->hidden << std::endl;
	std::cout << "net.output: " << net.get()->output << std::endl;
	std::cout << "net.hidden.weight: " << net.get()->hidden.get()->weight[0] << std::endl;
	// Next, we [store the parameters of the model as a file] with the name "mlp.params".
	torch::save(net, "mlp.pth");

	/*
	 * To recover the model, we instantiate a clone of the original MLP model.
	 * Instead of randomly initializing the model parameters, we [read the parameters stored in the file directly].
	 */

	MLP clone = net;
	std::cout << "before load, clone.hidden.weight: " << clone.get()->hidden.get()->weight[0] << std::endl;
	torch::load(clone, "mlp.pth");
	clone->eval(); //
	//clone.l (torch::load('mlp.params'))
	std::cout << "clone.hidden: " << clone.get()->hidden << std::endl;
	std::cout << "clone.output: " << clone.get()->output << std::endl;
	std::cout << "clone.hidden.weight: " << clone.get()->hidden.get()->weight[0] << std::endl;

	/*
	 * Since both instances have the same model parameters, the computational result of the same input X should be the same. Let us verify this
	 */
	auto Y_clone = clone->forward(X);
	std::cout << "Y: " << Y << std::endl;
	std::cout << "Y_clone: " << Y_clone << std::endl;
	std::cout << "Y_clone == Y: " << (Y_clone == Y) << std::endl;

	auto Y_net = net->forward(X);
	std::cout << "Y_net == Y: " << (Y_net == Y) << std::endl;

	std::cout << "Done!\n";
	return 0;
}




