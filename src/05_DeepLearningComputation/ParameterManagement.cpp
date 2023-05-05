#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

torch::nn::Sequential block1(void){
    return torch::nn::Sequential(torch::nn::Linear(4, 8), torch::nn::ReLU(), torch::nn::Linear(8, 4),
    		torch::nn::ReLU());
}

std::vector<torch::nn::Sequential> block2(void) {
    std::vector<torch::nn::Sequential> net;

    for(int i = 0; i < 4; i++ ) {
        //# Nested here
        net.push_back(block1());
    }

    return net;
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	auto net = torch::nn::Sequential(torch::nn::Linear(4, 8), torch::nn::ReLU(), torch::nn::Linear(8, 1));
	auto X = torch::randn({2, 4});
	std::cout << net->forward(X) << std::endl;

	// Parameter Access
	std::cout << net[2].get()->parameters() << std::endl;

	// Targeted Parameters
	std::cout << "key(): "  << net[2].get()->named_parameters()[1].key()   << std::endl;
	std::cout << "name(): " << net[2].get()->named_parameters()[1]->name() << std::endl;
	std::cout << "value(): " << net[2].get()->named_parameters()[1].value() << std::endl;

	// has_storage
	std::cout << net[2].get()->named_parameters()[0]->grad().defined() << std::endl;

	//All Parameters at Once	auto dict = net->named_parameters();
	auto dict = net->named_parameters();
	for (auto n = dict.begin(); n != dict.end(); n++) {
		std::cout << (*n).key() << std::endl;
	}

	auto tt =  block1();
	std::cout << "tt: " << tt.get()->size() << std::endl;
/*
	// Collecting Parameters from Nested Blocks
	auto rgnet = torch::nn::Sequential(block2(), torch::nn::Linear(4, 1));
	std::cout <<"rgnet: " << rgnet->forward(X) << std::endl;

	// Now that [we have designed the network, let us see how it is organized.]
	std::cout << rgnet << std::endl;
*/
	// Custom Initialization

	// Tied Parameters
	/*
	 * Often, we want to share parameters across multiple layers. Let us see how to do this elegantly.
	 * In the following we allocate a dense layer and then use its parameters specifically to set those of another layer.
	 */
	//# We need to give the shared layer a name so that we can refer to its
	//# parameters
	auto shared = torch::nn::Linear(8, 8);
	net = torch::nn::Sequential(torch::nn::Linear(4, 8), torch::nn::ReLU(), shared, torch::nn::ReLU(), shared,
			torch::nn::ReLU(), torch::nn::Linear(8, 1));
	std::cout << net->forward(X) << std::endl;

	//# Check whether the parameters are the same
	std::cout << (net[2].get()->named_parameters()[0].value()[0] == net[4].get()->named_parameters()[0].value()[0]) << std::endl;

	std::cout << net[2].get()->named_parameters()[0].key() << std::endl;//.weight.data[0, 0] = 100
	std::cout <<"-: " <<  net[2].get()->named_parameters()[0].value().index({0,0}) << std::endl;

	// torch::NoGradGuard no_grad;
	net[2].get()->named_parameters()[0].value().detach().index({0,0}) = 100;
	std::cout <<"+: " <<  net[2].get()->named_parameters()[0].value() << std::endl;

	//# Make sure that they are actually the same object rather than just having the
	//# same value
	//print(net[2].weight.data[0] == net[4].weight.data[0])
	std::cout << (net[2].get()->named_parameters()[0].value()[0] == net[4].get()->named_parameters()[0].value()[0]) << std::endl;

	std::cout << "Done!\n";
	return 0;
}
