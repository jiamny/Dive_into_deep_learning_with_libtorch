
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

torch::Device try_gpu(size_t i) {
    //"""Return gpu(i) if exists, otherwise return cpu()."""
    if( torch::cuda::device_count() >= (i + 1))
        return torch::Device(torch::kCUDA, i);
    else
    	return torch::kCPU;
}

std::vector<torch::Device> try_all_gpus() {
    //"""Return all available GPUs, or [cpu(),] if no GPU exists."""
	std::vector<torch::Device> devices;

    if( torch::cuda::device_count() > 0 ) {
    	for( size_t i = 0; i < torch::cuda::device_count(); i++ )
    		devices.push_back(torch::Device(torch::kCUDA, i));
    } else {
    	devices.push_back(torch::Device(torch::kCPU));
    }
    return devices;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	std::cout << torch::Device(torch::kCPU) << std::endl;
	std::cout << torch::cuda::is_available() << std::endl;
	std::cout << torch::Device(torch::kCUDA) << std::endl;
	std::cout << torch::Device(torch::kCUDA, 1) << std::endl;

	// We can (query the number of available GPUs.)
	std::cout << torch::cuda::device_count() << std::endl;

	std::cout << try_gpu(0) << std::endl;
	std::cout << try_gpu(10) << std::endl;
	std::cout << try_all_gpus() << std::endl;

	//Tensors and GPUs
	auto x = torch::tensor({1, 2, 3});
	std::cout << x.device() << std::endl;

	// Storage on the GPU
	auto X = torch::ones({2, 3}, torch::TensorOptions().device(try_gpu(0)));
	std::cout << X << std::endl;

	// Assuming that you have at least two GPUs, the following code will (create a random tensor on the second GPU.)
	auto Y = torch::ones({2, 3}, torch::TensorOptions().device(try_gpu(1)));
	std::cout << Y << std::endl;

	// Copying
	auto Z = X.cuda();
	std::cout << X << std::endl;
	std::cout << Z << std::endl;

	// Now that [the data are on the same GPU (both Z and Y are), we can add them up.]
	std::cout << (Z + Y) << std::endl;

	// Neural Networks and GPUs
	auto net = torch::nn::Sequential(torch::nn::Linear(3, 1));
	net->to(try_gpu(0));
	std::cout << net->forward(X) << std::endl;

	// Let us (confirm that the model parameters are stored on the same GPU.)
	std::cout << net[0].get()->named_parameters()[0].value().device() << std::endl;

	std::cout << "Done!\n";
	return 0;
}


