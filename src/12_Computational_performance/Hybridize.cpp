#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <chrono>


using namespace std::chrono;


const int add(const int a, const int b) {
    return a + b;
}

const int fancy_func(const int a, const int b, const int c, const int d) {
	const int  e = add(a, b);
	const int  f = add(c, d);
	const int  g = add(e, f);
    return g;
}

// Factory for networks
torch::nn::Sequential get_net() {
    auto net = torch::nn::Sequential(torch::nn::Linear(512, 256),
            torch::nn::ReLU(),
			torch::nn::Linear(256, 128),
			torch::nn::ReLU(),
			torch::nn::Linear(128, 2));
    return net;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	std::cout << fancy_func(1, 2, 3, 4) << "\n";

	// Hybridizing the Sequential Class
	auto x = torch::randn({1, 512});
	auto net = get_net();
	net->to(device);
	std::cout << "get_net: " << net->forward(x) << '\n';

//	net = torch::jit::script(net);
//	net->to(device);
//	std::cout << "torch::jit: " << net->forward(x) << '\n';

	auto start = high_resolution_clock::now();

	for(int i = 0; i < 1000; i++ )
		net->forward(x);

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "duration: " << (duration.count() / 1e6) << " sec.\n";

	std::cout << "Done!\n";
	return 0;
}









