#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <chrono>

#include "../utils.h"

using namespace std::chrono;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);
	auto start = high_resolution_clock::now();

	for(auto& i : range(10) ) {
	    auto a = torch::randn({1000, 1000}, device);
	    auto b = torch::mm(a, a);
	}
//	torch::cuda::synchronize(device.index());
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout << "Duration: " << (duration.count() / 1e6) << " sec.\n";

	auto x = torch::ones({1, 2}, device);
	auto y = torch::ones({1, 2}, device);
	auto z = x * y + 2;
	std::cout << "z: \n" << z << "\n";

	std::cout << "Done!\n";
	return 0;
}



