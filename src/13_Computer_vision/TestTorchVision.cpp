#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <torchvision/vision.h>

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	auto& ops = torch::jit::getAllOperators();
	std::cout << "torch jit operators \n";
	for (auto& op: ops) {
	    auto& name = op->schema().name();
	    if(name.find("torchvision") != std::string::npos)
	        std::cout << "op : " << op->schema().name() << "\n";
	}
	std::cout << "\n";

	std::cout << "Done!\n";
	return 0;
}

