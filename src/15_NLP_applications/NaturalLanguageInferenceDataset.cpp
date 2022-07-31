#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/ch_15_util.h"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	//Read the SNLI dataset into premises, hypotheses, and labels.

	const std::string data_dir = "./data/snli_1.0";
	const bool is_train = true;

	auto dt = read_snli(data_dir,is_train);

	for(int c = 0; c < 20; c++) {
		std::cout << std::get<0>(dt)[c] << '\n';
		std::cout << std::get<1>(dt)[c] << '\n';
		std::cout << std::get<2>(dt)[c] << '\n';
	}

	std::cout << "Done!\n";
}




