#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/Ch_13_util.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>



int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	const std::string data_dir = "./data/wikitext-2";
	std::vector<std::vector<std::string>> data = _read_wiki(data_dir);

	std::cout << data.size() << '\n';
	std::vector<std::string> lines = data[0];
	for(auto& s : lines)
		std::cout << s << '\n';





	std::cout << "Done!\n";
}



