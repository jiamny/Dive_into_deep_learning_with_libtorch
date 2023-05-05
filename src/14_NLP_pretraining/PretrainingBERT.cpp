
#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "../utils/ch_14_util.h"
#include "../TempHelpFunctions.hpp"


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	torch::Device device(torch::kCPU);
	//auto cuda_available = torch::cuda::is_available();
	//torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	//std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);




	std::cout << "Done!\n";
}





