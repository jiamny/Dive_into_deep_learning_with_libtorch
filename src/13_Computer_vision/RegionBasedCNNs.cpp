#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/ch_13_util.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

//#include <torchvision/vision.h>
//#include <torchvision/ops/roi_pool.h>

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);
/*
	// ----------------------------------------------------
	// computation of the region of interest pooling layer.
	// ----------------------------------------------------
	auto X = torch::arange(16.).reshape({1, 1, 4, 4}).to(torch::kFloat32);
	std::cout << "X:\n" << X << '\n';


	auto rois = torch::tensor({{0., 0., 0., 20., 20.}, {0., 0., 10., 30., 30.}}).to(torch::kFloat32);
*/
	/*
	 * Because the height and width of X are 1/10 of the height and width of the input image,
	 * the coordinates of the two region proposals are multiplied by 0.1 according to the specified
	 * spatial_scale argument. Then the two regions of interest are marked on X as X[:, :, 0:3, 0:3]
	 * and X[:, :, 1:4, 0:4], respectively. Finally in the 2×2 region of interest pooling, each region
	 * of interest is divided into a grid of sub-windows to further extract features of the same shape 2×2.
	 */
	// torchvision.ops.roi_pool(X, rois, output_size=(2, 2), spatial_scale=0.1)
/*
	std::tuple<at::Tensor, at::Tensor> rlt = vision::ops::roi_pool(X, rois, 0.1, 2, 2);

	std::cout << "torchvision.ops.roi_pool():\n"
		      << std::get<0>(rlt) << '\n';
			  //<< std::get<1>(rlt) << '\n';
*/
	std::cout << "Done!\n";
}




