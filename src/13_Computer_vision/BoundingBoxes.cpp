#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/Ch_13_util.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	auto rlt = readImg("./data/catdog.jpg");
	cv::Mat img = rlt.first;
	torch::Tensor imgT = rlt.second;
	std::cout << imgT.sizes() << '\n';

	std::cout << imgT.squeeze_().sizes() << '\n';

	imgT = imgT.permute({1,2,0}).mul(255).to(torch::kByte);

	std::cout << imgT.sizes() << std::endl;

	std::vector<uchar> z(imgT.size(0) * imgT.size(1) * imgT.size(2));
	std::memcpy(&(z[0]), imgT.data_ptr<uchar>(),sizeof(uchar)*imgT.numel());

	const uchar* zptr = &(z[0]);
	plt::title("cat and dog");
	plt::imshow(zptr, imgT.size(0), imgT.size(1), imgT.size(2));
	plt::show();
	plt::close();

	// -------------------------------------------------------------
	// define the bounding boxes of the dog and the cat in the image
	// -------------------------------------------------------------
	//Here `bbox` is the abbreviation for bounding box
	auto dog_bbox = {60.0, 45.0, 378.0, 516.0}, cat_bbox = {400.0, 112.0, 655.0, 493.0};

	// We can verify the correctness of the two bounding box conversion functions by converting twice.
	auto boxes = torch::tensor({{60.0, 45.0, 378.0, 516.0}, {400.0, 112.0, 655.0, 493.0}});
	auto is_equal = (box_center_to_corner(box_corner_to_center(boxes)) == boxes);
	std:: cout << is_equal << '\n';

	// ---------------------------------------------
	// draw the bounding boxes in the image
	// ---------------------------------------------
	plt::title("cat & dog with bbox");
	plt::imshow(zptr, imgT.size(0), imgT.size(1), imgT.size(2));
	auto dog_x = {60.0, 60.0,  378.0, 378.0, 60.0};
	auto dog_y = {45.0, 516.0, 516.0, 45.0, 45.0};
	plt::plot(dog_x, dog_y, "b-");
	auto cat_x = {400.0, 400.0,  655.0, 655.0, 400.0};
	auto cat_y = {112.0, 493.0, 493.0, 112.0, 112.0};
	plt::plot(cat_x, cat_y, "r-");
	plt::show();
	plt::close();

	std::cout << "Done!\n";
	return 0;
}




