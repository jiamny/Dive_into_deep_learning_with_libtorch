#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/ch_13_util.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <matplot/matplot.h>
using namespace matplot;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	torch::manual_seed(1000);

	auto rlt = readImg("./data/catdog.jpg");
	cv::Mat img = rlt.first;
	torch::Tensor imgT = rlt.second;
	std::cout << imgT.sizes() << '\n';

	std::cout << imgT.squeeze_().sizes() << '\n';

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);
	auto ax1 = F->nexttile();
	std::vector<std::vector<std::vector<unsigned char>>> z = tensorToMatrix4MatplotPP(imgT.squeeze().clone());
	matplot::imshow(ax1, z);
	matplot::show();


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

	auto dog_x = {60.0, 60.0,  378.0, 378.0, 60.0};
	auto dog_y = {45.0, 516.0, 516.0, 45.0, 45.0};
	auto cat_x = {400.0, 400.0,  655.0, 655.0, 400.0};
	auto cat_y = {112.0, 493.0, 493.0, 112.0, 112.0};

	F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);
	ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::imshow(ax1, z);
	matplot::plot(ax1, dog_x, dog_y, "b-")->line_width(2);
	matplot::plot(ax1, cat_x, cat_y, "r-")->line_width(2);
	matplot::hold(ax1, false);
	matplot::title("cat & dog with bbox");
	matplot::show();

	std::cout << "Done!\n";
	return 0;
}




