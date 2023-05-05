#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/ch_13_util.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


void display_anchors(int fmap_w, int fmap_h, int w, int h, cv::Mat img, std::vector<float> s) {

    // Values on the first two dimensions do not affect the output
	auto fmap = torch::zeros({1, 10, fmap_h, fmap_w});
	auto anchors = multibox_prior(fmap, s, {1, 2, 0.5});
	auto bbox_scale = torch::tensor({w, h, w, h});

	std::cout << anchors.sizes() << '\n';
	std::cout << anchors.squeeze_() * bbox_scale << '\n';

	show_bboxes(img, anchors.squeeze_() * bbox_scale, {}, {});

	cv::imshow("Add bboxes", img);
	cv::waitKey(-1);
	cv::destroyAllWindows();
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	torch::manual_seed(1000);

	auto rlt = readImg("./data/catdog.jpg");
	cv::Mat img = rlt.first;
	torch::Tensor imgT = rlt.second;
	std::cout << imgT.sizes() << '\n';

	cv::Mat kpImg = img.clone();

	int h = imgT.size(2), w = imgT.size(3);
	std::cout << "h: " << h << ", w: " << w << '\n';

	cv::imshow("Cat & dog", img);
	cv::waitKey(-1);
	cv::destroyAllWindows();

	// -----------------------------------------------
	// Multiscale Anchor Boxes
	// -----------------------------------------------
	std::vector<float> s = {0.15};
	int fmap_w=4, fmap_h=4;

	display_anchors(fmap_w, fmap_h, w, h, img, s);

	// We move on to [reduce the height and width of the feature map by half and use larger anchor boxes
	// to detect larger objects]. When the scale is set to 0.4, some anchor boxes will overlap with each other.
	img = kpImg.clone();
	fmap_w = 2;
	fmap_h = 2;
	std::vector<float> s2 = {0.4};

	display_anchors(fmap_w, fmap_h, w, h, img, s2);

	// Finally, we [further reduce the height and width of the feature map by half and increase the anchor
	// box scale to 0.8]. Now the center of the anchor box is the center of the image.
	fmap_w = 1;
	fmap_h = 1;
	std::vector<float> s3 = {0.8};

	display_anchors(fmap_w, fmap_h, w, h, kpImg, s3);

	std::cout << "Done!\n";
	return 0;
}





