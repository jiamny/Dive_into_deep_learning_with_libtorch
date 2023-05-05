#include <unistd.h>
#include <iomanip>
#include <utility>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <sstream>
#include <string>

#include "../utils/ch_13_util.h"


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	torch::manual_seed(1000);

	const std::string data_dir = "./data/banana-detection";
	bool is_train = true;
	int imgSize = 256;
	int batch_size = 32;

	auto data_targets = load_bananas_img_data(data_dir.c_str(), is_train, imgSize);

	auto train_set = BananasDataset(data_targets, imgSize).map(torch::data::transforms::Stack<>());

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
			          	  	  	  	  	  	  	  	  	  	  std::move(train_set), batch_size);

	auto batch = *train_loader->begin();
	auto data  = batch.data;
	auto y     = batch.target;
	std::cout << "data: " << data.sizes() << std::endl;
	std::cout << "y: " << y.sizes() << std::endl;

	// Demonstration
	std::vector<cv::Mat> imgs;

	auto bbox_scale = torch::tensor({imgSize, imgSize, imgSize, imgSize});

	for( int i = 0; i < 10; i ++ ) {
		auto anchors = y[i].squeeze(0);
		auto img_tensor = data[i];

		img_tensor.squeeze_();
		anchors = anchors.index({Slice(1, None)}).reshape({1, -1});

		std::cout << "anchors: " << anchors << '\n';

		torch::Tensor data_out = img_tensor.contiguous().detach().clone();

		auto rev_tensor = data_out.mul(255).to(torch::kByte).permute({1, 2, 0});

		// shape of tensor
		int64_t height = rev_tensor.size(0);
		int64_t width = rev_tensor.size(1);

		// Mat takes data form like {0,0,255,0,0,255,...} ({B,G,R,B,G,R,...})
		// so we must reshape tensor, otherwise we get a 3x3 grid
		auto tensor = rev_tensor.reshape({width * height * rev_tensor.size(2)});

		// CV_8UC3 is an 8-bit unsigned integer matrix/image with 3 channels
		cv::Mat rev_rgb_mat(cv::Size(width, height), CV_8UC3, tensor.data_ptr());
		cv::Mat rev_bgr_mat = rev_rgb_mat.clone();

		cv::cvtColor(rev_bgr_mat, rev_bgr_mat, cv::COLOR_RGB2BGR);

		if( i == 0 ) std::cout << "anchors: " << anchors * bbox_scale << std::endl;
		show_bboxes(rev_bgr_mat, anchors * bbox_scale, {}, {cv::Scalar(255, 255, 255)}, 2);

		imgs.push_back(rev_bgr_mat);
	}
	std::string tlt = "Demonstration";

	ShowManyImages(tlt, imgs);

	std::cout << "Done!\n";
}




