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


// DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

std::vector<std::pair<std::string, torch::Tensor>> load_img_data(const std::string data_dir,  bool is_train) {
    //Read the banana detection dataset images and labels
	std::vector<std::pair<std::string, torch::Tensor>> imgpath_tgt;
	std::ifstream file;

	std::string img_dir;
	std::string csv_fname = "";
	if( is_train ) {
	    csv_fname = data_dir + "/bananas_train/label.csv";
		img_dir = data_dir + "/bananas_train/images/";
	} else {
	    csv_fname = data_dir + "/bananas_val/label.csv";
	    img_dir = data_dir + "/bananas_val/images/";
	}

	file.open(csv_fname, std::ios_base::in);
	// Exit if file not opened successfully
	if( !file.is_open() ) {
		std::cout << "File not read successfully" << std::endl;
	    std::cout << "Path given: " << csv_fname << std::endl;
	    exit(-1);
	}

	// skip head line
	std::string line;
	std::getline(file, line);

	while( std::getline(file, line) ) {
		std::stringstream lineStream(line);
		std::string token;

		int cnt = 0;
		std::vector<float> t_data;
		std::string img_f;
		while (std::getline(lineStream, token, ',')) {
			if( token.length() > 0 ) {
				if( cnt == 0 ) {
					img_f = img_dir + token;
				} else {
					t_data.push_back(stof(token));
				}
				cnt++;
			}
		}

		auto image = cv::imread(img_f.c_str());
		// check if image is ok
		if(! image.empty() ) {
			torch::Tensor target = torch::from_blob(t_data.data(), {static_cast<long>(t_data.size())},
												at::TensorOptions(torch::kFloat)).clone().div_(256);
			//std::cout << "target: " << target.sizes() << '\n';
			imgpath_tgt.push_back(std::make_pair(img_f, target));
		}
	}
	std::cout << "size: " << imgpath_tgt.size() << '\n';
	return imgpath_tgt;
}

using mData = std::vector<std::pair<std::string, torch::Tensor>>;
using Example = torch::data::Example<>;

class CustomDataset:public torch::data::Dataset<CustomDataset>{
public:
	CustomDataset(const mData& data, int imgSize) : data_(data) { img_size = imgSize; }

    // Override get() function to return tensor at location index
    Example get(size_t index) override{

    	auto image = cv::imread(data_.at(index).first.c_str());
    	// ----------------------------------------------------------
    	// opencv BGR format change to RGB
    	// ----------------------------------------------------------
    	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    	cv::resize(image, image, cv::Size(img_size, img_size));

    	//cv::Mat img_float;
    	//image.convertTo(img_float, CV_32F, 1.0/255);
    	torch::Tensor img = torch::from_blob(image.data,
    					{image.rows, image.cols, image.channels()}, at::TensorOptions(torch::kByte)).clone(); // Channels x Height x Width

    	img = img.permute({2, 0, 1}).to(torch::kFloat).div_(255.0);
    	return {img, data_.at(index).second};
    }

    // Return the length of data
    torch::optional<size_t> size() const override {
        return data_.size();
    };

private:
    mData data_;
    int img_size;
};


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	const std::string data_dir = "./data/banana-detection";
	bool is_train = true;
	int imgSize = 256;
	int batch_size = 32;

	auto data_targets = load_img_data(data_dir, is_train);

	auto train_set = CustomDataset(data_targets, imgSize).map(torch::data::transforms::Stack<>());

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
			          	  	  	  	  	  	  	  	  	  	  std::move(train_set), batch_size);

	auto batch = *train_loader->begin();
	auto data  = batch.data.to(device);
	auto y     = batch.target.to(device);
	std::cout << "data: " << data.sizes() << std::endl;
	std::cout << "y: " << y.sizes() << std::endl;

	// Demonstration
	std::vector<cv::Mat> imgs;

	auto bbox_scale = torch::tensor({imgSize, imgSize, imgSize, imgSize});

	for( int i = 0; i < 10; i ++ ) {
		auto anchors = y[i];
		auto img_tensor = data[i];

		img_tensor.squeeze_();
		anchors = anchors.index({Slice(1, None)}).reshape({1, -1});

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




