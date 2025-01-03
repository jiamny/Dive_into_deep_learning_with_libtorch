
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdint.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <sstream>
#include <unistd.h>
#include <iomanip>

#include <dirent.h>           //get files in directory
#include <sys/stat.h>
#include <cmath>
#include <map>
#include <tuple>

#include "../utils/ch_13_util.h"
#include "../utils.h"
#include "../utils/transforms.hpp"              // transforms_Compose
#include "../utils/datasets.hpp"                // datasets::ImageFolderClassesWithPaths
#include "../utils/dataloader.hpp"              // DataLoader::ImageFolderClassesWithPaths

#include <matplot/matplot.h>
using namespace matplot;

using torch::indexing::Slice;
using torch::indexing::None;


torch::Tensor load_image(std::string path) {
    cv::Mat mat;

    //mat = cv::imread("./data/dog.jpg", cv::IMREAD_COLOR);
    mat = cv::imread(path.c_str(), cv::IMREAD_COLOR);

//    cv::imshow("origin BGR image", mat);
//    cv::waitKey(0);
//    cv::destroyAllWindows();

    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);

    int h = 224, w = 224;

    int im_h = mat.rows, im_w = mat.cols, chs = mat.channels();
    float res_aspect_ratio = w*1.0/h;
    float input_aspect_ratio = im_w*1.0/im_h;

    int dif = im_w;
    if( im_h > im_w ) int dif = im_h;

    int interpolation = cv::INTER_CUBIC;
    if( dif > static_cast<int>((h+w)*1.0/2) ) interpolation = cv::INTER_AREA;

    cv::Mat Y;

    if( input_aspect_ratio != res_aspect_ratio ) {
        if( input_aspect_ratio > res_aspect_ratio ) {
            int im_w_r = static_cast<int>(input_aspect_ratio*h);
            int im_h_r = h;

            cv::resize(mat, mat, cv::Size(im_w_r, im_h_r), (0,0), (0,0), interpolation);
            int x1 = static_cast<int>((im_w_r - w)/2);
            int x2 = x1 + w;
            mat(cv::Rect(x1, 0, w, im_h_r)).copyTo(Y);
        }

        if( input_aspect_ratio < res_aspect_ratio ) {
            int im_w_r = w;
            int im_h_r = static_cast<int>(w/input_aspect_ratio);
            cv::resize(mat, mat, cv::Size(im_w_r , im_h_r), (0,0), (0,0), interpolation);
            int y1 = static_cast<int>((im_h_r - h)/2);
            int y2 = y1 + h;
            mat(cv::Rect(0, y1, im_w_r, h)).copyTo(Y); // startX,startY,cols,rows
        }
    } else {
    	 cv::resize(mat, Y, cv::Size(w, h), interpolation);
    }

    torch::Tensor img_tensor = torch::from_blob(Y.data, {  Y.channels(), Y.rows, Y.cols }, torch::kByte); // Channels x Height x Width
//    img_tensor = img_tensor.permute({ 2, 0, 1 });
    /*
    std::vector<cv::Mat> channels(3);
    cv::split(Y, channels);
    auto R = torch::from_blob(
    	        	        channels[2].data,
    	        	        {Y.rows, Y.cols},
    	        	        torch::kUInt8);
    auto G = torch::from_blob(
    	        	        channels[1].data,
    	        	        {Y.rows, Y.cols},
    	        	        torch::kUInt8);

    auto B = torch::from_blob(
    	        	        channels[0].data,
    	        	        {Y.rows, Y.cols},
    	        	        torch::kUInt8);

    auto img_tensor = torch::cat({R, G, B})
    	        	                     .view({3, Y.rows, Y.cols})
    	        	                     .to(torch::kByte);
*/
    std::cout << "img_tensor1=" << img_tensor.sizes() << std::endl;

//    auto img_tensor = t_tensor.permute({2 , 0, 1}).clone();
//    std::cout << "img_tensor2=" << img_tensor.sizes() << std::endl;

    auto t = img_tensor.to(torch::kFloat).div_(255.0);
    std::cout << "t=" << t.sizes() << std::endl;
/*
    auto tt = t.detach().clone().mul(255).to(torch::kByte);
    auto t4mat = tt.clone().permute({1, 2, 0});

    int width = t4mat.size(0);
    int height = t4mat.size(1);
    cv::Mat mgm(cv::Size{ width, height }, CV_8UC3, t4mat.data_ptr<uchar>());

    cv::cvtColor(mgm, mgm, cv::COLOR_RGB2BGR);
    cv::imshow("converted color image", mgm.clone());
 	cv::waitKey(0);
 	cv::destroyAllWindows();
*/
    return t.clone();
}

void displayImage(std::string f1, std::string f2) {
	torch::manual_seed(0);

//	plt::figure_size(800, 500);
//	plt::subplot2grid(1, 2, 0, 0, 1, 1);

	torch::Tensor a = load_image(f1);
	torch::Tensor b = load_image(f2);

	torch::Tensor c = torch::stack({a, b}, 0);
/*
	 torch::Tensor a = torch::rand({3,4,4}).mul(255).clamp_max_(255).clamp_min_(0).to(torch::kByte);
//	 a = a.permute({2,0,1});
	 a = a.to(torch::kFloat).div_(255.0);
	 torch::Tensor b = torch::rand({3,4,4}).mul(255).clamp_max_(255).clamp_min_(0).to(torch::kByte);
//	 b = b.permute({2,0,1});
	 b = b.to(torch::kFloat).div_(255.0);
	 torch::Tensor c = torch::stack({a,b},0);

	 std::cout<<a<<std::endl;
	 std::cout<<b<<std::endl;
	 std::cout<<c[0]<<std::endl;
*/
	 a = a.permute({1,2,0}).mul(255).to(torch::kByte);
//	 a = a.to(torch::kByte);
	 std::cout << a.sizes() << std::endl;
/*
	 std::vector<uchar> z(a.size(0) * a.size(1) * a.size(2));
	 std::memcpy(&(z[0]), a.data_ptr<uchar>(),sizeof(uchar)*a.numel());

	 const uchar* zptr = &(z[0]);
	 plt::title("image a");
	 plt::imshow(zptr, a.size(0), a.size(1), a.size(2));

//	 auto aa = c[0].to(torch::kByte); //
	 auto aa = c[0].clone().permute({1,2,0}).mul(255).to(torch::kByte);
	 std::cout << aa.sizes() << std::endl;

	 std::vector<uchar> za(aa.size(0) * aa.size(1) * aa.size(2));
	 std::memcpy(&(za[0]), aa.data_ptr<uchar>(),sizeof(uchar)*aa.numel());

	 const uchar* zptra = &(za[0]);
	 plt::subplot2grid(1, 2, 0, 1, 1, 1);
	 plt::title("image aa");
	 plt::imshow(zptra, aa.size(0), aa.size(1), aa.size(2));
	 plt::show();
*/

	 auto t4mat = c[0].clone().permute({1,2,0}).mul(255).to(torch::kByte);

	 int width = t4mat.size(0);
	 int height = t4mat.size(1);
	 cv::Mat mgm(cv::Size{ width, height }, CV_8UC3, t4mat.data_ptr<uchar>());

	 cv::cvtColor(mgm, mgm, cv::COLOR_RGB2BGR);
	 cv::imshow("converted color image", mgm.clone());
	 cv::waitKey(0);
	 cv::destroyAllWindows();
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	// -----------------------------------
	// The Pokemon Dataset
	// -----------------------------------
	int64_t batch_size = 2;
	bool train_shuffle = true;
	int train_workers = 2;
	//std::string dataroot = "./data/pokemon";
	std::string dataroot = "./data/Test";
	std::vector<std::string> classes;
	//for(int i = 1; i < 722; i++)
	//	classes.push_back(std::to_string(i));

	classes.push_back(std::to_string(484));
	classes.push_back(std::to_string(485));

	std::vector<float> mean_ = {0.5, 0.5, 0.5};
	std::vector<float> std_  = {0.5, 0.5, 0.5};

	// Set Transforms
	std::vector<transforms_Compose> transform {
	        transforms_Resize(cv::Size(64, 64), cv::INTER_LINEAR),  // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
	        transforms_ToTensor(),                                  // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
			transforms_Normalize(mean_, std_)  						// Pixel Value Normalization for ImageNet
	};

	datasets::ImageFolderClassesWithPaths dataset;
	std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> mini_batch;
	DataLoader::ImageFolderClassesWithPaths dataloader;

	// Get Dataset
	dataset = datasets::ImageFolderClassesWithPaths(dataroot, transform, classes);
	dataloader = DataLoader::ImageFolderClassesWithPaths(dataset, batch_size, train_shuffle, train_workers);

	std::cout << "total training images : " << dataset.size() << std::endl;

	dataloader(mini_batch);
	torch::Tensor images = std::get<0>(mini_batch).to(device);
	auto label = std::get<1>(mini_batch).to(device);
	torch::Tensor imgT = images[0].clone().squeeze();

	std::cout << imgT.max() << " " << label[0] << '\n';

	torch::Tensor photo;
	bool use_photo = true;
	bool use_cv_imshow = true;
    bool normalize = true;

    if( normalize ) {
    	imgT = deNormalizeTensor(imgT, mean_, std_);
    }

	if( use_photo ) {
		std::string filename = "./data/pokemon/485/485-0.png";
		cv::Mat img = cv::imread(filename);

		cv::Mat cvtImg = img.clone();
		cv::cvtColor(img, cvtImg, cv::COLOR_BGR2RGB);

		cv::resize(cvtImg, cvtImg, cv::Size(64, 64), 0.0, 0.0, cv::INTER_LINEAR);

		photo = torch::from_blob(cvtImg.data, {cvtImg.rows, cvtImg.cols, cvtImg.channels()}, at::TensorOptions(torch::kByte)).clone();
		photo = photo.toType(torch::kFloat);
		photo = photo.permute({2, 0, 1});
		photo = photo.div_(255.0);

		photo = torch::unsqueeze(photo, 0);

		filename = "./data/pokemon/484/484-0.png";
		img = cv::imread(filename);

		cvtImg = img.clone();
		cv::cvtColor(img, cvtImg, cv::COLOR_BGR2RGB);
		cv::resize(cvtImg, cvtImg, cv::Size(64, 64), 0.0, 0.0, cv::INTER_LINEAR);

		torch::Tensor photo2 = torch::from_blob(cvtImg.data, {cvtImg.rows, cvtImg.cols, cvtImg.channels()}, at::TensorOptions(torch::kByte)).clone();
		photo2 = photo2.toType(torch::kFloat);
		photo2 = photo2.permute({2, 0, 1});
		photo2 = photo2.div_(255.0);

		photo2 = torch::unsqueeze(photo2, 0);
		photo = torch::cat({photo, photo2}, 0);

		imgT  = photo[0].clone().squeeze();
	}

	cv::Mat img2 = TensorToCvMat(imgT.clone());

	if( use_cv_imshow ) {
		std::string ty =  cvMatType2Str( img2.type() );
		std::cout << "img2_isContinuous: " << img2.isContinuous() << ", Type: " << ty << '\n';

		cv::resize(img2, img2, cv::Size(264, 264));
		cv::imshow("example", img2);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}

	torch::Tensor imgT2;

	if( ! use_photo) {
		imgT2 = images[1].clone().squeeze();
	} else {
		imgT2 = photo[1].clone().squeeze();
	}

    if( normalize ) {
    	imgT2 = deNormalizeTensor(imgT2, mean_, std_);
    }

	cv::Mat img3 = TensorToCvMat(imgT2.clone());

	if( use_cv_imshow ) {
		std::string ty =  cvMatType2Str( img3.type() );
		std::cout << "img3_isContinuous: " << img3.isContinuous() << ", Type: " << ty << '\n';

		cv::resize(img3, img3, cv::Size(264, 264));
		cv::imshow("example2", img3);
		cv::waitKey(0);
		cv::destroyAllWindows();
	} else {
		std::string ty =  cvMatType2Str( img2.type() );
		std::cout << "img2_isContinuous: " << img2.isContinuous() << ", Type: " << ty << '\n';
/*
		plt::figure_size(800, 400);

		std::vector<uint8_t> z1 = tensorToMatrix4Matplotlib(imgT.clone());
		uint8_t* zptr1 = &(z1[0]);

		plt::subplot2grid(1, 2, 0, 0, 1, 1);
		plt::imshow(zptr1, imgT.size(1), imgT.size(2), imgT.size(0));

		std::vector<uint8_t> array = tensorToMatrix4Matplotlib(imgT2.clone());
		uint8_t* zptr = &(array[0]);
		plt::subplot2grid(1, 2, 0, 1, 1, 1);
		plt::imshow(zptr, imgT2.size(1), imgT2.size(2), imgT2.size(0));
		plt::show();
		plt::close();
*/
	}

    std::cout << "Done!\n";
}

