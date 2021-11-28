
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

#include <dirent.h>           //get files in directory
#include <sys/stat.h>
#include <cmath>
#include <map>
#include <tuple>

#include "../utils.h"
#include "../utils/transforms.hpp"              // transforms_Compose
#include "../utils/datasets.hpp"                // datasets::ImageFolderClassesWithPaths
#include "../utils/dataloader.hpp"              // DataLoader::ImageFolderClassesWithPaths

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

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
	plt::figure_size(800, 500);
	plt::subplot(1, 2, 1);

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
	 plt::subplot(1, 2, 2);
	 plt::title("image aa");
	 plt::imshow(zptra, aa.size(0), aa.size(1), aa.size(2));
	 plt::show();


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

	//std::string f1 = "./data/dog.jpg";
	//std::string f2 = "./data/dog.jpg";
	//displayImage(f1, f2);

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	size_t img_size = 224;
	size_t batch_size = 16;
	std::vector<std::string> class_names = {"cat", "fish"};
	constexpr bool train_shuffle = true;  // whether to shuffle the training dataset
	constexpr size_t train_workers = 2;  // the number of workers to retrieve data from the training dataset

    // (4) Set Transforms
    std::vector<transforms_Compose> transform {
        transforms_Resize(cv::Size(img_size, img_size), cv::INTER_LINEAR),        // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        transforms_ToTensor()                                                     // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
//        transforms_Normalize(std::vector<float>{0.485, 0.456, 0.406}, std::vector<float>{0.229, 0.224, 0.225})  // Pixel Value Normalization for ImageNet
    };

	std::string dataroot = "./data/cat_fish/train";
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor loss, image, label, output;
    datasets::ImageFolderClassesWithPaths dataset;      // valid_dataset;
    DataLoader::ImageFolderClassesWithPaths dataloader; // valid_dataloader;


    // -----------------------------------
    // a1. Preparation
    // -----------------------------------

    // (1) Get Training Dataset
    dataset = datasets::ImageFolderClassesWithPaths(dataroot, transform, class_names);
    dataloader = DataLoader::ImageFolderClassesWithPaths(dataset, batch_size, /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);
    std::cout << "total training images : " << dataset.size() << std::endl;
    dataloader(mini_batch);
    image = std::get<0>(mini_batch).to(device);

    auto t4mat = image[1].detach().clone();
    t4mat = t4mat.permute({1,2,0}).mul(255).to(torch::kByte);

    std::cout << t4mat.sizes() << std::endl;

    int width = t4mat.size(0);
    int height = t4mat.size(1);
    cv::Mat mgm(cv::Size{ width, height }, CV_8UC3, t4mat.data_ptr<uchar>());

    cv::cvtColor(mgm, mgm, cv::COLOR_RGB2BGR);
    cv::imshow("converted color image", mgm.clone());
    cv::waitKey(0);
    cv::destroyAllWindows();


/*
    plt::figure_size(800, 500);
    plt::subplot(1, 2, 1);
    dataloader(mini_batch);
    image = std::get<0>(mini_batch).to(device);
    label = std::get<1>(mini_batch).to(device);
    auto fnames = std::get<2>(mini_batch);

    std::cout << "images : " << image.sizes() << std::endl;
    std::cout << "labels : " << label << std::endl;
    std::cout << "fnames : " << fnames[0] << std::endl;

    std::vector<unsigned char> z = tensorToMatrix(image[1]);
    const uchar* zptr = &(z[0]);
    int id = label[1].item<int64_t>();
    std::string tlt = class_names[id];
 //   std::string tlt = flowerLabels[t]; //cls[label];
    plt::title(tlt.c_str());
    plt::imshow(zptr, img_size, img_size, 3);

    plt::subplot(1, 2, 2);
    std::vector<unsigned char> z2 = tensorToMatrix(image[7]);
    const uchar* zptr2 = &(z2[0]);
    id = label[7].item<int64_t>();
    tlt = class_names[id];
//    tlt = flowerLabels[t]; //cls[label];
    plt::title(tlt.c_str());
    plt::imshow(zptr2, img_size, img_size, 3);
    plt::show();
*/
    auto class_match = std::vector<size_t>(class_names.size(), 0);
    for(auto& i : class_match) std::cout << i << std::endl;

    std::cout << "Done!\n";
}

