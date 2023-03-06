#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>

#include <matplot/matplot.h>
using namespace matplot;

using std::cout;
using std::map;
using std::string;
using std::vector;

int main() {
	std::string filename = "./data/front.jpg";
	cv::Mat Y = cv::imread(filename, cv::IMREAD_COLOR);

	cv::cvtColor(Y, Y, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> channels(Y.channels());
    cv::split(Y, channels);

    std::vector<std::vector<std::vector<unsigned char>>> image;

    for(size_t i = 0; i < Y.channels(); i++) {
    	std::vector<std::vector<unsigned char>> ch;
    	std::cout   << channels[i].rows << '\n';
    	std::cout   << channels[i].cols << '\n';
    	std::cout   << channels[i].row(0).size() << '\n';
    	for(size_t j = 0; j < channels[i].rows; j++) {
    		std::vector<unsigned char>  r = channels[i].row(j).reshape(1, 1);
    		std::cout   << r.size() << '\n';
    		ch.push_back(r);
    	}
    	image.push_back(ch);
    }

/*
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
    	        	                     .view({Y.channels(), Y.rows, Y.cols})
    	        	                     .to(torch::kByte)
*/
//	cv::imshow("", img.clone() );
//	cv::waitKey(0);
//	cv::destroyAllWindows();

/*
	std::vector<std::vector<std::vector<unsigned char>>> image = matplot::imread(filename);
	// image in (C, H, W) -- channels,  hight, width

	//imshow(const std::vector<std::vector<std::vector<unsigned char>>> &img);
	//std::vector<std::vector<std::vector<unsigned char>>> z;
	std::cout << "i = " << image.size() << "\n";
	for( size_t i = 0; i < image.size(); i++ ) {

		std::vector<std::vector<unsigned char>> simg = image[i];
		std::cout << "j = " << simg.size() << "\n";

		for( size_t j = 0; j < simg.size(); j++) {

			std::vector<unsigned char> ss = simg[j];
			std::cout << "k = " << ss.size() << "\n";

			for( size_t k = 0; k < ss.size(); k++)
				std::cout << static_cast<int>(ss[k]) << " ";
			std::cout << '\n';
		}
	}
	*/
    matplot::imshow(image);
    matplot::show();

    return 0;
}




