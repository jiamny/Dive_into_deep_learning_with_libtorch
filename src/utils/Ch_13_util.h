
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string>

#include "../utils.h"

using torch::indexing::Slice;
using torch::indexing::None;

#ifndef SRC_UTILS_CH_13_UTIL_H_
#define SRC_UTILS_CH_13_UTIL_H_


std::pair<cv::Mat, torch::Tensor> readImg( std::string filename );

torch::Tensor box_corner_to_center(torch::Tensor boxes);

torch::Tensor box_center_to_corner(torch::Tensor boxes);

torch::Tensor multibox_prior(torch::Tensor data, std::vector<float> sizes, std::vector<float> ratios);

void setLabel(cv::Mat& im, std::string label, cv::Scalar text_color, cv::Scalar text_bk_color, const cv::Point& pt);

void show_bboxes(cv::Mat& img, torch::Tensor bboxes, std::vector<std::string> labels, std::vector<cv::Scalar> colors);

void ShowManyImages(std::string title, int nArgs, ...);

template<typename T>
std::vector<T> make_list(std::vector<T> obj, std::vector<T> default_values) {
    if( obj.size() == 0 )
        obj = default_values;
    else {
    	obj.insert( obj.end(), default_values.begin(), default_values.end() );
    }
    return obj;
}

#endif /* SRC_UTILS_CH_13_UTIL_H_ */
