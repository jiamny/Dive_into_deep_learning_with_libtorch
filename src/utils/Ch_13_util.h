
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <string>
#include <map>
#include <random>

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

void show_bboxes(cv::Mat& img, torch::Tensor bboxes, std::vector<std::string> labels, std::vector<cv::Scalar> colors, int lineWidth=1);

void ShowManyImages(std::string title, std::vector<cv::Mat> imgs);

const std::map<std::string, int> IMAGE_MODEL = {{"UNCHANGED", 0}, {"GRAY", 1}, {"GRAY_ALPHA", 2},
												{"RGB", 3}, {"RGB_ALPHA", 4}};

torch::Tensor  CvMatToTensor(std::string imgf, std::vector<int> img_size);

cv::Mat TensorToCvMat(torch::Tensor img);

const uchar* tensorToMatrix4Matplotlib(torch::Tensor data);

torch::Tensor CvMatToTensorAfterFlip(std::string file, std::vector<int> img_size, double fP, int flip_axis=0);

torch::Tensor  CvMatToTensorChangeBrightness(std::string file, std::vector<int> img_size, double alpha, double beta);

torch::Tensor  CvMatToTensorChangeHue(std::string file, std::vector<int> img_size, int hue);

const std::vector<std::string> VOC_CLASSES = {"background", "aeroplane", "bicycle", "bird", "boat",
								               "bottle", "bus", "car", "cat", "chair", "cow",
								               "diningtable", "dog", "horse", "motorbike", "person",
								               "potted plant", "sheep", "sofa", "train", "tv/monitor"};

const std::vector<cv::Scalar> VOC_COLORMAP = {
			cv::Scalar(0, 0, 0), cv::Scalar(128, 0, 0), cv::Scalar(0, 128, 0), cv::Scalar(128, 128, 0),
			cv::Scalar(0, 0, 128), cv::Scalar(128, 0, 128), cv::Scalar(0, 128, 128), cv::Scalar(128, 128, 128),
			cv::Scalar(64, 0, 0), cv::Scalar(192, 0, 0), cv::Scalar(64, 128, 0), cv::Scalar(192, 128, 0),
			cv::Scalar(64, 0, 128), cv::Scalar(192, 0, 128), cv::Scalar(64, 128, 128), cv::Scalar(192, 128, 128),
			cv::Scalar(0, 64, 0), cv::Scalar(128, 64, 0), cv::Scalar(0, 192, 0), cv::Scalar(128, 192, 0),
			cv::Scalar(0, 64, 128)};

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
