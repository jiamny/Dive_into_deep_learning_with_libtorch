
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
#include <chrono>
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

torch::Tensor  CvMatToTensor2(cv::Mat img, std::vector<int> img_size);

cv::Mat TensorToCvMat(torch::Tensor img, bool is_float = true );

std::vector<uint8_t> tensorToMatrix4Matplotlib(torch::Tensor data, bool is_float=true);

torch::Tensor CvMatToTensorAfterFlip(std::string file, std::vector<int> img_size, double fP, int flip_axis=0);

torch::Tensor  CvMatToTensorChangeBrightness(std::string file, std::vector<int> img_size, double alpha, double beta);

torch::Tensor  CvMatToTensorChangeHue(std::string file, std::vector<int> img_size, int hue);

torch::Tensor deNormalizeTensor(torch::Tensor imgT, std::vector<float> mean_, std::vector<float> std_);

torch::Tensor NormalizeTensor(torch::Tensor imgT, std::vector<float> mean_, std::vector<float> std_);

std::string cvMatType2Str(int type);

const std::vector<std::string> VOC_CLASSES = {"background", "aeroplane", "bicycle", "bird", "boat",
								               "bottle", "bus", "car", "cat", "chair", "cow",
								               "diningtable", "dog", "horse", "motorbike", "person",
								               "potted plant", "sheep", "sofa", "train", "tv/monitor"};

const std::vector<std::vector<long>> VOC_COLORMAP = {
			{0, 0, 0}, {128, 0, 0}, {0, 128, 0}, {128, 128, 0},
			{0, 0, 128}, {128, 0, 128}, {0, 128, 128}, {128, 128, 128},
			{64, 0, 0}, {192, 0, 0}, {64, 128, 0}, {192, 128, 0},
			{64, 0, 128}, {192, 0, 128}, {64, 128, 128}, {192, 128, 128},
			{0, 64, 0}, {128, 64, 0}, {0, 192, 0}, {128, 192, 0},
			{0, 64, 128}};

std::vector<std::pair<std::string, std::string>> read_voc_images(const std::string voc_dir, bool is_train, int num_sample=0,
																	bool shuffle = false, std::vector<int> cropSize = {});
torch::Tensor voc_colormap2label(void);

torch::Tensor voc_label_indices(torch::Tensor colormap, torch::Tensor colormap2label);

// Randomly crop both feature and label images.
std::pair<torch::Tensor, torch::Tensor> voc_rand_crop(torch::Tensor feature, torch::Tensor label,
		int height, int width, std::vector<float> mean_, std::vector<float> std_);

using VocData = std::vector<std::pair<std::string, std::string>>;
using Example = torch::data::Example<>;

class VOCSegDataset:public torch::data::Dataset<VOCSegDataset>{
public:
	VOCSegDataset(const VocData& data, std::vector<int> imgSize, bool clrMaped = true) : data_(data) {
		img_size    = imgSize;
		colorMaped  = clrMaped;
	}

    // Override get() function to return tensor at location index
    Example get(size_t index) override{

    	if( colorMaped ) {
    		torch::Tensor imgT = CvMatToTensor(data_.at(index).first.c_str(), {});

    		imgT = NormalizeTensor(imgT, mean_, std_);

    		auto labT = CvMatToTensor(data_.at(index).second.c_str(), {});
    		labT = labT.mul(255).to(torch::kLong).clone();

    		auto dt = voc_rand_crop(imgT.clone(), labT.clone(), img_size[1], img_size[0], mean_, std_);

    		return {dt.first.clone(), voc_label_indices(dt.second.clone(), colormap2label)};
    	} else {

    		torch::Tensor imgT = CvMatToTensor(data_.at(index).first.c_str(), img_size);
    		imgT = NormalizeTensor(imgT, mean_, std_);

    		auto labT = CvMatToTensor(data_.at(index).second.c_str(), img_size);
    		labT = labT.mul(255).to(torch::kLong).clone();

    	    return {imgT.clone(), labT.clone()};
    	}
    }

    // Return the length of data
    torch::optional<size_t> size() const override {
        return data_.size();
    };

private:
    bool colorMaped;
    VocData data_;
    std::vector<int> img_size;
    torch::Tensor colormap2label = voc_colormap2label();
    bool is_train = true;
    std::vector<float> mean_ = {0.485, 0.456, 0.406};
    std::vector<float> std_  = {0.229, 0.224, 0.225};
};


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
