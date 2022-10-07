
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
using torch::indexing::Ellipsis;

#ifndef SRC_UTILS_CH_13_UTIL_H_
#define SRC_UTILS_CH_13_UTIL_H_

torch::Tensor box_iou(torch::Tensor boxes1, torch::Tensor boxes2);

torch::Tensor assign_anchor_to_bbox(torch::Tensor ground_truth, torch::Tensor anchors,
													torch::Device device, float iou_threshold=0.5);

torch::Tensor offset_boxes(torch::Tensor anchors, torch::Tensor assigned_bb, float eps=1e-6);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> multibox_target(torch::Tensor anchors, torch::Tensor labels);

torch::Tensor offset_inverse(torch::Tensor anchors, torch::Tensor offset_preds);

torch::Tensor nms(torch::Tensor boxes, torch::Tensor scores, float iou_threshold);

torch::Tensor multibox_detection(torch::Tensor cls_probs, torch::Tensor offset_preds, torch::Tensor anchors,
								float nms_threshold=0.5, float pos_threshold=0.009999999);

std::pair<cv::Mat, torch::Tensor> readImg( std::string filename, std::vector<int> imgSize = {} );

torch::Tensor box_corner_to_center(torch::Tensor boxes);

torch::Tensor box_center_to_corner(torch::Tensor boxes);

torch::Tensor multibox_prior(torch::Tensor data, std::vector<float> sizes, std::vector<float> ratios);

void setLabel(cv::Mat& im, std::string label, cv::Scalar text_color, cv::Scalar text_bk_color, const cv::Point& pt);

void show_bboxes(cv::Mat& img, torch::Tensor bboxes, std::vector<std::string> labels, std::vector<cv::Scalar> colors, int lineWidth=1);

void ShowManyImages(std::string title, std::vector<cv::Mat> imgs);

const std::map<std::string, int> IMAGE_MODEL = {{"UNCHANGED", 0}, {"GRAY", 1}, {"GRAY_ALPHA", 2},
												{"RGB", 3}, {"RGB_ALPHA", 4}};

torch::Tensor  CvMatToTensor(std::string imgf, std::vector<int> img_size);

torch::Tensor  CvMatToTensor2(cv::Mat img, std::vector<int> img_size, bool toRGB = true);

cv::Mat TensorToCvMat( torch::Tensor img, bool is_float = true, bool toBGR = true );

std::vector<uint8_t> tensorToMatrix4Matplotlib(torch::Tensor data, bool is_float=true, bool need_permute=true);

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

const std::vector<std::vector<int>> VOC_COLORMAP = {
			{0, 0, 0}, {128, 0, 0}, {0, 128, 0}, {128, 128, 0},
			{0, 0, 128}, {128, 0, 128}, {0, 128, 128}, {128, 128, 128},
			{64, 0, 0}, {192, 0, 0}, {64, 128, 0}, {192, 128, 0},
			{64, 0, 128}, {192, 0, 128}, {64, 128, 128}, {192, 128, 128},
			{0, 64, 0}, {128, 64, 0}, {0, 192, 0}, {128, 192, 0},
			{0, 64, 128}};

const std::vector<cv::Scalar> VOC_COLORMAP_SCALAR = {
		cv::Scalar(0, 0, 0), cv::Scalar(128, 0, 0), cv::Scalar(0, 128, 0), cv::Scalar(128, 128, 0),
		cv::Scalar(0, 0, 128), cv::Scalar(128, 0, 128), cv::Scalar(0, 128, 128), cv::Scalar(128, 128, 128),
		cv::Scalar(64, 0, 0), cv::Scalar(192, 0, 0), cv::Scalar(64, 128, 0), cv::Scalar(192, 128, 0),
		cv::Scalar(64, 0, 128), cv::Scalar(192, 0, 128), cv::Scalar(64, 128, 128), cv::Scalar(192, 128, 128),
		cv::Scalar(0, 64, 0), cv::Scalar(128, 64, 0), cv::Scalar(0, 192, 0), cv::Scalar(128, 192, 0),
		cv::Scalar(0, 64, 128)};

std::vector<std::pair<std::string, std::string>> read_voc_images(const std::string voc_dir, bool is_train, int num_sample=0,
																	bool shuffle = false, std::vector<int> cropSize = {});
std::unordered_map<std::string, int> voc_colormap2label(void);

torch::Tensor voc_label_indices(torch::Tensor colormap, std::unordered_map<std::string, int> colormap2label);

torch::Tensor decode_segmap(torch::Tensor pred, int nc);

// Randomly crop both feature and label images.
std::pair<torch::Tensor, torch::Tensor> voc_rand_crop(torch::Tensor feature, torch::Tensor label,
		int height, int width, std::vector<float> mean_, std::vector<float> std_, bool toRGB = true);

std::vector<std::pair<std::string, torch::Tensor>> load_bananas_img_data(const std::string data_dir,  bool is_train, int imgSize);

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
    		labT = labT.mul(255).to(torch::kByte).clone();

    		auto dt = voc_rand_crop(imgT.clone(), labT.clone(), img_size[1], img_size[0], mean_, std_, true);

    		return {dt.first.clone(), voc_label_indices(dt.second.clone(), colormap2label)};
    		//return {dt.first.clone(), dt.second.clone()};
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
    std::unordered_map<std::string, int> colormap2label = voc_colormap2label();
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

using mData = std::vector<std::pair<std::string, torch::Tensor>>;

class BananasDataset:public torch::data::Dataset<BananasDataset>{
public:
	BananasDataset(const mData& data, int imgSize) : data_(data) { img_size = imgSize; }

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


#endif /* SRC_UTILS_CH_13_UTIL_H_ */
