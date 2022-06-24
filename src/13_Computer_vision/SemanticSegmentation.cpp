#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>


#include <torch/script.h> // One-stop header.

#include "../utils/Ch_13_util.h"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	std::vector<float> mean_ = {0.485, 0.456, 0.406};
	std::vector<float> std_  = {0.229, 0.224, 0.225};

	bool is_train  = true;
	int batch_size = 32;
	const std::string voc_dir = "./data/VOCdevkit/VOC2012";
	std::vector<int> crop_size = {480,320};

	auto data_set  = read_voc_images(voc_dir, is_train, 0, false, {});

	auto train_set = VOCSegDataset(data_set, crop_size, false).map(torch::data::transforms::Stack<>());

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
				          	  	  	  	  	  	  	  	  	  	  std::move(train_set), batch_size);

	auto batch = *train_loader->begin();
	auto data  = batch.data.to(device);
	auto y     = batch.target.to(device);

	plt::figure_size(1000, 500);
	for(int r = 0; r < 2; r++) {
		for(int c = 0; c < 5; c++) {
			torch::Tensor img;
			if(r == 0) {
				img = data[c].clone().squeeze();
				img = deNormalizeTensor(img, mean_, std_);
			} else {
				img = y[c].clone().squeeze();
			}

			std::vector<uint8_t> z = tensorToMatrix4Matplotlib(img);
			const unsigned char* zptr = &(z[0]);

			plt::subplot2grid(2, 5, r, c, 1, 1);
			plt::imshow(zptr, static_cast<int>(img.size(1)),
					static_cast<int>(img.size(2)), static_cast<int>(img.size(0)));
		}
	}
	plt::show();
	plt::close();

	auto sgimg = CvMatToTensor("./data/2007_000032.png", {});

	sgimg = sgimg.squeeze().mul(255);
	std::cout << "sgimg.sizes: " << sgimg.sizes() << '\n';
	std::cout << "sgimg.max: " << sgimg.max() << '\n';

	auto ssgimg = sgimg.permute({1, 2, 0}).to(torch::kLong).clone();

	auto idx = ((ssgimg.index({Slice(), Slice(), 0}) * 256 + ssgimg.index({Slice(), Slice(), 1})) * 256
	           + ssgimg.index({Slice(), Slice(), 2}));
	std::cout << "idx.sizes: " << idx.sizes() << '\n';
	std::cout << idx.index({Slice(105,115), Slice(130,140)}) << '\n';

	auto lab = voc_label_indices(sgimg.clone(), voc_colormap2label());
	std::cout << lab.index({Slice(105,115), Slice(130,140)}) << '\n';

	// -----------------------------------------
	// Data Preprocessing
	// -----------------------------------------
	int height = 200, width = 300;

	torch::Tensor feature = data[0].clone().squeeze();
	torch::Tensor label = y[0].clone().squeeze();
/*
	std::cout << "labT: " << label.sizes() << '\n';
	std::cout << "labT.max: " << torch::max(label) << '\n';

	auto img = TensorToCvMat(label.clone(), false);
	cv::imshow("label", img);
	cv::waitKey(0);
	cv::destroyAllWindows();
*/
	plt::figure_size(1000, 500);
	for(int c = 0; c < 5; c++) {

		auto dt = voc_rand_crop(feature.clone(), label.clone(), height, width, mean_, std_);

		auto img = deNormalizeTensor(dt.first.clone(), mean_, std_);
		auto limg = dt.second.clone();

		std::vector<uint8_t> z = tensorToMatrix4Matplotlib(img);
		const unsigned char* zptr = &(z[0]);

		plt::subplot2grid(2, 5, 0, c, 1, 1);
		plt::imshow(zptr, static_cast<int>(img.size(1)),
					static_cast<int>(img.size(2)), static_cast<int>(img.size(0)));

		std::vector<uint8_t> lz = tensorToMatrix4Matplotlib(limg);
		const unsigned char* lzptr = &(lz[0]);
		plt::subplot2grid(2, 5, 1, c, 1, 1);
		plt::imshow(lzptr, static_cast<int>(limg.size(1)),
						static_cast<int>(limg.size(2)), static_cast<int>(limg.size(0)));
	}
	plt::show();
	plt::close();

	batch_size = 64;
	data_set  = read_voc_images(voc_dir, is_train, 0, false, crop_size);
	train_set = VOCSegDataset(data_set, crop_size).map(torch::data::transforms::Stack<>());

	train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
					          	  	  	  	  	  	  	  	  	  	  std::move(train_set), batch_size);
   	is_train = false;
	auto test_data   = read_voc_images(voc_dir, is_train, 0, false, crop_size);

	for(auto& batch_data : *train_loader) {
		auto X = batch.data.to(device);
		auto y = batch.target.to(device);
		std::cout << "X: " << X.sizes() << '\n';
		std::cout << "y: " << y.sizes() << '\n';
		break;
	}

	std::cout << "Done!\n";
}

