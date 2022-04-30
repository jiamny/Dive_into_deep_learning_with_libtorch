#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <algorithm>
#include <random>

#include "../utils/Ch_13_util.h"

torch::Tensor bilinear_kernel(int in_channels, int out_channels, int kernel_size) {
    auto factor = static_cast<int>((kernel_size + 1) / 2);
    float center;

    if( kernel_size % 2 == 1 )
        center = factor - 1;
    else
        center = factor - 0.5;

    std::vector<torch::Tensor> og = {torch::arange(kernel_size).reshape({-1, 1}),
    		   torch::arange(kernel_size).reshape({1, -1})};

    auto filt = (1 - torch::abs(og[0] - center) / factor) * (1 - torch::abs(og[1] - center) / factor);

    auto weight = torch::zeros({in_channels, out_channels, kernel_size, kernel_size});

    for( int i = 0; i < in_channels; i++ ) {
    	for(int j = 0; j < out_channels; j++) {
    		weight.index_put_({i, j, Slice(0, kernel_size), Slice(0, kernel_size)}, filt);
    	}
    }
    //weight.index_put_({Slice(0, in_channels), Slice(0, out_channels), Slice(), Slice()}, filt);

    return weight;
}

std::vector<std::pair<std::string, std::string>> read_voc_images(const std::string voc_dir,  bool is_train) {
    //Read the banana detection dataset images and labels
	std::vector<std::pair<std::string, std::string>> imgpaths;

	std::string txt_fname = "";
	if( is_train )
		txt_fname = voc_dir + "/ImageSets/Segmentation/train.txt";
	else
		txt_fname = voc_dir + "/ImageSets/Segmentation/val.txt";

    bool rgb = false;
    std::ifstream file;

    file.open(txt_fname, std::ios_base::in);
    // Exit if file not opened successfully
    if( !file.is_open() ) {
    	std::cout << "File not read successfully" << std::endl;
    	std::cout << "Path given: " << txt_fname << std::endl;
    	exit(-1);
    }

    std::string fname;
    int img_size = 0;

	while( std::getline(file, fname) ) {
		//std::cout << fname << '\n';

		std::string imgf = voc_dir + "/JPEGImages/" + fname + ".jpg";
		std::string labf = voc_dir + "/SegmentationClass/" + fname + ".png";
/*
		auto imgT = CvMatToTensor(labf.c_str(), img_size, true);
		auto rev_bgr_mat = TensorToCvMat(imgT, true);

		cv::imshow("rev_bgr_mat", rev_bgr_mat);
		cv::waitKey(-1);
		cv::destroyAllWindows();
*/
		imgpaths.push_back(std::make_pair(imgf, labf));
	}

	auto rng = std::default_random_engine {};
	std::shuffle(std::begin(imgpaths), std::end(imgpaths), rng);

	std::cout << "size: " << imgpaths.size() << '\n';
	return imgpaths;
}

using VocData = std::vector<std::pair<std::string, std::string>>;
using Example = torch::data::Example<>;

class VOCSegDataset:public torch::data::Dataset<VOCSegDataset>{
public:
	VOCSegDataset(const VocData& data, std::vector<int> imgSize) : data_(data) { img_size = imgSize; }

    // Override get() function to return tensor at location index
    Example get(size_t index) override{

    	auto imgT = CvMatToTensor(data_.at(index).first.c_str(), img_size);
    	auto labT = CvMatToTensor(data_.at(index).second.c_str(), img_size);

    	return {imgT, labT};
    }

    // Return the length of data
    torch::optional<size_t> size() const override {
        return data_.size();
    };

private:
    VocData data_;
    std::vector<int> img_size;
};

torch::Tensor voc_loss(torch::Tensor input, torch::Tensor target) {
	torch::Tensor ct = torch::nn::functional::cross_entropy(input, target,
			torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone));
	return ct.mean(1);
}

struct VocNetImpl : public torch::nn::Module {
	torch::nn::Sequential classifier{nullptr};
	torch::jit::script::Module features;

	VocNetImpl(torch::jit::script::Module& net, torch::nn::Sequential& cls) {
		features = net;
		classifier = cls;
		register_module("classifier", classifier);
	}

	torch::Tensor forward(torch::Tensor x) {

		// convert tensor to IValue
		std::vector<torch::jit::IValue> input;
		input.push_back(x);
		x = features.forward(input).toTensor();

		return classifier->forward(x);
	}
};
TORCH_MODULE(VocNet);


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	std::string mdlf = "./src/13_Computer_vision/resnet18_without_last_2_layers.pt";
	torch::jit::script::Module net;

	try {
	    // Deserialize the ScriptModule from a file using torch::jit::load().
	    net = torch::jit::load(mdlf);
	} catch (const c10::Error& e) {
	    std::cerr << e.backtrace() << "error loading the model\n";
	    return -1;
	}

	// Given an input with height and width of 320 and 480 respectively, the forward propagation of net
	// reduces the input height and width to 1/32 of the original, namely 10 and 15
	auto X = torch::rand({1, 3, 320, 480}).to(device);

	// convert tensor to IValue
	std::vector<torch::jit::IValue> input;
	input.push_back(X);

	auto output = net.forward(input).toTensor();

	std::cout << "output: " << output.sizes() << "\n";

	// Next, we [use a 1×1 convolutional layer to transform the number of output channels into the number of classes
	// (21) of the Pascal VOC2012 dataset.] Finally, we need to (increase the height and width of the feature maps
	// by 32 times) to change them back to the height and width of the input image.
	int num_classes = 21;
	torch::nn::Sequential classifier = torch::nn::Sequential(
			{{"final_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, num_classes, 1))},
			{"transpose_conv", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(
					num_classes, num_classes, 64).padding(16).stride(32))}} );

	// itializing Transposed Convolutional Layers
	std::cout << "bilinear: " << bilinear_kernel(3, 3, 4) << '\n';

	// experiment with upsampling of bilinear interpolation
	auto conv_trans = torch::nn::ConvTranspose2d(
			torch::nn::ConvTranspose2dOptions(3, 3, 4).padding(1).stride(2).bias(false));

//	torch::autograd::GradMode::set_enabled(false);  	// make parameters copying possible
/*
	for (auto& module : conv_trans->modules(true) ) {  //modules(include_self=false))
		std::cout << "III\n";
		if (auto M = dynamic_cast<torch::nn::ConvTranspose2dImpl*>(module.get())) {
			auto W = bilinear_kernel(3, 3, 4);
		    //M->weight.data().index_put_({Slice(), Slice(), Slice()}, W); // works!!!
			M->weight.data().copy_(W);
		}
	}
*/

	conv_trans.get()->weight.data().copy_(bilinear_kernel(3, 3, 4));

//	torch::autograd::GradMode::set_enabled(true);

	std::cout << "conv_trans: " << conv_trans << "\n";
	std::cout << "conv_trans.weights: " << conv_trans.get()->weight.sizes() << "\n";
	std::cout << "conv_trans.weights: " << conv_trans.get()->weight << "\n";

	std::string imgf = "./data/catdog.jpg";
	int img_size = 0;
	bool show_img = false;

	auto imgT = CvMatToTensor(imgf, {});

	auto inputImg  = imgT.detach().clone();
	std::cout << "input image sizes: " << inputImg.sizes() << "\n";

	auto cvImg = TensorToCvMat(inputImg);

	if( show_img ) {
		cv::imshow("image", cvImg);
		cv::waitKey(-1);
	}

	std::cout << "imgT: " << imgT.sizes() << "\n";
	auto Z = imgT.index({0, Slice(), Slice()});
	std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n";
	std::cout << "X.min " << torch::min(Z) << "X.max " << torch::max(Z) << "\n";
	Z = imgT.index({1, Slice(), Slice()});
	std::cout << "X.min " << torch::min(Z) << "X.max " << torch::max(Z) << "\n";
	Z = imgT.index({2, Slice(), Slice()});
	std::cout << "X.min " << torch::min(Z) << "X.max " << torch::max(Z) << "\n";

//	X = imgT.unsqueeze(0);
//	std::cout << "X: " << X.sizes() << "\n";
//	std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++\n" << X << "\n";
//	std::cout << "X.min " << torch::min(X) << "\n";
//	std::cout << "X.max " << torch::max(X) << "\n";

	auto Y = conv_trans(imgT.unsqueeze(0));
	std::cout << "\n\n\n";
	std::cout << "Y: " << Y.sizes() << "\n";
//	std::cout << "Y.data(): " << Y << "\n";

	auto out_imgT = Y[0].detach().clone();
	std::cout << "output image sizes: " << out_imgT.sizes() << "\n";
	std::cout << "output image sizes: " << torch::max(out_imgT).data().item<float>() << "\n";

	cvImg = TensorToCvMat(out_imgT);

	if( show_img ) {
		cv::imshow("image_2", cvImg);
		cv::waitKey(-1);
		cv::destroyAllWindows();
	}
/*
	// initialize the transposed convolutional layer with upsampling of bilinear interpolation.
	// For the 1×1 convolutional layer, we use Xavier initialization.

	torch::autograd::GradMode::set_enabled(false);
	auto W = bilinear_kernel(num_classes, num_classes, 64);

	for (auto& module : classifier->modules(true) ) { //modules(include_self=false))
		if (auto M = dynamic_cast<torch::nn::ConvTranspose2dImpl*>(module.get())) {
		    //std::cout << module->name() << std::endl;
		    M->weight.data().copy_(W);
		}
	}
	torch::autograd::GradMode::set_enabled(true);

	bool is_train = true;
	const std::string voc_dir = "./data/VOCdevkit/VOC2012";
	int batch_size = 32;

	auto data_set = read_voc_images(voc_dir, is_train);

	auto train_set = VOCSegDataset(data_set, {320, 480}).map(torch::data::transforms::Stack<>());

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
			          	  	  	  	  	  	  	  	  	  	  std::move(train_set), batch_size);

	auto batch = *train_loader->begin();
	auto data  = batch.data.to(device);
	auto y     = batch.target.to(device);
	std::cout << "data: " << data.sizes() << std::endl;
	std::cout << "y: " << y.sizes() << std::endl;

	// convert tensor to IValue
	input.clear();
	input.push_back(data);
	auto out = net.forward(input).toTensor();
std::cout << "out: " << out.sizes() << std::endl;
	auto y_hat = classifier->forward(out);
std::cout << "rlt: " << y_hat.sizes() << std::endl;

	if( y_hat.size(0) > 1 && y_hat.size(1) > 1 )
		y_hat = torch::argmax(y_hat, 1);

	y_hat = y_hat.to(y.dtype());
std::cout << "y_hat: " << y_hat.sizes() << std::endl;
std::cout << "y_hat_softmax[0:4]: " << y_hat.index({Slice(), Slice(0, 4), Slice(0, 4)}) << std::endl;


	// -------------------------------------------
	// Reading the Dataset
	// -------------------------------------------
	auto test_data = read_voc_images(voc_dir, false);

	auto test_set = VOCSegDataset(test_data, {320, 480}).map(torch::data::transforms::Stack<>());

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
				          	  	  	  	  	  	  	  	  	  	  std::move(test_set), batch_size);
	// -------------------------------------------
	// Training
	// -------------------------------------------

	auto model = VocNet(net, classifier);
	model->to(device);

	size_t num_epochs = 1;
	float lr = 0.001;
	double wd = 1e-3;
	auto trainer = torch::optim::SGD(model->parameters(), torch::optim::SGDOptions(lr).weight_decay(wd));

	bool first = true;
	std::vector<float> train_loss_ave;
	std::vector<float> train_epochs;

	for(size_t epoch = 1; epoch <= num_epochs; epoch++) {
		model->train();

		std::cout << "--------------- Training --------------------\n";
		first = true;
		float loss_sum = 0.0;
		int64_t num_correct_imgs = 0, total_imgs = 0;
		size_t num_batch = 0;

		for(auto& batch_data : *train_loader) {
			auto img_data  = batch_data.data.to(device);
			auto lab_data  = batch_data.target.to(device);

std::cout << "data: " << img_data.sizes() << std::endl;
std::cout << "y: " << lab_data.sizes() << std::endl;

			trainer.zero_grad();
			auto pred = model->forward(img_data);
			auto l = voc_loss(pred, lab_data);
			l.sum().backward();

			trainer.step();
			loss_sum += l.sum().data().item<float>();
std::cout << "loss: " << loss_sum << '\n';
			auto correct = accuracy(pred, lab_data);
			num_correct_imgs += correct;
			total_imgs += img_data.size(0);
			num_batch++;
		}
	}
*/
	std::cout << "Done!\n";
	return 0;
}



