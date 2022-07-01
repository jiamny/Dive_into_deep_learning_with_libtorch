#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <torch/script.h> // One-stop header.
#include <vector>
#include <map>

#include <iostream>
#include <memory>
#include <algorithm>
#include <random>

#include "../utils/Ch_13_util.h"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;


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

    auto in_idx = RangeToensorIndex(in_channels);
    auto out_idx = RangeToensorIndex(out_channels);
    weight.index_put_({in_idx, out_idx, Slice(0, kernel_size), Slice(0, kernel_size)}, filt);

    return weight;
}


torch::Tensor voc_loss(torch::Tensor input, torch::Tensor target) {
	torch::Tensor ct = torch::nn::functional::cross_entropy(input, target,
			torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone));
	return ct.mean().mean();
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

	torch::manual_seed(123);

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

	torch::autograd::GradMode::set_enabled(false);  	// make parameters copying possible

	for (auto& module : conv_trans->modules(true) ) {  //modules(include_self=false))
		if (auto M = dynamic_cast<torch::nn::ConvTranspose2dImpl*>(module.get())) {
			auto W = bilinear_kernel(3, 3, 4);
		    //M->weight.data().index_put_({Slice(), Slice(), Slice()}, W); // works!!!
			M->weight.data().copy_(W);
		}
	}

	conv_trans.get()->weight.data().copy_(bilinear_kernel(3, 3, 4));

	torch::autograd::GradMode::set_enabled(true);

	std::cout << "conv_trans: " << conv_trans << "\n";
	std::cout << "conv_trans.weights: " << conv_trans.get()->weight.sizes() << "\n";
	std::cout << "conv_trans.weights: " << conv_trans.get()->weight << "\n";
	std::cout << "conv_trans.bias: " << conv_trans.get()->bias << "\n";

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

	auto Y = conv_trans(imgT.unsqueeze(0));
	std::cout << "\n\n";
	std::cout << "Y: " << Y.sizes() << "\n";
//	std::cout << "Y.data(): " << Y << "\n";

	auto out_imgT = Y[0].detach().clone();
	std::cout << "output image sizes: " << out_imgT.sizes() << "\n";
	std::cout << "output max value: " << torch::max(out_imgT).data().item<float>() << "\n";

	cvImg = TensorToCvMat(out_imgT);

	if( show_img ) {
		cv::imshow("image_2", cvImg);
		cv::waitKey(-1);
		cv::destroyAllWindows();
	}

	// initialize the transposed convolutional layer with upsampling of bilinear interpolation.
	// For the 1×1 convolutional layer, we use Xavier initialization.
	std::cout << "Init---" << "\n";

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
	int batch_size = 8;
	std::vector<int> crop_size = {480,320};

	std::cout << "Read in data" << "\n";

	std::vector<float> mean_ = {0.485, 0.456, 0.406};
	std::vector<float> std_  = {0.229, 0.224, 0.225};

/*
	imgT = CvMatToTensor("./data/2007_001704.jpg", {});

	imgT = NormalizeTensor(imgT, mean_, std_);

	auto labT = CvMatToTensor("./data/2007_000063.png", {});
	labT = labT.mul(255).to(torch::kByte).clone();

	auto dt = voc_rand_crop(imgT.clone(), labT.clone(), crop_size[1], crop_size[0], mean_, std_, true);

	std::cout << "dt.first: " << dt.first.sizes() << '\n'; 		// (3, 320, 480)
	std::cout << "dt.second: " << dt.second.sizes() << '\n'; 	// (3, 320, 480)

	imgT = dt.first.clone();
	std::cout << imgT.index({Slice(105,115), Slice(130,140)}) << '\n';
	imgT = deNormalizeTensor(imgT, mean_, std_);
	std::cout << imgT.index({Slice(105,115), Slice(130,140)}) << '\n';
	imgT = imgT.permute({1, 2, 0}).clone();
	std::vector<uint8_t> z = tensorToMatrix4Matplotlib(imgT, true, false);
	const unsigned char* zptr = &(z[0]);
	plt::imshow(zptr, static_cast<int>(imgT.size(0)),
										static_cast<int>(imgT.size(1)), static_cast<int>(imgT.size(2)));
	plt::show();

	auto colormap = dt.second.clone();
	colormap = colormap.permute({1, 2, 0}).to(torch::kByte).clone();

	std::vector<uint8_t> lz = tensorToMatrix4Matplotlib(colormap, false, false);
	const unsigned char* lzptr = &(lz[0]);
	plt::imshow(lzptr, static_cast<int>(colormap.size(0)),
									static_cast<int>(colormap.size(1)), static_cast<int>(colormap.size(2)));
	plt::show();


	auto coded = voc_label_indices(dt.second.clone(), voc_colormap2label());
	std::cout << "coded: " << coded.index({Slice(105,115), Slice(130,140)}) << '\n';

	auto decoded = decode_segmap(coded, num_classes);

	lz = tensorToMatrix4Matplotlib(decoded, false, false);
	lzptr = &(lz[0]);
	plt::imshow(lzptr, static_cast<int>(decoded.size(0)),
										static_cast<int>(decoded.size(1)), static_cast<int>(decoded.size(2)));
	plt::show();
*/

	// -------------------------------------------
	// Reading the Dataset
	// -------------------------------------------
	auto data_set = read_voc_images(voc_dir, is_train, 0, false, crop_size);

	auto train_set = VOCSegDataset(data_set, crop_size).map(torch::data::transforms::Stack<>());

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
			          	  	  	  	  	  	 std::move(train_set), batch_size);

	is_train = false;

	batch_size = 5;
	auto test_data = read_voc_images(voc_dir, is_train, 10, false, crop_size);

	auto test_set = VOCSegDataset(test_data, crop_size).map(torch::data::transforms::Stack<>());

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
				          	  	  	  	  	  	  	  	  	  	  std::move(test_set), batch_size);
	// -------------------------------------------
	// Training
	// -------------------------------------------

	auto model = VocNet(net, classifier);
	model->to(device);

	size_t num_epochs = 30;
	float lr = 0.001;
	double wd = 1e-3;
	auto trainer = torch::optim::SGD(model->parameters(), torch::optim::SGDOptions(lr).weight_decay(wd));

	bool first = true;
	std::vector<float> train_loss_ave;
	std::vector<float> train_epochs;

	for(size_t epoch = 1; epoch <= num_epochs; epoch++) {
		model->train();

		std::cout << "--------------- Training -----------------> " << epoch << " / " << num_epochs << "\n";
		first = true;
		float loss_sum = 0.0;
		int64_t num_correct_imgs = 0, total_imgs = 0;
		size_t num_batch = 0;

		for(auto& batch_data : *train_loader) {
			auto img_data  = batch_data.data.to(device);
			auto lab_data  = batch_data.target.to(device);

			trainer.zero_grad();
			auto pred = model->forward(img_data);
			auto l = voc_loss(pred, lab_data.to(torch::kLong));  // label has to be long data type
			l.sum().backward();

			trainer.step();
			loss_sum += l.sum().data().item<float>();
/*
			auto correct = 0;
			if( pred.size(0) > 1 && pred.size(1) > 1 )
				pred = torch::argmax(pred, 1);

			pred = pred.to(lab_data.dtype());
			//std::cout << "y_hat: " << pred.sizes() << " y: " << lab_data.sizes() << std::endl;


			for( long i = 0; i < pred.size(0); i++ ) {
				auto pi = pred[i];
				auto yi = lab_data[i];
				if( torch::sum((pi != yi).to(torch::kInt)).data().item<int>() < 1 )
					correct++;
			}
			//std::cout << "accuracy:" << (correct*1.0/img_data.size(0)) << '\n';

			num_correct_imgs += correct;
*/


			total_imgs += img_data.size(0);
			num_batch++;
			std::cout << "num_batch: " << num_batch << '\n';
		}
		std::cout << "loss: " << (loss_sum*1.0/num_batch) << '\n';
	}

	model->eval();

	plt::figure_size(300 * batch_size, 900);
	for(auto& batch : *test_loader) {
		auto img_data  = batch.data.to(device);
		auto lab_data  = batch.target.to(device);

		//std::cout << "d: " << img_data.sizes() << '\n';
		//std::cout << "L: " << lab_data.sizes() << '\n';

		auto pred = model->forward(img_data);
		//std::cout << "P: " << pred.sizes() << '\n';
		auto l = voc_loss(pred, lab_data.to(torch::kLong));  // label has to be long data type

		for( int j = 0; j < pred.size(0); j++ ) {
			imgT = img_data[j].squeeze();
			auto jimg = deNormalizeTensor(imgT, mean_, std_);
			jimg = jimg.mul(255).permute({1, 2, 0}).to(torch::kByte).clone();
			std::vector<uint8_t> iz = tensorToMatrix4Matplotlib(jimg, false, false);
			const unsigned char* izptr = &(iz[0]);
			plt::subplot2grid(3, batch_size, 0, j, 1, 1);
			plt::title("Image");
			plt::imshow(izptr, static_cast<int>(jimg.size(0)),
								static_cast<int>(jimg.size(1)), static_cast<int>(jimg.size(2)));

			auto ppred = torch::argmax(pred[j].squeeze(), 0).detach().to(device);
			auto img = decode_segmap(ppred.squeeze(), num_classes);

			std::vector<uint8_t> z = tensorToMatrix4Matplotlib(img.clone(), false, false);
			const unsigned char* zptr = &(z[0]);

			plt::subplot2grid(3, batch_size, 1, j, 1, 1);
			plt::title("pred");
			plt::imshow(zptr, static_cast<int>(img.size(0)),
									  static_cast<int>(img.size(1)), static_cast<int>(img.size(2)));

			auto limg = decode_segmap(lab_data[j].squeeze(), num_classes);
			std::vector<uint8_t> lz = tensorToMatrix4Matplotlib(limg.clone(), false, false);
			const unsigned char* lzptr = &(lz[0]);
			plt::subplot2grid(3, batch_size, 2, j, 1, 1);
			plt::title("label");
			plt::imshow(lzptr, static_cast<int>(limg.size(0)),
							   static_cast<int>(limg.size(1)), static_cast<int>(limg.size(2)));
		}

		break;
	}
	plt::show();
	plt::close();

	std::cout << "Done!\n";
	return 0;
}



