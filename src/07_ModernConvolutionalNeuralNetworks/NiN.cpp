
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../utils.h"
#include "../utils/transforms.hpp"              // transforms_Compose
#include "../utils/datasets.hpp"                // datasets::ImageFolderClassesWithPaths
#include "../utils/dataloader.hpp"              // DataLoader::ImageFolderClassesWithPaths

#include <matplot/matplot.h>
using namespace matplot;

using Options = torch::nn::Conv2dOptions;


torch::nn::Sequential nin_block(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t strides, int64_t padding) {
    return torch::nn::Sequential(
    		torch::nn::Conv2d(Options(in_channels, out_channels, kernel_size).stride(strides).padding(padding)),
			torch::nn::ReLU(),
			torch::nn::BatchNorm2d(out_channels),
			torch::nn::Conv2d(Options(out_channels, out_channels, 1)),
			torch::nn::ReLU(),
			torch::nn::BatchNorm2d(out_channels),
			torch::nn::Conv2d(Options(out_channels, out_channels, 1)),
			torch::nn::ReLU());
}

// NiN Model
/*
 * The original NiN network was proposed shortly after AlexNet and clearly draws some inspiration.
 * NiN uses convolutional layers with window shapes of 11×11, 5×5, and 3×3, and the corresponding numbers of output channels
 * are the same as in AlexNet. Each NiN block is followed by a maximum pooling layer with a stride of 2 and a window shape of 3×3.
 *
 * One significant difference between NiN and AlexNet is that NiN avoids fully-connected layers altogether.
 */

struct NiNImpl : public torch::nn::Module {
	torch::nn::Sequential b1{nullptr}, b2{nullptr}, b3{nullptr}, b4{nullptr};
	torch::nn::MaxPool2d maxpool1{nullptr}, maxpool2{nullptr}, maxpool3{nullptr};
	torch::nn::AdaptiveAvgPool2d adpavpool{nullptr};
	torch::nn::Flatten flattern{nullptr};

	NiNImpl(int64_t num_classes) {
		b1 = nin_block(3, 96, 11, 4, 0);
		b2 = nin_block(96, 256, 5, 1, 2);
		b3 = nin_block(256, 384, 3, 1, 1);
		b4 = nin_block(384, num_classes, 3, 1, 1);

		register_module("b1", b1);
		register_module("b2", b2);
		register_module("b3", b3);
		register_module("b4", b4);

		maxpool1 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2));
		maxpool2 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2));
		maxpool3 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2));

		adpavpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
		flattern  = torch::nn::Flatten();

		// weights_init
		for (auto& m : modules(/*include_self=*/false)) {

			if ((typeid(m) == typeid(torch::nn::Conv2d)) || (typeid(m) == typeid(torch::nn::Conv2dImpl))) {
				auto p = m->named_parameters(false);
				auto w = p.find("weight");
				auto b = p.find("bias");
				if (w != nullptr) torch::nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.01);
				//if (w != nullptr) torch::nn::init::kaiming_normal_(*w, /*a=*/0.0, torch::kFanOut, torch::kReLU);
				if (b != nullptr) torch::nn::init::constant_(*b, /*bias=*/0.0);

			} else if ((typeid(m) == typeid(torch::nn::Linear)) || (typeid(m) == typeid(torch::nn::LinearImpl))){
				auto p = m->named_parameters(false);
				auto w = p.find("weight");
				auto b = p.find("bias");
				if (w != nullptr) torch::nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.01);
				if (b != nullptr) torch::nn::init::constant_(*b, /*bias=*/0.0);

			} else if ((typeid(m) == typeid(torch::nn::BatchNorm2d)) || (typeid(m) == typeid(torch::nn::BatchNorm2dImpl))){
				auto p = m->named_parameters(false);
				auto w = p.find("weight");
				auto b = p.find("bias");
				if (w != nullptr) torch::nn::init::constant_(*w, /*weight=*/1.0);
				if (b != nullptr) torch::nn::init::constant_(*b, /*bias=*/0.0);
			}
		}
	}

	torch::Tensor forward(torch::Tensor x) {
		x = b1->forward(x);
		x = maxpool1->forward(x);
		x = b2->forward(x);
		x = maxpool2->forward(x);
		x = b3->forward(x);
		x = maxpool3->forward(x);
		x = torch::dropout(x, 0.5, true);
		x = b4->forward(x);
		x = adpavpool->forward(x);

		return flattern->forward(x);
	}
};
TORCH_MODULE(NiN);

std::vector<std::string> Set_Class_Names(const std::string path, const size_t class_num) {
    // (1) Memory Allocation
    std::vector<std::string> class_names = std::vector<std::string>(class_num);

    // (2) Get Class Names
    std::string class_name;
    std::ifstream ifs(path, std::ios::in);
    size_t i = 0;
    if( ! ifs.fail() ) {
    	while( getline(ifs, class_name) ) {
//    		std::cout << class_name.length() << std::endl;
    		if( class_name.length() > 2 ) {
    			class_names.at(i) = class_name;
    			i++;
    		}
    	}
    } else {
    	std::cerr << "Error : can't open the class name file." << std::endl;
    	std::exit(1);
    }

    ifs.close();
    if( i != class_num ){
        std::cerr << "Error : The number of classes does not match the number of lines in the class name file." << std::endl;
        std::exit(1);
    }

    // End Processing
    return class_names;
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';
/*
	std::unordered_map<std::string, std::string> flowerLabels = getFlowersLabels("./data/flowers_cat_to_name.json");
	std::cout << flowerLabels["9"] <<std::endl;

	std::vector<std::string> class_names = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17"};

//	for(auto it= 0; it < class_names.size(); ++it){
//		std::cout << class_names[it] << " " << flowerLabels[class_names[it]] << std::endl;
//	}
*/
	auto tnet = NiN(17);
	std::cout << tnet->parameters(false) << std::endl;
	// We create a data example to see [the output shape of each block].
	auto X = torch::randn({1, 3, 224, 224});
	X = tnet->forward(X);
	std::cout << X.sizes()  << std::endl;
	std::cout << X  << std::endl;

	size_t img_size = 224;
	size_t batch_size = 32;
	const std::string path = "./data/17_flowers_name.txt";
	const size_t class_num = 17;
	const size_t valid_batch_size = 1;
	std::vector<std::string> class_names = Set_Class_Names( path, class_num);
	constexpr bool train_shuffle = true;    // whether to shuffle the training dataset
	constexpr size_t train_workers = 2;  	// the number of workers to retrieve data from the training dataset
    constexpr bool valid_shuffle = true;    // whether to shuffle the validation dataset
    constexpr size_t valid_workers = 2;     // the number of workers to retrieve data from the validation dataset


    // (4) Set Transforms
    std::vector<transforms_Compose> transform {
        transforms_Resize(cv::Size(img_size, img_size), cv::INTER_LINEAR),        // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        transforms_ToTensor(),                                                     // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
		transforms_Normalize(std::vector<float>{0.485, 0.456, 0.406}, std::vector<float>{0.229, 0.224, 0.225})  // Pixel Value Normalization for ImageNet
    };

	std::string dataroot = "./data/17_flowers/train";
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor loss, image, label, output;
    datasets::ImageFolderClassesWithPaths dataset, valid_dataset, test_dataset;      		// dataset;
    DataLoader::ImageFolderClassesWithPaths dataloader, valid_dataloader, test_dataloader; 	// dataloader;

    // -----------------------------------
    // a1. Preparation
    // -----------------------------------

    // (1) Get Dataset

    dataset = datasets::ImageFolderClassesWithPaths(dataroot, transform, class_names);
    dataloader = DataLoader::ImageFolderClassesWithPaths(dataset, batch_size, /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);

	std::cout << "total training images : " << dataset.size() << std::endl;

    std::string valid_dataroot = "./data/17_flowers/valid";
    valid_dataset = datasets::ImageFolderClassesWithPaths(valid_dataroot, transform, class_names);
    valid_dataloader = DataLoader::ImageFolderClassesWithPaths(valid_dataset, valid_batch_size, /*shuffle_=*/valid_shuffle, /*num_workers_=*/valid_workers);

    std::cout << "total validation images : " << valid_dataset.size() << std::endl;
    bool valid = true;
    bool test  = true;
    bool vobose = false;

    // (5) Define Network
    auto net = NiN(class_num);

	torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(1e-4).betas({0.5, 0.999}));

	auto criterion = torch::nn::NLLLoss(torch::nn::NLLLossOptions().ignore_index(-100).reduction(torch::kMean));

	net->to(device);

	size_t epoch;
	size_t total_iter = dataloader.get_count_max();
	size_t start_epoch, total_epoch;
	start_epoch = 1;
	total_iter = dataloader.get_count_max();
	total_epoch = 30;
	bool first = true;
	std::vector<double> train_loss_ave;
	std::vector<double> train_epochs;

	for (epoch = start_epoch; epoch <= total_epoch; epoch++) {
		net->train();
		std::cout << "--------------- Training --------------------\n";
		first = true;
		float loss_sum = 0.0;
		while (dataloader(mini_batch)) {
			image = std::get<0>(mini_batch).to(device);
			label = std::get<1>(mini_batch).to(device);

			if( first && vobose ) {
				for(size_t i = 0; i < label.size(0); i++)
					std::cout << label[i].item<int64_t>() << " ";
				std::cout << "\n";
				first = false;
			}

			image = std::get<0>(mini_batch).to(device);
			label = std::get<1>(mini_batch).to(device);
			output = net->forward(image);
			auto out = torch::nn::functional::log_softmax(output, /*dim=*/1);
			//std::cout << output.sizes() << "\n" << out.sizes() << std::endl;
			loss = criterion(out, label); //torch::mse_loss(out, label);

			optimizer.zero_grad();
			loss.backward();
			optimizer.step();

			loss_sum += loss.item<float>();
		}

		train_loss_ave.push_back(1.0*loss_sum/total_iter);
		train_epochs.push_back(epoch*1.0);
		std::cout << "epoch: " << epoch << "/"  << total_epoch << ", avg_loss: " << (loss_sum/total_iter) << std::endl;

		// ---------------------------------
		// validation
		// ---------------------------------
		if( valid && (epoch % 5 == 0) ) {
			std::cout << "--------------- validation --------------------\n";
			net->eval();
			size_t iteration = 0;
			float total_loss = 0.0;
			size_t total_match = 0, total_counter = 0;
			torch::Tensor responses;
			first = true;
			while (valid_dataloader(mini_batch)){

				image = std::get<0>(mini_batch).to(device);
				label = std::get<1>(mini_batch).to(device);
				size_t mini_batch_size = image.size(0);

				if( first && vobose ) {
				    for(size_t i = 0; i < label.size(0); i++)
				    	std::cout << label[i].item<int64_t>() << " ";
				    std::cout << "\n";
				    first = false;
				}

				output = net->forward(image);
				auto out = torch::nn::functional::log_softmax(output, /*dim=*/1);
				loss = criterion(out, label);

				responses = output.exp().argmax(/*dim=*/1);
				for (size_t i = 0; i < mini_batch_size; i++){
				    int64_t response = responses[i].item<int64_t>();
				    int64_t answer = label[i].item<int64_t>();

				    total_counter++;
				    if (response == answer) total_match++;
				}
				total_loss += loss.item<float>();
				iteration++;
			}
			// (3) Calculate Average Loss
			float ave_loss = total_loss / (float)iteration;

			// (4) Calculate Accuracy
			float total_accuracy = (float)total_match / (float)total_counter;
			std::cout << "\nValidation accuracy: " << total_accuracy << std::endl;
		}
	}

	if( test ) {
		std::string test_dataroot = "./data/17_flowers/test";
		test_dataset = datasets::ImageFolderClassesWithPaths(test_dataroot, transform, class_names);
		test_dataloader = DataLoader::ImageFolderClassesWithPaths(test_dataset, /*batch_size_=*/1, /*shuffle_=*/false, /*num_workers_=*/0);
		std::cout << "total test images : " << test_dataset.size() << std::endl << std::endl;

		float  ave_loss = 0.0;
		size_t match = 0;
		size_t counter = 0;
		std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> data;
		std::vector<size_t> class_match = std::vector<size_t>(class_num, 0);
		std::vector<size_t> class_counter = std::vector<size_t>(class_num, 0);
		std::vector<float> class_accuracy = std::vector<float>(class_num, 0.0);

		net->eval();
		while(test_dataloader(data)){
		    image = std::get<0>(data).to(device);
		    label = std::get<1>(data).to(device);
		    output = net->forward(image);
		    auto out = torch::nn::functional::log_softmax(output, /*dim=*/1);

		    loss = criterion(out, label);

		    ave_loss += loss.item<float>();

		    output = output.exp();
		    int64_t response = output.argmax(/*dim=*/1).item<int64_t>();
		    int64_t answer = label[0].item<int64_t>();
		    counter += 1;
		    class_counter[answer]++;

		    if (response == answer){
		        class_match[answer]++;
		        match += 1;
		    }
		}

		// (7.1) Calculate Average
		ave_loss = ave_loss / (float)dataset.size();

		// (7.2) Calculate Accuracy
		std::cout << "Test accuracy ==========\n";
		for (size_t i = 0; i < class_num; i++){
			class_accuracy[i] = (float)class_match[i] / (float)class_counter[i];
		    std::cout << class_names[i] << ": " << class_accuracy[i] << "\n";
		}
		float accuracy = (float)match / float(counter);
		std::cout << "\nTest accuracy: " << accuracy << std::endl;
	}

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::plot(ax1, train_epochs, train_loss_ave, "b")->line_width(2);
    matplot::xlabel(ax1, "epoch");
    matplot::ylabel(ax1, "loss");
    matplot::show();

	std::cout << "Done!\n";
	return 0;
}
