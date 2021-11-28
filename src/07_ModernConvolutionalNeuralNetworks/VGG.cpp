
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

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

using Options = torch::nn::Conv2dOptions;


// VGG Blocks
torch::nn::Sequential vgg_block(int64_t num_convs, int64_t in_channels, int64_t out_channels) {
	torch::nn::Sequential layers;
    for(int i = 0; i < num_convs; i++ ) {
    	layers->push_back(torch::nn::Conv2d(Options(in_channels, out_channels, 3).padding(1)));
    	layers->push_back(torch::nn::ReLU());
        in_channels = out_channels;
    }
    layers->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
    return layers;
}

// The following code implements VGG-11. This is a simple matter of executing a for-loop over conv_arch.
struct VGG : public torch::nn::Module {
	torch::nn::Sequential vb1{nullptr}, vb2{nullptr}, vb3{nullptr}, vb4{nullptr}, vb5{nullptr};
	torch::nn::Sequential classifier;

	VGG(int64_t num_classes, bool initialize_weights) {
		int64_t in_channels = 3;
		int64_t num_convs = 1;
		int64_t out_channels = 64;
		vb1 = vgg_block(num_convs, in_channels, out_channels);

		in_channels = out_channels;
		num_convs = 1;
		out_channels = 128;
		vb2 = vgg_block(num_convs, in_channels, out_channels);

		in_channels = out_channels;
		num_convs = 2;
		out_channels = 256;
		vb3 = vgg_block(num_convs, in_channels, out_channels);

		in_channels = out_channels;
		num_convs = 2;
		out_channels = 512;
		vb4 = vgg_block(num_convs, in_channels, out_channels);

		in_channels = out_channels;
		num_convs = 2;
		out_channels = 512;
		vb5 = vgg_block(num_convs, in_channels, out_channels);

		classifier = torch::nn::Sequential(
		        torch::nn::Flatten(),
		        // The fully-connected part
		        torch::nn::Linear(512 * 7 * 7, 4096),
				torch::nn::ReLU(),
				torch::nn::Dropout(0.5),
		        torch::nn::Linear(4096, 4096),
				torch::nn::ReLU(),
				torch::nn::Dropout(0.5),
		        torch::nn::Linear(4096, num_classes));

		register_module("vb1", vb1);
		register_module("vb2", vb2);
		register_module("vb3", vb3);
		register_module("vb4", vb4);
		register_module("vb5", vb5);
		register_module("classifier", classifier);

		if( initialize_weights )
			_initialize_weights();
	}

	torch::Tensor forward(torch::Tensor x) {
		x = vb1->forward(x);
		x = vb2->forward(x);
		x = vb3->forward(x);
		x = vb4->forward(x);
		x = vb5->forward(x);
	return classifier->forward(x);
	}

	void _initialize_weights() {
	  for (auto& module : modules(/*include_self=*/false)) {
	    if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
	      torch::nn::init::kaiming_normal_(
	          M->weight,
	          /*a=*/0,
	          torch::kFanOut,
	          torch::kReLU);
	      torch::nn::init::constant_(M->bias, 0);
	    } else if (
	        auto M = dynamic_cast<torch::nn::BatchNorm2dImpl*>(module.get())) {
	      torch::nn::init::constant_(M->weight, 1);
	      torch::nn::init::constant_(M->bias, 0);
	    } else if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
	      torch::nn::init::normal_(M->weight, 0, 0.01);
	      torch::nn::init::constant_(M->bias, 0);
	    }
	  }
	}
};


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	auto tnet = VGG(2, true);
	auto X = torch::randn({1,3,224,224});
	std::cout << tnet.forward(X) << std::endl;

	size_t img_size = 224;
	size_t batch_size = 16;
	size_t valid_batch_size = 16;
	std::vector<std::string> class_names = {"ants", "bees"};
	constexpr bool train_shuffle = true;   // whether to shuffle the training dataset
	constexpr size_t train_workers = 2;    // the number of workers to retrieve data from the training dataset
	constexpr bool valid_shuffle = true;   // whether to shuffle the validation dataset
	constexpr size_t valid_workers = 2;    // the number of workers to retrieve data from the validation dataset

    // (4) Set Transforms
    std::vector<transforms_Compose> transform {
        transforms_Resize(cv::Size(img_size, img_size), cv::INTER_LINEAR),        // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        transforms_ToTensor()                                                     // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
//        transforms_Normalize(std::vector<float>{0.485, 0.456, 0.406}, std::vector<float>{0.229, 0.224, 0.225})  // Pixel Value Normalization for ImageNet
    };

	std::string dataroot = "./data/hymenoptera_data/train", valid_dataroot="./data/hymenoptera_data/val";
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor loss, image, label, output;
    datasets::ImageFolderClassesWithPaths dataset,  valid_dataset;
    DataLoader::ImageFolderClassesWithPaths dataloader, valid_dataloader;

    // -----------------------------------
    // a1. Preparation
    // -----------------------------------

    // (1) Get Training Dataset
    dataset = datasets::ImageFolderClassesWithPaths(dataroot, transform, class_names);
    dataloader = DataLoader::ImageFolderClassesWithPaths(dataset, batch_size, /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);
    std::cout << "total training images : " << dataset.size() << std::endl;

    valid_dataset = datasets::ImageFolderClassesWithPaths(valid_dataroot, transform, class_names);
    valid_dataloader = DataLoader::ImageFolderClassesWithPaths(valid_dataset, valid_batch_size, /*shuffle_=*/valid_shuffle, /*num_workers_=*/valid_workers);
    std::cout << "total validation images : " << valid_dataset.size() << std::endl;


    // (5) Define Network
    VGG model = VGG(class_names.size(), true);
    model.to(device);


    std::cout << "Training Model..." << std::endl;

    auto criterion = torch::nn::CrossEntropyLoss();//torch::nn::NLLLoss(torch::nn::functional::NLLLossFuncOptions().reduction(torch::kMean).ignore_index(-100)); //torch::nn::CrossEntropyLoss();
    auto optimizer = torch::optim::Adam(model.parameters(), torch::optim::AdamOptions(1e-2).betas({0.5, 0.9}));

    // (1) Set Parameters
    size_t start_epoch = 1;
    size_t total_epoch = 30;

    // (2) Training per Epoch

    for(size_t epoch = start_epoch; epoch <= total_epoch; epoch++){

        model.train();
        std::cout << std::endl << "epoch:" << epoch << '/' << total_epoch << std::endl;

        size_t iter = 0; //dataloader.get_count_max();
        // -----------------------------------
        // b1. Mini Batch Learning
        // -----------------------------------
        while (dataloader(mini_batch)){
        	iter++;

            image = std::get<0>(mini_batch).to(device);
            label = std::get<1>(mini_batch).to(device);
            output = model.forward(image);
            loss = criterion(output, label);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // -----------------------------------
            // c2. Record Loss (iteration)
            // -----------------------------------
            if( iter % 10 == 0 ) std::cout  << "loss:" << loss.item<float>() << std::endl;
        }

        // (0) Initialization and Declaration
        size_t iteration;
        size_t mini_batch_size;
        size_t class_num;
        size_t total_match, total_counter;
        long int response, answer;
        float total_accuracy;
        float ave_loss, total_loss;

        std::vector<size_t> class_match, class_counter;
        std::vector<float> class_accuracy;
        std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> val_mini_batch;
        torch::Tensor val_loss, val_image, val_label, val_output, val_responses;

        // (1) Memory Allocation
        class_match = std::vector<size_t>(class_names.size(), 0);
        class_counter = std::vector<size_t>(class_names.size(), 0);
        class_accuracy = std::vector<float>(class_names.size(), 0.0);

        // (2) Tensor Forward per Mini Batch
        model.eval();
        iteration = 0;
        total_loss = 0.0;
        total_match = 0; total_counter = 0;

        while (valid_dataloader(val_mini_batch)){

        	val_image = std::get<0>(val_mini_batch).to(device);
            val_label = std::get<1>(val_mini_batch).to(device);
            auto fnames = std::get<2>(val_mini_batch);

            mini_batch_size = val_image.size(0);

            val_output = model.forward(val_image);
            val_loss = criterion(val_output, val_label);

            val_responses = val_output.exp().argmax(/*dim=*/1);
            auto agr = val_output.argmax(1);

            if( epoch == total_epoch ) {
            	std::cout << "val_label.size(0) = " << val_label.size(0) << std::endl;
            	for(size_t j = 0; j < val_label.size(0); j++ ) {
            		std::cout << "response: " << val_responses[j].item<long int>()
            			  << " agr: " << agr[j].item<long int>()
						  << " label: " << val_label[j].item<long int>()
						  << " fname: " << fnames[j] << std::endl;
            	}
            }

            for (size_t i = 0; i < mini_batch_size; i++){
            	response = val_responses[i].item<long int>();
                answer = val_label[i].item<long int>();
                class_counter[answer]++;

                if( response == answer ){
                	class_match[answer]++;
                    total_match++;
                }
            }

            total_loss += val_loss.item<float>();
            total_counter += mini_batch_size;
            iteration++;
        }

        // (3) Calculate Average Loss
        ave_loss = total_loss / iteration;

        // (4) Calculate Accuracy
        for (size_t i = 0; i < class_names.size(); i++){
           class_accuracy[i] = (class_match[i] * 1.0) / class_counter[i];
        }

        total_accuracy = (total_match * 1.0) / total_counter;

        // (5.1) Record Loss (Log/Loss)
        std::cout << "epoch:" << epoch << '/' << total_epoch << " ave_loss:" << ave_loss << " accuracy:" << total_accuracy << std::endl;

        std::cout << "class_accuracy" << " ";
        for (size_t i = 0; i < class_names.size(); i++){
            std::cout << class_names[i] << ": " << class_accuracy[i] << ", ";
        }
        std::cout << std::endl;
    }


	std::cout << "Done!\n";
	return 0;
}


