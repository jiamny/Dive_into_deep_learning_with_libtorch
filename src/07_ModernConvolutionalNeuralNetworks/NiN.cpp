
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

struct NiN : public torch::nn::Module {
	torch::nn::Sequential b1{nullptr}, b2{nullptr}, b3{nullptr}, b4{nullptr};
	torch::nn::MaxPool2d maxpool1{nullptr}, maxpool2{nullptr}, maxpool3{nullptr};
	torch::nn::AdaptiveAvgPool2d adpavpool{nullptr};
	torch::nn::Flatten flattern{nullptr};

	NiN(int64_t num_classes) {
		b1 = nin_block(3, 96, 11, 4, 0);
		b2 = nin_block(96, 256, 5, 1, 2);
		b3 = nin_block(256, 384, 3, 1, 1);
		b4 = nin_block(384, num_classes, 3, 1, 1);

		maxpool1 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2));
		maxpool2 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2));
		maxpool3 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2));

		adpavpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
		flattern  = torch::nn::Flatten();
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
	auto net = NiN(10);
	std::cout << net.parameters(false) << std::endl;
	// We create a data example to see [the output shape of each block].
	auto X = torch::randn({1, 3, 224, 224});
	X = net.forward(X);
	std::cout << X.sizes()  << std::endl;
	std::cout << X  << std::endl;

	size_t img_size = 224;
	size_t batch_size = 16;
	size_t valid_batch_size = 16;
	std::vector<std::string> class_names = {"cat", "fish"};
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

	std::string dataroot = "./data/cat_fish/train", valid_dataroot="./data/cat_fish/val";
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
    NiN model = NiN(class_names.size());
    model.to(device);

    std::cout << "Training Model..." << std::endl;

    auto criterion = torch::nn::CrossEntropyLoss();//torch::nn::NLLLoss(torch::nn::functional::NLLLossFuncOptions().reduction(torch::kMean).ignore_index(-100)); //torch::nn::CrossEntropyLoss();
    auto optimizer = torch::optim::Adam(model.parameters(), torch::optim::AdamOptions(1e-4).betas({0.5, 0.999}));

    // (1) Set Parameters
    size_t start_epoch = 1;
    size_t total_epoch = 18;

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
                if (response == answer){
                	class_match[answer]++;
                    total_match++;
                }
            }

            total_counter += mini_batch_size;
            total_loss += val_loss.item<float>();
            iteration++;
        }

        // (3) Calculate Average Loss
        ave_loss = total_loss / (float)iteration;

        // (4) Calculate Accuracy
        for (size_t i = 0; i < class_names.size(); i++){
           class_accuracy[i] = (float)class_match[i] / (float)class_counter[i];
        }
        total_accuracy = (float)total_match / (float)total_counter;

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
