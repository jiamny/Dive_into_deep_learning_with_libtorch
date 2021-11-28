
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

/*
 * DenseNet uses the modified "batch normalization, activation, and convolution" structure of ResNet (see the exercise in
 * :numref:sec_resnet). First, we implement this convolution block structure.
 */
struct conv_block : public torch::nn::SequentialImpl {

	conv_block(int64_t input_channels, int64_t num_channels) {
	    push_back(torch::nn::BatchNorm2d(input_channels));
	    push_back(torch::nn::ReLU());
	    push_back(torch::nn::Conv2d(Options(input_channels, num_channels, 3).padding(1)));
	}

	torch::Tensor forward(torch::Tensor x) {
		auto Y = torch::nn::SequentialImpl::forward(x);
		return torch::cat({x, Y}, 1);
	}
};

struct DenseBlock : public torch::nn::SequentialImpl {

	DenseBlock(int64_t num_convs, int64_t input_channels, int64_t num_channels ) {
        for( int64_t i = 0; i < num_convs; i++ ) {
        	 push_back(conv_block(num_channels * i + input_channels, num_channels));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        return torch::nn::SequentialImpl::forward(x);
    }
};

// Transition Layers]
struct transition_block : public torch::nn::SequentialImpl {
	transition_block(int64_t input_channels, int64_t num_channels) {
		push_back(torch::nn::BatchNorm2d(input_channels));
		push_back(torch::nn::ReLU());
		push_back(torch::nn::Conv2d(Options(input_channels, num_channels, 1)));
		push_back(torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2).stride(2)));
	}

	torch::Tensor forward(torch::Tensor x) {
	    return torch::nn::SequentialImpl::forward(x);
	}
};


// DenseNet Model
/*
 *  DenseNet uses four dense blocks. Similar to ResNet, we can set the number of convolutional layers used in each
 *  dense block. Here, we set it to 4, consistent with the ResNet-18 model in :numref:sec_resnet. Furthermore, we set
 *  the number of channels (i.e., growth rate) for the convolutional layers in the dense block to 32, so 128 channels
 *  will be added to each dense block.
 */
struct DensNet : public torch::nn::Module {
	torch::nn::Linear classifier{nullptr};
	//# `num_channels`: the current number of channels
	int64_t num_channels = 64, growth_rate = 32;
	std::vector<int64_t> num_convs_in_dense_blocks = {4, 4, 4, 4};
	//std::vector<DenseBlock> blks;
	//std::vector<torch::nn::Sequential> tblks;
	torch::nn::Sequential features = torch::nn::Sequential();

	DensNet(int64_t num_classes) {
		features->push_back(torch::nn::Conv2d(Options(3, 64, 7).stride(2).padding(3)));
		features->push_back(torch::nn::BatchNorm2d(64));
		features->push_back(torch::nn::ReLU());
		features->push_back(torch::nn::Functional(torch::max_pool2d, 3, 2, 1, 1, false));

		for( int i = 0; i < num_convs_in_dense_blocks.size(); i++ ) {
			int64_t num_convs = num_convs_in_dense_blocks[i];

			features->push_back(DenseBlock(num_convs, num_channels, growth_rate));
			//# This is the number of output channels in the previous dense block
			num_channels += num_convs * growth_rate;
			//# A transition layer that halves the number of channels is added between
		   	//# the dense blocks
			if( i != (num_convs_in_dense_blocks.size() - 1) ) {
				features->push_back(transition_block(num_channels, static_cast<int64_t>(num_channels / 2.0)));
				//tblks.push_back( transition_block(num_channels, static_cast<int64_t>(num_channels / 2.0)));
				num_channels = static_cast<int64_t>(num_channels / 2.0);
			}
		}

		// Final batch norm
		features->push_back(torch::nn::BatchNorm2d(num_channels));
		// Linear layer
		classifier = torch::nn::Linear(num_channels, num_classes);

		// Official init from torch repo.
		for (auto& module : modules(/*include_self=*/false)) {
		    if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get()))
		      torch::nn::init::kaiming_normal_(M->weight);
		    else if (auto M = dynamic_cast<torch::nn::BatchNorm2dImpl*>(module.get())) {
		      torch::nn::init::constant_(M->weight, 1);
		      torch::nn::init::constant_(M->bias, 0);
		    } else if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get()))
		      torch::nn::init::constant_(M->bias, 0);
		}
    }

    torch::Tensor forward(torch::Tensor x) {
    	auto features = this->features->forward(x);
    	auto out = torch::relu_(features);
    	out = torch::adaptive_avg_pool2d(out, {1, 1});

    	out = out.view({features.size(0), -1});
    	out = this->classifier->forward(out);
        return out;
    }
};

using torch::indexing::Slice;
using torch::indexing::None;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	auto flowerLabels = getFlowersLabels("./data/flowers_cat_to_name.json");

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	auto tnet = DensNet(2);
	auto X = torch::randn({1,3,224,224});
	std::cout << tnet.forward(X).sizes() << std::endl;

	size_t img_size = 224;
	size_t batch_size = 16;
	size_t valid_batch_size = 16;
	std::vector<std::string> class_names = {"cat", "fish"};
	constexpr bool train_shuffle = true;  // whether to shuffle the training dataset
	constexpr size_t train_workers = 2;  // the number of workers to retrieve data from the training dataset
	constexpr bool valid_shuffle = true;  // whether to shuffle the validation dataset
	constexpr size_t valid_workers = 2;  // the number of workers to retrieve data from the validation dataset

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
    DensNet model = DensNet(class_names.size());
    model.to(device);

    std::cout << "Training Model..." << std::endl;

    auto criterion = torch::nn::CrossEntropyLoss();//torch::nn::NLLLoss(torch::nn::functional::NLLLossFuncOptions().reduction(torch::kMean).ignore_index(-100)); //torch::nn::CrossEntropyLoss();
    auto optimizer = torch::optim::Adam(model.parameters(), torch::optim::AdamOptions(1e-4).betas({0.5, 0.999}));

    // (3) Set Optimizer Method
    //auto optimizer = torch::optim::SGD(model.parameters(), torch::optim::SGDOptions(learning_rate)
    //														.momentum(0.9)
	//														.weight_decay(1e-4));


    // Requires LibTorch >= 1.90
/*
    plt::figure_size(800, 500);
    plt::subplot(1, 2, 1);
    auto batch = *train_data_loader->begin();
    std::vector<unsigned char> z = tensorToMatrix(batch.data[1]);
    const uchar* zptr = &(z[0]);
    int label = batch.target[1].item<int64_t>();
    std::string t = cls[label];
    std::string tlt = flowerLabels[t]; //cls[label];
    plt::title(tlt.c_str());
    plt::imshow(zptr, img_size, img_size, 3);

    plt::subplot(1, 2, 2);
    std::vector<unsigned char> z2 = tensorToMatrix(batch.data[7]);
    const uchar* zptr2 = &(z2[0]);
    label = batch.target[7].item<int64_t>();
    t = cls[label];
    tlt = flowerLabels[t]; //cls[label];
    plt::title(tlt.c_str());
    plt::imshow(zptr2, img_size, img_size, 3);
    plt::show();
*/

    // -----------------------------------
    // a2. Training Model
    // -----------------------------------

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



