
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

struct Residual : public torch::nn::Module {

	torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
	torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
	bool use_1x1 = false;
    //"""The Residual block of ResNet."""
    explicit Residual(int64_t input_channels, int64_t num_channels, bool use_1x1conv, int64_t strides) { // false, 1
    	use_1x1 = use_1x1conv;
        conv1 = torch::nn::Conv2d(Options(input_channels, num_channels, 3).stride(strides).padding(1));
        conv2 = torch::nn::Conv2d(Options(num_channels, num_channels, 3).padding(1));

        if( use_1x1conv )
            conv3 = torch::nn::Conv2d(Options(input_channels, num_channels, 1).stride(strides));

        bn1 = torch::nn::BatchNorm2d(num_channels);
        bn2 = torch::nn::BatchNorm2d(num_channels);
    }

    torch::Tensor forward(torch::Tensor X) {
        auto Y = torch::relu(bn1->forward(conv1->forward(X)));
        Y = bn2->forward(conv2->forward(Y));
        if(use_1x1)
            X = conv3->forward(X);
        Y += X;
        return torch::relu(Y);
    }
};

/*
 * GoogLeNet uses four modules made up of Inception blocks. However, ResNet uses four modules made up of residual blocks,
 * each of which uses several residual blocks with the same number of output channels. The number of channels in the first
 * module is the same as the number of input channels. Since a maximum pooling layer with a stride of 2 has already been used,
 * it is not necessary to reduce the height and width. In the first residual block for each of the subsequent modules,
 * the number of channels is doubled compared with that of the previous module, and the height and width are halved.
 */

std::vector<Residual> resnet_block(int64_t input_channels, int64_t num_channels, int num_residuals, bool first_block) {
	std::vector<Residual> blk;
    for(int i= 0; i < num_residuals; i++ ){
        if( i == 0 && ! first_block )
            blk.push_back(
                Residual(input_channels, num_channels, true, 2));
        else
            blk.push_back(Residual(num_channels, num_channels, false, 1));
    }
    return blk;
}

// ResNet Model
/*
 * The first two layers of ResNet are the same as those of the GoogLeNet we described before: the 7×7 convolutional layer with
 * 64 output channels and a stride of 2 is followed by the 3×3 maximum pooling layer with a stride of 2. The difference is the
 * batch normalization layer added after each convolutional layer in ResNet.
*/

struct  ResNet : public torch::nn::Module {
	std::vector<Residual> b2, b3, b4, b5;
	torch::nn::Sequential b1{nullptr};
	torch::nn::Linear linear{nullptr};
	torch::nn::Sequential classifier{nullptr};

	ResNet(int64_t num_classes) {
		// We can now implement GoogLeNet piece by piece. The first module uses a 64-channel 7×7 convolutional layer.
		b1 = torch::nn::Sequential(torch::nn::Conv2d(Options(3, 64, 7).stride(2).padding(3)),
											torch::nn::BatchNorm2d(64),
											torch::nn::ReLU(),
											torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
		// Then, we add all the modules to ResNet. Here, two residual blocks are used for each module.
		b2 = resnet_block(64, 64, 2, true);
		b3 = resnet_block(64, 128, 2, false);
		b4 = resnet_block(128, 256, 2, false);
		b5 = resnet_block(256, 512, 2, false);
		classifier = torch::nn::Sequential(
				torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})),
                torch::nn::Flatten(), torch::nn::Linear(512, num_classes));
	}

	torch::Tensor forward(torch::Tensor x) {
		x = b1->forward(x);
		for(int i =0; i < b2.size(); i++)
			x = b2[i].forward(x);

		for(int i =0; i < b3.size(); i++)
			x = b3[i].forward(x);

		for(int i =0; i < b4.size(); i++)
			x = b4[i].forward(x);

		for(int i =0; i < b5.size(); i++)
			x = b5[i].forward(x);

		return classifier->forward(x);
	}
};


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	auto blk = Residual(3, 3, false, 1);
	auto X = torch::randn({4, 3, 6, 6});
	std::cout <<  blk.forward(X).sizes() << std::endl;

	blk = Residual(3, 6, true, 2);
	std::cout <<  blk.forward(X).sizes() << std::endl;

	ResNet net(2);

	X = torch::randn({1,3,224,224});
	std::cout << net.b1->forward(X).sizes() << std::endl;
	std::cout << net.forward(X).sizes() << std::endl;

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
    ResNet model = ResNet(class_names.size());
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




