
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

//Inception Blocks
/*
 * The first three paths use convolutional layers with window sizes of 1×1, 3×3, and 5×5 to extract information from different
 * spatial sizes. The middle two paths perform a 1×1 convolution on the input to reduce the number of channels, reducing the model's
 * complexity. The fourth path uses a 3×3 maximum pooling layer, followed by a 1×1 convolutional layer to change the number of channels.
 * The four paths all use appropriate padding to give the input and output the same height and width. Finally, the outputs along
 * each path are concatenated along the channel dimension and comprise the block's output. The commonly-tuned hyperparameters of
 * the Inception block are the number of output channels per layer.
 */
struct Inception : public torch::nn::Module {
	torch::nn::Conv2d p1_1{nullptr}, p2_1{nullptr}, p2_2{nullptr}, p3_1{nullptr}, p3_2{nullptr}, p4_2{nullptr};
	torch::nn::MaxPool2d p4_1{nullptr};

    //# `c1`--`c4` are the number of output channels for each path
	Inception(int64_t in_channels, int64_t c1, std::vector<int64_t> c2, std::vector<int64_t> c3, int64_t c4) {

        //# Path 1 is a single 1 x 1 convolutional layer
        p1_1 = torch::nn::Conv2d(Options(in_channels, c1, 1));
        //# Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        //# convolutional layer
        p2_1 = torch::nn::Conv2d(Options(in_channels, c2[0], 1));
        p2_2 = torch::nn::Conv2d(Options(c2[0], c2[1], 3).padding(1));
        //# Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        //# convolutional layer
        p3_1 = torch::nn::Conv2d(Options(in_channels, c3[0], 1));
        p3_2 = torch::nn::Conv2d(Options(c3[0], c3[1], 5).padding(2));
        //# Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        //# convolutional layer
        p4_1 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(1).padding(1));
        p4_2 = torch::nn::Conv2d(Options(in_channels, c4, 1));
	}

    torch::Tensor forward(torch::Tensor x) {
        auto p1 = torch::relu(p1_1->forward(x));
        auto p2 = torch::relu(p2_2->forward(torch::relu(p2_1->forward(x))));
        auto p3 = torch::relu(p3_2->forward(torch::relu(p3_1->forward(x))));
        auto p4 = torch::relu(p4_2->forward(p4_1->forward(x)));
        //# Concatenate the outputs on the channel dimension
        return torch::cat({p1, p2, p3, p4}, 1);
    }
};

//  GoogLeNet uses a stack of a total of 9 inception blocks and global average pooling to generate its estimates.
struct  GoogLeNet : public torch::nn::Module {
	torch::nn::Sequential b1{nullptr}, b2{nullptr}, b3{nullptr}, b4{nullptr}, b5{nullptr};
	torch::nn::Linear linear{nullptr};

	GoogLeNet(int64_t num_classes) {
		// We can now implement GoogLeNet piece by piece. The first module uses a 64-channel 7×7 convolutional layer.
		b1 = torch::nn::Sequential(
									torch::nn::Conv2d(Options(3, 64, 7).stride(2).padding(3)),
									torch::nn::ReLU(),
									torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));

		// The second module uses two convolutional layers: first, a 64-channel 1×1 convolutional layer,
		// then a 3×3 convolutional layer that triples the number of channels.
		b2 = torch::nn::Sequential(
									torch::nn::Conv2d(Options(64, 64, 1)),
									torch::nn::ReLU(),
									torch::nn::Conv2d(Options(64, 192, 3).padding(1)),
									torch::nn::ReLU(),
									torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));

		/*
		 * The third module connects two complete Inception blocks in series. The number of output channels of the first
		 * Inception block is 64+128+32+32=256, and the number-of-output-channel ratio among the four paths is 64:128:32:32=2:4:1:1.
		 * The second and third paths first reduce the number of input channels to 96/192=1/2 and 16/192=1/12, respectively,
		 * and then connect the second convolutional layer. The number of output channels of the second Inception block is
		 * increased to 128+192+96+64=480, and the number-of-output-channel ratio among the four paths is 128:192:96:64=4:6:3:2.
		 * The second and third paths first reduce the number of input channels to 128/256=1/2 and 32/256=1/8, respectively.
		 */
		b3 = torch::nn::Sequential(
									Inception(192, 64, std::vector<int64_t>({96, 128}), std::vector<int64_t>({16, 32}), 32),
									Inception(256, 128, std::vector<int64_t>({128, 192}), std::vector<int64_t>({32, 96}), 64),
									torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));

		/*
		 * The fourth module is more complicated. It connects five Inception blocks in series, and they have 192+208+48+64=512,
		 * 160+224+64+64=512, 128+256+64+64=512, 112+288+64+64=528, and 256+320+128+128=832 output channels, respectively.
		 * The number of channels assigned to these paths is similar to that in the third module: the second path with the 3×3 convolutional
		 * layer outputs the largest number of channels, followed by the first path with only the 1×1 convolutional layer,
		 * the third path with the 5×5 convolutional layer, and the fourth path with the 3×3 maximum pooling layer.
		 * The second and third paths will first reduce the number of channels according to the ratio. These ratios are slightly
		 * different in different Inception blocks.
		 */
		b4 = torch::nn::Sequential(
									Inception(480, 192, std::vector<int64_t>({96, 208}), std::vector<int64_t>({16, 48}), 64),
		                   	   	   	Inception(512, 160, std::vector<int64_t>({112, 224}), std::vector<int64_t>({24, 64}), 64),
									Inception(512, 128, std::vector<int64_t>({128, 256}), std::vector<int64_t>({24, 64}), 64),
									Inception(512, 112, std::vector<int64_t>({144, 288}), std::vector<int64_t>({32, 64}), 64),
									Inception(528, 256, std::vector<int64_t>({160, 320}), std::vector<int64_t>({32, 128}), 128),
									torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));

		/*
		 * The fifth module has two Inception blocks with 256+320+128+128=832 and 384+384+128+128=1024 output channels. The number of
		 * channels assigned to each path is the same as that in the third and fourth modules, but differs in specific values.
		 * It should be noted that the fifth block is followed by the output layer. This block uses the global average pooling layer
		 * to change the height and width of each channel to 1, just as in NiN. Finally, we turn the output into a two-dimensional
		 * array followed by a fully-connected layer whose number of outputs is the number of label classes.
		 */

		b5 = torch::nn::Sequential(
								Inception(832, 256, std::vector<int64_t>({160, 320}), std::vector<int64_t>({32, 128}), 128),
				                Inception(832, 384, std::vector<int64_t>({192, 384}), std::vector<int64_t>({48, 128}), 128)
								//torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})),
								//torch::nn::Flatten()
								);
		linear = torch::nn::Linear(1024, num_classes);

		register_module("b1", b1);
		register_module("b2", b2);
		register_module("b3", b3);
		register_module("b4", b4);
		register_module("b5", b5);
		register_module("fc", linear);

		// init_weights
		for (auto& module : modules(/*include_self=*/false)) {
		    if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get()))
		      torch::nn::init::normal_(M->weight); // Note: used instead of truncated
		                                           // normal initialization
		    else if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get()))
		      torch::nn::init::normal_(M->weight); // Note: used instead of truncated
		                                           // normal initialization
		    else if (auto M = dynamic_cast<torch::nn::BatchNorm2dImpl*>(module.get())) {
		      torch::nn::init::ones_(M->weight);
		      torch::nn::init::zeros_(M->bias);
		    }
		}
	}

	torch::Tensor forward(torch::Tensor x) {
		x = b1->forward(x);
		x = b2->forward(x);
		x = b3->forward(x);
		x = b4->forward(x);
		x = b5->forward(x);
		x = torch::adaptive_avg_pool2d(x, {1, 1});
		x = x.view({x.size(0), -1});
		torch::dropout(x, 0.2, is_training());
		x = linear->forward(x);
		return x;
	}
};

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	auto tnet = GoogLeNet(2);
	auto X = torch::randn({1,3,96,96});

	X = tnet.b1->forward(X);
	std::cout << "name: " << tnet.b1->name() << "output shape: \t" << X.sizes() << std::endl;
	X = tnet.b2->forward(X);
	std::cout << "name: " << tnet.b2->name() << "output shape: \t" << X.sizes() << std::endl;
	X = tnet.b3->forward(X);
	std::cout << "name: " << tnet.b3->name() << "output shape: \t" << X.sizes() << std::endl;
	X = tnet.b4->forward(X);
	std::cout << "name: " << tnet.b4->name() << "output shape: \t" << X.sizes() << std::endl;
	X = tnet.b5->forward(X);
	std::cout << "name: " << tnet.b5->name() << "output shape: \t" << X.sizes() << std::endl;

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
	GoogLeNet model = GoogLeNet(class_names.size());
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

