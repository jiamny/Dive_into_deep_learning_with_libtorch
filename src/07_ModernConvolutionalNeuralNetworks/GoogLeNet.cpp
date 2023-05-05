
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
struct  GoogLeNetImpl : public torch::nn::Module {
	torch::nn::Sequential b1{nullptr}, b2{nullptr}, b3{nullptr}, b4{nullptr}, b5{nullptr};
	torch::nn::Linear linear{nullptr};

	GoogLeNetImpl(int64_t num_classes) {
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
		    if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
		      torch::nn::init::normal_(M->weight); // Note: used instead of truncated normal initialization
		      torch::nn::init::zeros_(M->bias);
		    } else if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
		      torch::nn::init::normal_(M->weight); // Note: used instead of truncated normal initialization
		      torch::nn::init::zeros_(M->bias);
		    } else if (auto M = dynamic_cast<torch::nn::BatchNorm2dImpl*>(module.get())) {
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
TORCH_MODULE(GoogLeNet);

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

	auto tnet = GoogLeNet(17);
	auto X = torch::randn({1,3,224,224});
	std::cout << tnet->forward(X).sizes() << std::endl;

	X = tnet->b1->forward(X);
	std::cout << "name: " << tnet->b1->name() << "output shape: \t" << X.sizes() << std::endl;
	X = tnet->b2->forward(X);
	std::cout << "name: " << tnet->b2->name() << "output shape: \t" << X.sizes() << std::endl;
	X = tnet->b3->forward(X);
	std::cout << "name: " << tnet->b3->name() << "output shape: \t" << X.sizes() << std::endl;
	X = tnet->b4->forward(X);
	std::cout << "name: " << tnet->b4->name() << "output shape: \t" << X.sizes() << std::endl;
	X = tnet->b5->forward(X);
	std::cout << "name: " << tnet->b5->name() << "output shape: \t" << X.sizes() << std::endl;

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

    auto net = GoogLeNet(class_num);

	torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(1e-4).betas({0.5, 0.999}));

	auto criterion = torch::nn::NLLLoss(torch::nn::NLLLossOptions().ignore_index(-100).reduction(torch::kMean));

	net->to(device);

	size_t epoch;
	size_t total_iter = dataloader.get_count_max();
	size_t start_epoch, total_epoch;
	start_epoch = 1;
	total_iter = dataloader.get_count_max();
	total_epoch = 40;
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


