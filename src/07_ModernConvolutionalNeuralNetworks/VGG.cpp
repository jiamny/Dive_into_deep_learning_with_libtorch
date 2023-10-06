
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

void _initialize_weights(torch::nn::Module &m) {
	if ((typeid(m) == typeid(torch::nn::Conv2d)) || (typeid(m) == typeid(torch::nn::Conv2dImpl))) {
	        auto p = m.named_parameters(false);
	        auto w = p.find("weight");
	        auto b = p.find("bias");
	        //if (w != nullptr) nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.01);
	        if (w != nullptr) torch::nn::init::kaiming_normal_(*w, /*a=*/0, torch::kFanOut, torch::kReLU);
	        if (b != nullptr) torch::nn::init::constant_(*b, /*bias=*/0.0);
	} else if ((typeid(m) == typeid(torch::nn::Linear)) || (typeid(m) == typeid(torch::nn::LinearImpl))){
	        auto p = m.named_parameters(false);
	        auto w = p.find("weight");
	        auto b = p.find("bias");
	        if (w != nullptr) torch::nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.01);
	        if (b != nullptr) torch::nn::init::constant_(*b, /*bias=*/0.0);
	} else if ((typeid(m) == typeid(torch::nn::BatchNorm2d)) || (typeid(m) == typeid(torch::nn::BatchNorm2dImpl))){
	        auto p = m.named_parameters(false);
	        auto w = p.find("weight");
	        auto b = p.find("bias");
	        if (w != nullptr) torch::nn::init::constant_(*w, /*weight=*/1.0);
	        if (b != nullptr) torch::nn::init::constant_(*b, /*bias=*/0.0);
	}
	return;
}


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
struct VGGImpl : public torch::nn::Module {

private:
	torch::nn::Sequential vb1{nullptr}, vb2{nullptr}, vb3{nullptr}, vb4{nullptr}, vb5{nullptr};
	torch::nn::Sequential classifier;

public:
	VGGImpl(int64_t num_classes) {
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
	}

	torch::Tensor forward(torch::Tensor x) {
		x = vb1->forward(x);
		x = vb2->forward(x);
		x = vb3->forward(x);
		x = vb4->forward(x);
		x = vb5->forward(x);
	return classifier->forward(x);
	}

	void init() {
		this->apply(_initialize_weights);
		return;
	}
};

TORCH_MODULE(VGG);

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

	auto tnet = VGG(17);
	tnet->init();
	auto X = torch::randn({1,3,224,224});
	std::cout << tnet->forward(X) << std::endl;

	size_t img_size = 224;
	size_t batch_size = 8;
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
    auto net = VGG(class_num);

	torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(1e-4).betas({0.5, 0.999}));

	auto criterion = torch::nn::NLLLoss(torch::nn::NLLLossOptions().ignore_index(-100).reduction(torch::kMean));

	net->init();
	net->to(device);

	size_t epoch;
	size_t total_iter = dataloader.get_count_max();
	size_t start_epoch, total_epoch;
	start_epoch = 1;
	total_iter = dataloader.get_count_max();
	total_epoch = 5;
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


