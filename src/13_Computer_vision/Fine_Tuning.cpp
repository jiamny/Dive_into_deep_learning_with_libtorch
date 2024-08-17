#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include <torch/script.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <torch/script.h>

#include <iostream>
#include <memory>
#include <algorithm>
#include <random>

#include "resnet.h"
#include "../utils/ch_13_util.h"
#include "../utils/transforms.hpp"              // transforms_Compose
#include "../utils/datasets.hpp"                // datasets::ImageFolderClassesWithPaths
#include "../utils/dataloader.hpp"              // DataLoader::ImageFolderClassesWithPaths


#include <matplot/matplot.h>
using namespace matplot;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	//torch::Device device(torch::kCPU);
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	std::vector<std::string> class_names = {"hotdog", "not-hotdog"};
	std::vector<float> rgb_mean = {0.485, 0.456, 0.406};
	std::vector<float> rgb_std = {0.229, 0.224, 0.225};

	size_t img_size = 224;
	size_t batch_size = 128;
	const size_t class_num = 2;
	const size_t valid_batch_size = 1;

	constexpr bool train_shuffle = true;    // whether to shuffle the training dataset
	constexpr size_t train_workers = 2;  	// the number of workers to retrieve data from the training dataset
	constexpr bool valid_shuffle = true;    // whether to shuffle the validation dataset
	constexpr size_t valid_workers = 2;     // the number of workers to retrieve data from the validation dataset

    // Set Transforms
    std::vector<transforms_Compose> transform {
        transforms_Resize(cv::Size(img_size, img_size), cv::INTER_LINEAR),  // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        transforms_ToTensor(),                                              // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
		transforms_Normalize(rgb_mean, rgb_std)  							// Pixel Value Normalization for ImageNet
    };

	std::string dataroot = "./data/hotdog/train";
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor loss, image, label, output;
    datasets::ImageFolderClassesWithPaths dataset, valid_dataset, test_dataset;      		// dataset;
    DataLoader::ImageFolderClassesWithPaths dataloader, valid_dataloader, test_dataloader; 	// dataloader;

    // Get Dataset
    dataset = datasets::ImageFolderClassesWithPaths(dataroot, transform, class_names);
    std::cout << "total training images : " << dataset.size() << std::endl;

    std::string fname;
    torch::Tensor img;
    torch::Tensor cid;
    std::tuple<torch::Tensor, torch::Tensor, std::string> itdata = std::make_tuple(img, cid, fname);
    int hdog = 0;
    int nhdog = 0;

	auto f = figure(true);
	f->width(f->width() * 2);
	f->height(f->height() * 1.5);
	f->x_position(0);
	f->y_position(0);

    for( int64_t i = 0; i < dataset.size(); i++ ) {
    	dataset.get(i, itdata);
    	if( std::get<1>(itdata).data().item<long>() == 0 && hdog < 5 ) {
    		torch::Tensor imgT = std::get<0>(itdata);
    		imgT.to(device);
    		imgT = deNormalizeTensor(imgT, rgb_mean, rgb_std);
    		imgT = torch::clamp(imgT, 0, 1);

    		matplot::subplot(2, 5, hdog);
			std::vector<std::vector<std::vector<unsigned char>>> z = tensorToMatrix4MatplotPP(imgT.clone());
			matplot::imshow(z);
    		hdog++;
    	} else {
        	if( std::get<1>(itdata).data().item<long>() == 1 && nhdog < 5 ) {
        		torch::Tensor imgN = std::get<0>(itdata);
        		imgN.to(device);
        		imgN = deNormalizeTensor(imgN, rgb_mean, rgb_std);
        		imgN = torch::clamp(imgN, 0, 1);

        		matplot::subplot(2, 5, 5 + nhdog);
    			std::vector<std::vector<std::vector<unsigned char>>> z = tensorToMatrix4MatplotPP(imgN.clone());
    			matplot::imshow(z);
        		nhdog++;
        	}
    	}
    }

	matplot::show();

	ResNet18 net(1000, true);
	net->to(device);

	/*
	std::string mdlf = "./src/13_Computer_vision/resnet18_jit_model.pt";

	torch::jit::script::Module net;
	try {
	    // Deserialize the ScriptModule from a file using torch::jit::load().
	    net = torch::jit::load(mdlf.c_str(), device);
	} catch (const c10::Error& e) {
	    std::cerr << "error loading the model\n";
	    return -1;
	}

	auto X = torch::rand({1, 3, 224, 224}).to(device);
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(X);

	auto ot = net.forward(inputs).toTensor();

	std::cout << "ot: " << ot.sizes() << "\n";
	std::cout << net.attr("fc") << '\n';
	*/
	auto X = torch::rand({1, 3, 224, 224}).to(device);
	auto ot = net->forward(X);
	std::cout << "ot: " << ot.sizes() << "\n";
	std::cout << (*(net->named_modules())["fc"]) << '\n';
	//torch::nn::Linear fc = (*(net->named_modules()["fc"]));
	//fc->parameters();

	auto dict = net->named_parameters();
	for (auto n = dict.begin(); n != dict.end(); n++) {
		std::cout << (*n).key() << "\n"; // << (*n).value() << std::endl;
	}

    std::string valid_dataroot = "./data/hotdog/test";

    valid_dataset = datasets::ImageFolderClassesWithPaths(valid_dataroot, transform, class_names);
    std::cout << "total validation images : " << valid_dataset.size() << std::endl;

    dataloader = DataLoader::ImageFolderClassesWithPaths(dataset, batch_size, train_shuffle, train_workers);
    valid_dataloader = DataLoader::ImageFolderClassesWithPaths(valid_dataset, valid_batch_size, valid_shuffle, valid_workers);

    bool param_group = false;
    float lr = 5e-4;

    torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(lr).weight_decay(0.001));

    const unsigned step_size = 2;
    const double gamma = 0.9;

    auto scheduler = torch::optim::StepLR(optimizer, step_size, gamma);

    auto criterion = torch::nn::NLLLoss(torch::nn::NLLLossOptions().ignore_index(-100).reduction(torch::kMean));

    size_t epoch;
    size_t total_iter = dataloader.get_count_max();
    size_t start_epoch, total_epoch;
    start_epoch = 1;
    total_iter = dataloader.get_count_max();
    total_epoch = 20;
    bool first = true;
    bool valid = true;
    //bool test  = true;
    bool vobose = false;
    std::vector<float> train_loss_ave;
    std::vector<float> train_epochs;
    std::vector<float> valid_loss_ave;

    net->to(device);

    for (epoch = start_epoch; epoch <= total_epoch; epoch++) {
       	net->train();
       	std::cout << "--------------- Training --------------------\n";
       	first = true;
       	float loss_sum = 0.0;
       	size_t total_match = 0, total_counter = 0;
       	torch::Tensor responses;
       	while (dataloader(mini_batch)) {

       		image = std::get<0>(mini_batch).to(device);
       		label = std::get<1>(mini_batch).to(device);
       		size_t mini_batch_size = image.size(0);

       		if( first && vobose ) {
       			for(size_t i = 0; i < label.size(0); i++)
       				std::cout << label[i].item<int64_t>() << " ";
       			std::cout << "\n";
       			first = false;
       		}

       		image = std::get<0>(mini_batch).to(device);
       		label = std::get<1>(mini_batch).to(device);
       		output = net->forward(image);

       		auto out = torch::nn::functional::log_softmax(output, 1);
       		//std::cout << output.sizes() << "\n" << out.sizes() << std::endl;
       		loss = criterion(out, label); //torch::mse_loss(out, label)

       		optimizer.zero_grad();
       		loss.backward();
       		optimizer.step();

       		loss_sum += loss.item<float>();
        }

        train_loss_ave.push_back(loss_sum/total_iter);
        train_epochs.push_back(epoch*1.0);
       	std::cout << "epoch: " << epoch << "/"  << total_epoch << ", avg_loss: "
       			  << (loss_sum/total_iter) << std::endl;

       	// ---------------------------------
       	// validation
       	// ---------------------------------
       	if( valid ) {
       		std::cout << "--------------- validation --------------------\n";
       		torch::NoGradGuard nograd;

       		net->eval();
       		size_t iteration = 0;
       		float total_loss = 0.0;
       		total_match = 0;
       		total_counter = 0;
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

       			// evaluate_loss
       			output = net->forward(image);
       			auto out = torch::nn::functional::log_softmax(output, 1);
       			loss = criterion(out, label);

       			total_loss += loss.item<float>();
       			iteration++;
       		}

       		// Optimizer scheduler
       		scheduler.step();

       		// Calculate Average Loss
       		float ave_loss = total_loss / (float)iteration;
       		valid_loss_ave.push_back(ave_loss);
       		std::cout << "Valid loss: " << ave_loss << "\n\n";
       	}
    }

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::legend();
	matplot::hold(ax1, true);
	matplot::plot(ax1, train_epochs, train_loss_ave, "b")->line_width(2)
					.display_name("Train loss");
	matplot::plot(ax1, train_epochs, valid_loss_ave, "m-.")->line_width(2)
						.display_name("Valid loss");
	matplot::hold(ax1, false);
   	matplot::xlabel("epoch");
   	matplot::show();

	std::cout << "Done!\n";
}



