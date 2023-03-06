#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "../utils/transforms.hpp"              // transforms_Compose
#include "../utils/datasets.hpp"                // datasets::ImageFolderClassesWithPaths
#include "../utils/dataloader.hpp"              // DataLoader::ImageFolderClassesWithPaths

#include <matplot/matplot.h>
using namespace matplot;


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

        register_module("conv1", conv1);
        register_module("conv2", conv2);

        if( use_1x1conv ) {
            conv3 = torch::nn::Conv2d(Options(input_channels, num_channels, 1).stride(strides));
            register_module("conv3", conv3);
        }

        bn1 = torch::nn::BatchNorm2d(num_channels);
        bn2 = torch::nn::BatchNorm2d(num_channels);
        register_module("bn1", bn1);
        register_module("bn2", bn2);
    }

    torch::Tensor forward(torch::Tensor X) {
        auto Y = torch::relu(bn1->forward(conv1->forward(X)));
        Y = bn2->forward(conv2->forward(Y));
        if( ! conv3.is_empty() )
            X = conv3->forward(X);
        Y += X;
        return torch::relu(Y);
    }
};


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
struct  ResNetImpl : public torch::nn::Module {
	std::vector<Residual> b2, b3, b4, b5;
	torch::nn::Sequential b1{nullptr};
	torch::nn::Linear linear{nullptr};
	torch::nn::Sequential classifier{nullptr};

	ResNetImpl(int64_t num_classes) {
		// The first module uses a 64-channel 7×7 convolutional layer.
		b1 = torch::nn::Sequential(torch::nn::Conv2d(Options(3, 64, 7).stride(2).padding(3)),
											torch::nn::BatchNorm2d(64),
											torch::nn::ReLU(),
											torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));

		register_module("b1", b1);
		// Then, we add all the modules to ResNet. Here, two residual blocks are used for each module.
		b2 = resnet_block(64, 64, 2, true);
		b3 = resnet_block(64, 128, 2, false);
		b4 = resnet_block(128, 256, 2, false);
		b5 = resnet_block(256, 512, 2, false);
		classifier = torch::nn::Sequential(torch::nn::Linear(512*6*6, num_classes));

		register_module("classifier", classifier);

		// weights_init
		for (auto& module : modules(/*include_self=*/false)) {
			/*
		    if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get()))
		      torch::nn::init::kaiming_normal_(
		          M->weight,
		          0, // a = 0
		          torch::kFanOut,
		          torch::kReLU);
		    else if (auto M = dynamic_cast<torch::nn::BatchNorm2dImpl*>(module.get())) {
		      torch::nn::init::constant_(M->weight, 1);
		      torch::nn::init::constant_(M->bias, 0);
		    }
		    */
		    if(auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
		           auto p = M->named_parameters(false);
		           auto w = p.find("weight");
		           auto b = p.find("bias");
		           //if (w != nullptr) torch::nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.01);
		           if (w != nullptr) torch::nn::init::kaiming_normal_(*w, /*a=*/0.0, torch::kFanOut, torch::kReLU);
		           if (b != nullptr) torch::nn::init::constant_(*b, /*bias=*/0.0);

		    } else if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())){
		           auto p = M->named_parameters(false);
		           auto w = p.find("weight");
		           auto b = p.find("bias");
		           if (w != nullptr) torch::nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.01);
		           if (b != nullptr) torch::nn::init::constant_(*b, /*bias=*/0.0);

		    } else if (auto M = dynamic_cast<torch::nn::BatchNorm2dImpl*>(module.get())){
		           auto p = M->named_parameters(false);
		           auto w = p.find("weight");
		           auto b = p.find("bias");
		           if (w != nullptr) torch::nn::init::constant_(*w, /*weight=*/1.0);
		           if (b != nullptr) torch::nn::init::constant_(*b, /*bias=*/0.0);
		    }
		}
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

		//torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})),
		//torch::nn::Flatten()
		x = torch::adaptive_avg_pool2d(x, {6, 6});
		x = x.view({x.size(0), -1});

		return classifier->forward(x);
	}
};

TORCH_MODULE(ResNet);

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	torch::Device device(torch::kCPU);
//	auto cuda_available = torch::cuda::is_available();
//	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
//	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);
	size_t img_size = 40;
	size_t batch_size = 32;
	const size_t class_num = 10;
	const size_t valid_batch_size = 1;
	std::vector<std::string> class_names = {"airplane", "automobile", "bird", "cat", "deer",
											"dog", "frog", "horse", "ship", "truck"};
	constexpr bool train_shuffle = true;    // whether to shuffle the training dataset
	constexpr size_t train_workers = 2;  	// the number of workers to retrieve data from the training dataset
	constexpr bool valid_shuffle = true;    // whether to shuffle the validation dataset
	constexpr size_t valid_workers = 2;     // the number of workers to retrieve data from the validation dataset


    // (4) Set Transforms
    std::vector<transforms_Compose> transform {
        transforms_Resize(cv::Size(img_size, img_size), cv::INTER_LINEAR),        // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        transforms_ToTensor(),                                                     // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
		transforms_Normalize(std::vector<float>{0.4914, 0.4822, 0.4465}, std::vector<float>{0.2023, 0.1994, 0.2010})  // Pixel Value Normalization for ImageNet
    };

	std::string dataroot = "./data/kaggle_cifar10_tiny/train_valid_test/train";
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor loss, image, label, output;
    datasets::ImageFolderClassesWithPaths dataset, valid_dataset, test_dataset;      		// dataset;
    DataLoader::ImageFolderClassesWithPaths dataloader, valid_dataloader, test_dataloader; 	// dataloader;

    // Get Dataset
    dataset = datasets::ImageFolderClassesWithPaths(dataroot, transform, class_names);
    dataloader = DataLoader::ImageFolderClassesWithPaths(dataset, batch_size, /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);

	std::cout << "total training images : " << dataset.size() << std::endl;

    std::string valid_dataroot = "./data/kaggle_cifar10_tiny/train_valid_test/valid";
    valid_dataset = datasets::ImageFolderClassesWithPaths(valid_dataroot, transform, class_names);
    valid_dataloader = DataLoader::ImageFolderClassesWithPaths(valid_dataset, valid_batch_size, /*shuffle_=*/valid_shuffle, /*num_workers_=*/valid_workers);

    std::cout << "total validation images : " << valid_dataset.size() << std::endl;

    bool valid = true;
    bool test  = true;
    bool vobose = false;
    float lr = 2e-4, wd = 5e-4;

    // Deep Convolutional Neural Networks (AlexNet)
    //ResNetBB tnet = ResNet18(class_num);
    ResNet tnet(class_num);

    auto X = torch::randn({1, 3, 40, 40});
    std::cout << tnet->forward(X).sizes() << std::endl;

    ResNet net(class_num);
    const unsigned step_size = 4;
	const double gamma = 0.9;

    torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(lr).momentum(0.9).weight_decay(wd));
    auto scheduler = torch::optim::StepLR(optimizer, step_size, gamma);

    auto criterion = torch::nn::NLLLoss(torch::nn::NLLLossOptions().ignore_index(-100).reduction(torch::kMean));

    net->to(device);

    size_t epoch;
    size_t total_iter = dataloader.get_count_max();
    size_t start_epoch, total_epoch;
    start_epoch = 1;
    total_iter = dataloader.get_count_max();
    total_epoch = 50;
    bool first = true;
    std::vector<float> train_loss_ave;
    std::vector<float> train_epochs;
    std::vector<float> valid_accs;
    std::vector<float> train_accs;

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

    		responses = output.exp().argmax(1);

    		for (size_t i = 0; i < mini_batch_size; i++){
    		    int64_t response = responses[i].item<int64_t>();
    		    int64_t answer = label[i].item<int64_t>();

    		    total_counter++;
    		    if (response == answer) total_match++;
    		}

    		optimizer.zero_grad();
    		loss.backward();
    		optimizer.step();

    		loss_sum += loss.item<float>();
    	}

    	train_loss_ave.push_back(loss_sum/total_iter);
    	train_epochs.push_back(epoch*1.0);
    	std::cout << "epoch: " << epoch << "/"  << total_epoch << ", avg_loss: "
    			  << (loss_sum/total_iter) << ", acc: " << ((float)total_match / (float)total_counter) << std::endl;

    	train_accs.push_back( (float)total_match / (float)total_counter );

    	// ---------------------------------
    	// validation
    	// ---------------------------------
    	if( valid ) {
    		std::cout << "--------------- validation --------------------\n";
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

    			output = net->forward(image);
    			auto out = torch::nn::functional::log_softmax(output, 1);
    			loss = criterion(out, label);

    			responses = output.exp().argmax(1);
    			for (size_t i = 0; i < mini_batch_size; i++){
    				int64_t response = responses[i].item<int64_t>();
    				int64_t answer = label[i].item<int64_t>();

    				total_counter++;
    				if (response == answer) total_match++;
    			}
    			total_loss += loss.item<float>();
    			iteration++;
    		}

    		// Optimizer scheduler
    		scheduler.step();

    		// Calculate Average Loss
    		float ave_loss = total_loss / (float)iteration;

    		// Calculate Accuracy
    		float total_accuracy = (float)total_match / (float)total_counter;
    		std::cout << "Validation accuracy: " << total_accuracy << "\n\n";
    		valid_accs.push_back(total_accuracy);
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
	matplot::plot(ax1, train_epochs, train_accs, "m--")->line_width(2)
				.display_name("Train acc");
	matplot::plot(ax1, train_epochs, valid_accs, "g-.")->line_width(2)
				.display_name("Valid acc");
	matplot::hold(ax1, false);
	matplot::xlabel("epoch");
	matplot::show();

	std::cout << "Done!\n";
}

