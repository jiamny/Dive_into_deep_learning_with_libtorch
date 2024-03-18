#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../utils/ch_10_util.h"
#include "../utils.h"
#include "../TempHelpFunctions.hpp"
#include "../fashion.h"

#include "../utils/transforms.hpp"              // transforms_Compose
#include "../utils/datasets.hpp"                // datasets::ImageFolderClassesWithPaths
#include "../utils/dataloader.hpp"              // DataLoader::ImageFolderClassesWithPaths

#include <matplot/matplot.h>
using namespace matplot;

struct PatchEmbeddingImpl : public torch::nn::Module {
	int64_t num_patches;
	torch::nn::Conv2d conv{nullptr};

	PatchEmbeddingImpl(std::vector<int64_t> img_sz, std::vector<int64_t> patch_sz, int64_t num_hd=512, int64_t in_channels=3) {

		num_patches = static_cast<int64_t>(img_sz[0] / patch_sz[0]) *
					  static_cast<int64_t>(img_sz[1] / patch_sz[1]);

		//conv = torch::nn::LazyConv2d(num_hiddens, kernel_size=patch_size,stride=patch_size);
		conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, num_hd,
				{patch_sz[0], patch_sz[1]}).stride({patch_sz[0], patch_sz[1]}));

		register_module("conv", conv);
	}

	torch::Tensor forward(torch::Tensor X) {
		// Output shape: (batch size, no. of patches, no. of channels)
        return conv->forward(X).flatten(2).transpose(1, 2);
	}
};
TORCH_MODULE(PatchEmbedding);

// Vision Transformer Encoder
struct ViTMLPImpl : public torch::nn::Module {
	torch::nn::Linear dense1{nullptr}, dense2{nullptr};
	torch::nn::GELU gelu{nullptr};
	torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};
	ViTMLPImpl(int64_t mlp_inputs, int64_t mlp_num_hiddens, int64_t mlp_num_outputs, double dropp=0.5) {

        dense1 = torch::nn::Linear(torch::nn::LinearOptions(mlp_inputs, mlp_num_hiddens));
        gelu = torch::nn::GELU();
        dropout1 = torch::nn::Dropout(dropp);
        dense2 = torch::nn::Linear(torch::nn::LinearOptions(mlp_num_hiddens,mlp_num_outputs));
        dropout2 = torch::nn::Dropout(dropp);
        register_module("dense1", dense1);
        register_module("dense2", dense2);
        register_module("dropout1", dropout1);
        register_module("dropout2", dropout2);
        register_module("gelu", gelu);
	}

    torch::Tensor forward(torch::Tensor x) {
    	x = dense1->forward(x);
    	x = gelu->forward(x);
        return dropout2->forward(dense2->forward(dropout1->forward(x)));
    }
};
TORCH_MODULE(ViTMLP);


struct ViTBlockImpl : public torch::nn::Module {
	torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
	MultiHeadAttention attention{nullptr};
	ViTMLP mlp{nullptr};

	ViTBlockImpl(int64_t key_size, int64_t query_size, int64_t value_size, int64_t num_inputs,
			int64_t num_hiddens, std::vector<int64_t> norm_shape, int64_t mlp_num_hiddens,
			int64_t num_heads, double dropout, bool use_bias=false) {

        ln1 = torch::nn::LayerNorm(torch::nn::LayerNormOptions(norm_shape));
        attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads,
                                                dropout, use_bias);
        ln2 = torch::nn::LayerNorm(torch::nn::LayerNormOptions(norm_shape));
        mlp = ViTMLP(num_inputs, mlp_num_hiddens, num_hiddens, dropout);
        register_module("ln1", ln1);
        register_module("ln2", ln2);
        register_module("attention", attention);
        register_module("mlp", mlp);
	}

    torch::Tensor forward(torch::Tensor X) {
    	torch::Tensor valid_lens=torch::empty(0);
    	torch::Tensor t = ln1->forward(X);
        X = X + attention(t.clone(), t.clone(), t, valid_lens);
        return X + mlp->forward(ln2->forward(X));
    }
};
TORCH_MODULE(ViTBlock);


struct ViTImpl : public torch::nn::Module {
    //Vision Transformer.
	torch::Tensor cls_token, pos_embedding;
	PatchEmbedding patch_embedding{nullptr};
	int64_t num_steps = 0;
	torch::nn::Sequential blks{nullptr}, head{nullptr};
	torch::nn::Dropout dropout{nullptr};

	ViTImpl(int64_t key_size, int64_t query_size, int64_t value_size, int64_t num_inputs, torch::Device device,
			std::vector<int64_t> img_size, std::vector<int64_t> patch_size, int64_t num_hiddens,
			std::vector<int64_t> norm_shape, int64_t mlp_num_hiddens, int64_t num_heads, int64_t num_blks,
			double emb_dropout, double blk_dropout, bool use_bias=false, int64_t num_classes=10) {
        //self.save_hyperparameters()
        patch_embedding = PatchEmbedding(img_size, patch_size, num_hiddens, num_inputs);
        cls_token = torch::zeros({1, 1, num_hiddens}).to(device);
        num_steps = patch_embedding->num_patches + 1;  // Add the cls token

        // Positional embeddings are learnable
        pos_embedding = torch::randn({1, num_steps, num_hiddens}).to(device);
        dropout = torch::nn::Dropout(emb_dropout);
        blks = torch::nn::Sequential();

        for( int64_t i = 0; i < num_blks; i++) {
            blks->push_back(std::to_string(i), ViTBlock(
            		key_size, query_size, value_size, num_hiddens,
            		num_hiddens, norm_shape, mlp_num_hiddens,
            		num_heads, blk_dropout, use_bias));
        }
        head = torch::nn::Sequential(
        			torch::nn::LayerNorm(torch::nn::LayerNormOptions(norm_shape)),
                    torch::nn::Linear(torch::nn::LinearOptions(num_hiddens, num_classes)));

        register_module("patch_embedding", patch_embedding);
        register_module("dropout", dropout);
        register_module("blks", blks);
        register_module("head", head);
	}

    torch::Tensor forward(torch::Tensor X) {
        X = patch_embedding->forward(X);
        X = torch::cat({cls_token.expand({X.size(0), -1, -1}), X}, 1);
        X = dropout->forward(X.add(pos_embedding));
        for(auto& blk : *(blks.ptr())) {
        //    X = blk(X)
        	X = blks->forward(X);
        }

        return head->forward(X.index({Slice(), 0}));
    }
};
TORCH_MODULE(ViT);


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

	//-----------------------------------------------------------
	// Patch Embedding
	//-----------------------------------------------------------
	std::vector<int64_t> img_size = {96, 96};
	std::vector<int64_t> patch_size = {16, 16};
	int64_t num_hiddens = 512, batch_size = 4, in_channels = 3;

	auto patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens, in_channels);
	patch_emb->to(device);
	torch::Tensor X = torch::zeros({batch_size, in_channels, img_size[0], img_size[1]}).to(device);
	X = patch_emb->forward(X);
	std::cout << "X.shape: " << X.sizes() << '\n';

	int64_t ss = std::pow(static_cast<int64_t>(img_size[0] / patch_size[0]), 2);
	std::vector<int64_t> ref =  {batch_size, ss, num_hiddens};

	std::cout << "check_shape: " << check_shape(X, ref) << '\n';

	//-----------------------------------------------------------
	// Vision Transformer Encoder
	//-----------------------------------------------------------
	X = torch::ones({2, 100, 24}).to(device);
	std::vector<int64_t> norm_shape = {24};
	ViTBlock encoder_blk = ViTBlock(24, 24, 24, 24, 24, norm_shape, 48, 8, 0.5);
	encoder_blk->to(device);
	encoder_blk->eval();
	std::cout << "encoder_blk(X).shape: " << encoder_blk->forward(X).sizes() <<  " X: " <<  X.sizes() << '\n';
	std::cout << "check_shape: " << check_shape(encoder_blk->forward(X), X.sizes()) << '\n';


	//-----------------------------------------------------------
	// Putting It All Together
	//-----------------------------------------------------------
	norm_shape.clear();
	num_hiddens = 512;
	batch_size = 4;
	int64_t mlp_num_hiddens = 2048, num_heads = 8, num_blks = 2;
	double emb_dropout = 0.1, blk_dropout = 0.1, lr = 0.1;
	norm_shape.push_back(num_hiddens);

	int64_t key_size = 512, query_size = 512, value_size = 512, num_inputs = 3;

	X = torch::randn({1, 3, 96, 96}).to(device);

	ViT model = ViT(key_size, query_size, value_size, num_inputs, device, img_size, patch_size,
					num_hiddens, norm_shape, mlp_num_hiddens, num_heads, num_blks, emb_dropout,
					blk_dropout, false, 17);
	model->to(device);
	model->eval();
	auto tt = model->forward(X);
	std::cout << "model(X): " << tt.sizes() << '\n';

	//-----------------------------------------------------------
	// Training
	//-----------------------------------------------------------
	const std::string path = "./data/17_flowers_name.txt";
	const size_t class_num = 17;
	const size_t valid_batch_size = 1;
	std::vector<std::string> class_names = Set_Class_Names( path, class_num);
	constexpr bool train_shuffle = true;    // whether to shuffle the training dataset
	constexpr size_t train_workers = 2;  	// the number of workers to retrieve data from the training dataset
    constexpr bool valid_shuffle = true;    // whether to shuffle the validation dataset
    constexpr size_t valid_workers = 2;     // the number of workers to retrieve data from the validation dataset

    std::vector<transforms_Compose> transform {
        transforms_Resize(cv::Size(img_size[0], img_size[1]), cv::INTER_LINEAR),        // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        transforms_ToTensor(),                                                     // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
		transforms_Normalize(std::vector<float>{0.485, 0.456, 0.406}, std::vector<float>{0.229, 0.224, 0.225})  // Pixel Value Normalization for ImageNet
    };

	std::string dataroot = "./data/17_flowers/train";
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor image, label, output;
    datasets::ImageFolderClassesWithPaths dataset, valid_dataset, test_dataset;      		// dataset;
    DataLoader::ImageFolderClassesWithPaths train_loader, valid_loader; 	// dataloader;

    //Get Dataset

    dataset = datasets::ImageFolderClassesWithPaths(dataroot, transform, class_names);
    train_loader = DataLoader::ImageFolderClassesWithPaths(dataset, batch_size, /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);

	std::cout << "total training images : " << dataset.size() << std::endl;

    std::string valid_dataroot = "./data/17_flowers/valid";
    valid_dataset = datasets::ImageFolderClassesWithPaths(valid_dataroot, transform, class_names);
    valid_loader = DataLoader::ImageFolderClassesWithPaths(valid_dataset, valid_batch_size, /*shuffle_=*/valid_shuffle, /*num_workers_=*/valid_workers);

    std::cout << "total validation images : " << valid_dataset.size() << std::endl;

    ViT net = ViT(key_size, query_size, value_size, num_inputs, device, img_size, patch_size,
    					num_hiddens, norm_shape, mlp_num_hiddens, num_heads, num_blks, emb_dropout,
    					blk_dropout, false, class_names.size());
    net->to(device);

	int64_t num_epochs = 30;
	int64_t total_iter = train_loader.get_count_max();

	torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(1e-4).betas({0.5, 0.999}));
	auto criterion = torch::nn::NLLLoss(torch::nn::NLLLossOptions().ignore_index(-100).reduction(torch::kMean));

	std::vector<double> train_loss;
	std::vector<double> train_acc;
	std::vector<double> val_acc;
	std::vector<double> xx;

	for(int64_t epoch = 1; epoch <= num_epochs; epoch++) {
		net->train();
		std::cout << "--------------- Training --------------------\n";
		double epoch_loss = 0.0;
		int64_t total_match = 0;
		int64_t total_counter = 0;
		int64_t num_batch = 0;
		torch::Tensor responses;

		while(train_loader(mini_batch)) {
			image = std::get<0>(mini_batch).to(device);
			label = std::get<1>(mini_batch).to(device);
			size_t mini_batch_size = image.size(0);

			output = net->forward(image);

			auto out = torch::nn::functional::log_softmax(output, /*dim=*/1);
			auto l = criterion(out, label);

			responses = output.exp().argmax(/*dim=*/1);
			for (size_t i = 0; i < mini_batch_size; i++){
				int64_t response = responses[i].item<int64_t>();
				int64_t answer = label[i].item<int64_t>();

				total_counter++;
				if (response == answer) total_match++;
			}
			epoch_loss += l.item<float>();

			optimizer.zero_grad();
			l.backward();
			optimizer.step();

			num_batch++;
		}

	    auto sample_mean_loss = epoch_loss / num_batch;
	    auto tr_acc = static_cast<double>(total_match) / total_counter;

	    std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
	    					            << sample_mean_loss << ", Accuracy: " << tr_acc << '\n';

	    train_loss.push_back((sample_mean_loss*1.0));
	    train_acc.push_back(tr_acc);

		// ---------------------------------
		// validation
		// ---------------------------------
		std::cout << "--------------- validation --------------------\n";
		torch::NoGradGuard no_grad;

		total_match = 0;
		total_counter = 0;

		while(valid_loader(mini_batch)) {
			image = std::get<0>(mini_batch).to(device);
			label = std::get<1>(mini_batch).to(device);
			size_t mini_batch_size = image.size(0);

			auto output = net->forward(image);
			responses = output.exp().argmax(/*dim=*/1);
			for (size_t i = 0; i < mini_batch_size; i++){
				int64_t response = responses[i].item<int64_t>();
				int64_t answer = label[i].item<int64_t>();

				total_counter++;
				if (response == answer) total_match++;
			}
		}

		std::cout << "Valid finished!\n";

		auto val_accuracy = static_cast<double>(total_match) / total_counter;
		val_acc.push_back(val_accuracy);

		std::cout << "Validation - accuracy: " << val_accuracy << '\n';
		xx.push_back((epoch + 1));
	}

	auto F = figure(true);
	F->size(1200, 500);
	F->add_axes(false);
	F->reactive_mode(false);

    auto ax1 = subplot(1, 2, 0);
    ax1->xlabel("epoch");
    ax1->ylabel("loss");
    ax1->title("VisionTransformers train loss");

    plot(xx, train_loss, "-o")->line_width(2).display_name("train loss");
    legend({});

   	auto ax2 = subplot(1, 2, 1);
   	plot(xx, train_acc, "m--")->line_width(2).display_name("train acc");
   	hold(on);
   	plot(xx, val_acc, "r-.")->line_width(2).display_name("val acc");
   	hold(on);
    legend({});
   	ax2->xlabel("epoch");
   	ax2->ylabel("acc");
   	ax2->title("VisionTransformers train & val acc");
   	hold( off);
   	F->draw();
   	show();

	std::cout << "Done!\n";
}

