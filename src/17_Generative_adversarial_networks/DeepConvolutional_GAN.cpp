#include <unistd.h>
#include <iomanip>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

//#include "../utils.h"
#include "../TempHelpFunctions.hpp" // range()
#include "../utils/ch_13_util.h"

#include "../utils/transforms.hpp"              // transforms_Compose
#include "../utils/datasets.hpp"                // datasets::ImageFolderClassesWithPaths
#include "../utils/dataloader.hpp"              // DataLoader::ImageFolderClassesWithPaths

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

using torch::indexing::Slice;
using torch::indexing::None;

struct G_blockImpl : public torch::nn::Module {
	torch::nn::ConvTranspose2d conv2d_trans{nullptr};
	torch::nn::BatchNorm2d batch_norm{nullptr};
	torch::nn::ReLU activation{nullptr};

	G_blockImpl(int out_channels, int in_channels=3, int kernel_size=4, int strides=2, int padding=1) {
        conv2d_trans = torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(in_channels, out_channels,
                                kernel_size).stride(strides).padding(padding).bias(false));

        batch_norm = torch::nn::BatchNorm2d(out_channels);
        activation = torch::nn::ReLU();

        register_module("conv2d_trans", conv2d_trans);
        register_module("batch_norm", batch_norm);
        register_module("activation", activation);
	}

    torch::Tensor forward(torch::Tensor X) {
        return activation->forward(batch_norm->forward(conv2d_trans->forward(X)));
    }
};
TORCH_MODULE(G_block);

struct D_blockImpl : public torch::nn::Module {
	torch::nn::LeakyReLU activation{nullptr};
	torch::nn::BatchNorm2d batch_norm{nullptr};
	torch::nn::Conv2d conv2d{nullptr};

	D_blockImpl(int out_channels, int in_channels=3, int kernel_size=4, int strides=2,
    	                int padding=1, double alpha=0.2) {
    	        //super(D_block, self).__init__(**kwargs)
		conv2d = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
    	                                .stride(strides).padding(padding).bias(false));

		batch_norm = torch::nn::BatchNorm2d(out_channels);
    	activation = torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(alpha).inplace(true));

    	register_module("conv2d", conv2d);
    	register_module("batch_norm", batch_norm);
    	register_module("activation", activation);
	}

	torch::Tensor forward(torch::Tensor X) {
    	        return activation->forward(batch_norm->forward(conv2d->forward(X)));
	}
};
TORCH_MODULE(D_block);



torch::Tensor update_D(torch::Tensor X, torch::Tensor Z, torch::nn::Sequential& net_D, torch::nn::Sequential& net_G,
									torch::nn::BCEWithLogitsLoss& loss, torch::optim::Adam& trainer_D) {
    //"""Update discriminator."""
    int batch_size = X.size(0);
    auto ones = torch::ones({batch_size,}).to(X.device());
    auto zeros = torch::zeros({batch_size,}).to(X.device());
    trainer_D.zero_grad();
    auto real_Y = net_D->forward(X);
    auto fake_X = net_G->forward(Z);
    // Do not need to compute gradient for `net_G`, detach it from
    // computing gradients.
    auto fake_Y = net_D->forward(fake_X.detach());
    torch::Tensor loss_D = (loss(real_Y, ones.reshape(real_Y.sizes())) +
              loss(fake_Y, zeros.reshape(fake_Y.sizes()))) / 2;
    loss_D.backward();
    trainer_D.step();
    return loss_D;
}

torch::Tensor  update_G(torch::Tensor Z, torch::nn::Sequential& net_D, torch::nn::Sequential& net_G,
									torch::nn::BCEWithLogitsLoss& loss, torch::optim::Adam& trainer_G) {
    //"""Update generator."""
    int batch_size = Z.size(0);
    auto ones = torch::ones({batch_size,}).to(Z.device());
    trainer_G.zero_grad();
    // We could reuse `fake_X` from `update_D` to save computation
    auto fake_X = net_G->forward(Z);
    // Recomputing `fake_Y` is needed since `net_D` is changed
    auto fake_Y = net_D->forward(fake_X);
    torch::Tensor loss_G = loss(fake_Y, ones.reshape(fake_Y.sizes()));
    loss_G.backward();
    trainer_G.step();
    return loss_G;
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	// -----------------------------------
	// The Pokemon Dataset
	// -----------------------------------
	int64_t batch_size = 256;
	bool train_shuffle = true;
	int train_workers = 2;
	std::string dataroot = "./data/pokemon";
	std::vector<std::string> classes;
	int num_class = 721;
	for(int i = 0; i < num_class; i++)
		classes.push_back(std::to_string(i+1));

	std::vector<float> mean_ = {0.5, 0.5, 0.5};
	std::vector<float> std_  = {0.5, 0.5, 0.5};

	// Set Transforms
	std::vector<transforms_Compose> transform {
	        transforms_Resize(cv::Size(64, 64), cv::INTER_LINEAR),    // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
	        transforms_ToTensor(),                                    // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
			transforms_Normalize(mean_, std_)  						  // Pixel Value Normalization for ImageNet
	};

	datasets::ImageFolderClassesWithPaths dataset;
	std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> mini_batch;
	DataLoader::ImageFolderClassesWithPaths dataloader;

	// Get Dataset
	dataset = datasets::ImageFolderClassesWithPaths(dataroot, transform, classes);
	dataloader = DataLoader::ImageFolderClassesWithPaths(dataset, batch_size, train_shuffle, train_workers);

	std::cout << "total training images : " << dataset.size() << std::endl;

	dataloader(mini_batch);
	torch::Tensor images = std::get<0>(mini_batch).to(device);
	auto label = std::get<1>(mini_batch).to(device);
	torch::Tensor imgT = images[0].clone().squeeze();

	std::cout << imgT.max() << " " << label[0] << '\n';
	std::cout << "imgT sizes: " << imgT.sizes() << '\n';

	plt::figure_size(900, 800);
	int img_c = 10;
	for(int r = 0; r < 4; r++) {
		for(int c = 0; c < 4; c++) {
			plt::subplot2grid(4, 4, r, c, 1, 1);

			torch::Tensor imgT = images[img_c].clone().squeeze();
			imgT = deNormalizeTensor(imgT, mean_, std_);

			std::vector<uint8_t> z = tensorToMatrix4Matplotlib(imgT.clone());
			const uchar* zptr = &(z[0]);
			plt::imshow(zptr, imgT.size(1), imgT.size(2), imgT.size(0));
			img_c += 5;
		}
	}
	plt::show();
	plt::close();

	// -------------------------------------------
	auto x = torch::zeros({2, 3, 16, 16});
	auto g_blk = G_block(20);
	g_blk->to(device);
	std::cout << g_blk->forward(x).sizes() << '\n';

	// If changing the transposed convolution layer to a 4×4 kernel, 1×1 strides and zero padding.
	// With a input size of 1×1, the output will have its width and height increased by 3 respectively.

	x = torch::zeros({2, 3, 1, 1});
	g_blk = G_block(20, 3, 4, 1, 0);
	g_blk->to(device);
	std::cout << g_blk->forward(x).sizes() << '\n';

	/*
	 * The generator consists of four basic blocks that increase input's both width and height from 1 to 32.
	 * At the same time, it first projects the latent variable into 64×8 channels, and then halve the channels each time.
	 * At last, a transposed convolution layer is used to generate the output. It further doubles the width and height
	 * to match the desired 64×64 shape, and reduces the channel size to 3. The tanh activation function is applied
	 * to project output values into the (−1,1) range.
	 */

	int n_G = 64;
	auto net_G = torch::nn::Sequential(
	    G_block(n_G*8, 100, 4, 							//in_channels=100, out_channels=n_G*8,
	            1, 0),                  				// Output: (64 * 8, 4, 4)
	    G_block(n_G*4, n_G*8), 							// Output: (64 * 4, 8, 8) in_channels=n_G*8, out_channels=n_G*4
	    G_block(n_G*2, n_G*4), 							// Output: (64 * 2, 16, 16) in_channels=n_G*4, out_channels=n_G*2
	    G_block(n_G, n_G*2),   							// Output: (64, 32, 32) in_channels=n_G*2, out_channels=n_G
	    torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(n_G, 3,
                4).stride(2).padding(1).bias(false)),
	    torch::nn::Tanh()); // Output: (3, 64, 64)

	x = torch::zeros({1, 100, 1, 1});
	net_G->to(device);
	std::cout << net_G->forward(x).sizes() << '\n';

	// --------------------------------------------
	// Discriminator
	// --------------------------------------------

	std::vector<float> alphas = {0., .2, .4, .6, .8, 1.};
	x = torch::arange(-2, 1, 0.1);

	std::vector<float> xx(x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
	std::vector<std::string> formats = {"b-", "m--", "g-.", "r:", "c--", "y-."};

	plt::figure_size(700, 550);
	for( int a = 0; a < alphas.size(); a++ ) {
		auto y = torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(alphas[a]))->forward(x);
		std::vector<float> z(y.data_ptr<float>(), y.data_ptr<float>() + y.numel());
		plt::named_plot(to_string_with_precision(alphas[a], 2).c_str(), xx, z, formats[a].c_str());
	}
	plt::xlabel("x");
	plt::ylabel("y");
	plt::legend();
	plt::show();
	plt::close();

	x = torch::zeros({2, 3, 16, 16});
	auto d_blk = D_block(20);
	d_blk->to(device);
	std::cout << d_blk->forward(x).sizes() << '\n';

	// The discriminator is a mirror of the generator.
	int n_D = 64;

	auto net_D = torch::nn::Sequential(
	    D_block(n_D),  															// Output: (64, 32, 32)
	    D_block(n_D*2, n_D),  													// Output: (64 * 2, 16, 16)
	    D_block(n_D*4, n_D*2),  												// Output: (64 * 4, 8, 8)
	    D_block(n_D*8, n_D*4),  												// Output: (64 * 8, 4, 4)
	    torch::nn::Conv2d(torch::nn::Conv2dOptions(n_D*8, 1, 4).bias(false)));  // Output: (1, 1, 1)

	// It uses a convolution layer with output channel 1 as the last layer to obtain a single prediction value.
	x = torch::zeros({1, 3, 64, 64});
	net_D->to(device);
	std::cout << net_D->forward(x).sizes() << '\n';

	// -----------------------------------------
	// Training
	// -----------------------------------------

	dataloader = DataLoader::ImageFolderClassesWithPaths(dataset, batch_size, train_shuffle, train_workers);

	int latent_dim = 100, num_epochs = 20;
	float lr = 0.005;

	auto loss = torch::nn::BCEWithLogitsLoss(torch::nn::BCEWithLogitsLossOptions().reduction(torch::kSum));

	for (auto& module : net_G->modules(true) ) {  //modules(include_self=false))
		if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
			//M->weight.data().normal_(0.0, 0.02);
			torch::nn::init::normal_(M->weight, 0.0, 0.02);
		}
	}

	for (auto& module : net_D->modules(true) ) {  //modules(include_self=false))
		if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
			torch::nn::init::normal_(M->weight, 0.0, 0.02);
		}
	}


	net_D->to(device);
	net_G->to(device);

	auto trainer_D = torch::optim::Adam(net_D->parameters(), torch::optim::AdamOptions(0.005).betas({0.5, 0.999}));
	auto trainer_G = torch::optim::Adam(net_G->parameters(), torch::optim::AdamOptions(0.005).betas({0.5, 0.999}));

	std::vector<torch::Tensor> fake_X;
	std::vector<float> d_loss, g_loss, ept;
	std::vector<torch::Tensor> imgs;

	net_D->train(true);
	net_G->train(true);
	int epoch = 0;
	for( epoch = 0; epoch < num_epochs; epoch++ ) {

		float loss_d_tot =0.0, loss_g_tot = 0.0;
		int tot_size = 0;

		while (dataloader(mini_batch)) {
			auto X = std::get<0>(mini_batch).to(device);
			auto b_size = X.size(0);
			auto Z = torch::normal(0, 1, {b_size, latent_dim, 1, 1});
			Z.to(device);

			auto loss_D = update_D(X, Z, net_D, net_G, loss, trainer_D);
			auto loss_G = update_G(Z, net_D, net_G, loss, trainer_G);

			//The losses
	        loss_d_tot += loss_D.data().item<float>();
	        loss_g_tot += loss_G.data().item<float>();
	        tot_size  += batch_size;
		}

		if( (epoch + 1) == num_epochs ) {
			// Show generated examples
	        auto Z = torch::normal(0, 1, {21, latent_dim, 1, 1}).to(device);
	        // Normalize the synthetic data to N(0, 1)
	        //torch::Tensor fake_x = net_G->forward(Z).permute({0, 2, 3, 1}) / 2 + 0.5;
	        torch::Tensor fake_x = net_G->forward(Z) / 2 + 0.5;

	        int L = static_cast<int>(fake_x.size(0) / 7.0);

	        for(int i = 0; i < L; i++) {
	        	for(int j = 0; j < 7; j++) {
	        		imgs.push_back( fake_x[i * 7 + j].cpu().detach().clone() );
	        	}
	        }
		}

        std::cout << "epoch: " << (epoch + 1) << ", loss_D: " << (loss_d_tot/tot_size)
        				  << ", loss_G: " << (loss_g_tot/tot_size) <<'\n';
        d_loss.push_back(loss_d_tot/tot_size);
        g_loss.push_back(loss_g_tot/tot_size);
        ept.push_back(epoch*1.0);
	}

	plt::figure_size(500, 450);
	//plt::subplot2grid(2, 1, 0, 0, 1, 1);
	plt::named_plot("discriminator", ept, d_loss, "b-");
	plt::named_plot("generated", ept, g_loss, "m-.");
	plt::xlabel("epoch");
	plt::ylabel("loss");
	plt::legend();
	plt::show();
	plt::close();

	plt::figure_size(1000, 700);
	for(auto& img : imgs) {
		for( int r = 0; r < 3; r++) {
			for( int c = 0; c < 7; c++ ) {
				plt::subplot2grid(3, 7, r, c, 1, 1);
				torch::Tensor imgT = img.squeeze();
				imgT = deNormalizeTensor(imgT, mean_, std_);

				std::vector<uint8_t> z = tensorToMatrix4Matplotlib(imgT.clone());
				const uchar* zptr = &(z[0]);
				plt::imshow(zptr, imgT.size(1), imgT.size(2), imgT.size(0));
			}
		}
	}
	plt::show();
	plt::close();

	std::cout << "Done!\n";
}




