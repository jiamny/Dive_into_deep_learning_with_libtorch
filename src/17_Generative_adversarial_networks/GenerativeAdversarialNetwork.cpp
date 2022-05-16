#include <unistd.h>
#include <iomanip>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <tuple>
#include <random>
#include "../utils.h"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;


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
	int data_size = 1000;

	// -------------------------------------
	// Generate Some "Real" Data
	// -------------------------------------
	auto X = torch::normal(0.0, 1.0, {data_size, 2});
	auto A = torch::tensor({{1.0, 2.0}, {-0.1, 0.5}});
	auto b = torch::tensor({1.0, 2.0});
	auto data = torch::matmul(X, A) + b;

	std::vector<float> x, y;
	for( int64_t i = 0; i < 100; i++) {
		x.push_back(data[i][0].item<float>());
		y.push_back(data[i][1].item<float>());
	}

	plt::figure_size(450, 400);
	plt::scatter(x, y, 10.0);
	plt::show();
	plt::close();

	std::cout << "The covariance matrix is\n" << torch::matmul(A.t(), A) << '\n';

	int64_t batch_size = 8;
	//data_iter = d2l.load_array((data,), batch_size)

	data.to(device);
	torch::Tensor fake_label = torch::zeros({data_size});
	auto dataset = LRdataset(std::make_pair(data, fake_label))
					.map(torch::data::transforms::Stack<>());

	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		        std::move(dataset), batch_size);

	// -------------------------------------
	// Generator
	// -------------------------------------
	auto net_G = torch::nn::Sequential(torch::nn::Linear(2, 2));

	// -------------------------------------
	// Discriminator
	// -------------------------------------
	auto net_D = torch::nn::Sequential(
			torch::nn::Linear(2, 5), torch::nn::Tanh(),
			torch::nn::Linear(5, 3), torch::nn::Tanh(),
			torch::nn::Linear(3, 1));

	// -------------------------------------
	// Training
	// -------------------------------------
	float lr_D = 0.05, lr_G = 0.005;
	int latent_dim = 2, num_epochs = 20;
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

	auto trainer_D = torch::optim::Adam(net_D->parameters(), lr_D);

	auto trainer_G = torch::optim::Adam(net_G->parameters(), lr_G);

	std::vector<torch::Tensor> fake_X;
	std::vector<float> d_loss, g_loss, ept;

	for( int epoch = 0; epoch < num_epochs; epoch++ ) {
		//Train one epoch
		float loss_d_tot =0.0, loss_g_tot = 0.0;
		int tot_size = 0;
		for (auto &batch : *data_loader) {
			auto X = batch.data;
			batch_size = X.size(0);
			// Visualize generated examples
			auto Z = torch::normal(0, 1, {batch_size, latent_dim});

			auto loss_D = update_D(X, Z, net_D, net_G, loss, trainer_D);
			auto loss_G = update_G(Z, net_D, net_G, loss, trainer_G);

			if( epoch == (num_epochs - 1) ) {
				auto fkt = net_G->forward(Z).detach();
				fake_X.push_back(fkt);
			}
			//The losses
	        loss_d_tot += loss_D.data().item<float>();
	        loss_g_tot += loss_G.data().item<float>();
	        tot_size  += batch_size;

		}

		std::cout << "epoch: " << (epoch + 1) << ", loss_D: " << (loss_d_tot/tot_size)
				  << ", loss_G: " << (loss_g_tot/tot_size) <<'\n';
		d_loss.push_back(loss_d_tot/tot_size);
		g_loss.push_back(loss_g_tot/tot_size);
		ept.push_back(epoch*1.0);
	}

	auto fkdata = torch::cat(fake_X, 0);
	std::cout << "fkdata\n" << fkdata.sizes() << '\n';

	std::vector<float> px, py;
	for( int64_t i = 0; i < 100; i++) {
		px.push_back(fkdata[i][0].item<float>());
		py.push_back(fkdata[i][1].item<float>());
	}

	plt::figure_size(450, 800);
	plt::subplot2grid(2, 1, 0, 0, 1, 1);
	plt::named_plot("discriminator", ept, d_loss, "b-");
	plt::named_plot("generated", ept, g_loss, "m-.");
	plt::xlabel("epoch");
	plt::ylabel("loss");
	plt::legend();

	plt::subplot2grid(2, 1, 1, 0, 1, 1);
	plt::scatter(x, y, 10.0, {{"color", "blue"}, {"label", "real"}});
	plt::scatter(px, py, 10.0, {{"color", "orange"}, {"label", "generated"}});
	plt::legend();
	plt::show();
	plt::close();

	std::cout << "Done!\n";
}




