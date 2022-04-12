#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <chrono>

#include "../csvloader.h"
#include "../utils.h"
#include "../utils/ch_11_util.h"

using namespace std::chrono;

using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;

std::pair<std::vector<float>, std::vector<float>> train_sgd(float lr, int64_t num_epochs,
												  torch::Tensor t_data, torch::Tensor t_label, int64_t batch_size ) {
	// Initialization
    torch::Tensor w = torch::normal(0.0, 0.01, {t_data.size(1), 1}).requires_grad_(true);
    torch::Tensor b = torch::zeros({1}, torch::requires_grad(true));

    torch::Tensor loss, t;

    auto start = high_resolution_clock::now();
    std::vector<float> epochs, losses;

	for( int64_t  epoch = 0; epoch < num_epochs; epoch++ ) {
		std::list<std::pair<torch::Tensor, torch::Tensor>> data_iter = get_data_ch11(t_data, t_label, batch_size);

		float t_loss = 0.0;
		int64_t b_cnt = 0;
		for (auto &batch : data_iter) {

			auto X = batch.first;
			auto y = batch.second;

	    	t = linreg(X, w, b);
	        loss = squared_loss(t, y).mean();

	        // Compute gradient on `l` with respect to [`w`, `b`]
	        loss.backward();

	        sgd(w, b, lr, X.size(0));  // Update parameters using their gradient

	        t_loss += loss.item<float>();
	        b_cnt++;
	    }

		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		std::cout << "Epoch: " << (epoch + 1) << ", loss: " << (t_loss/b_cnt)  << ", duration: " << (duration.count() / 1e6) << " sec.\n";
		epochs.push_back(epoch*1.0);
		losses.push_back((t_loss/b_cnt));
	}

	plt::figure_size(800, 600);
	plt::named_plot("train", epochs, losses, "b");
	plt::xlabel("epoch");
	plt::ylabel("loss");
	plt::legend();
	plt::show();
	plt::close();

	return {epochs, losses};
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	auto A = torch::zeros({256, 256});
	auto B = torch::randn({256, 256});
	auto C = torch::randn({256, 256});

	// Element-wise assignment simply iterates over all rows and columns of 𝐁 and 𝐂 respectively to assign the value to 𝐀.
	// Compute A = BC one element at a time
	auto start = high_resolution_clock::now();

	for(int i = 0; i < 256; i++) {
    	for(int j = 0; j < 256; j++ ) {
    		A[i, j] = torch::dot(B.index({i, Slice()}), C.index({Slice(), j}));
    	}
	}

	// Get ending timepoint
	auto stop = high_resolution_clock::now();

	// use duration cast method
	auto duration = duration_cast<microseconds>(stop - start);

	std::cout << (duration.count() / 1e6) << " sec.\n";

	// A faster strategy is to perform column-wise assignment.
	start = high_resolution_clock::now();

	for(int j = 0; j < 256; j++ ) {
		A.index_put_({Slice(), j}, torch::mv(B, C.index({Slice(), j}))); // A[:, j]
	}

	stop = high_resolution_clock::now();
	duration = duration_cast<microseconds>(stop - start);
	std::cout << (duration.count() / 1e6) << " sec.\n";

	// Last, the most effective manner is to perform the entire operation in one block.
	// Let us see what the respective speed of the operations is.
	start = high_resolution_clock::now();
	A = torch::mm(B, C);
	stop = high_resolution_clock::now();
	duration = duration_cast<microseconds>(stop - start);
	std::cout << (duration.count() / 1e6) << " sec.\n";
/*
	start = high_resolution_clock::now();
	for(int j = 0; j < 256; j = j+64) {
		// A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
		A.index_put_({Slice(), Slice(j, j+64)}, torch::mv(B, C.index({Slice(), Slice(j, j+64)})));
	}
	stop = high_resolution_clock::now();
	duration = duration_cast<microseconds>(stop - start);
	std::cout << (duration.count() / 1e6) << " sec.\n";
*/
	// Load train CSV data
	std::ifstream file;
	std::string path = "./data/airfoil_self_noise.csv";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		file.close();
		return -1;
	}

	int num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	std::cout << "records in file = " << num_records << '\n';

	// set file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);

	std::pair<std::vector<float>, std::vector<float>> data_label = process_data(file, true, true);

	int64_t batch_size = 1500;

	std::vector<float> datas;
	std::vector<float> labels;

	datas  = data_label.first;
	labels = data_label.second;
	std::cout << datas.size() << std::endl;


	// Convert vectors to a tensor
	auto t_label = torch::from_blob(labels.data(), {int(labels.size()), 1}).clone();
	auto t_data = torch::from_blob(datas.data(), {int(labels.size()), int(datas.size()/labels.size())}).clone();
/*
	std::cout << "size = " << t_label.data().sizes() << "\n";
	std::cout << "sizes = " << t_data.data().sizes() << "\n";
	std::cout << "t_data[0:20,:]\n" << t_data.index({Slice(0, 20), Slice()}) << "\n";
	std::cout << "t_label[0:20]\n" << t_label.index({Slice(0, 20)}) << "\n";
*/
	int64_t num_epochs = 10;
	float lr = 1.0;

	std::pair<std::vector<float>, std::vector<float>> gd, sgd_res, mini1_res, mini2_res;

	gd = train_sgd(lr, num_epochs, t_data, t_label, batch_size);

	printf("\n\n");
	lr = 0.005;
	batch_size = 1;
	sgd_res = train_sgd(lr, num_epochs, t_data, t_label, batch_size);

	printf("\n\n");
	lr = 0.4;
	batch_size = 100;
	mini1_res = train_sgd(lr, num_epochs, t_data, t_label, batch_size);

	printf("\n\n");
	lr = 0.05;
	batch_size = 10;
	mini2_res = train_sgd(lr, num_epochs, t_data, t_label, batch_size);

	plt::figure_size(800, 600);
	plt::named_plot("gd", gd.first, gd.second, "b");
	plt::named_plot("sgd", sgd_res.first, sgd_res.second, "m--");
	plt::named_plot("batch size=100", mini1_res.first, mini1_res.second, "g-.");
	plt::named_plot("batch size=10", mini2_res.first, mini2_res.second, "r:");
	plt::xlabel("epoch");
	plt::ylabel("loss");
	plt::legend();
	plt::show();
	plt::close();

	// ---------------------------------------------
	// Concise Implementation
	// ---------------------------------------------
	batch_size = 10;

	auto net = torch::nn::Sequential( torch::nn::Linear(5, 1) );

	// init weight
	for(auto& module : net->modules(false)) { 			// include_self= false
		if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
			torch::nn::init::normal_(M->weight, 0.01);  // torch.nn.init.normal_(m.weight, std=0.01)
		}
	}

	auto optimizer = torch::optim::SGD(net->parameters(), 0.01);
	auto loss = torch::nn::MSELoss(torch::nn::MSELossOptions(torch::kNone));

	start = high_resolution_clock::now();
	std::vector<float> epochs, losses;

	for( int64_t  epoch = 0; epoch < num_epochs; epoch++ ) {
		std::list<std::pair<torch::Tensor, torch::Tensor>> data_iter = get_data_ch11(t_data, t_label, batch_size);

		float t_loss = 0.0;
		int64_t b_cnt = 0;
		for (auto &batch : data_iter) {
			auto X = batch.first;
			auto y = batch.second;

			optimizer.zero_grad();

			auto out = net->forward(X);

			y = y.reshape(out.sizes());
			auto l = loss(out, y).mean();
			l.backward();

			optimizer.step();

		    t_loss += l.item<float>();
		    b_cnt++;
		}

		stop = high_resolution_clock::now();
		duration = duration_cast<microseconds>(stop - start);
		std::cout << "Epoch: " << (epoch + 1) << ", loss: " << (t_loss/b_cnt)  << ", duration: " << (duration.count() / 1e6) << " sec.\n";
		epochs.push_back(epoch*1.0);
		losses.push_back((t_loss/b_cnt));
	}

	plt::figure_size(800, 600);
	plt::plot(epochs, losses, "b");
	plt::xlabel("epoch");
	plt::ylabel("loss");
	plt::show();
	plt::close();

	std::cout << "Done!\n";
	return 0;
}








