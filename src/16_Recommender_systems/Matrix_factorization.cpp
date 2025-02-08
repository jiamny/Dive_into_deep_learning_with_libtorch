#include <torch/utils.h>
#include "../utils/ch_16_util.h"
#include "../utils.h"
#include "../TempHelpFunctions.hpp"

#include <matplot/matplot.h>
using namespace matplot;

struct MatrixFactorizationImpl : public torch::nn::Module {
	bool sparse;
	torch::nn::Embedding user_embedding{nullptr}, user_bias{nullptr}, item_embedding{nullptr}, item_bias{nullptr};

	MatrixFactorizationImpl(int embedding_dims, int num_users, int num_items,
                 bool _sparse=false) {

        sparse = _sparse;
        user_embedding = torch::nn::Embedding(torch::nn::EmbeddingOptions(num_users, embedding_dims).sparse(sparse));
        user_bias = torch::nn::Embedding(torch::nn::EmbeddingOptions(num_users, 1).sparse(sparse));

        item_embedding = torch::nn::Embedding(torch::nn::EmbeddingOptions(num_items, embedding_dims).sparse(sparse));
        item_bias = torch::nn::Embedding(torch::nn::EmbeddingOptions(num_items, 1).sparse(sparse));

        for(auto& param : parameters()) {
            torch::nn::init::normal_(param, 0.0, 0.01);
        }
        register_module("user_embedding", user_embedding);
        register_module("user_bias", user_bias);
        register_module("item_embedding", item_embedding);
        register_module("item_bias", item_bias);
	}

    torch::Tensor forward(torch::Tensor user_id, torch::Tensor item_id) {
    	torch::Tensor Q = user_embedding->forward(user_id);
		torch::Tensor bq = user_bias->forward(user_id).flatten();

		torch::Tensor I = item_embedding->forward(item_id);
		torch::Tensor bi = item_bias->forward(item_id).flatten();
		c10::OptionalArrayRef<long int> d = {-1};
        return (Q * I).sum(d) + bq + bi;
    }
};
TORCH_MODULE(MatrixFactorization);

std::tuple<std::vector<double>, std::vector<double>> train(MatrixFactorization& model, torch::Tensor X_train,
		torch::Tensor y_train, torch::Tensor X_valid, torch::Tensor y_valid, torch::nn::MSELoss& loss_func,
		int num_epochs, float learning_rate, float weight_decay, int batch_size, torch::Device device) {
	std::vector<double> train_ls, valid_ls;

	auto dataset = LRdataset(X_train, y_train)
					   .map(torch::data::transforms::Stack<>());

	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		                   std::move(dataset), batch_size);


    auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(weight_decay));
    //#optimizer = torch.optim.SparseAdam(model.parameters(), lr=learning_rate)
    model->to(device);

    for(int epoch= 0; epoch < num_epochs; epoch++) {
        model->train();
        double total_loss = 0.0;
        int total_len =  0;
		for (auto &batch : *data_loader) {

			auto X = batch.data.to(device);
			auto x_u = X.index({Slice(), 0});
			auto x_i = X.index({Slice(), 1});
			auto y = batch.target.to(device);
            auto y_pred = model->forward(x_u, x_i);
            auto l = loss_func(y_pred, y.flatten()).sum();
            optimizer.zero_grad();
            l.backward();
            optimizer.step();

            total_loss += l.data().item<double>();
            total_len += y.size(0);
		}
        train_ls.push_back(total_loss / total_len);

        if(X_valid.numel() > 0 ) {
            model->eval();
            torch::NoGradGuard no_grad;
            int n = y_valid.size(0);
            X_valid = X_valid.to(device);
            y_valid = y_valid.to(device);
            auto valid_loss = loss_func(model(X_valid.index({Slice(), 0}), X_valid.index({Slice(), 0})), y_valid.flatten());
            valid_ls.push_back(valid_loss.data().item<double>() / n);
        }
        printf("epoch %3d, train mse %.4f, valid mse %.4f\n", epoch + 1, train_ls[epoch], valid_ls[epoch]);
    }
    return std::make_tuple(train_ls, valid_ls);
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	//torch::Device device(torch::kCPU);
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);
	int batch_size = 1024;
	int num_epochs = 300;
	float learning_rate = 0.0006;
	float weight_decay = 0.1;

	// The MovieLens Dataset: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
	std::string file_name = "./data/ml-latest-small/ratings.csv";

	std::string line;
	std::ifstream fL(file_name.c_str());
	std::vector<long int> X_r;
	std::vector<float> y_r;

	int r = 0, c = 2;
	if( fL.is_open() ) {
		 std::getline(fL, line); // skip header

		while ( std::getline(fL, line) ) {
			line = strip(line);
			std::vector<std::string> strs = stringSplit(line, ',');
			X_r.push_back(std::atol(strip(strs[0]).c_str()));
			X_r.push_back(std::atol(strip(strs[1]).c_str()));
			y_r.push_back(std::atof(strip(strs[2]).c_str()));
			r++;
		}
	}
	fL.close();

	torch::Tensor X = torch::from_blob(X_r.data(), {r, c}, at::TensorOptions(torch::kInt64)).clone();
	torch::Tensor y = torch::from_blob(y_r.data(), {static_cast<int>(y_r.size()), 1}, at::TensorOptions(torch::kFloat32)).clone();
	std::cout << X.index({Slice(0,10), 0}).sizes() << '\n' << y.sizes() << '\n';

    torch::Tensor x_train, x_test, y_train, y_test;
    std::tie(x_train, x_test, y_train, y_test) = train_test_split(X, y, 0.2, true);
    std::cout << x_train.index({Slice(0,10), Slice()}) << '\n' << y_train.index({Slice(0,10), Slice()}) << '\n';

    float mean_rating = y.mean().data().item<float>();
    int num_users = X.index({Slice(), 0}).max().data().item<int>() + 1;
    int num_items = X.index({Slice(), 1}).max().data().item<int>() + 1;
    MatrixFactorization model = MatrixFactorization(30, num_users, num_items, false);
    auto lsf = torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kSum));

    std::vector<double> train_ls, valid_ls;
    std::tie(train_ls, valid_ls) = train(model, x_train, y_train, x_test, y_test, lsf,
    		num_epochs, learning_rate, weight_decay, batch_size, device);

    //printVector(train_ls);
    //printVector(valid_ls);

    std::vector<double> xx;
    for(int i = 1; i <= train_ls.size(); i++)
    	xx.push_back(1.0 * i);

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::semilogy(ax1, xx, train_ls, "b-")->line_width(2).display_name("train loss");
	matplot::semilogy(ax1, xx, valid_ls, "m--")->line_width(2).display_name("valid loss");
	matplot::xlabel(ax1, "epoch");
	matplot::ylabel(ax1, "loss");
	matplot::legend(ax1, {});
	matplot::show();

	std::cout << "Done!\n";
}

