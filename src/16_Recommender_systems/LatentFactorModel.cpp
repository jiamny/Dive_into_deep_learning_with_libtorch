#include <torch/utils.h>
#include "../utils/ch_16_util.h"
#include "../utils.h"
#include "../TempHelpFunctions.hpp"

#include <matplot/matplot.h>
using namespace matplot;

struct LFMModelImpl : public torch::nn::Module {
	torch::nn::Embedding u_emb{nullptr}, i_emb{nullptr};

	LFMModelImpl(int user_num, int item_num, int k=10) {
        u_emb = torch::nn::Embedding(torch::nn::EmbeddingOptions(user_num, k));
        i_emb = torch::nn::Embedding(torch::nn::EmbeddingOptions(item_num, k));

        u_emb->weight.data().uniform_(0, 0.005);
        i_emb->weight.data().uniform_(0, 0.005);
        register_module("u_emb", u_emb);
        register_module("i_emb", i_emb);
	}

	torch::Tensor forward(torch::Tensor uid, torch::Tensor mid) {
		torch::Tensor t = torch::mul(u_emb->forward(uid), i_emb->forward(mid));
		c10::OptionalArrayRef<long int> d = {1};
        return t.sum(d);
    }
};
TORCH_MODULE(LFMModel);

torch::Tensor  mean_squared_error(torch::Tensor y_true, torch::Tensor y_pred) {
    // Returns the mean squared error between y_true and y_pred
	torch::Tensor mse = torch::mean(torch::pow(y_true - y_pred, 2));
    return mse;
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	//torch::Device device(torch::kCPU);
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	std::string file_name = "./data/ml-100k/u.data";

	std::string line;
	std::ifstream fL(file_name.c_str());
	std::vector<int> X_r;
	std::vector<int> y_r;

	int r = 0, c = 2;
	if( fL.is_open() ) {
		 std::getline(fL, line); // skip header

		while ( std::getline(fL, line) ) {
			line = strip(line);
			std::vector<std::string> strs = stringSplit(line, '\t');
			printVector(strs);
			X_r.push_back(std::atoi(strip(strs[0]).c_str()));
			X_r.push_back(std::atoi(strip(strs[1]).c_str()));
			y_r.push_back(std::atoi(strip(strs[2]).c_str()));
			r++;
		}
	}
	fL.close();

	torch::Tensor X = torch::from_blob(X_r.data(), {r, c}, at::TensorOptions(torch::kInt32)).clone();
	torch::Tensor y = torch::from_blob(y_r.data(), {r, 1}, at::TensorOptions(torch::kInt32)).clone();
	//X = X.to(torch::kFloat32);

	std::cout << X.sizes() << '\n' << y.sizes() << '\n';

    torch::Tensor x_train, x_test, y_train, y_test;
    std::tie(x_train, x_test, y_train, y_test) = train_test_split(X, y, 0.3, true);
    std::cout << x_train.index({Slice(0,10), Slice()}) << '\n' << y_train.index({Slice(0,10), Slice()}) << '\n';

    int epochs = 200;
    int batch_size = 1024;

	auto dataset = LRdataset(x_train, y_train)
					   .map(torch::data::transforms::Stack<>());

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		                   std::move(dataset), batch_size);

	auto tst_dataset = LRdataset(x_train, y_train)
					   .map(torch::data::transforms::Stack<>());

	auto tst_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		                   std::move(tst_dataset), batch_size);

    int num_users = X.index({Slice(), 0}).max().data().item<int>() + 1;
    int num_items = X.index({Slice(), 1}).max().data().item<int>() + 1;
    std::cout << "num_users: " << num_users << " num_items: " << num_items << '\n';
    LFMModel model = LFMModel(num_users, num_items, 20);
    model->to(device);
/*
	auto batch = *train_loader->begin();
	auto data = batch.data;
	auto x_u = data.index({Slice(), 0}).to(device);
	auto x_i = data.index({Slice(), 1}).to(device);
	auto lab  = batch.target.to(device);
	auto y_pred = model->forward(x_u, x_i);
	std::cout << "lab: " << lab.sizes() << std::endl;
*/

    torch::optim::Adam optim = torch::optim::Adam(model->parameters(), 1e-3);
    torch::nn::MSELoss loss_func = torch::nn::MSELoss();

    std::vector<double> train_ls, test_mse, xx;

    for(auto& epoch : range(epochs, 0)) {

        model->train();
        double total_loss = 0.;
        int cnt = 0;
        for(auto &batch : *train_loader) {

        	auto data = batch.data;
        	auto x_u = data.index({Slice(), 0}).to(device);
        	auto x_i = data.index({Slice(), 1}).to(device);
            auto lab = batch.target.to(torch::kFloat32);
            lab = lab.to(device);
            auto y_pred = model->forward(x_u, x_i);

            auto l = loss_func(y_pred, lab.flatten());

            optim.zero_grad();
            l.backward();
            optim.step();

            total_loss += l.data().item<double>();
            cnt += 1;
        }
        train_ls.push_back(total_loss / cnt);

        model->eval();
        torch::NoGradGuard no_grad;

        std::vector<torch::Tensor> labels, predicts;
        for(auto &batch : *tst_loader) {
        	auto data = batch.data;
        	auto x_u = data.index({Slice(), 0}).to(device);
        	auto x_i = data.index({Slice(), 1}).to(device);
            auto lab = batch.target.to(torch::kFloat32);
            lab = lab.to(device);
        	auto predict = model->forward(x_u, x_i);

        	predicts.push_back(predict);
        	labels.push_back(lab.flatten());
        }

        torch::Tensor all_pred = torch::cat(predicts, 0);
        torch::Tensor all_label = torch::cat(labels, 0);
        torch::Tensor mse = mean_squared_error(all_label, all_pred);
        test_mse.push_back(mse.data().item<double>());
        xx.push_back((epoch + 1)*1.0);

        printf("Epoch = %4d, train loss = %.5f, validation mse = %.5f\n", epoch + 1, (total_loss / cnt), mse.data().item<double>());
    }

    double min = *min_element(test_mse.begin(), test_mse.end());
    std::cout << "min test mse = " << min << '\n';

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::semilogy(ax1, xx, train_ls, "b-")->line_width(2).display_name("train loss");
	matplot::semilogy(ax1, xx, test_mse, "m--")->line_width(2).display_name("validation mse");
	matplot::xlabel(ax1, "epoch");
	matplot::legend(ax1, {});
	matplot::show();

	std::cout << "Done!\n";
}




