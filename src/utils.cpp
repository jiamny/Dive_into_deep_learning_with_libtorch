
#include "utils.h"


LRdataset::LRdataset(std::pair<torch::Tensor, torch::Tensor> data_and_labels) {

    features_ = std::move(data_and_labels.first);
    labels_ = std::move(data_and_labels.second);
}

LRdataset::LRdataset(torch::Tensor data, torch::Tensor labels) {

    features_ = data;
    labels_ = labels;
}

torch::data::Example<> LRdataset::get(size_t index) {
    return {features_[index], labels_[index]};
}

torch::optional<size_t> LRdataset::size() const {
    return features_.size(0);
}

const torch::Tensor& LRdataset::features() const {
    return features_;
}

const torch::Tensor& LRdataset::labels() const {
    return labels_;
}

std::pair<torch::Tensor, torch::Tensor> synthetic_data(torch::Tensor true_w, float true_b, int64_t num_samples) {

	auto X = torch::normal(0.0, 1.0, {num_samples, true_w.size(0)});
	auto y = torch::matmul(X, true_w) + true_b;
	y += torch::normal(0.0, 0.01, y.sizes());
	y = torch::reshape(y, {-1, 1});

	//return torch::cat({X, y}, 1);
	return {X, y};
 }

// # Defined in file: ./chapter_linear-networks/linear-regression-scratch.md
torch::Tensor linreg(torch::Tensor X, torch::Tensor w, torch::Tensor b) {
	// The linear regression model
	return torch::matmul(X, w) + b;
}

// # Defined in file: ./chapter_linear-networks/linear-regression-scratch.md
torch::Tensor squared_loss(torch::Tensor y_hat, torch::Tensor y) {
	// Squared loss
	auto rlt = torch::pow(y_hat - torch::reshape(y, y_hat.sizes()), 2) / 2;
	return rlt;
}

void sgd(torch::Tensor& w, torch::Tensor& b, float lr, int64_t batch_size) {
	//Minibatch stochastic gradient descent.
	torch::NoGradGuard no_grad_guard;
	// SGD
	w -= lr * w.grad() / batch_size;
	w.grad().zero_();

	b -= lr * b.grad() / batch_size;
	b.grad().zero_();
}


std::unordered_map<int, std::string> get_fashion_mnist_labels(void){
	// Create an unordered_map to hold label names
	std::unordered_map<int, std::string> fashionMap;
	fashionMap.insert({0, "T-shirt/top"});
	fashionMap.insert({1, "Trouser"});
	fashionMap.insert({2, "Pullover"});
	fashionMap.insert({3, "Dress"});
	fashionMap.insert({4, "Coat"});
	fashionMap.insert({5, "Sandal"});
	fashionMap.insert({6, "Short"});
	fashionMap.insert({7, "Sneaker"});
	fashionMap.insert({8, "Bag"});
	fashionMap.insert({9, "Ankle boot"});

	return fashionMap;
}


torch::Tensor softmax(torch::Tensor X) {
	auto X_exp = torch::exp(X);
    auto partition = X_exp.sum(1, true);
    return (X_exp / partition);  // The broadcasting mechanism is applied here
}

int64_t accuracy(torch::Tensor y_hat, torch::Tensor y) {
	if( y_hat.size(0) > 1 && y_hat.size(1) > 1 )
		y_hat = torch::argmax(y_hat, 1);

	y_hat = y_hat.to(y.dtype());

	auto cmp = (y_hat == y );

	return torch::sum(cmp.to(y.dtype())).item<int64_t>();

}

torch::Tensor d2l_relu(torch::Tensor x) {
	auto a = torch::zeros_like(x);
    return torch::max(x, a);
}

torch::Tensor l2_penalty(torch::Tensor x) {
	return (torch::sum(x.pow(2)) / 2);
}

std::pair<torch::Tensor, torch::Tensor> init_params(int64_t num_inputs) {
	auto w = torch::empty({num_inputs, 1}, torch::TensorOptions().requires_grad(true));
	torch::nn::init::normal_(w, 0.0, 1.0);
	auto b = torch::zeros(1, torch::TensorOptions().requires_grad(true));
	return {w, b};
}


