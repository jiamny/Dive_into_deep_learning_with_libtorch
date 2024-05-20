
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
	w -= (lr * w.grad() / batch_size);
	w.grad().zero_();

	b -= (lr * b.grad() / batch_size);
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
	torch::Tensor val_max, _;
	std::tie(val_max, _) = torch::max(X, -1, true);
	torch::Tensor X_exp = torch::exp(X - val_max);

	c10::OptionalArrayRef<long int> dim = {1};
	torch::Tensor partition = torch::sum(X_exp, dim, true);
	return (X_exp / partition);
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

std::unordered_map<std::string, std::string> getFlowersLabels(std::string jsonFile) {

	std::unordered_map<std::string, std::string> labelMap;
	//Json::Reader reader;  	  //for reading the data
	Json::Value newValue; 	  //for modifying and storing new values

	//opening file using fstream
	//std::ifstream file(jsonFile);
	std::ifstream ifs;
	ifs.open(jsonFile);
	Json::CharReaderBuilder builder;
	builder["collectComments"] = true;
	std::__cxx11::basic_string<char,std::char_traits<char>,std::allocator<char>> errs;

	// check if there is any error is getting data from the json file
	//if ( ! reader.parse(file, newValue) ) {
	if (! parseFromStream(builder, ifs, &newValue, &errs)) {
		//std::cout << reader.getFormattedErrorMessages();
		std::cout << errs << std::endl;
		exit(1);
	} else {

		for( int i = 0; i < newValue.size(); i++ ) {
			//std::cout << newValue.getMemberNames()[i].c_str() << " " << newValue[newValue.getMemberNames()[i].c_str()].asCString() << std::endl;
			labelMap.insert({ std::string(newValue.getMemberNames()[i].c_str()),  newValue[newValue.getMemberNames()[i].c_str()].asCString()});
		}
	//		std::string t = "20";
	//		std::cout << labelMap[t] << std::endl;
	}

	return labelMap;
}



// data batch indices
std::list<torch::Tensor> data_index_iter(int64_t num_examples, int64_t batch_size, bool shuffle) {

	std::list<torch::Tensor> batch_indices;
	// data index
	std::vector<int64_t> index;
	for (int64_t i = 0; i < num_examples; ++i) {
		index.push_back(i);
	}
	// shuffle index
	if( shuffle ) std::random_shuffle(index.begin(), index.end());

	for (int64_t i = 0; i < index.size(); i +=batch_size) {
		std::vector<int64_t>::const_iterator first = index.begin() + i;
		std::vector<int64_t>::const_iterator last = index.begin() + std::min(i + batch_size, num_examples);
		std::vector<int64_t> indices(first, last);

		int64_t idx_size = indices.size();
		torch::Tensor idx = (torch::from_blob(indices.data(), {idx_size}, at::TensorOptions(torch::kInt64))).clone();

		//auto batch_x = X.index_select(0, idx);
		//auto batch_y = Y.index_select(0, idx);

		batch_indices.push_back(idx);
	}
	return( batch_indices );
}

torch::Tensor RangeToensorIndex(int64_t num) {
	std::vector<int64_t> idx;
	for( int64_t i = 0; i < num; i++ )
		idx.push_back(i);

	torch::Tensor RngIdx = (torch::from_blob(idx.data(), {num}, at::TensorOptions(torch::kInt64))).clone();
	return RngIdx;
}

bool isNumberRegex(const std::string& str) {
	std::regex numberRegex("^[-+]?([0-9]*\\.[0-9]+[0-9]+)$");
	return std::regex_match(str, numberRegex);
}


