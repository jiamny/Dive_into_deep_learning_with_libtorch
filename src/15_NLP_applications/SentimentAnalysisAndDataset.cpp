#include <unistd.h>
#include <iomanip>
#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/utils.h>
#include <vector>

#include "../utils/ch_15_util.h"

#include <matplot/matplot.h>
using namespace matplot;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	torch::Device device(torch::kCPU);

	torch::manual_seed(123);

	std::string data_dir = "./data/aclImdb";
	bool is_train = true;

	auto acimdb = read_imdb(data_dir, is_train);
	auto data = acimdb.first;
	auto labels = acimdb.second;
	std::cout << data.size() <<'\n';
	for( int i = 0; i < 3; i++ )
		std::cout << "label: " << labels[i] << " review: " << data[i].substr(0, 60) << '\n';

//	train_tokens = d2l.tokenize(train_data[0], token='word')
//	vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])

	//-------------------------------------------------------------
	// split words and extract first token upto max_tokens
	//-------------------------------------------------------------
	std::vector<std::string> ttokens = tokenize(data, "word", false);
	std::cout << ttokens.size() << '\n';
	int64_t max_tokens = ttokens.size() - 1;
	std::cout << "ttokens: " << ttokens[max_tokens] <<'\n';

	std::vector<std::string> tokens(&ttokens[0], &ttokens[max_tokens]);

	std::vector<std::pair<std::string, int64_t>> counter = count_corpus( tokens );

	std::vector<std::string> reserved_tokens;
	reserved_tokens.push_back("<pad>");
	auto vocab = Vocab(counter, 5.0, reserved_tokens);

	std::cout << "the: " << vocab["the"] << "\n";
	std::cout << "counter: " << counter[0].second << "\n";
	std::vector<int> tknum;
	for(int i = 0; i < data.size(); i++)
		tknum.push_back(count_num_tokens(data[i]).second);

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::hist(tknum);
	matplot::xlabel("# tokens per review");
	matplot::ylabel("count");
	matplot::show();

	size_t num_steps = 500;  // sequence length
	auto t = vocab[count_num_tokens(data[0]).first];
	printVector(t);

	std::vector<torch::Tensor> tensors;
//	auto dt = truncate_pad(vocab[count_num_tokens(data[0]).first], num_steps, vocab["<pad>"]);
//	auto TT = torch::from_blob(dt.data(), {1, num_steps}, torch::TensorOptions(torch::kLong));
//	std::cout << TT << '\n';

	for( int i = 0; i < data.size(); i++ ) {
		auto dt = truncate_pad(vocab[count_num_tokens(data[i]).first], num_steps, vocab["<pad>"]);
		//std::cout << dt.size() << '\n';
		//printVector(dt);

		// Create an array of size equivalent to vector
		//int arr[dt.size()];
		// Copy all elements of vector to array
		//std::transform( dt.begin(), dt.end(), arr, [](const auto & elem){ return elem; });
	    //std::copy( dt.begin(), dt.end(), arr);

		auto TT = torch::from_blob(dt.data(), {1, static_cast<long>(num_steps)}, torch::TensorOptions(torch::kLong)).clone();
		//std::cout << TT.sizes() << '\n';
		//std::cout << TT << '\n';

		tensors.push_back(TT);
	}
	//std::cout << tensors[0] << '\n';
	auto features = torch::concat(tensors, 0).to(torch::kLong);
	//std::cout << "===============================================" << '\n';
	//std::cout << features[0] << '\n';
	std::cout << features.sizes() << '\n';
	//printVector(labels);

	auto rlab = torch::from_blob(labels.data(), {static_cast<long>(labels.size()), 1}, torch::TensorOptions(torch::kLong));
	std::cout << rlab << '\n';

	int64_t batch_size = 32;

	std::cout << "features: " << features.sizes() << ", rlab: " << rlab.sizes() << '\n';

	auto dataset = LRdataset(std::make_pair(features, rlab)).map(torch::data::transforms::Stack<>());
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		        std::move(dataset), batch_size);

	int num_batch = 0;

	for (auto& batch : *data_loader) {
		auto X = batch.data;
		auto y = batch.target;
		if( num_batch == 0 ) {
			std::cout << "X: " << X.sizes() << ", y: " << y.sizes() << '\n';
			std::cout << X << '\n';
		}
		num_batch++;
	}

	std::cout << "# batches: " << num_batch << '\n';

	std::cout << "Done!\n";
}




