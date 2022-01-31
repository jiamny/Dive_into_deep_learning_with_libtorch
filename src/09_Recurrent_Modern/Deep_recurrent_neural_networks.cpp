#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../utils/ch_8_9_util.h"
#include "../utils.h"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;


struct RNNModeldeeprnn : public torch::nn::Module {
    //"""The deep rnn model."""
	torch::nn::LSTM rnn{nullptr};
	int64_t vocab_size;
	int64_t num_hiddens;
	int64_t num_directions;
	torch::nn::Linear linear{nullptr};

	RNNModeldeeprnn( torch::nn::LSTM rnn_layer, int64_t vocab_size ) {
        rnn = rnn_layer;
        this->vocab_size = vocab_size;
        num_hiddens = rnn.get()->options.hidden_size();
        // If the RNN is bidirectional (to be introduced later),
        // `num_directions` should be 2, else it should be 1.
        if( ! rnn.get()->options.bidirectional() ) {
            num_directions = 1;
            linear = torch::nn::Linear(num_hiddens, vocab_size);
        } else {
            num_directions = 2;
            linear = torch::nn::Linear(num_hiddens * 2, vocab_size);
        }
        register_module("rnn_layer", rnn_layer);
	}

	std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> forward(torch::Tensor inputs,
															std::tuple<torch::Tensor, torch::Tensor> state ) {
        auto X = torch::one_hot(inputs.transpose(0, 1), vocab_size); //(inputs.T.long(), self.vocab_size)
        X = X.to(torch::kFloat32);
        torch::Tensor Y;

        std::tie(Y, state) = rnn->forward(X, state);

        // The fully connected layer will first change the shape of `Y` to
        // (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        // (`num_steps` * `batch_size`, `vocab_size`).
        Y = linear(Y.reshape({-1, Y.size(-1)})); // Y.shape[-1]
		return std::make_tuple(Y, state);
	}

	std::tuple<torch::Tensor, torch::Tensor> begin_state(int64_t batch_size, torch::Device device) {
		// `nn.GRU` takes a tensor as hidden state
		return std::make_tuple(torch::zeros( {num_directions * rnn.get()->options.num_layers(),
		                                batch_size, num_hiddens}, device),
				torch::zeros( {num_directions * rnn.get()->options.num_layers(),
            							batch_size, num_hiddens}, device));
	}
};



int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(7);

	int64_t batch_size=32, num_steps = 35;
	int64_t max_tokens = 10000;

	std::vector<std::string> lines = read_time_machine("./data/timemachine.txt");

	//-------------------------------------------------------------
	// split words and extract first token upto max_tokens
	//-------------------------------------------------------------
	std::vector<std::string> ttokens = tokenize(lines, "char", true);
	std::vector<std::string> tokens(&ttokens[0], &ttokens[max_tokens]); // first 10000 tokens

	std::vector<std::pair<std::string, int64_t>> counter = count_corpus( tokens );

	std::vector<std::string> rv(0);
	auto vocab = Vocab(counter, 0.0, rv);

	std::cout << vocab["t"] << "\n";

	// -----------------------------------------------------------
	// Training and Predicting
	// -----------------------------------------------------------
	//=============================================================
	// Let us [check whether the outputs have the correct shapes]
	auto X = torch::arange(10).reshape({2, 5});
	int num_hiddens = 256;
	int num_layers  = 2;

	torch::nn::LSTM lstm_layer(torch::nn::LSTMOptions(vocab.length(), num_hiddens).num_layers(num_layers));

	auto net = RNNModeldeeprnn(lstm_layer, vocab.length());
	net.to(device);

	std::tuple<torch::Tensor, torch::Tensor> state = net.begin_state(X.size(0), device);

	std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>  rlt = net.forward(X.to(device), state);

	auto Z = std::get<0>(rlt);
	std::tuple<torch::Tensor, torch::Tensor> new_state = std::get<1>(rlt);

	std::cout << "Z: " << Z.sizes() << std::endl;
	//std::cout << Z << std::endl;
	std::cout << "new_state H: " << std::get<0>(new_state).sizes() << std::endl;
	std::cout << "new_state C: " << std::get<1>(new_state).sizes() << std::endl;
	//std::cout << new_state << std::endl;

	//================================================
	// Deep RNN concise
	//================================================
	// Let us [first define the prediction function to generate new characters following the user-provided prefix]
	std::string s = "time traveller ";
	std::vector<char> v(s.begin(), s.end());
	std::vector<std::string> prefix;
	for(int i = 0; i < v.size(); i++ ) {
		std::string tc(1, v[i]);
		prefix.push_back(tc);
	}
	std::string prd = predict_ch9(prefix, 10, net, vocab, device);
	std::cout << prd << std::endl;

	std::vector<int> tokens_ids;
	for( size_t i = 0; i < tokens.size(); i++ )
		tokens_ids.push_back(vocab[tokens[i]]);

	int64_t num_epochs = 900;
	float lr = 1.0;
	bool use_random_iter = false;

	// Training and Predicting
	std::vector<std::pair<torch::Tensor, torch::Tensor>> ctrain_iter = seq_data_iter_random(tokens_ids, batch_size, num_steps);

	torch::nn::LSTM clstm_layer(torch::nn::LSTMOptions(vocab.length(), num_hiddens).num_layers(num_layers));
	RNNModeldeeprnn cnet(clstm_layer, vocab.length());
	cnet.to(device);

	std::pair<std::vector<double>, std::vector<double>> ctrlt = train_ch9( cnet, ctrain_iter, vocab, device, lr,
			num_epochs, use_random_iter);

	plt::figure_size(700, 500);
	plt::subplot(1, 1, 1);
	plt::named_plot("train", ctrlt.first, ctrlt.second, "b");
	plt::xlabel("epoch");
	plt::ylabel("perplexity");
	plt::title("RNNModeldeeprnn");
	plt::legend();
	plt::show();

	std::cout << "Done!\n";
	return 0;
}




