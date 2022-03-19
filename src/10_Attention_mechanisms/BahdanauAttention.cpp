#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../utils/ch_10_util.h"

// ----------------------------------------------------
// implement the RNN decoder with Bahdanau attention
// ----------------------------------------------------
struct Seq2SeqAttentionDecoder : public torch::nn::Module  {
	AdditiveAttention attention;
	torch::nn::Embedding embedding{nullptr};
	torch::nn::GRU rnn{nullptr};
	torch::nn::Linear dense{nullptr};
	std::vector<torch::Tensor> _attention_weights;

	Seq2SeqAttentionDecoder(int64_t vocab_size, int64_t embed_size, int64_t num_hiddens, int64_t num_layers, float dropout=0) {
        attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout);
        embedding = torch::nn::Embedding(vocab_size, embed_size);
        rnn = torch::nn::GRU( torch::nn::GRUOptions(embed_size + num_hiddens, num_hiddens).num_layers(num_layers).dropout(dropout));
        		//embed_size + num_hiddens, num_hiddens, num_layers, dropout);
        dense = torch::nn::Linear(num_hiddens, vocab_size);
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>  init_state(std::tuple<torch::Tensor, torch::Tensor> enc_outputs,
			torch::Tensor enc_valid_lens) {
        // Shape of `outputs`: (`num_steps`, `batch_size`, `num_hiddens`).
        // Shape of `hidden_state[0]`: (`num_layers`, `batch_size`, `num_hiddens`)
		torch::Tensor outputs, hidden_state;
        std::tie(outputs, hidden_state) = enc_outputs;
        return {outputs.permute({1, 0, 2}), hidden_state, enc_valid_lens};
    }

    std::pair<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> forward(torch::Tensor X,
    															std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> state) {
        // Shape of `enc_outputs`: (`batch_size`, `num_steps`, `num_hiddens`).
        // Shape of `hidden_state[0]`: (`num_layers`, `batch_size`, `num_hiddens`)
		torch::Tensor enc_outputs, hidden_state, enc_valid_lens;
        std::tie(enc_outputs, hidden_state, enc_valid_lens) = state;
        // Shape of the output `X`: (`num_steps`, `batch_size`, `embed_size`)
        X = embedding->forward(X).permute({1, 0, 2});
        _attention_weights.clear();
        std::vector<torch::Tensor> outputs;
        //std::cout << "X---1: " << X.sizes() << "\n";
        //std::cout << "enc_outputs---1: " << enc_outputs.sizes() << "\n";
        //std::cout << "hidden_state---1: " << hidden_state.sizes() << "\n";

        for( int i = 0; i < X.size(0); i++ ) {
        	auto x = X[i].clone();
        	//std::cout << "x---2: " << x.sizes() << "\n";
            // Shape of `query`: (`batch_size`, 1, `num_hiddens`)
        	//std::cout << "hidden_state[-1]---2: " << hidden_state[-1].sizes() << "\n";
            auto query = torch::unsqueeze(hidden_state[-1], /*dim=*/1);
            //std::cout << "query---2: " << query.sizes() << "\n";

            //std::cout << "enc_outputs---2: " << enc_outputs.sizes() << "\n";
            // Shape of `context`: (`batch_size`, 1, `num_hiddens`)
            auto context = attention.forward(query, enc_outputs, enc_outputs, enc_valid_lens);
            //std::cout << "context---2: " << context.sizes() << "\n";

            // Concatenate on the feature dimension
            x = torch::cat({context, torch::unsqueeze(x, 1)}, -1);
            //std::cout << "x_cat---2: " << x.sizes() << "\n";
            // Reshape `x` as (1, `batch_size`, `embed_size` + `num_hiddens`)
            torch::Tensor out;
            std::tie(out, hidden_state) = rnn->forward(x.permute({1, 0, 2}), hidden_state);
            //std::cout << "out---2: " << out.sizes() << "\n";
            outputs.push_back(out);
            _attention_weights.push_back(attention.attention_weights);
        }
        // After fully-connected layer transformation, shape of `outputs`:
        // (`num_steps`, `batch_size`, `vocab_size`)
        torch::Tensor output = dense->forward(torch::cat(outputs, 0));
        //std::cout << "output---2: " << output.sizes() << "\n";
        //std::cout << "output.permute({1, 0, 2})---2: " << output.permute({1, 0, 2}).sizes() << "\n";
        //return {torch::empty({0}), std::make_tuple(enc_outputs, hidden_state,enc_valid_lens)};
        return {output.permute({1, 0, 2}), std::make_tuple(enc_outputs, hidden_state,enc_valid_lens)};
    }

    std::vector<torch::Tensor> attention_weights() {
        return _attention_weights;
    }
};


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	// test the implemented decoder with Bahdanau attention using a minibatch of 4 sequence inputs of 7 time steps.
	auto encoder = Seq2SeqEncoder(10, 8, 16, 2);
	encoder->eval();

	auto decoder = Seq2SeqAttentionDecoder(10, 8, 16, 2);
	decoder.eval();

	auto X = torch::zeros({4, 7}).to(torch::kLong); 	// (`batch_size`, `num_steps`)
	std::tuple<torch::Tensor, torch::Tensor> enc_outputs = encoder->forward(X);
	std::cout << std::get<0>(enc_outputs).sizes() << std::endl;
	std::cout << std::get<1>(enc_outputs).sizes() << std::endl;

	torch::Tensor val_lens; // ! defined()
	torch::Tensor val_lens2 = torch::empty({0});
	std::cout << "defined: " << val_lens.defined() << ", numel(): " << val_lens2.numel() << "\n";

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> state = decoder.init_state(enc_outputs, val_lens);

	std::cout << std::tuple_size<decltype(state)>::value << std::endl;
	std::cout << std::get<0>(state).sizes() << std::endl;
	std::cout << std::get<1>(state).sizes() << std::endl;
	std::cout << std::get<2>(state).numel() << std::endl;


	torch::Tensor output;
	std::tie(output, state) = decoder.forward(X, state);
	std::cout << output.sizes() << std::endl;

	// Training

	std::cout << "Done!\n";
	return 0;
}



