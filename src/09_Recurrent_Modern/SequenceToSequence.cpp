
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

using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;

// ---------------------------
// Encoder
// ---------------------------
struct Seq2SeqEncoderImpl : public torch::nn::Module {
	torch::nn::Embedding embedding{nullptr};
	torch::nn::GRU rnn{nullptr};
    //The RNN encoder for sequence to sequence learning.
	Seq2SeqEncoderImpl(int64_t vocab_size, int64_t embed_size, int64_t num_hiddens, int64_t num_layers, float dropout=0){
        // Embedding layer
        embedding = torch::nn::Embedding(vocab_size, embed_size);
        rnn = torch::nn::GRU(torch::nn::GRUOptions(embed_size, num_hiddens).num_layers(num_layers).dropout(dropout)); //embed_size, num_hiddens, num_layers, dropout
	}

	std::tuple<torch::Tensor, torch::Tensor>  forward(torch::Tensor X) {
        // The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        X = embedding->forward(X);
        // In RNN models, the first axis corresponds to time steps
        X = X.permute({1, 0, 2});
        // When state is not mentioned, it defaults to zeros
        torch::Tensor output, state;
        std::tie(output, state) = rnn->forward(X);
        // `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        // `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return std::make_tuple(output, state);
    }
};

TORCH_MODULE(Seq2SeqEncoder);

// ---------------------------
// Decoder
// ---------------------------
struct Seq2SeqDecoderImpl : public torch::nn::Module  {
	torch::nn::Embedding embedding{nullptr};
	torch::nn::GRU rnn{nullptr};
	torch::nn::Linear dense{nullptr};

    // The RNN decoder for sequence to sequence learning.
	Seq2SeqDecoderImpl(int64_t vocab_size, int64_t embed_size, int64_t num_hiddens, int64_t num_layers, float dropout=0){

		embedding = torch::nn::Embedding(vocab_size, embed_size);
		rnn = torch::nn::GRU(torch::nn::GRUOptions(embed_size + num_hiddens, num_hiddens).num_layers(num_layers).dropout(dropout));
        dense = torch::nn::Linear(num_hiddens, vocab_size);
	}

	torch::Tensor init_state(std::tuple<torch::Tensor, torch::Tensor> enc_outputs) {
        return std::get<1>(enc_outputs);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor X,
    		torch::Tensor state) {
        // The output `X` shape: (`num_steps`, `batch_size`, `embed_size`)
        X = embedding->forward(X).permute({1, 0, 2});

        // Broadcast `context` so it has the same `num_steps` as `X`
        auto context = state[-1].repeat({X.size(0), 1, 1});
        auto X_and_context = torch::cat({X, context}, 2);
        torch::Tensor output;
        std::tie(output, state) = rnn->forward(X_and_context, state);
        output = dense->forward(output).permute({1, 0, 2});
        // `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
        // `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return std::make_tuple(output, state);
    }
};

TORCH_MODULE(Seq2SeqDecoder);

struct EncoderDecoderImpl : public torch::nn::Module {
    //The base class for the encoder-decoder architecture.
    //Defined in :numref:`sec_encoder-decoder`
	Seq2SeqDecoder decoder{nullptr};
	Seq2SeqEncoder encoder{nullptr};

	EncoderDecoderImpl(Seq2SeqEncoder encoder, Seq2SeqDecoder decoder) {
        this->encoder = encoder;
        this->decoder = decoder;
	}

	std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor enc_X, torch::Tensor dec_X) {
		std::tuple<torch::Tensor, torch::Tensor> enc_outputs = this->encoder->forward(enc_X);
		torch::Tensor dec_state = this->decoder->init_state(enc_outputs);
        return this->decoder->forward(dec_X, dec_state);
	}

};
TORCH_MODULE(EncoderDecoder);

void xavier_init_weights(torch::nn::Module &m) {

	torch::NoGradGuard no_grad;

    if(typeid(m) == typeid(torch::nn::Linear) ) {
    	auto p = m.named_parameters(false);
    	auto w = p.find("weight");
        torch::nn::init::xavier_uniform_(*w);
    }
    if(typeid(m) == typeid(torch::nn::GRU) ) {
        for( auto& param : m.named_parameters(true)) { 					// _flat_weights_names() ) {
            if( param.pair().first.find("weight") ) {
                torch::nn::init::xavier_uniform_(param.pair().second);	// m.parameters[param]
            }
        }
    }
}

// ---------------------------------
// Loss Function
// ---------------------------------
torch::Tensor sequence_mask(torch::Tensor X, torch::Tensor  valid_len, float value=0) {
    //Mask irrelevant entries in sequences.
    int64_t maxlen = X.size(1);
    auto mask = torch::arange((maxlen),
    		torch::TensorOptions().dtype(torch::kFloat32).device(X.device())).index({None, Slice()}) < valid_len.index({Slice(), None});
    //std::cout << torch::ones_like(mask) << std::endl;
    // (if B - boolean tensor) at::Tensor not_B = torch::ones_like(B) ^ B;
    // std::cout << (torch::ones_like(mask) ^ mask).sizes() <<std::endl;
    X.index_put_({torch::ones_like(mask) ^ mask}, value);

    //std::cout << (torch::ones_like(mask) ^ mask).sizes() << "\n";
    //std::cout << X.sizes() << "\n";

    return X;
}

class MaskedSoftmaxCELoss : public torch::nn::CrossEntropyLoss {
public:
	MaskedSoftmaxCELoss() {
		loss = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().reduction(torch::kNone));
	}
    /*The softmax cross-entropy loss with masks.
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    */
    torch::Tensor forward(torch::Tensor pred, torch::Tensor label, torch::Tensor valid_len){
    	torch::Tensor weights = torch::ones_like(label);
        weights = sequence_mask(weights, valid_len);
        //reduction ='none'
        // auto loss = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().reduction(torch::kNone));
        torch::Tensor unweighted_loss = loss->forward(pred.permute({0, 2, 1}), label);
        torch::Tensor weighted_loss = (unweighted_loss * weights).mean(1);
        //std::cout << "unweighted_loss:\n" << unweighted_loss << "\n";
        //std::cout << "weighted_loss:\n" << weighted_loss << "\n";
        return weighted_loss;
   }
private:
   torch::nn::CrossEntropyLoss loss{nullptr};
};


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(7);

	// test encoder
	auto encoder = Seq2SeqEncoder(10, 8, 16, 2);
	encoder->eval();
	auto X = torch::zeros({4, 7}).to(torch::kLong);
	std::tuple<torch::Tensor, torch::Tensor> erlt = encoder->forward(X);
	std::cout << std::get<0>(erlt).sizes() << std::endl;
	std::cout << std::get<1>(erlt).sizes() << std::endl;

	// test decoder
	X = torch::zeros({4, 7}).to(torch::kLong);
	auto decoder = Seq2SeqDecoder(10, 8, 16, 2);
	decoder->eval();
	torch::Tensor state = decoder->init_state(encoder->forward(X));

	std::tuple<torch::Tensor, torch::Tensor> drlt = decoder->forward(X, state);
	std::cout << "output.shape: " << std::get<0>(drlt).sizes() << std::endl;
	std::cout << "state.shape: " << std::get<1>(drlt).sizes() << std::endl;

	// test sequence_mask
	X = torch::tensor({{1, 2, 3}, {4, 5, 6}});
	auto Y = sequence_mask(X, torch::tensor({1, 2}));
	std::cout << "sequence_mask:\n" << Y << std::endl;
	std::cout << "X:\n" << X.sizes() << std::endl;
	std::cout << "v_len:\n" << torch::tensor({1, 2}).sizes() << std::endl;

	// (We can also mask all the entries across the last few axes.)
	// If you like, you may even specify to replace such entries with a non-zero value.
	X = torch::ones({2, 3, 4});
	std::cout << "X.shape: " << X << std::endl;
	Y = sequence_mask(X, torch::tensor({1, 2}), -1);
	std::cout << "sequence_mask:\n" << Y << std::endl;

	auto loss_fn = MaskedSoftmaxCELoss();
	torch::Tensor weighted_loss = loss_fn.forward(torch::ones({3, 4, 10}), torch::ones({3, 4}).to(torch::kLong),
										torch::tensor({4, 2, 0}));

	std::cout << "weighted_loss:\n" << weighted_loss << std::endl;

	// -------------------------------
	// Training
	// -------------------------------

	int embed_size = 32, num_hiddens = 32, num_layers = 2;
	float dropout = 0.1, lr = 0.005;
	int batch_size = 64, num_steps = 10;
	int64_t num_epochs = 300;

	std::string filename = "./data/fra-eng/fra.txt";
	std::pair<torch::Tensor, torch::Tensor> data_arrays;
	Vocab src_vocab, tgt_vocab;

	std::tie(data_arrays, src_vocab, tgt_vocab) = load_data_nmt(filename, num_steps, 600);
	encoder = Seq2SeqEncoder(src_vocab.length(), embed_size, num_hiddens, num_layers, dropout);
	decoder = Seq2SeqDecoder(tgt_vocab.length(), embed_size, num_hiddens, num_layers, dropout);
	auto net = EncoderDecoder(encoder, decoder);

//	std::cout << data_arrays.first << std::endl;
	int nWorkers = 2;

	auto dataset = Nmtdataset(data_arrays)
		    				.map(torch::data::transforms::Stack<>());
	auto data_iter = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		    	        std::move(dataset), torch::data::DataLoaderOptions().batch_size(batch_size)
	                    													.workers(nWorkers)
																			.drop_last(true));//Shuffle by default

	//Train a model for sequence to sequence.
	net->apply(xavier_init_weights);
	net->to(device);
	auto optimizer = torch::optim::Adam(net->parameters(), lr);
	//torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(lr).betas({0.5, 0.999}));

	loss_fn = MaskedSoftmaxCELoss();
	net->train();

	std::vector<double> epochs, plsum;
	std::vector<int64_t> wtks;
	torch::Tensor Y_hat, stat;

	for( int64_t epoch = 0;  epoch < num_epochs; epoch++ ) {

	    for(auto& batch : *data_iter) {
	        optimizer.zero_grad();
	        auto features = batch.data;
	        auto labels = batch.target;
	        auto X = features.index({Slice(), Slice(None, num_steps)});
	        auto X_valid_len = labels.index({Slice(), 0});
	        auto Y = features.index({Slice(), Slice(num_steps, None)});
	        auto Y_valid_len = labels.index({Slice(), 1});
	        X.to(device);
	        X_valid_len.to(device);
	        Y.to(device);
	        Y_valid_len.to(device);

	        std::vector<int64_t> tmp;
	        for(int i = 0; i < Y.size(0); i++)
	        	tmp.push_back(tgt_vocab["<bos>"]);

	        auto options = torch::TensorOptions().dtype(torch::kLong);
	        torch::Tensor bos = torch::from_blob(tmp.data(), {1, Y.size(0)}, options).clone();
	        bos = bos.clone().to(device).reshape({-1, 1});

	        torch::Tensor dec_input = torch::cat({bos, Y.index({Slice(), Slice(None, -1)})}, 1); // Y[:, :-1]

	        std::tie(Y_hat, stat) = net->forward(X, dec_input); // , X_valid_len

	        auto l = loss_fn.forward(Y_hat, Y, Y_valid_len);
	        l.sum().backward(); 								 // Make the loss scalar for `backward`

	        //"""Clip the gradient."""
	        torch::nn::utils::clip_grad_norm_(net->parameters(), 0.5);

	        auto num_tokens = Y_valid_len.sum();

	        optimizer.step();
	        torch::NoGradGuard no_grad;
	        //torch::no_grad();
	        //metric.add(l.sum(), num_tokens);
	        plsum.push_back((l.sum()).item<float>());
	        wtks.push_back(num_tokens.item<long>());
	        epochs.push_back(1.0*epoch);

	    }

	    if((epoch + 1) % 10 == 0)
	    	std::cout << "loss: " << plsum[epoch]/wtks[epoch] << std::endl;
	}

	std::cout << "Done!\n";
	return 0;
}

