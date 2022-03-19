
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <cmath>

#include "../utils/ch_8_9_util.h"
#include "../utils.h"
#include "../utils/ch_10_util.h"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;

// Seq2SeqEncoder and sequence_mask() defined in ch_10_util.h

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
        register_module("Dembedding", embedding);
        register_module("Drnn", rnn);
        register_module("Ddense", dense);
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
        register_module("decoder", decoder);
        register_module("encoder", encoder);
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
    	weights = torch::ones_like(label);
        weights = sequence_mask(weights, valid_len);
        //reduction ='none'
        // auto loss = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().reduction(torch::kNone));
        unweighted_loss = loss->forward(pred.permute({0, 2, 1}), label);
        weighted_loss = (unweighted_loss * weights).mean(1);
        //std::cout << "unweighted_loss:\n" << unweighted_loss << "\n";
        //std::cout << "weighted_loss:\n" << weighted_loss << "\n";
        return weighted_loss;
   }
private:
   torch::nn::CrossEntropyLoss loss{nullptr};
   torch::Tensor weights, unweighted_loss, weighted_loss; //mast???!!!
};

// ----------------------------------------------------------
// Prediction
// To predict the output sequence token by token,
// at each decoder time step the predicted token from
//the previous time step is fed into the decoder as an input.
// -----------------------------------------------------------

template<typename T>
std::string predict_seq2seq(T net, std::string src_sentence, Vocab src_vocab, Vocab tgt_vocab, int64_t num_steps,
					torch::Device device, bool save_attention_weights=false) {

	auto options = torch::TensorOptions().dtype(torch::kLong);
    // Predict for sequence to sequence.
	net->eval();
    std::vector<std::string> etks = tokenize({src_sentence}, "word", false);

	std::vector<int64_t> vec;
	std::vector<int64_t> a = src_vocab[etks];
	a.push_back(src_vocab["<eos>"]);
	auto c = truncate_pad( a, num_steps, src_vocab["<pad>"]);
	for(auto i : c)
		vec.push_back(i);

	torch::Tensor src_ar = torch::from_blob(vec.data(), {1, num_steps}, options);
	src_ar = src_ar.clone().to(torch::kLong);
	torch::Tensor src_val_len = (src_ar != src_vocab["<pad>"]).to(torch::kLong).sum(1);

	std::tuple<torch::Tensor, torch::Tensor> enc_outputs = net->encoder->forward(src_ar);
	torch::Tensor dec_state = net->decoder->init_state(enc_outputs);

    torch::Tensor prd;
	std::tie(prd, dec_state) = net->decoder->forward(src_ar, dec_state );
	prd = prd.argmax(2);
//	std::cout << "pred: " << prd << "\n";

	auto r_ptr = prd.data_ptr<int64_t>();
	std::vector<int64_t> idx{r_ptr, r_ptr + prd.numel()};
//	std::for_each(std::begin(idx), std::end(idx), [](const auto & element) { std::cout << element << " "; });
//	std::cout << std::endl;

	int64_t L = src_val_len.data().item<int64_t>() - 1;
	std::string out = "";
	for( int64_t i = 0; i < L; i++ ) {
		std::string tk = tgt_vocab.idx_to_token[idx[i]];
		if( out == "" )
			out = tk;
		else
			if( tk != "<eos>" ) out += (" " + tk);
	}
    return out;
}

// join list of strings
std::string join(int i, int n, std::vector<std::string> label_tokens) {
	std::string ret;
	for(int c = i; c < (i + n); c++) {
		if(! ret.empty())
			ret += " ";
		ret += label_tokens[c];
	}
	return(ret);
}

// ----------------------------------------------------------------------------
// Evaluation of Predicted Sequences
// We can evaluate a predicted sequence by comparing it with the label sequence
// (the ground-truth). BLEU (Bilingual Evaluation Understudy)
// ------------------------------------------------------------------------------
double bleu(std::string pred_seq, std::string label_seq, int64_t k) {
    //Compute the BLEU.
	std::vector<std::string> pred_tokens = tokenize({pred_seq}, "word", false);
	std::vector<std::string> label_tokens = tokenize({label_seq}, "word", false);

	int64_t len_pred = pred_tokens.size();
	int64_t len_label = label_tokens.size();
	double score = std::exp(std::min(0.0, 1.0 - (len_label*1.0 / len_pred)));

	for( int n = 1; n < (k + 1); n++ ) {
		int num_matches = 0;
		std::map<std::string, int> label_subs;
		for( int i = 0; i < (len_label - n + 1); i++ ) {
			std::string s = join(i, n, label_tokens);
			label_subs[s] += 1;
		}
		//std::cout << "-------------------------------------------\n";
		//std::for_each(std::begin(label_subs), std::end(label_subs), [](const auto & element) { std::cout << element << " "; });
		//std::cout << std::endl;

		for( int i = 0; i < (len_pred - n + 1); i++ ) {
			std::string s = join(i, n, label_tokens);
			if(label_subs[s] > 0 ){
				num_matches += 1;
				label_subs[s] -= 1;
			}
		}
		score *= std::pow(num_matches / (len_pred - n + 1), std::pow(0.5, n));
	}
	return score;
}


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
	int64_t num_epochs = 400;

	std::string filename = "./data/fra-eng/fra.txt";
	//std::pair<torch::Tensor, torch::Tensor> data_arrays;
	Vocab src_vocab, tgt_vocab;
	torch::Tensor src_array, src_valid_len, tgt_array, tgt_valid_len;

	std::tie(src_array, src_valid_len, tgt_array, tgt_valid_len, src_vocab, tgt_vocab) = load_data_nmt(filename, num_steps, 600);
	encoder = Seq2SeqEncoder(src_vocab.length(), embed_size, num_hiddens, num_layers, dropout);
	decoder = Seq2SeqDecoder(tgt_vocab.length(), embed_size, num_hiddens, num_layers, dropout);
	auto net = EncoderDecoder(encoder, decoder);

	//Train a model for sequence to sequence.
	net->apply(xavier_init_weights);
	net->to(device);
	//auto optimizer = torch::optim::Adam(net->parameters(), lr);
	torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(lr).betas({0.5, 0.999}));

	loss_fn = MaskedSoftmaxCELoss();
	net->train(true);

	std::vector<float> epochs, plsum;
	std::vector<int64_t> wtks;
	torch::Tensor Y_hat, stat;

	for( int64_t epoch = 0;  epoch < num_epochs; epoch++ ) {
		// get shuffled batch data indices
		std::list<torch::Tensor> idx_iter = data_index_iter(src_array.size(0), batch_size, true);

		for(auto& batch_idx : idx_iter) {

	        optimizer.zero_grad();

	        auto X = src_array.index_select(0, batch_idx);
	        auto X_valid_len = src_valid_len.index_select(0, batch_idx);
	        auto Y = tgt_array.index_select(0, batch_idx);
	        auto Y_valid_len = tgt_valid_len.index_select(0, batch_idx);
	        X.to(device);
	        X_valid_len.to(device);
	        Y.to(device);
	        Y_valid_len.to(device);

	        std::vector<int64_t> tmp;
	        for(int i = 0; i < Y.size(0); i++)
	        	tmp.push_back(tgt_vocab["<bos>"]);

	        auto options = torch::TensorOptions().dtype(torch::kLong);
	        torch::Tensor bos = torch::from_blob(tmp.data(), {1, Y.size(0)}, options).clone();
	        bos = bos.to(device).reshape({-1, 1});

	        torch::Tensor dec_input = torch::cat({bos, Y.index({Slice(), Slice(None, -1)})}, 1); // Y[:, :-1]

	        std::tie(Y_hat, stat) = net->forward(X, dec_input); // X_valid_len

	        auto l = loss_fn.forward(Y_hat, Y, Y_valid_len);
	        l.sum().backward(); 							    // Make the loss scalar for `backward`

	        //"""Clip the gradient."""
	        torch::nn::utils::clip_grad_norm_(net->parameters(), 1.0);
	        auto num_tokens = Y_valid_len.sum();

	        optimizer.step();
	        torch::NoGradGuard no_grad;
	        //torch::no_grad();
	        plsum.push_back((l.sum()).item<float>());
	        wtks.push_back(num_tokens.item<long>());
	        epochs.push_back(1.0*epoch);
	    }

	    if((epoch + 1) % 10 == 0)
	    	std::cout << "loss: " << plsum[epoch]/wtks[epoch] << std::endl;
	}

	plt::figure_size(800, 600);
	plt::named_plot("train", epochs, plsum, "b");
	plt::legend();
	plt::xlabel("epoch");
	plt::ylabel("loss");
	plt::show();
	plt::close();

	printf("\n\n");
	// Prediction
	std::vector<std::string> engs = {"go .", "i lost .", "he\'s calm .", "i\'m home ."};
	std::vector<std::string> fras = {"va !", "j\'ai perdu .", "il est calme .", "je suis chez moi ."};
	int t = 0;
	for(  ; t < engs.size(); t++ ) {
		std::string translation = predict_seq2seq(net, engs[t], src_vocab, tgt_vocab, num_steps, device);
		std::cout << "translation: " << translation << "\n";
		std::cout << "target: " << fras[t] << "\n";

		auto score = bleu(translation, fras[t],  2);
		std::cout << "Bleu score: " << score << "\n";
	}
	std::cout << "Done!\n";
	return 0;
}

