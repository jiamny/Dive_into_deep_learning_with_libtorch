
#ifndef SRC_08_RECURRENTNEURALNETWORKS_UTIL_H_
#define SRC_08_RECURRENTNEURALNETWORKS_UTIL_H_

#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <regex>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <map>
#include <set>
#include <random>

#include "../utils.h"

using torch::indexing::Slice;
using torch::indexing::None;


// The comparison function for sorting the set by increasing order of its pair's
// second value. If the second value is equal, order by the pair's first value
struct comp {
    template<typename T>
    bool operator()(const T &l, const T &r) const
    {
        if (l.second != r.second) {
            return l.second < r.second;
        }

        return l.first < r.first;
    }
};


std::vector<std::string> read_time_machine( std::string filename );

std::vector<std::pair<std::string, int64_t>> count_corpus( std::vector<std::string> tokens );

std::vector<std::string> tokenize(const std::vector<std::string> lines, const std::string token, bool max_cut=false);

std::vector<std::string> tokenize_str(const std::string line, char delim = ' ');

// Vocabulary
/*
 * build a dictionary, often called vocabulary as well, to map string tokens into numerical indices starting from 0
 * we first count the unique tokens in all the documents from the training set, namely a corpus, and then assign a numerical
 * index to each unique token according to its frequency. Rarely appeared tokens are often removed to reduce the complexity.
 * Any token that does not exist in the corpus or has been removed is mapped into a special unknown token “<unk>”.
 * We optionally add a list of reserved tokens, such as “<pad>” for padding, “<bos>” to present the beginning for a sequence,
 * and “<eos>” for the end of a sequence.
 */

class Vocab {

public:
	std::map<int64_t, std::string> idx_to_token;
	std::set<std::pair<std::string, int64_t>, comp> order_token_to_idx;

	Vocab() {}	// default constructor

    //Vocabulary for text.
	Vocab(std::vector<std::pair<std::string, int64_t>> corpus, float min_freq,
			std::vector<std::string> reserved_tokens);

	Vocab(std::vector<std::pair<std::string, int64_t>> token_freqs);

	~Vocab(void) {}

	int64_t length(void);

    int64_t unk(void);

    std::vector<std::string> to_tokens( std::vector<int64_t> indices );

    std::vector<std::pair<std::string, int64_t>> token_freqs(void);

    // Overload the [] operator
    int64_t operator [] (const std::string s);
    std::vector<int64_t> operator [] (const std::vector<std::string> ss );

private:
    std::map<std::string, int64_t> token_to_idx;
    std::vector<std::pair<std::string, int64_t>> _token_freqs;
};


// Reading Long Sequence Data
// Random Sampling
template<typename T>
std::vector<std::pair<torch::Tensor, torch::Tensor>>  seq_data_iter_random(std::vector<T> my_seq, int batch_size, int num_steps) {
    // Generate a minibatch of subsequences using random sampling
    // Start with a random offset (inclusive of `num_steps - 1`) to partition a sequence
	std::random_device rd;
	std::default_random_engine eng(rd());
	std::uniform_int_distribution<int> distr(0, num_steps - 1);
	//std::cout << distr(eng) << std::endl;

	std::vector<int> sub_seq;
	for( int i = distr(eng); i < my_seq.size(); i++ ) {
		sub_seq.push_back(my_seq[i]);
	}

	//Subtract 1 since we need to account for labels
    int num_subseqs = static_cast<int>((my_seq.size() - 1) * 1.0 / num_steps);

    //The starting indices for subsequences of length `num_steps`
    std::vector<int> initial_indices;
    //= list(range(0, num_subseqs * num_steps, num_steps))
//    std::cout << "initial_indices:\n";
    for(int i  = 0; i < num_subseqs; i ++ ) {
    	auto idx = i * num_steps;
    	initial_indices.push_back(idx);
//    	std::cout << idx << " ";
    }
//    printf("\n");

    //In random sampling, the subsequences from two adjacent random
    //minibatches during iteration are not necessarily adjacent on the  original sequence
    std::random_shuffle(initial_indices.begin(), initial_indices.end());

    //std::copy(initial_indices.begin(), initial_indices.end(), std::ostream_iterator<int>(std::cout, " "));
    //std::cout << "\n";

    int64_t num_batches = static_cast<int>(num_subseqs * 1.0 / batch_size);
    //std::cout << "num_batches: " << num_batches << std::endl;

    std::vector<int64_t> batch_initial_indices;

//    std::cout << "batch_initial_indices:\n";
    for(int i  = 0; i < num_batches; i ++ ) {
    	//Here, `initial_indices` contains randomized starting indices for subsequences
//    	std::cout << i * batch_size << std::endl;
    	batch_initial_indices.push_back( i * batch_size );
    }
//    printf("\n");

    auto opts = torch::TensorOptions().dtype(torch::kInt);

    std::vector<std::pair<torch::Tensor, torch::Tensor>> outpairs;

    for(int i  = 0; i < batch_initial_indices.size(); i ++ ) {
    	torch::Tensor X = torch::zeros({batch_size, num_steps}, opts);
    	torch::Tensor Y = torch::zeros({batch_size, num_steps}, opts);

    	for(int r = 0; r < batch_size; r++ )
    		for(int j = initial_indices[i + r], c= 0; j < (initial_indices[i + r] + num_steps); j++, c++ )
    			X.index({r, c}) = my_seq[j];
    	//std::cout << X.sizes() << std::endl;

    	for(int r = 0; r < batch_size; r++ )
    		for(int j = initial_indices[i + r] + 1, c=0; j < (initial_indices[i + r] + 1 + num_steps); j++, c++ )
    			Y.index({r, c}) = my_seq[j];

    	outpairs.push_back({X.clone().to(torch::kInt64), Y.clone().to(torch::kInt64)});
    }

	return outpairs;
}

// Sequential Partitioning
/*
 * we can also ensure that the subsequences from two adjacent minibatches during iteration are adjacent on the original sequence
 */
template<typename T>
std::vector<std::pair<torch::Tensor, torch::Tensor>>  seq_data_iter_sequential(std::vector<T> my_seq, int batch_size, int num_steps) {
    //Generate a minibatch of subsequences using sequential partitioning.
    // Start with a random offset to partition a sequence
	std::random_device rd;
	std::default_random_engine eng(rd());
	std::uniform_int_distribution<int> distr(0, num_steps);
    int offset = distr(eng);

    int num_tokens = static_cast<int>(((my_seq.size() - offset - 1)*1.0 / batch_size)) * batch_size;

    auto opts = torch::TensorOptions().dtype(torch::kInt);

    std::vector<std::pair<torch::Tensor, torch::Tensor>> outpairs;

    torch::Tensor Xs = torch::zeros({num_tokens}, opts);
    torch::Tensor Ys = torch::zeros({num_tokens}, opts);

    for( int i = 0; i < num_tokens; i++ ) {
    	Xs.index({i}) = my_seq[offset + i];
    	Ys.index({i}) = my_seq[offset + 1 + i];
    }
    Xs = Xs.reshape({batch_size,-1});
    Ys = Ys.reshape({batch_size,-1});

	int num_batches = static_cast<int>(Xs.size(1) * 1.0 / num_steps);   //.shape[1] // num_steps

    for(int i = 0; i < num_batches; i++ ) {
    	int idx = i * num_steps;
        auto X = Xs.index({Slice(None),  Slice(i, i + num_steps)});
        auto Y = Ys.index({Slice(None),  Slice(i, i + num_steps)});
        outpairs.push_back({X.clone().to(torch::kInt64), Y.clone().to(torch::kInt64)});
    }

	return outpairs;
}


//================================================
// Prediction
//================================================
template<typename T>
std::string predict_ch9(std::vector<std::string> prefix, int64_t num_preds, T net, Vocab vocab, torch::Device device) {
    //"""Generate new characters following the `prefix`."""
	std::tuple<torch::Tensor, torch::Tensor> state = net.begin_state(1, device);

//	std::cout << "H: " << std::get<0>(state).sizes() << std::endl;
//	std::cout << "C: " << std::get<1>(state).sizes() << std::endl;

    std::vector<int64_t> outputs;
    outputs.push_back(vocab[prefix[0]]);

    for( int i = 1; i < prefix.size(); i ++ ) { //# Warm-up period
    	std::string y = prefix[i];
    	torch::Tensor xx;
    	// get_inpt
    	torch::Tensor X = torch::tensor({{outputs[i-1]}}, device).reshape({1,1});

    	std::tie(xx, state) = net.forward(X, state);

    	outputs.push_back(vocab[prefix[i]]);
    }
    //for y in prefix[1:]:  			//# Warm-up period
    //    _, state = net(get_input(), state)
    //    outputs.append(vocab[y])

    for( int i = 0; i < num_preds; i ++ ) {
    	torch::Tensor y;

    	int j = outputs.size();
    	torch::Tensor X = torch::tensor({{outputs[j-1]}}, device).reshape({1,1});

    	std::tie(y, state) = net.forward(X, state);
    	// ---------------------------------------------------
    	// transfer data to CPU
    	// ---------------------------------------------------
    	outputs.push_back(static_cast<int>(y.argmax(1, 0).reshape({1}).to(torch::kCPU).item<int>()));
    }
    //for _ in range(num_preds):  	//# Predict `num_preds` steps
    //    y, state = net(get_input(), state)
    //    outputs.append(int(y.argmax(dim=1).reshape(1)))
    std::string pred("");
    for( auto& p : outputs )
    	pred += vocab.idx_to_token[p];

    return pred; // ''.join([vocab.idx_to_token[i] for i in outputs]);
}


template<typename T>
std::pair<double, double> train_epoch_ch9(T& net, std::vector<std::pair<torch::Tensor, torch::Tensor>> train_iter,
		torch::nn::CrossEntropyLoss loss, torch::optim::Optimizer& updater, torch::Device device,
		float lr, bool use_random_iter) {
	//***********************************************
	//two ways of setting the precision
	std::streamsize ss = std::cout.precision();
	std::cout.precision(15);
	// another way std::setprecision(N)

	double ppx = 0.0;
	int64_t tot_tk = 0;

	precise_timer timer;
	std::tuple<torch::Tensor, torch::Tensor> state = std::make_tuple(torch::empty({0}).to(device),
			torch::empty({0}).to(device));

	for( int i = 0; i < train_iter.size(); i++ ) {
	    auto X = train_iter[i].first.to(device);
	    auto Y = train_iter[i].second.to(device);

	    if( std::get<0>(state).numel() == 0 || use_random_iter ) {
	    	state = net.begin_state(X.size(0), device);
	    } else {
	    	std::get<0>(state).detach_();
	    	std::get<1>(state).detach_();
	    }

	    torch::Tensor y = Y.transpose(0, 1).reshape({-1});
	    X = X.to(device),
	    y = y.to(device);
	    torch::Tensor y_hat;
	    std::tie(y_hat, state) = net.forward(X, state);

	    auto l = loss(y_hat, y.to(torch::kLong)).mean();

	    updater.zero_grad();
	    l.backward();

	    //"""Clip the gradient."""
	    torch::nn::utils::clip_grad_norm_(net.parameters(), 0.5);

	    updater.step();
    	// ---------------------------------------------------
    	// transfer data to CPU
    	// ---------------------------------------------------
	    ppx += (l.data().to(torch::kCPU).item<float>() * y.to(torch::kCPU).numel());
	    tot_tk += y.to(torch::kCPU).numel();
	}
	unsigned int dul = timer.stop<unsigned int, std::chrono::microseconds>();
	auto t = (dul/1000000.0);

	//****************************************
	//change back the precision
	std::cout.precision(ss);
	// another way std::setprecision(ss)

	return { std::exp(ppx*1.0 / tot_tk), (tot_tk * 1.0 / t) };
}

template<typename T>
std::pair<std::vector<double>, std::vector<double>> train_ch9(T& net, std::vector<std::pair<torch::Tensor, torch::Tensor>> train_iter,
		Vocab vocab, torch::Device device, float lr, int64_t num_epochs, bool use_random_iter) {

	std::string s = "time traveller", s2 = "traveller";
	std::vector<char> v(s.begin(), s.end()), v2(s2.begin(), s2.end());
	std::vector<std::string> prefix, prefix2;
	for(int i = 0; i < v.size(); i++ ) {
	    std::string tc(1, v[i]);
	    prefix.push_back(tc);
	}

	for(int i = 0; i < v2.size(); i++ ) {
		std::string tc(1, v[i]);
		prefix2.push_back(tc);
	}

    //"""Train a model (defined in Chapter 8)."""
	auto loss = torch::nn::CrossEntropyLoss();
	auto updater = torch::optim::SGD(net.parameters(), lr);

    std::vector<double> epochs, ppl;
    // Train and predict
    double eppl, speed;
    for(int64_t  epoch = 0; epoch < num_epochs; epoch++ ) {

        auto rlt = train_epoch_ch9(net, train_iter, loss, updater, device, lr, use_random_iter);
        eppl = rlt.first;
        speed = rlt.second;
        if( (epoch + 1) % 10 == 0 ) {
        	std::cout << predict_ch9(prefix, 50, net, vocab, device) << std::endl;

            ppl.push_back(eppl);
            epochs.push_back((epoch+1)*1.0);
        }
    }

    printf("perplexity: %.1f, speed: %.1f tokens/sec on %s\n", eppl, speed, device.str().c_str());
    std::cout << predict_ch9(prefix, 50, net, vocab, device) << std::endl;
    std::cout << predict_ch9(prefix2, 50, net, vocab, device) << std::endl;

    return {epochs, ppl};
}

std::string read_data_nmt(const std::string filename);

bool no_space(char c, char pc);

std::string preprocess_nmt( std::string raw_test);

std::tuple<std::vector<std::vector<std::string>>, std::vector<std::vector<std::string>>> tokenize_nmt(std::string processed,
																										size_t num_examples);

std::tuple<torch::Tensor, torch::Tensor> build_array_nmt(std::vector<std::vector<std::string>> lines,
														 Vocab vocab, int num_steps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, Vocab, Vocab> load_data_nmt(std::string filename,
														 int num_steps, int num_examples);

torch::Tensor sequence_mask(torch::Tensor X, torch::Tensor  valid_len, float value=0);

// --------------------------------------------
// Masked Softmax Operation
// --------------------------------------------
torch::Tensor masked_softmax(torch::Tensor X, torch::Tensor valid_lens);

void xavier_init_weights(torch::nn::Module &m);

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
    	weights = torch::ones_like(label).to(pred.device());
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


// ---------------------------
// Seq2Seq Encoder
// ---------------------------
struct Seq2SeqEncoderImpl : public torch::nn::Module {
	torch::nn::Embedding embedding{nullptr};
	torch::nn::GRU rnn{nullptr};
    //The RNN encoder for sequence to sequence learning.
	Seq2SeqEncoderImpl(int64_t vocab_size, int64_t embed_size, int64_t num_hiddens, int64_t num_layers, float dropout=0){
        // Embedding layer
        embedding = torch::nn::Embedding(vocab_size, embed_size);
        rnn = torch::nn::GRU(torch::nn::GRUOptions(embed_size, num_hiddens).num_layers(num_layers).dropout(dropout)); //embed_size, num_hiddens, num_layers, dropout
        register_module("Eembedding", embedding);
        register_module("Ernn", rnn);
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

std::string join(int i, int j, std::vector<std::string> label_tokens);

double bleu(std::string pred_seq, std::string label_seq, int64_t k);

#endif /* SRC_08_RECURRENTNEURALNETWORKS_UTIL_H_ */
