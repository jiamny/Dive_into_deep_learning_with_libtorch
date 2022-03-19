
#ifndef SRC_UTILS_CH_10_UTIL_H_
#define SRC_UTILS_CH_10_UTIL_H_

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <iostream>
#include <iomanip>

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;

void plot_heatmap(torch::Tensor tsr, std::string xlab, std::string ylab);

torch::Tensor sequence_mask(torch::Tensor X, torch::Tensor  valid_len, float value=0);

// --------------------------------------------
// Masked Softmax Operation
// --------------------------------------------
torch::Tensor masked_softmax(torch::Tensor X, torch::Tensor valid_lens);

// -----------------------------------------------
// Scaled Dot-Product Attention
// -----------------------------------------------
struct DotProductAttention : public torch::nn::Module {
    // Scaled dot product attention.
	DotProductAttention() {}
	DotProductAttention(float dropout) {
        dpout = torch::nn::Dropout(dropout);
	}

    // Shape of `queries`: (`batch_size`, no. of queries, `d`)
    // Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    // Shape of `values`: (`batch_size`, no. of key-value pairs, value dimension)
    // Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)
    torch::Tensor forward(torch::Tensor queries, torch::Tensor keys, torch::Tensor values, torch::Tensor valid_lens) {
        int n_shape = (queries.sizes()).size();
    	auto d = queries.sizes()[n_shape - 1];
        // Set `transpose_b=True` to swap the last two dimensions of `keys`
        auto scores = torch::bmm(queries, keys.transpose(1, 2)) / std::sqrt(d);
        attention_weights = masked_softmax(scores, valid_lens);
        return torch::bmm(dpout(attention_weights), values);
    }
    torch::nn::Dropout dpout{nullptr};
    torch::Tensor attention_weights;
};

// ---------------------------------------------
// Additive Attention]
// ---------------------------------------------
struct AdditiveAttention : public torch::nn::Module {
    //Additive attention
	AdditiveAttention() {}
    AdditiveAttention(int64_t key_size, int64_t query_size, int64_t num_hiddens, float dropout) {
        W_k = torch::nn::Linear(torch::nn::LinearOptions(key_size, num_hiddens).bias(false));
        W_q = torch::nn::Linear(torch::nn::LinearOptions(query_size, num_hiddens).bias(false));
        W_v = torch::nn::Linear(torch::nn::LinearOptions(num_hiddens, 1).bias(false));
        dpout = torch::nn::Dropout(dropout);
    }
    ~AdditiveAttention() {}

    torch::Tensor forward(torch::Tensor queries, torch::Tensor keys, torch::Tensor values, torch::Tensor valid_lens) {
        queries = W_q->forward(queries);
		keys = W_k->forward(keys);
        // After dimension expansion, shape of `queries`: (`batch_size`, no. of
        // queries, 1, `num_hiddens`) and shape of `keys`: (`batch_size`, 1,
        // no. of key-value pairs, `num_hiddens`). Sum them up with broadcasting

//		std::cout << "queries: " << queries.sizes() << "\n";
//		std::cout << "keys: " << keys.sizes() << "\n";
        auto features = queries.unsqueeze(2) + keys.unsqueeze(1);
        features = torch::tanh(features);

//        std::cout << "features: " << features.sizes() << "\n";
		// There is only one output of `self.w_v`, so we remove the last
        // one-dimensional entry from the shape. Shape of `scores`:
        // (`batch_size`, no. of queries, no. of key-value pairs)
        auto scores = W_v->forward(features).squeeze(-1); //squeeze() 不加参数的，把所有为1的维度都压缩
//        std::cout << "scores: " << scores.sizes() << "\n";
//        std::cout << "valid_lens: " << valid_lens.numel() << "\n";

        attention_weights = masked_softmax(scores, valid_lens);
//        std::cout << "attention_weights: " << attention_weights.sizes() << "\n";

        // Shape of `values`: (`batch_size`, no. of key-value pairs, value dimension)
        return torch::bmm(dpout->forward(attention_weights), values);
    }
    torch::nn::Linear W_k{nullptr}, W_q{nullptr}, W_v{nullptr};
    torch::nn::Dropout dpout{nullptr};
    torch::Tensor attention_weights;
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

// -------------------------------------
torch::Tensor transpose_qkv(torch::Tensor X, int64_t num_heads);
torch::Tensor transpose_output(torch::Tensor X, int64_t num_heads);

// -------------------------------------
// Model Implementation
// -------------------------------------
struct MultiHeadAttention : public torch::nn::Module {
	int64_t num_heads;
	DotProductAttention attention;
	torch::nn::Linear W_k{nullptr}, W_q{nullptr}, W_v{nullptr}, W_o{nullptr};

    //Multi-head attention.
	MultiHeadAttention(int64_t key_size, int64_t query_size, int64_t value_size, int64_t num_hiddens,
                 int64_t n_heads, float dropout, bool bias=false) {

        num_heads = n_heads;
        attention = DotProductAttention(dropout);
        W_q = torch::nn::Linear(torch::nn::LinearOptions(query_size, num_hiddens).bias(bias));
        W_k = torch::nn::Linear(torch::nn::LinearOptions(key_size, num_hiddens).bias(bias));
        W_v = torch::nn::Linear(torch::nn::LinearOptions(value_size, num_hiddens).bias(bias));
        W_o = torch::nn::Linear(torch::nn::LinearOptions(num_hiddens, num_hiddens).bias(bias));
        register_module("W_q",W_q);
        register_module("W_k",W_k);
        register_module("W_v",W_v);
        register_module("W_o",W_o);
	}

    torch::Tensor forward(torch::Tensor queries, torch::Tensor keys, torch::Tensor values, torch::Tensor valid_lens) {
        // Shape of `queries`, `keys`, or `values`:
        // (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        // Shape of `valid_lens`:
        // (`batch_size`,) or (`batch_size`, no. of queries)
        // After transposing, shape of output `queries`, `keys`, or `values`:
        // (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        // `num_hiddens` / `num_heads`)
        queries = transpose_qkv(W_q->forward(queries), num_heads);
        keys    = transpose_qkv(W_k->forward(keys), num_heads);
        values  = transpose_qkv(W_v->forward(values), num_heads);

        if( valid_lens.defined() ) {
            // On axis 0, copy the first item (scalar or vector) for
            // `num_heads` times, then copy the next item, and so on
            valid_lens = torch::repeat_interleave(valid_lens, /*repeats=*/num_heads, /*dim=*/0);
        }

        // Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        // `num_hiddens` / `num_heads`)
        auto output = attention.forward(queries, keys, values, valid_lens);

        // Shape of `output_concat`:
        // (`batch_size`, no. of queries, `num_hiddens`)
        auto output_concat = transpose_output(output, num_heads);
        return W_o->forward(output_concat);
    }
};


#endif /* SRC_UTILS_CH_10_UTIL_H_ */
