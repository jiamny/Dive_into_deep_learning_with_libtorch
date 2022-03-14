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

// -------------------------------------
// To allow for [parallel computation of multiple heads], the above MultiHeadAttention class uses two transposition
// functions as defined below. Specifically, the transpose_output function reverses the operation of the transpose_qkv function.
// -------------------------------------
torch::Tensor transpose_qkv(torch::Tensor X, int64_t num_heads) {
    // Transposition for parallel computation of multiple attention heads.
    // Shape of input `X`:
    // (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    // Shape of output `X`:
    // (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    // `num_hiddens` / `num_heads`)
    X = X.reshape({X.size(0), X.size(1), num_heads, -1});

    // Shape of output `X`:
    // (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    // `num_hiddens` / `num_heads`)
    X = X.permute({0, 2, 1, 3});

    // Shape of `output`:
    // (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    // `num_hiddens` / `num_heads`)
    return X.reshape({-1, X.size(2), X.size(3)});
}


torch::Tensor transpose_output(torch::Tensor X, int64_t num_heads) {
    // Reverse the operation of `transpose_qkv`.
    X = X.reshape({-1, num_heads, X.size(1), X.size(2)});
    X = X.permute({0, 2, 1, 3});
    return X.reshape({X.size(0), X.size(1), -1});
}

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


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	// Test MultiHeadAttention
	int64_t num_hiddens = 100, num_heads = 5;

	/*
	auto queries = torch::normal(0, 1, {10, 4, 20});
	auto keys = torch::normal(0, 1, {10, 6, 20});
	auto values = torch::normal(0, 1, {10, 6, 20});
	auto valid_lens = torch::tensor({3,2});
	std::cout << valid_lens.defined() << std::endl;

	if( valid_lens.defined() ) {
		valid_lens = torch::repeat_interleave(valid_lens, num_heads, 0);
	}

	auto dattention = DotProductAttention(0.5);
	dattention.eval();

	std::cout << "queries:\n" << queries.sizes() << std::endl;
	std::cout << "keys:\n" << keys.sizes() << std::endl;
	std::cout << "values:\n" << values.sizes() << std::endl;
	std::cout << "valid_lens:\n" << valid_lens.sizes() << std::endl;

	auto DA = dattention.forward(queries, keys, values, valid_lens);
	std::cout << "demonstrate DotProductAttention class:\n" << DA.sizes() << std::endl;
	*/

	auto attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
	                               num_hiddens, num_heads, 0.5);
	attention.eval();

	std::cout << attention << std::endl;

	int64_t batch_size = 2, num_queries = 4, num_kvpairs = 6;
	auto valid_lens = torch::tensor({3, 2});

	auto X = torch::ones({batch_size, num_queries, num_hiddens});
	auto Y = torch::ones({batch_size, num_kvpairs, num_hiddens});
	std::cout << attention.forward(X, Y, Y, valid_lens).sizes() << std::endl;

	std::cout << "Done!\n";
	return 0;
}





