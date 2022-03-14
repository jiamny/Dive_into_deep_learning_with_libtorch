
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


#endif /* SRC_UTILS_CH_10_UTIL_H_ */
