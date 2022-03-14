#include "ch_10_util.h"

void plot_heatmap(torch::Tensor tsr, std::string xlab, std::string ylab) {
	int nrows = tsr.size(0), ncols = tsr.size(1);

	std::vector<float> z(ncols * nrows);
	for( int j=0; j<nrows; ++j ) {
	    for( int i=0; i<ncols; ++i ) {
	            z.at(ncols * j + i) = (tsr.index({j, i})).item<float>();
	     }
	}

	const float* zptr = &(z[0]);
	const int colors = 1;

	plt::title("heatmap");
	plt::imshow(zptr, nrows, ncols, colors);
	plt::xlabel(xlab);
	plt::ylabel(ylab);
	//plt::colorbar();
	plt::show();
}

torch::Tensor sequence_mask(torch::Tensor X, torch::Tensor  valid_len, float value) {
    //Mask irrelevant entries in sequences.
    int64_t maxlen = X.size(1);
    auto mask = torch::arange((maxlen),
    		torch::TensorOptions().dtype(torch::kFloat32).device(X.device())).index({None, Slice()}) < valid_len.index({Slice(), None});

    // (if B - boolean tensor) at::Tensor not_B = torch::ones_like(B) ^ B;
    // std::cout << (torch::ones_like(mask) ^ mask).sizes() <<std::endl;
    X.index_put_({torch::ones_like(mask) ^ mask}, value);

    return X;
}

// --------------------------------------------
// Masked Softmax Operation
// --------------------------------------------
torch::Tensor masked_softmax(torch::Tensor X, torch::Tensor valid_lens) {
    // Perform softmax operation by masking elements on the last axis.
    // `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if( ! valid_lens.defined()) { 								// None
        return torch::nn::functional::softmax(X, /*dim=*/-1);
    } else {
        auto shape = X.sizes();

        if( valid_lens.dim() == 1) {
            valid_lens = torch::repeat_interleave(valid_lens, shape[shape.size() - 2]);
        } else {
            valid_lens = valid_lens.reshape(-1);
        }

        // On the last axis, replace masked elements with a very large negative value, whose exponentiation outputs 0
        //std::cout << X.reshape({-1, shape[shape.size() - 1]}).sizes()  << "\n";
        X = sequence_mask(X.reshape({-1, shape[shape.size() - 1]}), valid_lens, /*value=*/ -1e6);

        return torch::nn::functional::softmax(X.reshape(shape), /*dim=*/-1);

    }
}

