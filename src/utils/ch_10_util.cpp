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
	PyObject* mat;

	plt::title("heatmap");
	plt::imshow(zptr, nrows, ncols, colors, {}, &mat);
	plt::xlabel(xlab);
	plt::ylabel(ylab);
	plt::colorbar(mat);
	plt::show();
    plt::close();
    Py_DECREF(mat);
}


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


