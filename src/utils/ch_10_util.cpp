#include "ch_10_util.h"

void plot_heatmap(torch::Tensor tsr, std::string xlab, std::string ylab, std::string tlt) {
	tsr = tsr.cpu().squeeze().to(torch::kDouble);
	int nrows = tsr.size(0), ncols = tsr.size(1);

	std::vector<std::vector<double>> C;
	for( int i = 0; i < nrows; i++ ) {
		std::vector<double> c;
		for( int j = 0; j < ncols; j++ ) {
			c.push_back(tsr[i][j].item<double>());
		}
		C.push_back(c);
	}

	auto h = figure(true);
	h->size(800, 600);
	h->add_axes(false);
	h->reactive_mode(false);
	h->tiledlayout(1, 1);
	h->position(0, 0);

	auto ax = h->nexttile();
	//ax->axis(false);
	matplot::heatmap(ax, C);
	matplot::colorbar(ax);
    matplot::xlabel(ax, xlab);
    matplot::ylabel(ax, ylab);
    if( tlt.length() > 2 ) {
    	matplot::title(ax, tlt.c_str());
    } else {
    	matplot::title(ax, "heatmap");
    }
    matplot::show();
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


