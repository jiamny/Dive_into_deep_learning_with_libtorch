#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../utils/ch_10_util.h"

using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;

void binary(int64_t n) {
    /* step 1 */
    if (n > 1)
        binary(n / 2);
    /* step 2 */
    std::cout << n % 2;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	int64_t num_hiddens = 100, num_heads = 5;
	auto attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
	                                   num_hiddens, num_heads, 0.5);
	attention->eval();
	std::cout << attention << "\n";

	int64_t batch_size = 2, num_queries = 4;
	auto valid_lens =  torch::tensor({3, 2});

	auto X = torch::ones({batch_size, num_queries, num_hiddens});
	std::cout << attention->forward(X, X, X, valid_lens).sizes() << std::endl;

	int64_t encoding_dim = 32, num_steps = 60;

	auto pos_encoding = PositionalEncoding(encoding_dim, 0);
	pos_encoding->eval();
	X = pos_encoding->forward(torch::zeros({1, num_steps, encoding_dim}).to(device));

	auto P = pos_encoding->P.index({Slice(), Slice(0, X.size(1)), Slice()});
	std::cout << "P: " << P.sizes() << "\n";

	auto xL = torch::arange(num_steps).to(torch::kFloat); //torch::arange(num_steps).to(torch::kFloat);
	std::vector<float> xx(xL.data_ptr<float>(), xL.data_ptr<float>() + xL.numel());
	auto yP = P.index({0, Slice(), Slice(6,10)}).clone();

	std::cout << "yP: " << yP.sizes() << "\n";

	auto yL = yP.index({Slice(), 0}).clone();
	std::vector<float> col6(yL.data_ptr<float>(), yL.data_ptr<float>() + yL.numel());
	yL = yP.index({Slice(), 1}).clone();
	std::vector<float> col7(yL.data_ptr<float>(), yL.data_ptr<float>() + yL.numel());
	yL = yP.index({Slice(), 2}).clone();
	std::vector<float> col8(yL.data_ptr<float>(), yL.data_ptr<float>() + yL.numel());
	yL = yP.index({Slice(), 3}).clone();
	std::vector<float> col9(yL.data_ptr<float>(), yL.data_ptr<float>() + yL.numel());

	std::for_each(std::begin(col7), std::end(col7), [](const auto & element) { std::cout << element << " "; });
	std::cout << std::endl;

	plt::figure_size(800, 600);
	plt::named_plot("Col 6", xx, col6, "b");
	plt::named_plot("Col 7", xx, col7, "g--");
	plt::named_plot("Col 8", xx, col8, "r-.");
	plt::named_plot("Col 9", xx, col9, "c:");
	plt::xlabel("Row (position)");
	plt::legend();
	plt::show();
	plt::close();


	// Absolute Positional Information
	for( int64_t i = 0; i < 8; i++) {
		binary(i);
		std::cout << "\n";
	}

	auto tsr = P.index({0, Slice(), Slice()}).squeeze(0);
	std::cout << tsr.sizes() << "\n";
	std::string xlab = "Column (encoding dimension)";
	std::string ylab = "Row (position)";
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

	std::cout << "Done!\n";
	return 0;
}




