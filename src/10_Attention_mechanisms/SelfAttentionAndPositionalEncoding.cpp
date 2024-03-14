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
	std::cout << (cuda_available ? "CUDA available. Using GPU." : "Using CPU.") << '\n';

	torch::manual_seed(1000);

	int64_t num_hiddens = 100, num_heads = 5;
	auto attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
	                                   num_hiddens, num_heads, 0.5);
	attention->to(device);
	attention->eval();
	std::cout << attention << "\n";

	int64_t batch_size = 2, num_queries = 4;
	auto valid_lens =  torch::tensor({3, 2}).to(device);

	auto X = torch::ones({batch_size, num_queries, num_hiddens}).to(device);
	std::cout << attention->forward(X, X, X, valid_lens).sizes() << std::endl;

	int64_t encoding_dim = 32, num_steps = 60;

	auto pos_encoding = PositionalEncoding(encoding_dim, 0);
	pos_encoding->to(device);
	pos_encoding->eval();

	X = pos_encoding->forward(torch::zeros({1, num_steps, encoding_dim}).to(device));

	auto P = pos_encoding->P.index({Slice(), Slice(0, X.size(1)), Slice()});
	std::cout << "P: " << P.sizes() << "\n";

	auto xL = torch::arange(num_steps).to(torch::kDouble); //torch::arange(num_steps).to(torch::kFloat);
	std::vector<double> xx(xL.data_ptr<double>(), xL.data_ptr<double>() + xL.numel());

	auto yP = P.index({0, Slice(), Slice(6,10)}).clone().to(torch::kDouble).to(torch::kCPU);
	std::cout << "yP: " << yP.sizes() << "\n";

	auto yL = yP.index({Slice(), 0}).clone();
	std::vector<double> col6(yL.data_ptr<double>(), yL.data_ptr<double>() + yL.numel());

	yL = yP.index({Slice(), 1}).clone();
	std::vector<double> col7(yL.data_ptr<double>(), yL.data_ptr<double>() + yL.numel());

	yL = yP.index({Slice(), 2}).clone();
	std::vector<double> col8(yL.data_ptr<double>(), yL.data_ptr<double>() + yL.numel());

	yL = yP.index({Slice(), 3}).clone();
	std::vector<double> col9(yL.data_ptr<double>(), yL.data_ptr<double>() + yL.numel());

	std::for_each(std::begin(col7), std::end(col7), [](const auto & element) { std::cout << element << " "; });
	std::cout << std::endl;

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::plot(ax1,  xx, col6, "b")->line_width(2);
	matplot::plot(ax1, xx, col7, "g--")->line_width(2);
	matplot::plot(ax1, xx, col8, "r-.")->line_width(2);
	matplot::plot(ax1, xx, col9, "m:")->line_width(2);
    matplot::hold(ax1, false);
    matplot::xlabel(ax1, "Row (position)");
    matplot::legend(ax1, {"Col 6", "Col 7", "Col 8", "Col 9"});
    matplot::show();

	// Absolute Positional Information
	for( int64_t i = 0; i < 8; i++) {
		binary(i);
		std::cout << "\n";
	}

	auto tsr = P.index({0, Slice(), Slice()}).squeeze(0);
	std::cout << tsr.sizes() << "\n";
	std::string xlab = "Column (encoding dimension)";
	std::string ylab = "Row (position)";

	plot_heatmap(tsr, xlab, ylab);

	std::cout << "Done!\n";
	return 0;
}




