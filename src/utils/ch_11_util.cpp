#include "ch_11_util.h"
#include <string>

std::list<std::pair<torch::Tensor, torch::Tensor>> get_data_ch11(torch::Tensor X,
																		torch::Tensor Y, int64_t batch_size) {

	int64_t num_examples = X.size(0);
	std::list<std::pair<torch::Tensor, torch::Tensor>> batched_data;
	// data index
	std::vector<int64_t> index;
	for (int64_t i = 0; i < num_examples; ++i) {
			index.push_back(i);
	}
	std::random_shuffle(index.begin(), index.end());

	for (int64_t i = 0; i < index.size(); i +=batch_size) {
		std::vector<int64_t>::const_iterator first = index.begin() + i;
		std::vector<int64_t>::const_iterator last = index.begin() + std::min(i + batch_size, num_examples);
		std::vector<int64_t> indices(first, last);

		int64_t idx_size = indices.size();
		torch::Tensor idx = (torch::from_blob(indices.data(), {idx_size}, torch::kInt64)).clone();

		auto batch_x = X.index_select(0, idx);
		auto batch_y = Y.index_select(0, idx);

		batched_data.push_back(std::make_pair(batch_x, batch_y));
	}
	return batched_data;
}

double f_2d(double x1, double x2) {
    return 0.1 * x1 * x1 + 2 * x2 * x2;
}

void show_trace_2d( std::pair<std::vector<double>, std::vector<double>> rlt, std::string tlt ) {

//	std::for_each( rlt.first.begin(), rlt.first.end(), [](const auto & elem ) {std::cout << elem << " "; });
//	printf("\n");

	auto h = figure(true);
	h->size(800, 600);
	h->add_axes(false);
	h->reactive_mode(false);
	h->tiledlayout(1, 1);
	h->position(0, 0);

	auto ax = h->nexttile();
	matplot::plot(ax, rlt.first, rlt.second, "om-")->line_width(2);
	matplot::hold(ax, true);

	std::vector<std::vector<double>> x, y, z;
	for (double i = -5.5; i <= 1.0;  i += 0.1) {
	    std::vector<double> x_row, y_row, z_row;
	    for (double j = -3.0; j <= 1.0; j += 0.1) {
	            x_row.push_back(i);
	            y_row.push_back(j);
	            z_row.push_back(f_2d(i, j));
	    }
	    x.push_back(x_row);
	    y.push_back(y_row);
	    z.push_back(z_row);
	}

	matplot::contour(ax, x, y, z)->line_width(2);
	matplot::hold(ax, false);
	matplot::xlabel(ax, "x1");
	matplot::ylabel(ax, "x2");
	if( tlt.length() > 1 )
		matplot::title(ax, tlt);
	matplot::show();
}
