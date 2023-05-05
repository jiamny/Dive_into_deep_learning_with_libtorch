#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../utils/ch_10_util.h"

// Define some kernels
torch::Tensor gaussian(torch::Tensor x) {
    return torch::exp(-1* torch::pow(x, 2) / 2);
}

torch::Tensor  boxcar(torch::Tensor x) {
    return torch::abs(x) < 1.0;
}

torch::Tensor  constant(torch::Tensor x) {
    return 1.0 + 0 * x;
}

torch::Tensor epanechikov(torch::Tensor x) {
    return torch::max(1 - torch::abs(x), torch::zeros_like(x));
}

torch::Tensor f(torch::Tensor x) {
    return 2 * torch::sin(x) + x;
}

torch::Tensor gaussian_with_width(torch::Tensor x, double sigma) {
	    return torch::exp(-1*x.pow(2) / (2*std::pow(sigma,2)));
}

torch::Tensor (*functptr[])(torch::Tensor) = { &gaussian, &boxcar, &constant, &epanechikov } ;

std::pair<torch::Tensor, torch::Tensor> nadaraya_watson(torch::Tensor x_train, torch::Tensor y_train,
		torch::Tensor x_val, int j, std::vector<double> sigma ) {
    auto dists = x_train.reshape({-1, 1}) - x_val.reshape({1, -1});
    // Each column/row corresponds to each query/key
    torch::Tensor k;
    if( sigma.size() > 0 )
    	k = gaussian_with_width(dists, sigma[j]);
    else
    	k = (*functptr[j])(dists).to(dists.dtype());

    // Normalization over keys for each query
    torch::Tensor attention_w = k / k.sum(0);
    torch::Tensor y_hat = torch::matmul(y_train, attention_w);		//y_train@attention_w;
    return {y_hat, attention_w};
}

void plot(torch::Tensor x_train, torch::Tensor y_train, torch::Tensor x_val, torch::Tensor y_val,
							std::vector<std::string> names, bool attention=false, std::vector<double> sigma = {}) {

	auto f = figure(true);
	f->width(f->width() * 2.2);
	f->height(f->height() * 1.2);
	f->x_position(0);
	f->y_position(0);

	x_val = x_val.to(torch::kDouble);
	std::vector<double> xx(x_val.data_ptr<double>(), x_val.data_ptr<double>()+x_val.numel());
	y_val = y_val.to(torch::kDouble);
	std::vector<double> yy(y_val.data_ptr<double>(), y_val.data_ptr<double>()+y_val.numel());

	x_train = x_train.to(torch::kDouble);
	y_train = y_train.to(torch::kDouble);
	std::vector<double> xt(x_train.data_ptr<double>(), x_train.data_ptr<double>()+x_train.numel());
	std::vector<double> yt(y_train.data_ptr<double>(), y_train.data_ptr<double>()+y_train.numel());

	if( ! attention ) matplot::legend();

	torch::Tensor y_hat, attention_w;
	for( int i = 0; i < names.size(); i++ ) {

		std::tie(y_hat, attention_w) = nadaraya_watson(x_train, y_train, x_val, i, sigma);

		y_hat = y_hat.to(torch::kDouble);
		attention_w = attention_w.to(torch::kDouble);

		matplot::subplot(1, 4, i);
		if(attention) {
			attention_w = attention_w.cpu().squeeze().to(torch::kDouble);

			int nrows = attention_w.size(0), ncols = attention_w.size(1);

			std::vector<std::vector<double>> C;
			for( int i = 0; i < nrows; i++ ) {
				std::vector<double> c;
				for( int j = 0; j < ncols; j++ ) {
					c.push_back(attention_w[i][j].item<double>());
				}
				C.push_back(c);
			}
			matplot::heatmap(C);

		} else {
			std::vector<double> yh(y_hat.data_ptr<double>(), y_hat.data_ptr<double>()+y_hat.numel());
			matplot::hold(true);
			matplot::plot(xx, yh)->line_width(2)
					.display_name("y_hat");
			matplot::plot(xx, yy, "m--")->line_width(2)
					.display_name("y");
			matplot::scatter(xt, yt, 6);
			matplot::hold(false);
		}
		matplot::title(names[i]);
	}

	if(attention)
	    matplot::colorbar();

	matplot::show();
}



int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	torch::Device device(torch::kCPU);

	torch::manual_seed(1000);

	std::vector<std::string> names = {"Gaussian", "Boxcar", "Constant", "Epanechikov"};

	auto x = torch::arange(-2.5, 2.5, 0.1).to(torch::kDouble);

	std::cout << x.sizes() <<'\n';


	auto h = figure(true);
	h->width(h->width() * 2.2);
	h->height(h->height() * 1.2);
	h->x_position(0);
	h->y_position(0);

	std::vector<double> xx(x.data_ptr<double>(), x.data_ptr<double>()+x.numel());
	for( int i = 0; i < names.size(); i++ ) {
			matplot::subplot(1, 4, i);
			torch::Tensor y = (*functptr[i])(x);
			y = y.to(torch::kDouble);
			std::vector<double> yy(y.data_ptr<double>(), y.data_ptr<double>()+y.numel());
			matplot::plot(xx, yy)->line_width(2);
			matplot::title(names[i]);
		}
		matplot::show();


	int  n = 40;
	auto r = torch::sort(torch::rand(n) * 5);
	auto x_train = std::get<0>(r);
	auto y_train = f(x_train) + torch::randn(n);
	torch::Tensor x_val = torch::arange(0, 5, 0.1);
	torch::Tensor y_val = f(x_val);

	// ------------------------------------------------------------------------------
	// Attention Pooling via Nadaraya-Watson Regression
	// ------------------------------------------------------------------------------
	plot(x_train, y_train, x_val, y_val, names);

	plot(x_train, y_train, x_val, y_val, names, true);

	// ------------------------------------------------------------------------------
	// Adapting Attention Pooling
	// ------------------------------------------------------------------------------
	std::vector<double> sigmas = {0.1, 0.2, 0.5, 1.0};
	std::vector<std::string> sigs = {"0.1", "0.2", "0.5", "1.0"};
	std::stringstream ss;

	for( int i = 0; i < names.size(); i++ ) {
		names[i] = names[i] + " " + sigs[i];
		std::cout << names[i] << "\n";
	}

	plot(x_train, y_train, x_val, y_val, names, false, sigmas);

	plot(x_train, y_train, x_val, y_val, names, true, sigmas);

	std::cout << "Done!\n";
}



