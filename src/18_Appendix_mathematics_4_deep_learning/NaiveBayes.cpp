#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <unistd.h>
#include <iomanip>
#include <vector>
#include <iostream>
#include <string>

#include "../fashion.h"
#include <matplot/matplot.h>
using namespace matplot;

using torch::indexing::Slice;
using torch::indexing::None;

void showImages(torch::Tensor data, int nr, int nc, int start, std::map<int, std::string> fmap, std::vector<long> labels) {

//	plt::figure_size(1600, 400);
	auto f = figure(true);
	f->width(f->width() * 2);
	f->height(f->height() * 2);
	f->x_position(0);
	f->y_position(0);

	int i = start;
	int ncols = 28, nrows = 28;

	for( int r = 0; r < nr; r ++ ) {
		for( int c = 0; c < nc; c++ ) {
			auto img = data[i];

        	std::vector<std::vector<double>> C;
        	for( int i = 0; i < nrows; i++ ) {
        		std::vector<double> c;
        		for( int j = 0; j < ncols; j++ )
        			c.push_back(img[i][j].item<double>());
        		C.push_back(c);
        	}

        	matplot::subplot(nr, nc, r*nc + c);
        	if( ! labels.empty() )
        		matplot::title(fmap[labels[i]].c_str());
        	matplot::image(C);
        	matplot::axis(false);
        	f->draw();
			i++;
		}
	}
	matplot::show();
}

torch::Tensor bayes_pred(torch::Tensor x, torch::Tensor P_xy, torch::Tensor P_y) {
    x = x.unsqueeze(0);  						// (28, 28) -> (1, 28, 28)
    auto p_xy = P_xy * x + (1 - P_xy)*(1 - x);
    p_xy = p_xy.reshape({10, -1}).prod(1);		// p(x|y)
    return p_xy * P_y;
}

// stable version
torch::Tensor bayes_pred_stable(torch::Tensor x, torch::Tensor log_P_xy, torch::Tensor log_P_xy_neg, torch::Tensor log_P_y) {
	x = x.unsqueeze(0);
	auto p_xy = log_P_xy * x + log_P_xy_neg * (1 - x);
	p_xy = (p_xy.reshape({10, -1})).sum(1); 		// p(x|y)
	return p_xy + log_P_y;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	std::map<int, std::string> fmap;
	fmap[0] = "T-shirt/top";
	fmap[1] = "Trouser";
	fmap[2] = "Pullover";
	fmap[3] = "Dress";
	fmap[4] = "Coat";
	fmap[5] = "Sandal";
	fmap[6] = "Shirt";
	fmap[7] = "Sneaker";
	fmap[8] = "Bag";
	fmap[9] = "Ankle boot";

	int64_t batch_size = 6000;

	const std::string FASHION_data_path("./data/fashion/");

	// fashion custom dataset
	auto train_dataset = FASHION(FASHION_data_path, FASHION::Mode::kTrain)
				    		.map(torch::data::transforms::Stack<>());

	auto test_dataset = FASHION(FASHION_data_path, FASHION::Mode::kTest)
			                .map(torch::data::transforms::Stack<>());

	// Number of samples in the training set
	auto num_train_samples = train_dataset.size().value();
	std::cout << "num_train_samples: " << num_train_samples << std::endl;

	// Reading a Minibatch
	// Data loader
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
						         std::move(train_dataset), batch_size);

	// Number of samples in the testset
	auto num_test_samples = test_dataset.size().value();
	std::cout << "num_test_samples: " << num_test_samples << std::endl;

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
									std::move(test_dataset), batch_size);

	auto tbatch = *train_loader->begin();

	std::cout << "tbatch: " << tbatch.data.sizes() << std::endl;

	auto train_dt = tbatch.data.squeeze();
	auto train_lb = tbatch.target.squeeze();
	std::cout << "t-dt: " << train_dt.sizes() << std::endl;
	std::cout << "t-lb: " << train_lb.sizes() << std::endl;

	std::cout << train_lb.sizes() << '\n';

	std::vector<long> labels(train_lb.data_ptr<long>(), train_lb.data_ptr<long>() + train_lb.numel());
	showImages(train_dt, 2, 9, 10, fmap, labels);


	auto n_y = torch::zeros(10);
	for(int y = 0; y < 10; y++ ) {
	    n_y.index_put_({y}, (train_lb == y).sum().data().item<int>());
	}
	std::cout << "n_y: " << n_y << '\n';

	auto P_y = n_y.div(n_y.sum().data().item<int>()*1.0);
	std::cout << "P_y: " << P_y << '\n';

	// ----------------------------------------------------------------
	auto n_x = torch::zeros({10, 28, 28}).to(torch::kFloat);

	for(int y = 0; y < 10; y++ ) {

		std::vector<torch::Tensor> idx = torch::where(train_lb == y);
		auto ty = train_dt.index_select(0, idx[0]);
		std::cout << ty.sizes() << '\n';

		for( int r = 0; r < 28; r++ ) {
			for( int c = 0; c < 28; c++ ) {
				n_x.index_put_({y, r, c}, ty.index({Slice(), r, c}).sum().data().item<float>());
			}
		}
		//n_x[y] = torch.tensor(X.numpy()[Y.numpy() == y].sum(axis=0))
	}
	std::cout <<"n_x: " << n_x.sizes() << '\n';

	auto P_xy = (n_x + 1).div( (n_y + 1).reshape({10, 1, 1}));
	std::cout <<"P_xy: " << P_xy.sizes() << '\n';

	showImages(P_xy, 2, 5, 0, fmap, {});

	auto tstbch = *test_loader->begin();
	auto test_dt = tstbch.data.squeeze();
	auto test_lb = tstbch.target.squeeze();
	std::cout << "tst-dt: " << test_dt.sizes() << std::endl;
	std::cout << "tst-lb: " << test_lb.sizes() << std::endl;
	auto image = test_dt[0];

	auto pred = bayes_pred(image, P_xy, P_y);
	std::cout << "pred: " << pred << ", label: "<< test_lb[0] << std::endl;

	double a = 0.1;
	std::cout << "underflow: " << std::pow(a, 784) << '\n';
	std::cout << "logarithm is normal: " <<  784*std::log(a) << '\n';

	// We can implement the following stable version:

	auto log_P_xy = torch::log(P_xy);
	auto log_P_xy_neg = torch::log(1 - P_xy);
	auto log_P_y = torch::log(P_y);

	auto py = bayes_pred_stable(image, log_P_xy, log_P_xy_neg, log_P_y);
	std::cout << "py: " << py << ", label: "<< test_lb[0] << std::endl;

	// We may now check if the prediction is correct.
	std::cout << (py.argmax(0) == test_lb[0]) << std::endl;

	// Finally, let us compute the overall accuracy of the classifier.
	int correct = 0;
	std::vector<long> preds;
	for( int i = 0; i < test_dt.size(0); i++ ) {
		py = bayes_pred_stable(test_dt[i], log_P_xy, log_P_xy_neg, log_P_y);

		if( py.argmax(0).data().item<int>() == test_lb[i].data().item<int>() )
			correct++;
		preds.push_back(py.argmax(0).data().item<long>());
	}

	std::cout << "Validation accuracy: " <<  (correct*1.0/test_dt.size(0)) << '\n';

	// Show a few validation examples, we can see the Bayes classifier works pretty well.
	showImages(test_dt, 2, 9, 10, fmap, preds);

	std::cout << "Done!\n";
}




