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

// Generating the Dataset
torch::Tensor f(torch::Tensor x) {
    return( 2 * torch::sin(x) + torch::pow(x, 0.8) );
}

void plot_kernel_reg(torch::Tensor y_hat, torch::Tensor y_train,
		torch::Tensor y_truth, torch::Tensor x_test, std::string tlt) {
	std::vector<float> yhat(y_hat.data_ptr<float>(), y_hat.data_ptr<float>() + y_hat.numel());
	std::vector<float> ytrain(y_train.data_ptr<float>(), y_train.data_ptr<float>() + y_train.numel());
	std::vector<float> ytruth(y_truth.data_ptr<float>(), y_truth.data_ptr<float>() + y_truth.numel());
	std::vector<float> xtest(x_test.data_ptr<float>(), x_test.data_ptr<float>() + x_test.numel());

	plt::figure_size(800, 600);
	plt::named_plot("Truth", xtest, ytruth, "b");
	plt::named_plot("Pred", xtest, yhat, "m--");
	plt::plot(xtest, ytrain, "yo");
	plt::xlim(0, 5);
	plt::ylim(-1, 5);
	plt::xlabel("x");
	plt::ylabel("y");
	plt::title(tlt.c_str());
	plt::legend();
	plt::show();
	plt::close();
}

// ------------------------------------------------
// Defining the Model
// ------------------------------------------------
class NWKernelRegressionImpl : public torch::nn::Module {
public:
	NWKernelRegressionImpl() {
		//w = torch::nn::Parameter(torch::rand({1}, torch::requires_grad(true)));
		w = torch::rand({1}, torch::requires_grad(true));
		register_parameter("w", w);
	}

    torch::Tensor forward(torch::Tensor queries, torch::Tensor keys, torch::Tensor values) {
        // Shape of the output `queries` and `attention_weights`:
        // (no. of queries, no. of key-value pairs)
        queries = queries.repeat_interleave(keys.size(1)).reshape({-1, keys.size(1)});
        auto attention_weights = torch::nn::functional::softmax(-1*torch::pow(((queries - keys)*w), 2) / 2, /*dim=*/1);
		// attention_weights = rorch::nn::functional::softmax(-1 * torch::pow((queries - keys)* w.item<float>(), 2) / 2, /*dim=*/1);
        // Shape of `values`: (no. of queries, no. of key-value pairs)
        return torch::bmm(attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1);
    }
private:
    torch::Tensor w; // w parameter
};
TORCH_MODULE(NWKernelRegression);


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	// generate an artificial dataset
	int64_t n_train = 50; 								//  No. of training examples
	auto tuple = torch::sort(torch::rand(n_train) * 5); //  Training inputs
	torch::Tensor x_train = std::get<0>(tuple);

	auto y_train = f(x_train) + torch::normal(0.0, 0.5, (n_train));  // Training outputs
	auto x_test = torch::arange(0, 5, 0.1);							  // Testing examples
	auto y_truth = f(x_test); 										  // Ground-truth outputs for the testing examples
	auto n_test =  x_test.size(0);  								  // No. of testing examples
	std::cout << n_test << "\n";

	/* Average Pooling
	 * using average pooling to average over all the training outputs:
	 */
	auto y_hat = torch::repeat_interleave(y_train.mean(), n_test);

	plot_kernel_reg( y_hat, y_train, y_truth, x_test, "Average Pooling");

	/* Nonparametric Attention Pooling
	 * Notably, Nadaraya-Watson kernel regression is a nonparametric model;
	 * thus :eqref:eq_nadaraya-watson-gaussian is an example of nonparametric attention pooling
	 */

	// Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the
	// same testing inputs (i.e., same queries)
	auto X_repeat = (x_test.repeat_interleave(n_train)).reshape({-1, n_train});
	std::cout << X_repeat.sizes() << "\n";
	std::cout << x_train.sizes() << "\n";
	// Note that `x_train` contains the keys. Shape of `attention_weights`:
	// (`n_test`, `n_train`), where each row contains attention weights to be
	// assigned among the values (`y_train`) given each query
	auto attention_weights = torch::nn::functional::softmax(-1*torch::pow((X_repeat - x_train), 2) / 2, /*dim=*/1);
	//std::cout << attention_weights << "\n";

	// Each element of `y_hat` is weighted average of values, where weights are attention weights
	y_hat = torch::matmul(attention_weights, y_train);

	plot_kernel_reg( y_hat, y_train, y_truth, x_test, "Nonparametric Attention Pooling");


	// Now let us take a look at the [attention weights]
	float maxV = attention_weights.max().item<float>();
	std::cout << "maxV: " <<  maxV << "\n";

	int nrows = attention_weights.size(0), ncols = attention_weights.size(1);

	std::vector<float> z(ncols * nrows);
	for( int j=0; j<nrows; ++j ) {
		for( int i=0; i<ncols; ++i ) {
		    z.at(ncols * j + i) = (attention_weights.index({j, i})).item<float>();
		}
	}
    std::cout << "max: " << *std::max_element(z.begin(), z.end()) << "\n";
    std::cout << "min: " << *std::min_element(z.begin(), z.end()) << "\n";

    const float* zptr = &(z[0]);
    const int colors = 1;
    PyObject* mat;

    plt::title("heatmap");
    plt::imshow(zptr, nrows, ncols, colors, {}, &mat);
    plt::xlabel("Sorted training inputs");
    plt::ylabel("Sorted testing inputs");
    plt::colorbar(mat);
    plt::show();
    plt::close();
    Py_DECREF(mat);

    //plot_heatmap(attention_weights, "Sorted training inputs", "Sorted testing inputs");

	// Parametric Attention Pooling
    auto X = torch::ones({2, 1, 4});
    auto Y = torch::ones({2, 4, 6});
    std::cout << "Batch Matrix Multiplication:\n" << (torch::bmm(X, Y)).sizes() << "\n";

    // use minibatch matrix multiplication to compute weighted averages of values in a minibatch.
    auto weights = torch::ones({2, 10}) * 0.1;
    auto vals    = torch::arange(20.0).reshape({2, 10});
    std::cout << "Batch Matrix Multiplication to compute weighted averages of values in a minibatch:\n"
    		<< torch::bmm(weights.unsqueeze(1), vals.unsqueeze(-1)) << "\n";

    // Training
    // Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the same training inputs
    auto X_tile = x_train.repeat({n_train, 1});

    // Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the same training outputs
    auto Y_tile = y_train.repeat({n_train, 1});

    // Shape of `keys`: ('n_train', 'n_train' - 1)
    auto keys = torch::masked_select(X_tile, (1 - torch::eye(n_train)).to(torch::kBool)).reshape({n_train, -1});
    std::cout << keys.sizes() << "\n";

	// Shape of `values`: ('n_train', 'n_train' - 1)
    auto values = torch::masked_select(Y_tile, (1 - torch::eye(n_train)).to(torch::kBool)).reshape({n_train, -1});
    std::cout << values.sizes() << "\n";

    // Using the squared loss and stochastic gradient descent, we [train the parametric attention model].
    auto net = NWKernelRegression();
    auto loss = torch::nn::MSELoss(torch::nn::MSELossOptions(torch::kNone));
    auto trainer = torch::optim::SGD(net->parameters(), 0.5);

    std::vector<float> v_epoch, v_loss;

    for( int epoch = 0; epoch < 5; epoch++ ) {
        trainer.zero_grad();
        auto l = loss( net->forward(x_train, keys, values), y_train );
        l.sum().backward();
        trainer.step();
        std::cout << "epoch: " << (epoch + 1) << ", loss: " << l.sum().item<float>() << std::endl;
        v_epoch.push_back((epoch + 1)*1.0);
        v_loss.push_back(l.sum().item<float>());
    }

    plt::figure_size(700, 500);
    plt::named_plot("train", v_epoch, v_loss, "b");
    plt::xlabel("epoch");
    plt::ylabel("loss");
    plt::title("Train the parametric attention model");
    plt::legend();
    plt::show();
    plt::close();

    // Shape of `keys`: (`n_test`, `n_train`), where each column contains the same training inputs (i.e., same keys)
    keys = x_train.repeat({n_test, 1});
    // Shape of `value`: (`n_test`, `n_train`)
    values = y_train.repeat({n_test, 1});
    y_hat = net->forward(x_test, keys, values).view({-1}).detach();

    plot_kernel_reg( y_hat, y_train, y_truth, x_test, "parametric attention model");

	std::cout << "Done!\n";
	return 0;
}

