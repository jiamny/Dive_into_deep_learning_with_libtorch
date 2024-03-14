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
	auto hat = y_hat.to(torch::kDouble).to(torch::kCPU);
	auto train = y_train.to(torch::kDouble).to(torch::kCPU);
	auto truth = y_truth.to(torch::kDouble).to(torch::kCPU);
	auto test  = x_test.to(torch::kDouble).to(torch::kCPU);
	std::vector<double> yhat(hat.data_ptr<double>(), hat.data_ptr<double>() + hat.numel());
	std::vector<double> ytrain(train.data_ptr<double>(), train.data_ptr<double>() + train.numel());
	std::vector<double> ytruth(truth.data_ptr<double>(), truth.data_ptr<double>() + truth.numel());
	std::vector<double> xtest(test.data_ptr<double>(), test.data_ptr<double>() + test.numel());

	auto F = figure(true);
	F->size(1800, 500);
	F->add_axes(false);
	F->reactive_mode(false);

	subplot(1, 3, 0);
	plot(xtest, ytruth, "b")->line_width(2).display_name("Truth");
	xlabel("x");
	legend({});
	subplot(1, 3, 1);
	plot(xtest, yhat, "m--")->line_width(2).display_name("Pred");
	ylabel("y");
	legend({});
	title(tlt.c_str());
	subplot(1, 3, 2);
	plot(xtest, ytrain, "ro")->line_width(2).display_name("Train");
	legend({});
	F->draw();
    show();
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
	std::cout << (cuda_available ? "CUDA available. Using GPU." : "Using CPU.") << '\n';

	torch::manual_seed(1000);

	// generate an artificial dataset
	int64_t n_train = 50; 								//  No. of training examples
	auto tuple = torch::sort(torch::rand(n_train) * 5); //  Training inputs
	torch::Tensor x_train = std::get<0>(tuple).to(device);

	std::cout << "f()\n";
	auto y_train = f(x_train) + torch::normal(0.0, 0.5, (n_train)).to(device);  // Training outputs
	auto x_test = torch::arange(0, 5, 0.1).to(device);							  // Testing examples
	auto y_truth = f(x_test).to(device); 										  // Ground-truth outputs for the testing examples
	auto n_test =  x_test.size(0);  								  // No. of testing examples
	std::cout << n_test << "\n";

	/* Average Pooling
	 * using average pooling to average over all the training outputs:
	 */
	auto y_hat = torch::repeat_interleave(y_train.mean(), n_test).to(device);

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

    plot_heatmap(attention_weights, "Sorted training inputs", "Sorted testing inputs");

	// Parametric Attention Pooling
    auto X = torch::ones({2, 1, 4}).to(device);
    auto Y = torch::ones({2, 4, 6}).to(device);
    std::cout << "Batch Matrix Multiplication:\n" << (torch::bmm(X, Y)).sizes() << "\n";

    // use minibatch matrix multiplication to compute weighted averages of values in a minibatch.
    auto weights = (torch::ones({2, 10}) * 0.1).to(device);
    auto vals    = torch::arange(20.0).reshape({2, 10}).to(device);
    std::cout << "Batch Matrix Multiplication to compute weighted averages of values in a minibatch:\n"
    		<< torch::bmm(weights.unsqueeze(1), vals.unsqueeze(-1)) << "\n";

    std::cout << "Training\n";
    // Training
    // Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the same training inputs
    auto X_tile = x_train.repeat({n_train, 1});

    // Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the same training outputs
    auto Y_tile = y_train.repeat({n_train, 1});

    // Shape of `keys`: ('n_train', 'n_train' - 1)
    auto keys = torch::masked_select(X_tile, (1 - torch::eye(n_train)).to(torch::kBool).to(device)).reshape({n_train, -1});
    std::cout << keys.sizes() << "\n";

	// Shape of `values`: ('n_train', 'n_train' - 1)
    auto values = torch::masked_select(Y_tile, (1 - torch::eye(n_train)).to(torch::kBool).to(device)).reshape({n_train, -1});
    std::cout << values.sizes() << "\n";

    // Using the squared loss and stochastic gradient descent, we [train the parametric attention model].
    auto net = NWKernelRegression();
    net->to(device);
    auto loss = torch::nn::MSELoss(torch::nn::MSELossOptions(torch::kNone));
    auto trainer = torch::optim::SGD(net->parameters(), 0.5);

    std::vector<double> v_epoch, v_loss;

    for( int epoch = 0; epoch < 10; epoch++ ) {
        trainer.zero_grad();
        std::cout << "x_train: " << x_train.device() << keys.device() << " " << values.device() << " " << y_train.device() << '\n';
        auto l = loss( net->forward(x_train, keys, values), y_train );
        l.sum().backward();
        trainer.step();
        std::cout << "epoch: " << (epoch + 1) << ", loss: " << l.sum().item<float>() << std::endl;
        v_epoch.push_back((epoch + 1)*1.0);
        v_loss.push_back(1.0*l.sum().item<float>());
    }

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::plot(ax1, v_epoch, v_loss, "b")->line_width(2);
    matplot::xlabel(ax1, "epoch");
    matplot::ylabel(ax1, "loss");
    matplot::title(ax1, "Train the parametric attention model");
    matplot::show();

    // Shape of `keys`: (`n_test`, `n_train`), where each column contains the same training inputs (i.e., same keys)
    keys = x_train.repeat({n_test, 1});
    // Shape of `value`: (`n_test`, `n_train`)
    values = y_train.repeat({n_test, 1});
    y_hat = net->forward(x_test, keys, values).view({-1}).detach();

    plot_kernel_reg( y_hat, y_train, y_truth, x_test, "parametric attention model");

	std::cout << "Done!\n";
	return 0;
}

