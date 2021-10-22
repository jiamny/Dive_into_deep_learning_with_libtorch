#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../fashion.h"
#include "../utils.h"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

torch::Tensor dropout_layer(torch::Tensor X, float dropout) {
    assert( 0 <= dropout <= 1);

    //In this case, all elements are dropped out
    if( dropout == 1 )
        return torch::zeros_like(X);

    // In this case, all elements are kept
    if( dropout == 0 )
        return X;
    auto mask = (torch::rand(X.sizes(), dtype(torch::kFloat32)) > dropout);
    return mask * X / (1.0 - dropout);
}

struct NetCh04Impl : public torch::nn::Module {

public:
	int64_t num_inputs;
	NetCh04Impl(int64_t num_inputs, int64_t num_outputs, int64_t num_hiddens1, int64_t num_hiddens2, bool is_training);

    torch::Tensor forward(torch::Tensor X);
private:
    bool training=false;
    torch::nn::Linear lin1{nullptr}, lin2{nullptr}, lin3{nullptr};
};

TORCH_MODULE(NetCh04);

NetCh04Impl::NetCh04Impl(int64_t num_inputs, int64_t num_outputs, int64_t num_hiddens1, int64_t num_hiddens2, bool is_training) {
	this->training = is_training;
	this->num_inputs = num_inputs;
	this->lin1 = torch::nn::Linear(num_inputs, num_hiddens1);
	this->lin2 = torch::nn::Linear(num_hiddens1, num_hiddens2);
	this->lin3 = torch::nn::Linear(num_hiddens2, num_outputs);
	register_module("fc1", lin1);
	register_module("fc2", lin2);
	register_module("fc3", lin3);
}

torch::Tensor NetCh04Impl::forward(torch::Tensor x) {
	float dropout1 = 0.2, dropout2 = 0.5;

//	auto H1 = torch::nn::functional::relu(lin1->forward(x.view({x.size(0), -1})));
	auto H1 = torch::nn::functional::relu(lin1->forward(x.reshape({-1, num_inputs})));
//	auto H1 = torch::nn::functional::relu(lin1->forward(x));
	// Use dropout only when training the model
	if( training ) {
		// Add a dropout layer after the first fully connected layer
		H1 = dropout_layer(H1, dropout1);
	}

	auto H2 = torch::nn::functional::relu(lin2->forward(H1));

	if( training ) {
		// Add a dropout layer after the second fully connected layer
		H2 = dropout_layer(H2, dropout2);
	}
	auto out = lin3->forward(H2);
	return out;
}

struct NeuralNetImpl : public torch::nn::Module {
 public:
    NeuralNetImpl(int64_t num_inputs, int64_t num_outputs, int64_t num_hiddens1, int64_t num_hiddens2);

    torch::Tensor forward(torch::Tensor x);

 private:
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};
};

TORCH_MODULE(NeuralNet);


NeuralNetImpl::NeuralNetImpl(int64_t num_inputs, int64_t num_outputs, int64_t num_hiddens1, int64_t num_hiddens2) :
	fc1(num_inputs, num_hiddens1), fc2(num_hiddens1, num_hiddens2), fc3(num_hiddens2, num_outputs) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
}

torch::Tensor NeuralNetImpl::forward(torch::Tensor x) {
	float dropout1 = 0.2, dropout2 = 0.5;

    x = torch::nn::functional::relu(fc1->forward(x));
    x = torch::nn::functional::dropout(x, torch::nn::functional::DropoutFuncOptions().p(dropout1));
    x = torch::nn::functional::relu(fc2->forward(x));
    x = torch::nn::functional::dropout(x, torch::nn::functional::DropoutFuncOptions().p(dropout2));
    auto out = fc3->forward(x);
    return out;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	// Implementation from Scratch
	/*
	 * To implement the dropout function for a single layer, we must draw as many samples from a Bernoulli (binary) random variable as our layer has dimensions,
	 * where the random variable takes value 1 (keep) with probability 1−𝑝 and 0 (drop) with probability 𝑝. One easy way to implement this is to first draw
	 * samples from the uniform distribution 𝑈[0,1]. Then we can keep those nodes for which the corresponding sample is greater than 𝑝, dropping the rest.

	 * In the following code, we (implement a dropout_layer function that drops out the elements in the tensor input X with probability dropout), rescaling
	 * the remainder as described above: dividing the survivors by 1.0-dropout.
	 */
	auto X = torch::arange(16, dtype(torch::kFloat32)).reshape({2, 8});

	std::cout << "X:\n" << X << std::endl;
	std::cout << "dropout_layer(X, 0.):\n" << dropout_layer(X, 0.) << std::endl;
	std::cout << "dropout_layer(X, 0.5):\n" << dropout_layer(X, 0.5) << std::endl;
	std::cout << "dropout_layer(X, 1.):\n" << dropout_layer(X, 1.) << std::endl;

	// Defining Model Parameters
	/*
	 * Again, we work with the Fashion-MNIST dataset introduced in :numref:sec_fashion_mnist. We [define an MLP with two hidden layers containing 256 units each.]
	 */
	int64_t num_inputs = 784, num_outputs = 10, num_hiddens1 = 256,  num_hiddens2 =256;

	// Defining the Model
	/*
	 * The model below applies dropout to the output of each hidden layer (following the activation function). We can set dropout probabilities for each layer
	 * separately. A common trend is to set a lower dropout probability closer to the input layer. Below we set it to 0.2 and 0.5 for the first and second
	 * hidden layers, respectively. We ensure that dropout is only active during training.
	 */
	//auto net = NeuralNetImpl(num_inputs, num_outputs, num_hiddens1, num_hiddens2);
	//auto net = NetCh04Impl(num_inputs, num_outputs, num_hiddens1, num_hiddens2, true);
	//NetCh04Impl net = NetCh04Impl(num_inputs, num_outputs, num_hiddens1, num_hiddens2, true); // using member .
	NetCh04 net = NetCh04(num_inputs, num_outputs, num_hiddens1, num_hiddens2, true);           // using ptr ->

	/*
	 * This is similar to the training and testing of MLPs described previously.
	 */
	int64_t num_epochs = 10;
	float lr = 0.5;
	int64_t batch_size = 256;

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

	auto criterion = torch::nn::CrossEntropyLoss();
	auto trainer = torch::optim::SGD(net->parameters(), lr);

/*
	auto batch = *train_loader->begin();
	auto data  = batch.data.to(device);
	auto target = batch.target.to(device);

	auto data_hat = net->forward(data);

	auto loss = criterion(data_hat, target);

	loss.backward();

	auto prediction = data_hat.argmax(1);
	std::cout << "prediction: \n" << prediction << std::endl;
	std::cout << "target: \n" << target << std::endl;
	// Update number of correctly classified samples
	auto correct = (prediction == target).sum().item<int>(); //prediction.eq(y).sum().item<int64_t>();


	std::cout << loss << std::endl;
	std::cout << loss.item<double>() << std::endl;
	std::cout << correct << std::endl;
	std::cout << target.numel() << std::endl;

	std::cout << "accuracy: " << accuracy(data_hat, target) << std::endl;
*/
	/*
	* Train a model
	*/

	std::vector<double> train_loss;
	std::vector<double> train_acc;
	std::vector<double> test_loss;
	std::vector<double> test_acc;
	std::vector<double> xx;

	for( size_t epoch = 0; epoch < num_epochs; epoch++ ) {
		net->train(true);
		torch::AutoGradMode enable_grad(true);

		// Initialize running metrics
		double epoch_loss = 0.0;
		int64_t epoch_correct = 0;
	    int64_t num_train_samples = 0;

		for(auto &batch : *train_loader) {
			//auto x = batch.data.view({batch.data.size(0), -1}).to(device);
			auto x = batch.data.to(device);
			auto y = batch.target.to(device);

			//std::cout << "x.sizes(): " << x.sizes() << std::endl;

			auto y_hat = net->forward(x);
			auto loss = criterion(y_hat, y); //torch::cross_entropy_loss(y_hat, y);
				//std::cout << loss.item<double>() << std::endl;

			// Update running loss
			epoch_loss += loss.item<double>() * x.size(0);

			// Calculate prediction
			// Update number of correctly classified samples
			epoch_correct += accuracy(y_hat, y); //(prediction == y).sum().item<int>(); //prediction.eq(y).sum().item<int64_t>();
			//std::cout << epoch_correct << std::endl;
			trainer.zero_grad();
			loss.backward();
			trainer.step();
			num_train_samples += x.size(0);
		}
//		std::cout << epoch_loss << std::endl;
//		std::cout << num_train_samples << std::endl;
		auto sample_mean_loss = (epoch_loss / num_train_samples);
		auto tr_acc = static_cast<double>(epoch_correct *1.0 / num_train_samples);

		std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
				            << sample_mean_loss << ", Accuracy: " << tr_acc << '\n';

		train_loss.push_back((sample_mean_loss));
		train_acc.push_back(tr_acc);

		std::cout << "Training finished!\n\n";
		std::cout << "Testing...\n";

		// Test the model
		net->eval();
		torch::NoGradGuard no_grad;

		double tst_loss = 0.0;
		epoch_correct = 0;
		int64_t num_test_samples = 0;

		for( auto& batch : *test_loader) {
			//auto data = batch.data.view({batch.data.size(0), -1}).to(device);
			auto data = batch.data.to(device);
			auto target = batch.target.to(device);

			//std::cout << data.sizes() << std::endl;

			auto output = net->forward(data);

			auto loss = criterion(output, target); //torch::nn::functional::cross_entropy(output, target);
			tst_loss += loss.item<double>() * data.size(0);

			//auto prediction = output.argmax(1);
			epoch_correct += accuracy(output, target); //prediction.eq(target).sum().item<int64_t>();

			num_test_samples += data.size(0);
		}

		std::cout << "Testing finished!\n";

		auto test_accuracy = static_cast<double>(epoch_correct * 1.0 / num_test_samples);
		auto test_sample_mean_loss = tst_loss / num_test_samples;

		test_loss.push_back((test_sample_mean_loss));
		test_acc.push_back(test_accuracy);

		std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
		xx.push_back((epoch + 1));
	}

	plt::figure_size(800, 600);
	plt::subplot(1, 1, 1);
	plt::ylim(0.2, 0.9);
	plt::named_plot("Train loss", xx, train_loss, "b");
	plt::named_plot("Test loss", xx, test_loss, "c:");
	plt::named_plot("Train acc", xx, train_acc, "g--");
	plt::named_plot("Test acc", xx, test_acc, "r-.");
	plt::xlabel("epoch");
	plt::title("Define an MLP with two hidden layers");
	plt::legend();
	plt::show();

	// Concise Implementation
	/*
	 * With high-level APIs, all we need to do is add a Dropout layer after each fully-connected layer, passing in the dropout probability as the only
	 * argument to its constructor. During training, the Dropout layer will randomly drop out outputs of the previous layer (or equivalently, the inputs
	 * to the subsequent layer) according to the specified dropout probability. When not in training mode, the Dropout layer simply passes the data
	 * through during testing.
	 */
	float dropout1 = 0.2, dropout2 = 0.5;

	auto net_concise = torch::nn::Sequential(
		    torch::nn::Flatten(), torch::nn::Linear(784, 256), torch::nn::ReLU(),
		    // Add a dropout layer after the first fully connected layer
		    torch::nn::Dropout(dropout1), torch::nn::Linear(256, 256), torch::nn::ReLU(),
		    // Add a dropout layer after the second fully connected layer
		    torch::nn::Dropout(dropout2), torch::nn::Linear(256, 10));

	// initialize the weights at random with zero mean and standard deviation 0.01
	if (auto M = dynamic_cast<torch::nn::LinearImpl*>(net_concise.get())) {
		torch::nn::init::normal_(M->weight, 0, 0.01);
	}

	auto ttrainer = torch::optim::SGD(net_concise->parameters(), lr);

	/*
	* Train a model
	*/

	train_loss.clear();
	train_acc.clear();
	test_loss.clear();
	test_acc.clear();
	xx.clear();

	for( size_t epoch = 0; epoch < num_epochs; epoch++ ) {
		net_concise->train(true);
		torch::AutoGradMode enable_grad(true);

		// Initialize running metrics
		double epoch_loss = 0.0;
		int64_t epoch_correct = 0;
	    int64_t num_train_samples = 0;

		for(auto &batch : *train_loader) {
			auto x = batch.data.to(device);
			auto y = batch.target.to(device);

			//std::cout << "x.sizes(): " << x.sizes() << std::endl;

			auto y_hat = net_concise->forward(x);
			auto loss = criterion(y_hat, y); //torch::cross_entropy_loss(y_hat, y);

			// Update running loss
			epoch_loss += loss.item<double>() * x.size(0);;
			epoch_correct += accuracy(y_hat, y);

			ttrainer.zero_grad();
			loss.backward();
			ttrainer.step();
			num_train_samples += x.size(0);
		}
		std::cout << epoch_loss << std::endl;
		std::cout << num_train_samples << std::endl;
		auto sample_mean_loss = (epoch_loss / num_train_samples);
		auto train_accuracy = static_cast<double>(epoch_correct *1.0 / num_train_samples);

		std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
				            << sample_mean_loss << ", Accuracy: " << train_accuracy << '\n';

		train_loss.push_back((sample_mean_loss));
		train_acc.push_back(train_accuracy);

		std::cout << "Training finished!\n\n";
		std::cout << "Testing...\n";

		torch::NoGradGuard no_grad;

		double tst_loss = 0.0;
		epoch_correct = 0;
		int64_t num_test_samples = 0;

		for(auto& batch : *test_loader) {
			auto data = batch.data.to(device);
			auto target = batch.target.to(device);

			auto output = net_concise->forward(data);

			auto loss = criterion(output, target);
			tst_loss += loss.item<double>() * data.size(0);

			epoch_correct += accuracy( output, target );

			num_test_samples += data.size(0);
		}

		std::cout << "Testing finished!\n";

		auto test_accuracy = static_cast<double>(epoch_correct) / num_test_samples;
		auto test_sample_mean_loss = tst_loss / num_test_samples;

		test_loss.push_back((test_sample_mean_loss));
		test_acc.push_back(test_accuracy);

		std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';
		xx.push_back((epoch + 1));
	}

	plt::figure_size(800, 600);
	plt::subplot(1, 1, 1);
	plt::ylim(0.2, 0.9);
	plt::named_plot("Train loss", xx, train_loss, "b");
	plt::named_plot("Test loss", xx, test_loss, "c:");
	plt::named_plot("Train acc", xx, train_acc, "g--");
	plt::named_plot("Test acc", xx, test_acc, "r-.");
	plt::title("Concise implementation");
	plt::xlabel("epoch");
	plt::legend();
	plt::show();

	std::cout << "Done!\n";
	return 0;
}



