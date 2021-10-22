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

using namespace torch::autograd;
namespace plt = matplotlibcpp;


void show_images(torch::Tensor batch_data, torch::Tensor target,
		size_t num_cols, size_t num_rows, double scale, int img_size, int batch_size,
		torch::TensorOptions dtype_option, std::unordered_map<int, std::string> fmap) {

	plt::figure_size( static_cast<size_t>(num_cols * scale * img_size), static_cast<size_t>(num_rows * scale * img_size));

	for( int i = 0; i < batch_size; i++ ) {

		plt::subplot(num_rows, num_cols, (i+1));

		auto image = batch_data.data()[i].view({-1,1}).to(dtype_option);
		//std::cout << image.data().sizes() << "\n";

		int type_id = target.data()[i].item<int>();
		//std::cout << "type_id = " << type_id << " name = " << fmap.at(type_id) << "\n";

		int ncols = img_size, nrows = img_size;
		std::vector<float> z(image.data_ptr<float>(), image.data_ptr<float>() + image.numel());;
		const float* zptr = &(z[0]);
		const int colors = 1;
		plt::imshow(zptr, nrows, ncols, colors);
		plt::title(fmap.at(type_id));
	}

	plt::show();
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	// -------------------------------------------
	// Reading the Dataset
	// -------------------------------------------
	std::string data_path = "./data/fashion/";
	auto dtype_option = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

	const std::string FASHION_data_path = data_path;

	// fashion custom dataset
	auto train_dataset = FASHION(FASHION_data_path, FASHION::Mode::kTrain)
	    			.map(torch::data::transforms::Stack<>());

	// Number of samples in the training set
	auto num_train_samples = train_dataset.size().value();

	std::cout << "num_train_samples: " << num_train_samples << std::endl;

	auto test_dataset = FASHION(FASHION_data_path, FASHION::Mode::kTest)
	                .map(torch::data::transforms::Stack<>());

	// Number of samples in the testset
	auto num_test_samples = test_dataset.size().value();

	std::cout << "num_test_samples: " << num_test_samples << std::endl;

	/*
	 * The following function converts between numeric label indices and their names in text.
	 */
	std::unordered_map<int, std::string> fmap = get_fashion_mnist_labels();

	const int64_t batch_size = 18;

	/*
	 * Reading a Minibatch
	 */
	// Data loader
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
	         std::move(train_dataset), batch_size);

	// We can now create a function to visualize these examples.
	auto it = *train_loader->begin();
	std::cout << "data.size() = " << it.data.size(0) << "\n";
	std::cout << "target.size() = " << it.data.size(0) << "\n";

	auto batch_data = it.data;
	auto target     = it.target;

	std::cout << batch_data.data().sizes() << "\n";

	size_t num_cols = 6;
	size_t num_rows = 3;
	double scale  = 10.2;
	size_t img_size = 28;
	/*
	 * We can now create a function to visualize these examples.
	 */
	show_images(batch_data, target, num_cols, num_rows, scale, img_size, batch_size, dtype_option, fmap);

	/*
	 * Reading a Minibatch
	 *
	 * To make our life easier when reading from the training and test sets, we use the built-in data iterator rather than creating one
	 * from scratch. Recall that at each iteration, a data iterator [reads a minibatch of data with size batch_size each time.]
	 * We also randomly shuffle the examples for the training data iterator.
	 */
	auto first_batch = *train_loader->begin();

	/*
	 * Let us look at the time it takes to read the training data.
	 */
	precise_timer timer;
	for(auto& batch : *train_loader) {
    	auto X = batch.data.to(device);
    	auto y = batch.target.to(device);
	    continue;
	}
	std::setprecision(2);
	unsigned int dul = timer.stop<unsigned int, std::chrono::microseconds>();
	std::cout << "Roughly takes " << (dul/1000000.0) << " seconds\n";

	/*
	 * Putting All Things Together
	 */

	//const size_t batch_sz = 32;
	//auto tt_loader =  load_data_fashion_mnist(batch_sz, data_path, true);

	auto batch = *train_loader->begin();
	auto X = batch.data.to(device);
	auto y = batch.target.to(device);

	std::cout << "X shape = " << X.sizes() << " X.dtype = " << X.dtype() << std::endl;
	std::cout << "y shape = " << y.sizes() << " y.dtype = " << y.dtype() << std::endl;

	std::cout << "Done!\n";
	return 0;
}





