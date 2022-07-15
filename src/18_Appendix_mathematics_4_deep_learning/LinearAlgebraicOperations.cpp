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
#include <opencv2/imgcodecs.hpp>
#include <vector>

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

//#include "../utils.h"
#include "../fashion.h"

torch::Tensor angle(torch::Tensor v, torch::Tensor w) {
    return torch::acos(v.dot(w) / (torch::norm(v) * torch::norm(w)));
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	// -----------------------------------
	// Dot Products and Angles
	// -----------------------------------

	std::cout << angle(torch::tensor({0, 1, 2}).to(torch::kFloat32), torch::tensor({2.0, 3.0, 4.0})) << '\n';

	// -----------------------------------
	// Hyperplanes
	// -----------------------------------
	// Load in the dataset

	bool useCvMat = false;

	std::string data_path = "./data/fashion/";
	int64_t batch_size = 256;

	// fashion custom dataset
	auto train_dataset = FASHION(data_path, FASHION::Mode::kTrain)
			    			.map(torch::data::transforms::Stack<>());

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
									         std::move(train_dataset), batch_size);

	auto test_dataset = FASHION(data_path, FASHION::Mode::kTest)
					                .map(torch::data::transforms::Stack<>());

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
						         std::move(test_dataset), batch_size);

	auto batch = *train_loader->begin();
	auto data  = batch.data.to(device);
	std::cout << data[0].sizes() << '\n';

	auto xtrain_0 = data[0];

	torch::Tensor data_out = xtrain_0.contiguous().detach().clone();
	auto rev_tensor = data_out.mul(255).to(torch::kByte).permute({1, 2, 0});

	if( ! useCvMat ) {
		std::vector<uchar> z(rev_tensor.data_ptr<uchar>(), rev_tensor.data_ptr<uchar>() + rev_tensor.numel());
		plt::title("Xtrain_0");
		plt::imshow(&(z[0]), rev_tensor.size(0), rev_tensor.size(1), rev_tensor.size(2), {{"cmap", "Greys"}});
		plt::show();
		plt::close();
	} else {

		auto tensor = rev_tensor.reshape({ rev_tensor.size(0) * rev_tensor.size(1) * rev_tensor.size(2)});

		// CV_8UC1 is an 8-bit unsigned integer matrix/image with 1 channels
		cv::Mat cvmat(cv::Size(rev_tensor.size(0), rev_tensor.size(1)), CV_8UC1, tensor.data_ptr());

		cv::imshow("Xtrain_0", cvmat);
		cv::waitKey(-1);
		cv::destroyAllWindows();
	}

	// Invertibility
	auto M = torch::tensor({{1, 2}, {1, 4}}).to(torch::kFloat32);
	auto M_inv = torch::tensor({{2.0, -1.0}, {-0.5, 0.5}});
	std::cout <<  M_inv.mm( M ) << '\n';

	// -------------------------
	// Expressing in Code
	// -------------------------
	// Define tensors
	auto B = torch::tensor({{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}});
	auto A = torch::tensor({{1, 2}, {3, 4}});
	auto v = torch::tensor({1, 2});

	// Print out the shapes
	std::cout << A.sizes() << '\n'
			  << B.sizes() << '\n'
			  << v.sizes() << '\n';

	// Reimplement matrix multiplication
	std::cout << torch::einsum("ij, j -> i", {A, v}) << '\n';
	std::cout << torch::matmul(A, v) << '\n';

	std::cout << torch::einsum("ijk, il, j -> kl", {B, A, v}) << '\n';

	std::cout << "Done!\n";
}



