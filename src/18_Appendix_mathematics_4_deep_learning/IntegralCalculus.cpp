#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/ch_18_util.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	// --------------------------------------------------
	// Geometric Interpretation
	// --------------------------------------------------

	auto xT = torch::arange(-2, 2, 0.01);
	auto fT = torch::exp(-1*torch::pow(xT, 2));

	std::vector<float> x(xT.data_ptr<float>(), xT.data_ptr<float>() + xT.numel());
	std::vector<float> f(fT.data_ptr<float>(), fT.data_ptr<float>() + fT.numel());

	std::vector<float> y0;
	for( size_t i = 0; i < x.size(); i++ )
		y0.push_back(0.0);

	std::map<std::string, std::string> fill_parameters;
	fill_parameters["color"] = "blue";
	fill_parameters["alpha"] = "0.5";

	plt::figure_size(500, 400);
	plt::plot(x, f, "b-");
	plt::fill_between(x, y0, f, fill_parameters);
	plt::show();
	plt::close();

	// In most cases, this area will be infinite or undefined (consider the area under f(x)=x2),
	// so people will often talk about the area between a pair of ends, say a and b.
	plt::figure_size(500, 400);
	plt::plot(x, f, "k-");
	plt::fill_between(vector_slice(x, 50, 250), vector_slice(y0, 50, 250), vector_slice(f, 50, 250), fill_parameters);
	plt::show();
	plt::close();

	//  Let us take a look at an example doing this in code. We will see how to get the true value in a later section.
	float epsilon = 0.05;
	int a = 0;
	int b = 2;

	xT = torch::arange(a, b, epsilon);
	fT = xT / (1 + torch::pow(xT, 2));

	std::vector<float> xx(xT.data_ptr<float>(), xT.data_ptr<float>() + xT.numel());
	std::vector<float> ff(fT.data_ptr<float>(), fT.data_ptr<float>() + fT.numel());

	auto approx = torch::sum(epsilon*fT);
	auto tru = torch::log(torch::tensor({5.})) / 2;

	std::map<std::string, std::string> bar_parameters;
	bar_parameters["width"] = "0.05";
	bar_parameters["align"] = "edge";

	plt::figure_size(500, 400);
	plt::plot(xx, ff, "k-");
	plt::bar(xx, ff, "b", "-", 1.0, bar_parameters);
	plt::ylim(0.0, 1.0);
	plt::show();
	plt::close();

	std::cout << "approximation: " << approx << "\ntruth: " << tru << '\n';

	// ----------------------------------------------
	// Multiple Integrals
	// ----------------------------------------------
	// Construct grid and compute function

	std::vector<std::vector<float>> x_, y_, z_;

	for( float i = -2.0; i < 2.0; i += 0.04 ) {
		std::vector<float> x_row, y_row, z_row;
		for( float j = -2.0; j < 2.0; j += 0.04 ) {
	            x_row.push_back(i);
	            y_row.push_back(j);
	            z_row.push_back(std::exp(-1 * std::pow(i, 2) - std::pow(j, 2)));
		}
		x_.push_back(x_row);
		y_.push_back(y_row);
		z_.push_back(z_row);
	}

	// Plot function
	plt::plot_surface(x_, y_, z_);
	plt::xlabel("x");
	plt::ylabel("y");
	plt::xlim(-2, 2);
	plt::ylim(-2, 2);
	plt::show();
	plt::close();

	std::cout << "Done!\n";
}




