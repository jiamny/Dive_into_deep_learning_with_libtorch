#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/Ch_18_util.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

torch::Tensor f(torch::Tensor x, torch::Tensor y) {
    return torch::log(torch::exp(x) + torch::exp(y));
}

torch::Tensor grad_f(torch::Tensor x, torch::Tensor y) {
    return torch::cat({torch::exp(x) / (torch::exp(x) + torch::exp(y)),
                     torch::exp(y) / (torch::exp(x) + torch::exp(y))});
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	// --------------------------------------------------
	// Higher-Dimensional Differentiation
	// --------------------------------------------------

	auto epsilon = torch::tensor({0.01, -0.03});

	auto grad_approx = f(torch::tensor({0.}), torch::log(
	    torch::tensor({2.}))) + epsilon.dot(
	    grad_f(torch::tensor({0.}), torch::log(torch::tensor({2.}))));

	auto true_value = f(torch::tensor({0.}) + epsilon[0], torch::log(
	    torch::tensor({2.})) + epsilon[1]);

	std::cout << "approximation: " << grad_approx << ", true Value: " << true_value << '\n';

	// -------------------------------------------------
	//  A Note on Mathematical Optimization
	// -------------------------------------------------
	// f(x)=3x^4−4x^3−12x^2. => has derivative: 12x(x−2)(x+1).

	auto xf = torch::arange(-2, 3, 0.01);
	auto yf = (3 * torch::pow(xf, 4)) - (4 * torch::pow(xf, 3)) - (12 * torch::pow(xf, 2));

	std::vector<float> xx(xf.data_ptr<float>(), xf.data_ptr<float>() + xf.numel());
	std::vector<float> yy(yf.data_ptr<float>(), yf.data_ptr<float>() + yf.numel());

	plt::figure_size(500, 400);
	plt::plot(xx, yy, "b-");
	plt::xlabel("x");
	plt::ylabel("f(x)");
	plt::show();
	plt::close();

	// -----------------------------------------------
	// Multivariate Chain Rule
	// -----------------------------------------------
	// Compute the value of the function from inputs to outputs
	double w = -1.0, x = 0.0, y = -2.0, z = 1.0;
	double a = std::pow((w + x + y + z), 2);
	double b = std::pow((w + x - y - z), 2);
	double u = std::pow((a + b), 2);
	double v = std::pow((a - b), 2);

	double f = std::pow((u + v), 2);

	std::cout << "f at " << w << " " << x << " " << y << " " <<  z << " is " << f << '\n';

	// Compute the single step partials
	double df_du = 2*(u + v), df_dv =  2*(u + v);
	double du_da = 2*(a + b), du_db = 2*(a + b), dv_da = 2*(a - b), dv_db = -2*(a - b);
	double da_dw = 2*(w + x + y + z), db_dw =  2*(w + x - y - z);

	// Compute the final result from inputs to outputs
	double du_dw = du_da*da_dw + du_db*db_dw, dv_dw = dv_da*da_dw + dv_db*db_dw;
	double df_dw = df_du*du_dw + df_dv*dv_dw;

	std::cout << "df/dw at " << w << " " << x << " " << y << " " << z << " is " << df_dw << '\n';

	// then keeping track of how f changes when we change any node in the entire network. Let us implement it.
	// # Compute the value of the function from inputs to outputs
	w = -1.0, x = 0.0, y = -2.0, z = 1.0;
	a = std::pow((w + x + y + z), 2);
	b = std::pow((w + x - y - z), 2);
	u = std::pow((a + b), 2);
	v = std::pow((a - b), 2);

	f = std::pow((u + v), 2);

	std::cout << "f at " << w << " " << x << " " << y << " " <<  z << " is " << f << '\n';

	// Compute the derivative using the decomposition above
	// First compute the single step partials
	df_du = 2*(u + v), df_dv = 2*(u + v);
	du_da = 2*(a + b), du_db = 2*(a + b), dv_da = 2*(a - b), dv_db = -2*(a - b);
	da_dw = 2*(w + x + y + z), db_dw = 2*(w + x - y - z);
	double da_dx = 2*(w + x + y + z), db_dx = 2*(w + x - y - z);
	double da_dy = 2*(w + x + y + z), db_dy = -2*(w + x - y - z);
	double da_dz = 2*(w + x + y + z), db_dz = -2*(w + x - y - z);

	// Now compute how f changes when we change any value from output to input
	double df_da = df_du*du_da + df_dv*dv_da, df_db = df_du*du_db + df_dv*dv_db;
	df_dw = df_da*da_dw + df_db*db_dw;
	double df_dx = df_da*da_dx + df_db*db_dx, df_dy = df_da*da_dy + df_db*db_dy, df_dz = df_da*da_dz + df_db*db_dz;

	std::cout << "df/dw at " << w << " " << x << " " << y << " " << z << " is " << df_dw << '\n';
	std::cout << "df/dx at " << w << " " << x << " " << y << " " << z << " is " << df_dx << '\n';
	std::cout << "df/dy at " << w << " " << x << " " << y << " " << z << " is " << df_dy << '\n';
	std::cout << "df/dz at " << w << " " << x << " " << y << " " << z << " is " << df_dz << '\n';

	// To see how to encapsulated this, let us take a quick look at this example.
	// Initialize as ndarrays, then attach gradients
	auto wT = torch::tensor({-1.}).requires_grad_(true);
	auto xT = torch::tensor({0.}).requires_grad_(true);
	auto yT = torch::tensor({-2.}).requires_grad_(true);
	auto zT = torch::tensor({1.}).requires_grad_(true);
	// Do the computation like usual, tracking gradients
	auto aT = torch::pow((wT + xT + yT+ zT), 2), bT = torch::pow((wT + xT - yT - zT), 2);
	auto uT = torch::pow((aT + bT), 2), vT = torch::pow((aT - bT), 2);
	auto Tf = torch::pow((uT + vT), 2);

	// Execute backward pass
	Tf.backward();

	std::cout << "df/dw at " << wT.data().item<float>() << " " << xT.data().item<float>() << " " << yT.data().item<float>()
			  << " " << zT.data().item<float>() << " is " << wT.grad().data().item<float>() << '\n';

	std::cout << "df/dx at " << wT.data().item<float>() << " " << xT.data().item<float>() << " " << yT.data().item<float>()
	          << " " << zT.data().item<float>() << " is " << xT.grad().data().item<float>() << '\n';

	std::cout << "df/dy at " << wT.data().item<float>() << " " << xT.data().item<float>() << " " << yT.data().item<float>()
	    	  << " " << zT.data().item<float>() << " is " << yT.grad().data().item<float>() << '\n';

	std::cout << "df/dz at " << wT.data().item<float>() << " " << xT.data().item<float>() << " " << yT.data().item<float>()
	          << " " << zT.data().item<float>() << " is " << zT.grad().data().item<float>() << '\n';

	// --------------------------------------------
	// Hessians
	// --------------------------------------------
	// Construct grid and compute function

	std::vector<std::vector<float>> x_, y_, z_;

	for( float i = -2.0; i < 2.0; i += 0.2 ) {
		std::vector<float> x_row, y_row, z_row;
		for( float j = -2.0; j < 2.0; j += 0.2 ) {
		            x_row.push_back(i);
		            y_row.push_back(j);
		        	//z = x*torch.exp(- x**2 - y**2)
		            z_row.push_back(std::exp(-1 * std::pow(i, 2) - std::pow(j, 2)));
		}
		// Compute approximating quadratic with gradient and Hessian at (1, 0)
		// w = torch.exp(torch.tensor([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)
		for( float j = -2.0; j < 2.0; j += 0.2 ) {
			x_row.push_back(i);
			y_row.push_back(j);
			z_row.push_back(std::exp( -1 )* (-1 - (i + 1) + 2 * std::pow((i + 1), 2) + 2 * std::pow(j, 2)));
		}
		x_.push_back(x_row);
		y_.push_back(y_row);
		z_.push_back(z_row);
	}

	// Plot function
	plt::plot_surface(x_, y_, z_);
	plt::xlabel("x");
	plt::ylabel("y");
	plt::set_zlabel("z");
	plt::xlim(-2, 2);
	plt::ylim(-2, 2);
	plt::show();
	plt::close();

	std::cout << "Done!\n";
}



