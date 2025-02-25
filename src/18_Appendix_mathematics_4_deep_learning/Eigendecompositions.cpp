#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>

#include <matplot/matplot.h>
using namespace matplot;

using torch::indexing::Slice;
using torch::indexing::None;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(42);
	// ----------------------------------------
	// Finding Eigenvalues
	// ----------------------------------------

	auto eigRlts = torch::linalg_eig(torch::tensor({{2, 1}, {2, 3}}).to(torch::kFloat64));

	std::cout << "eig values:\n" << std::get<0>(eigRlts) << '\n';

	std::cout << "eig vectors:\n" << std::get<1>(eigRlts) << '\n';

	// ----------------------------------------
	// Decomposing Matrices
	// ----------------------------------------
	torch::Tensor X = torch::tensor({{1.0, 0.1, 0.1, 0.1},
	              {0.1, 3.0, 0.2, 0.3},
	              {0.1, 0.2, 5.0, 0.5},
	              {0.1, 0.3, 0.5, 9.0}});

	eigRlts = torch::linalg_eig(X);
	std::cout << "eig values:\n" << std::get<0>(eigRlts) << '\n';

	// -------------------------------------------------
	// A Useful Application: The Growth of Iterated Maps
	// -------------------------------------------------

	// Eigenvectors as Long Term Behavior

	int k = 5;
	torch::Tensor A = torch::randn({k, k}).to(torch::kDouble);
	std::cout << "A:\n" << A << '\n';

	// Behavior on Random Data
	// Calculate the sequence of norms after repeatedly applying `A`
	auto v_in = torch::randn({k, 1}).to(torch::kDouble);

	std::vector<double> norm_list, x;
	norm_list.push_back(torch::norm(v_in).item<double>());
	x.push_back(0.0);

	for(int i = 1;  i < 100; i++ ) {
    	//v_in = A @ v_in;
		v_in = torch::matmul(A, v_in);
		double vv = torch::norm(v_in).item<double>();
    	norm_list.push_back(vv);
    	x.push_back(i*1.0);
    }

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::plot(ax1, x, norm_list, "b")->line_width(2);
    matplot::xlabel(ax1, "Iteration");
    matplot::ylabel(ax1, "Value");
    matplot::title("Calculate the sequence of norms");
    matplot::show();

	// The norm is growing uncontrollably! Indeed if we take the list of quotients, we will see a pattern.
	// Compute the scaling factor of the norms
	std::vector<double>  norm_ratio_list;
	x.clear();
	//double cum = norm_list[0];
	for(int i = 1; i < 100; i++) {
		double vv = (norm_list[i]*1.0/norm_list[i-1]);
	    norm_ratio_list.push_back(vv);
	    x.push_back(i*1.0);
	//    cum += norm_list[i];
	}

	F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	ax1 = F->nexttile();
	matplot::plot(ax1, x, norm_ratio_list, "b")->line_width(2);
    matplot::xlabel(ax1, "Iteration");
    matplot::ylabel(ax1, "Ratio");
    matplot::title("Compute the scaling factor of the norms");
    matplot::show();

	// ----------------------------------------
	// Relating Back to Eigenvectors
	// ----------------------------------------
	// Compute the eigenvalues
	auto eigs = std::get<0>(torch::linalg_eig(A));
	eigs = torch::abs(eigs);
	//norm_eigs = [torch.abs(torch.tensor(x)) for x in eigs]
	auto TT = eigs.sort();
	auto norm_eigs = std::get<0>(TT);
	auto T2 = std::get<1>(TT);
	std::cout << "norms of eigenvalues:\n" << norm_eigs << '\n';
	std::cout << "get<1>:\n" << T2 << '\n';

	// ---------------------------------------
	// Fixing the Normalization
	// ---------------------------------------
	// Rescale the matrix `A`
	A /= torch::max(norm_eigs).data().item<double>();

	// Do the same experiment again
	v_in = torch::randn({k, 1}).to(torch::kDouble);

	norm_list.clear();
	x.clear();
	norm_list.push_back(torch::norm(v_in).item<double>());
	x.push_back(0.0);

	for(int i = 1;  i < 100; i++ ) {
	    //v_in = A @ v_in;
		v_in = torch::matmul(A, v_in);
	    norm_list.push_back(torch::norm(v_in).item<double>());
	    x.push_back(i*1.0);
	}

	F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	ax1 = F->nexttile();
	matplot::plot(ax1, x, norm_list, "b")->line_width(2);
    matplot::xlabel(ax1, "Iteration");
    matplot::ylabel(ax1, "Value");
    matplot::title("Fixing the Normalization");
    matplot::show();

	norm_ratio_list.clear();
	x.clear();

	for(int i = 1; i < 100; i++) {
		double vv = (norm_list[i]*1.0/norm_list[i - 1]);
		norm_ratio_list.push_back(vv);
		x.push_back(i*1.0);
	}

	F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	ax1 = F->nexttile();
	matplot::plot(ax1, x, norm_ratio_list, "b")->line_width(2);
    matplot::xlabel(ax1, "Iteration");
    matplot::ylabel(ax1, "Ratio");
    matplot::title("The ratio between consecutive norms");
    matplot::show();

	std::cout << "Done!\n";
}


