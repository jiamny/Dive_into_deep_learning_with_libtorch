#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/ch_13_util.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <matplot/matplot.h>
using namespace matplot;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	std::cout << "Original image\n";

	torch::Tensor imgT = CvMatToTensor("data/cat1.jpg", {});
	std::cout << imgT.sizes() << '\n';

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);
	auto ax1 = F->nexttile();
	std::vector<std::vector<std::vector<unsigned char>>> z = tensorToMatrix4MatplotPP(imgT.squeeze().clone());
	matplot::imshow(ax1, z);
	matplot::show();

	// -------------------------------------
	// Flipping and Cropping
	// -------------------------------------
	int num_rows=2, num_cols=4;
	double scale=1.5;
	cv::Mat img1, img2;

	// flips an image left and right with a 50%
	std::cout << "Flips an image left and right with a 50%\n";

	auto f = figure(true);
	f->width(f->width() * 2);
	f->height(f->height() * 2);
	f->x_position(0);
	f->y_position(0);

	for( int r = 0; r < num_rows; r++ ) {
		for( int c = 0; c < num_cols; c++ ) {
			torch::Tensor imgt = CvMatToTensorAfterFlip("data/cat1.jpg", {}, 0.5, 0).squeeze();

			matplot::subplot(num_rows, num_cols, r*num_cols + c);
			std::vector<std::vector<std::vector<unsigned char>>> z = tensorToMatrix4MatplotPP(imgt.clone());
			matplot::imshow(z);
		}
		f->draw();
	}
	matplot::show();

	// flip an image up and down with a 50% chance
	std::cout << "Flips an image up and down with a 50% chance\n";

	f = figure(true);
	f->width(f->width() * 2);
	f->height(f->height() * 2);
	f->x_position(0);
	f->y_position(0);

	for( int r = 0; r < num_rows; r++ ) {
		for( int c = 0; c < num_cols; c++ ) {
			torch::Tensor imgt = CvMatToTensorAfterFlip("data/cat1.jpg", {}, 0.5, 1).squeeze();

			matplot::subplot(num_rows, num_cols, r*num_cols + c);
			std::vector<std::vector<std::vector<unsigned char>>> z = tensorToMatrix4MatplotPP(imgt.clone());
			matplot::imshow(z);
		}
		f->draw();
	}
	matplot::show();

	// ----------------------------------------
	// Changing Colors
	// ----------------------------------------

	//randomly change the brightness
    double alpha = 1.0;   	// Simple contrast control
    double beta = 0.5;      // Simple brightness control
    std::cout << "Randomly change the brightness\n";

	f = figure(true);
	f->width(f->width() * 2);
	f->height(f->height() * 2);
	f->x_position(0);
	f->y_position(0);

    for( int r = 0; r < num_rows; r++ ) {
    	for( int c = 0; c < num_cols; c++ ) {
    		torch::Tensor imgt = CvMatToTensorChangeBrightness("data/cat1.jpg", {}, alpha, beta).squeeze();

			matplot::subplot(num_rows, num_cols, r*num_cols + c);
			std::vector<std::vector<std::vector<unsigned char>>> z = tensorToMatrix4MatplotPP(imgt.clone());
			matplot::imshow(z);
    	}
    	f->draw();
    }
    matplot::show();


    // randomly change the hue
    int hue = 90;	// H is between 0-180 in OpenCV

    std::cout << "Randomly change the hue\n";

	f = figure(true);
	f->width(f->width() * 2);
	f->height(f->height() * 2);
	f->x_position(0);
	f->y_position(0);

    for( int r = 0; r < num_rows; r++ ) {
    	for( int c = 0; c < num_cols; c++ ) {
    		torch::Tensor imgt = CvMatToTensorChangeHue("data/cat1.jpg", {}, hue).squeeze();

			matplot::subplot(num_rows, num_cols, r*num_cols + c);
			std::vector<std::vector<std::vector<unsigned char>>> z = tensorToMatrix4MatplotPP(imgt.clone());
			matplot::imshow(z);
    	}
    	f->draw();
    }

    matplot::show();

	std::cout << "Done!\n";
}



