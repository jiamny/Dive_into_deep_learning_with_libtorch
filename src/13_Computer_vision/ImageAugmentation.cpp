#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/Ch_13_util.h"
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

	std::cout << "Original image\n";

	torch::Tensor imgT = CvMatToTensor("data/cat1.jpg", {});
	std::cout << imgT.sizes() << '\n';
	auto mimg = imgT.permute({1,2,0}).mul(255).to(torch::kByte).clone();

	std::vector<uchar> z(mimg.numel());
	std::memcpy(&(z[0]), mimg.data_ptr<unsigned char>(),sizeof(uchar)*mimg.numel());
	const uchar* zptr = &(z[0]);

	plt::figure_size(700, 500);
	plt::imshow(zptr, imgT.size(1), imgT.size(2), imgT.size(0));
	plt::show();
	plt::close();

	// -------------------------------------
	// Flipping and Cropping
	// -------------------------------------
	int num_rows=2, num_cols=4;
	double scale=1.5;
	cv::Mat img1, img2;

	// flips an image left and right with a 50%
	std::cout << "Flips an image left and right with a 50%\n";
	plt::figure_size(1200, 600);
	for( int r = 0; r < num_rows; r++ ) {
		for( int c = 0; c < num_cols; c++ ) {
			torch::Tensor imgt = CvMatToTensorAfterFlip("data/cat1.jpg", {}, 0.5, 0);
			const uchar* zptr = tensorToMatrix4Matplotlib(imgt.squeeze());
			plt::subplot2grid(num_rows, num_cols, r, c, 1, 1);
			plt::imshow(zptr, imgt.size(1), imgt.size(2), imgt.size(0));
		}
	}
	plt::show();
	plt::close();

	// flip an image up and down with a 50% chance
	std::cout << "Flips an image up and down with a 50% chance\n";
	plt::figure_size(1200, 600);
	for( int r = 0; r < num_rows; r++ ) {
		for( int c = 0; c < num_cols; c++ ) {
			torch::Tensor imgt = CvMatToTensorAfterFlip("data/cat1.jpg", {}, 0.5, 1);
			const uchar* zptr = tensorToMatrix4Matplotlib(imgt.squeeze());
			plt::subplot2grid(num_rows, num_cols, r, c, 1, 1);
			plt::imshow(zptr, imgt.size(1), imgt.size(2), imgt.size(0));
		}
	}
	plt::show();
	plt::close();

	// ----------------------------------------
	// Changing Colors
	// ----------------------------------------

	//randomly change the brightness
    double alpha = 1.0;   	// Simple contrast control
    double beta = 0.5;      // Simple brightness control
    std::cout << "Randomly change the brightness\n";
    plt::figure_size(1200, 600);
    for( int r = 0; r < num_rows; r++ ) {
    	for( int c = 0; c < num_cols; c++ ) {
    		torch::Tensor imgt = CvMatToTensorChangeBrightness("data/cat1.jpg", {}, alpha, beta);
    		const uchar* zptr = tensorToMatrix4Matplotlib(imgt.squeeze());
    		plt::subplot2grid(num_rows, num_cols, r, c, 1, 1);
    		plt::imshow(zptr, imgt.size(1), imgt.size(2), imgt.size(0));
    	}
    }
    plt::show();
    plt::close();

    // randomly change the hue
    int hue = 90;	// H is between 0-180 in OpenCV

    std::cout << "Randomly change the hue\n";
    plt::figure_size(1200, 600);
    for( int r = 0; r < num_rows; r++ ) {
    	for( int c = 0; c < num_cols; c++ ) {
    		torch::Tensor imgt = CvMatToTensorChangeHue("data/cat1.jpg", {}, hue);
    		const uchar* zptr = tensorToMatrix4Matplotlib(imgt.squeeze());
    		plt::subplot2grid(num_rows, num_cols, r, c, 1, 1);
    		plt::imshow(zptr, imgt.size(1), imgt.size(2), imgt.size(0));
    	}
    }
    plt::show();
    plt::close();

	std::cout << "Done!\n";
}



