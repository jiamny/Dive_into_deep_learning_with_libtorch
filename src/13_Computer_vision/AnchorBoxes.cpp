#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/ch_13_util.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "../TempHelpFunctions.hpp"


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	// -----------------------------------------------------
	// Generating Multiple Anchor Boxes
	// -----------------------------------------------------

	auto rlt = readImg("./data/catdog.jpg");
	cv::Mat img = rlt.first;
	torch::Tensor imgT = rlt.second;
	std::cout << imgT.sizes() << '\n';

	int h = imgT.size(2), w = imgT.size(3);
	std::cout << "h: " << h << ", w: " << w << '\n';

	auto X = torch::rand({1, 3, h, w});		//Construct input data
	auto Y = multibox_prior(X, {0.75, 0.5, 0.25}, {1, 2, 0.5});
	std::cout << Y.sizes() << '\n';

	auto boxes = Y.reshape({h, w, 5, 4});
	std::cout << "boxes.index({250, 250, 0, Slice()}): " << boxes.index({250, 250, 0, Slice()}) << '\n';

	cv::Mat kpImg = img.clone();
	// show all the anchor boxes centered on one pixel in the image
	auto bbox_scale = torch::tensor({w, h, w, h});
	auto bboxes = boxes.index({250, 250, Slice(), Slice()}) * bbox_scale;

	show_bboxes(img, bboxes, {"s=0.75, r=1", "s=0.5, r=1", "s=0.25, r=1", "s=0.75, r=2",
	             "s=0.75, r=0.5"}, {});

	cv::imshow("exp_1", img);
	cv::waitKey(-1);

	// An Example
	auto ground_truth = torch::tensor({{0.0, 0.1, 0.08, 0.52, 0.92},
	                         {1.0, 0.55, 0.2, 0.9, 0.88}});

	auto anchors = torch::tensor({{0.0, 0.1, 0.2, 0.3}, {0.15, 0.2, 0.4, 0.4},
	                    {0.63, 0.05, 0.88, 0.98}, {0.66, 0.45, 0.8, 0.8},
	                    {0.57, 0.3, 0.92, 0.9}});

	img = kpImg.clone();
	bboxes = ground_truth.index({Slice(), Slice(1, None)}) * bbox_scale;

	show_bboxes(img, bboxes, {"dog", "cat"}, {cv::Scalar(0,0,0)});
	cv::imshow("ground_truth", img);
	cv::waitKey(-1);

	show_bboxes(img, anchors * bbox_scale, {"0", "1", "2", "3", "4"}, {});
	cv::imshow("anchors", img);
	cv::waitKey(-1);

	std::cout << "anchors.unsqueeze(0): " <<  anchors.unsqueeze(0).sizes() << '\n';
	std::cout << "ground_truth.unsqueeze(0): " <<  ground_truth.unsqueeze(0).sizes() << '\n';

	auto labels = multibox_target(anchors.unsqueeze(0), ground_truth.unsqueeze(0));

	std::cout << "labels[2]: " << std::get<2>(labels) << '\n';

	std::cout << "labels[1]: " << std::get<1>(labels) << '\n';

	// -------------------------------------------------------
	// Predicting Bounding Boxes with Non-Maximum Suppression
	// -------------------------------------------------------

	// Now let us [apply the above implementations to a concrete example with four anchor boxes].
	// For simplicity, we assume that the predicted offsets are all zeros. This means that the predicted
	// bounding boxes are anchor boxes. For each class among the background, dog, and cat, we also
	// define its predicted likelihood.

	anchors = torch::tensor({{0.1, 0.08, 0.52, 0.92},
							{0.08, 0.2, 0.56, 0.95},
	                        {0.15, 0.3, 0.62, 0.91},
							{0.55, 0.2, 0.9, 0.88}});
	std::cout << anchors.numel() << '\n';

	auto offset_preds = torch::tensor({0.0, 0.0, 0.0, 0.0,
									   0.0, 0.0, 0.0, 0.0,
									   0.0, 0.0, 0.0, 0.0,
									   0.0, 0.0, 0.0, 0.0});
	auto cls_probs = torch::tensor({{0.0, 0.0, 0.0, 0.0},  		// Predicted background likelihood
	                      	  	  	{0.9, 0.8, 0.7, 0.1},  		// Predicted dog likelihood
									{0.1, 0.2, 0.3, 0.9}}); 	// Predicted cat likelihood

	// plot these predicted bounding boxes with their confidence on the image
	img = kpImg.clone();
	bboxes = anchors * bbox_scale;

	show_bboxes(img, bboxes, {"dog=0.9", "dog=0.8", "dog=0.7", "cat=0.9"}, {});
	cv::imshow("four anchor boxes", img);
	cv::waitKey(-1);

	// output the final predicted bounding box kept by non-maximum suppression
	auto output = multibox_detection(cls_probs.unsqueeze(0), offset_preds.unsqueeze(0), anchors.unsqueeze(0), 0.5);
	std::cout << "output: " << output[0] << '\n';

	img = kpImg.clone();
	output.squeeze_();

	std::vector<std::string> tlabels = {"dog=", "cat="};

	for(int k = 0; k < output.size(0); k++ ) {
		auto t = output[k];
	    if( t[0].item<int>() == -1 )
	        continue;

	    int lidx = t[0].item<float>();
	    auto tbox = t.index({Slice(2, None)}) * bbox_scale;

	    show_bboxes(img, tbox.unsqueeze(0), {tlabels[lidx] + to_string_with_precision(t[1].item<float>(), 2)}, {});
	}

	cv::imshow("The final predicted bounding box", img);
	cv::waitKey(-1);
	cv::destroyAllWindows();

	std::cout << "Done!\n";
	return 0;
}


