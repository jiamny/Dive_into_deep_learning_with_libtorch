#include "Ch_13_util.h"


#include <opencv2/opencv.hpp>
#include <opencv2/core/hal/interface.h>

void ShowManyImages(std::string title, std::vector<cv::Mat> imgs) {
	int size;
	int i;
	int m, n;
	int x, y;

	// w - Maximum number of images in a row
	// h - Maximum number of images in a column
	int w, h;

	// scale - How much we have to resize the image
	float scale;
	int max;

	// If the number of arguments is lesser than 0 or greater than 12
	// return without displaying
	if(imgs.size() <= 0) {
		std::cout << "Number of arguments too small....\n";
		return;
	}
	// Determine the size of the image,
	// and the number of rows/cols
	// from number of arguments
	else if (imgs.size() == 1) {
		w = h = 1;
		size = 300;
	} else if (imgs.size() == 2) {
		w = 2; h = 1;
		size = 300;
	} else if (imgs.size() == 3 || imgs.size() == 4) {
		w = 2; h = 2;
    	size = 300;
	} else if (imgs.size() == 5 || imgs.size() == 6) {
		w = 3; h = 2;
		size = 200;
	} else if (imgs.size() == 7 || imgs.size() == 8) {
		w = 4; h = 2;
		size = 200;
	} else {
		w = 4; h = 3;
		size = 150;
	}

	// Create a new 3 channel image
	cv::Mat DispImage = cv::Mat::zeros(cv::Size(100 + size*w, 60 + size*h), CV_8UC3);

	// Loop for nArgs number of arguments
	for (i = 0, m = 20, n = 20; i < imgs.size(); i++, m += (20 + size)) {
		// Get the Pointer to the IplImage
		cv::Mat img = imgs[i];

		// Check whether it is NULL or not
		// If it is NULL, release the image, and return
		if(img.empty()) {
			std::cout << "Invalid arguments\n";
			return;
		}

		// Find the width and height of the image
		x = img.cols;
		y = img.rows;

		// Find whether height or width is greater in order to resize the image
		max = (x > y)? x: y;

		// Find the scaling factor to resize the image
		scale = (float) ( (float) max / size );

		// Used to Align the images
		if( i % w == 0 && m!= 20) {
			m = 20;
			n+= 20 + size;
		}

		// Set the image ROI to display the current image
		// Resize the input image and copy the it to the Single Big Image
		cv::Rect ROI(m, n, (int)( x/scale ), (int)( y/scale ));
		cv::Mat temp;
		cv::resize(img, temp, cv::Size(ROI.width, ROI.height));
		temp.copyTo(DispImage(ROI));
	}

	// Create a new window, and show the Single Big Image
	cv::namedWindow( title, 1 );
	cv::imshow( title, DispImage);
	cv::waitKey(-1);
	cv::destroyAllWindows();
}

std::pair<cv::Mat, torch::Tensor> readImg( std::string filename ) {

	cv::Mat img = cv::imread(filename);
	int h = img.rows;
	int w = img.cols;

	cv::Mat cvtImg = img.clone();
	cv::cvtColor(img, cvtImg, cv::COLOR_BGR2RGB);
	torch::Tensor photo = torch::from_blob(cvtImg.data, {1, h, w, cvtImg.channels()}, at::TensorOptions(torch::kByte)).clone();
	photo = photo.toType(torch::kFloat);
	photo = photo.permute({0, 3, 1, 2});
	photo = photo.div_(255.0);

	return std::make_pair(img, photo);
}

torch::Tensor box_corner_to_center(torch::Tensor boxes) {
    //Convert from (upper-left, lower-right) to (center, width, height).
    //Defined in :numref:`sec_bbox`"""
    auto x1 = boxes.index({Slice(), 0});
	auto y1 = boxes.index({Slice(), 1});
	auto x2 = boxes.index({Slice(), 2});
	auto y2 = boxes.index({Slice(), 3});

    auto cx = (x1 + x2) / 2;
    auto cy = (y1 + y2) / 2;
    auto w = x2 - x1;
    auto h = y2 - y1;
    boxes = torch::stack({cx, cy, w, h}, -1);
    return boxes;
}

torch::Tensor  box_center_to_corner(torch::Tensor boxes) {
    //Convert from (center, width, height) to (upper-left, lower-right).
    //Defined in :numref:`sec_bbox`"""
    auto cx = boxes.index({Slice(), 0});
	auto cy = boxes.index({Slice(), 1});
	auto w = boxes.index({Slice(), 2});
	auto h = boxes.index({Slice(), 3});
    auto x1 = cx - 0.5 * w;
    auto y1 = cy - 0.5 * h;
    auto x2 = cx + 0.5 * w;
    auto y2 = cy + 0.5 * h;
    boxes = torch::stack({x1, y1, x2, y2}, -1);
    return boxes;
}

torch::Tensor multibox_prior(torch::Tensor data, std::vector<float> sizes, std::vector<float> ratios) {
    //Generate anchor boxes with different shapes centered on each pixel.
    int64_t in_height = data.size(2), in_width = data.size(3);		//data.shape[-2:]
    torch::Device device = data.device();
    int num_sizes = sizes.size(), num_ratios = ratios.size();

    int boxes_per_pixel = (num_sizes + num_ratios - 1);
    auto size_tensor  = torch::tensor(sizes).to(device);
    auto ratio_tensor = torch::tensor(ratios).to(device);

    // Offsets are required to move the anchor to the center of a pixel. Since
    // a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    float offset_h = 0.5, offset_w = 0.5;
    float steps_h = 1.0 / in_height;		// Scaled steps in y axis
    float steps_w = 1.0 / in_width;			// Scaled steps in x axis

    // Generate all center points for the anchor boxes
    auto center_h = (torch::arange(in_height).to(device) + offset_h) * steps_h;
    auto center_w = (torch::arange(in_width).to(device) + offset_w) * steps_w;

	auto T =  torch::meshgrid({center_h, center_w}, "ij");
	auto shift_y = T[0];
	auto shift_x = T[1];
    /*
	std::vector<torch::Tensor> rlt_y, rlt_x;
	for(int i = 0; i < in_width; i++ )
		rlt_y.push_back(center_h);
	for(int i = 0; i < in_height; i++ )
		rlt_x.push_back(center_w);
    auto shift_y = torch::stack(rlt_y, 0).t();
	auto shift_x = torch::stack(rlt_x, 0);
	*/

    shift_y = shift_y.reshape(-1);
    shift_x = shift_x.reshape(-1);

    // Generate `boxes_per_pixel` number of heights and widths that are later
    // used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    auto w = torch::cat({size_tensor * torch::sqrt(ratio_tensor[0]), sizes[0] * torch::sqrt(ratio_tensor.index({Slice(1, None)}))}, 0)
                   * in_height / in_width;  // Handle rectangular inputs

    auto h = torch::cat({size_tensor / torch::sqrt(ratio_tensor[0]), sizes[0] / torch::sqrt(ratio_tensor.index({Slice(1, None)}))}, 0);

    // Divide by 2 to get half height and half width
    auto anchor_manipulations = (torch::stack({-1*w, -1*h, w, h}).transpose(1, 0)).repeat({in_height * in_width, 1}) / 2;

    // Each center point will have `boxes_per_pixel` number of anchor boxes, so
    // generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    auto out_grid = torch::stack({shift_x, shift_y, shift_x, shift_y}, 1).repeat_interleave(boxes_per_pixel, 0);

    auto output = out_grid + anchor_manipulations;
    return output.unsqueeze(0);
}

//In order to [show all the anchor boxes centered on one pixel in the image],
//we define the following show_bboxes function to draw multiple bounding boxes on the image.

void setLabel(cv::Mat& im, std::string label, cv::Scalar text_color, cv::Scalar text_bk_color, const cv::Point& pt) {
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.4;
    int thickness = 1;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width + 5, -text.height), text_bk_color, cv::FILLED);
    cv::putText(im, label, pt, fontface, scale, text_color, thickness, cv::LINE_AA);
}

void show_bboxes(cv::Mat& img, torch::Tensor bboxes, std::vector<std::string> labels, std::vector<cv::Scalar> colors) {
    //Show bounding boxes.
    labels = make_list(labels, {});
    colors = make_list(colors, {cv::Scalar( 255, 0, 0 ) , cv::Scalar( 0, 255, 0 ), cv::Scalar( 0, 0, 255 ),
    							cv::Scalar(255,0,255), cv::Scalar(255,255,0)});

    for( int i = 0; i < bboxes.size(0); i++ ) {
    	auto bbox = bboxes[i];
        auto color = colors[i % colors.size()];

        int x1 = bbox[0].item<int>(), y1 = bbox[1].item<int>(), x2 = bbox[2].item<int>(), y2 = bbox[3].item<int>();
        // to complete show the box
        if( x1 <= 0 ) x1 = 4;
        if( y1 <= 20 ) y1 = 20;

        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 1, 4);

        // if there is label for the bbox
        if( labels.size() > 0 &&  labels.size() > i ) {
        	auto text_color = cv::Scalar(255,255,255);
        	if( color == cv::Scalar(255,255,255) ) text_color = cv::Scalar(0,0,0);

        	const cv::Point pt = cv::Point(x1, y1-5);
        	setLabel(img, labels[i], text_color, color, pt);
        }
    }
}


