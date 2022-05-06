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

void show_bboxes(cv::Mat& img, torch::Tensor bboxes, std::vector<std::string> labels, std::vector<cv::Scalar> colors, int lineWidth) {
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

        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, lineWidth, 4);

        // if there is label for the bbox
        if( labels.size() > 0 &&  labels.size() > i ) {
        	auto text_color = cv::Scalar(255,255,255);
        	if( color == cv::Scalar(255,255,255) ) text_color = cv::Scalar(0,0,0);

        	const cv::Point pt = cv::Point(x1, y1-5);
        	setLabel(img, labels[i], text_color, color, pt);
        }
    }
}

cv::Mat TensorToCvMat(torch::Tensor img) {

	float maxV = torch::max(img).data().item<float>();
	if( maxV > 1.0 ) img.div_(maxV);

	torch::Tensor data_out = img.contiguous().detach().clone();
	auto rev_tensor = data_out.mul(255).to(torch::kByte).permute({1, 2, 0});

	// shape of tensor
	int64_t height = rev_tensor.size(0);
	int64_t width = rev_tensor.size(1);

	// Mat takes data form like {0,0,255,0,0,255,...} ({B,G,R,B,G,R,...})
	// so we must reshape tensor, otherwise we get a 3x3 grid
	auto tensor = rev_tensor.reshape({width * height * rev_tensor.size(2)});

	// CV_8UC3 is an 8-bit unsigned integer matrix/image with 3 channels
	cv::Mat rev_rgb_mat(cv::Size(width, height), CV_8UC3, tensor.data_ptr());
	cv::Mat rev_bgr_mat = rev_rgb_mat.clone();

	cv::cvtColor(rev_bgr_mat, rev_bgr_mat, cv::COLOR_RGB2BGR);

	return rev_bgr_mat;
}


torch::Tensor  CvMatToTensor(std::string imgf, std::vector<int> img_size) {
	auto image = cv::imread(imgf.c_str());

	// ----------------------------------------------------------
	// opencv BGR format change to RGB
	// ----------------------------------------------------------
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

	if( img_size.size() > 0 ) cv::resize(image, image, cv::Size(img_size[0], img_size[1]));

	torch::Tensor imgT = torch::from_blob(image.data,
						{image.rows, image.cols, image.channels()}, at::TensorOptions(torch::kByte)).clone(); // Channels x Height x Width

	imgT = imgT.permute({2, 0, 1}).to(torch::kFloat).div_(255.0);

	return imgT;
}

const uchar* tensorToMatrix4Matplotlib(torch::Tensor imgT) {
	// OpenCV is BGR, Pillow is RGB
	torch::Tensor mimg = imgT.permute({1,2,0}).mul(255).to(torch::kByte).clone();

	std::vector<uchar> z(mimg.numel());
	std::memcpy(&(z[0]), mimg.data_ptr<unsigned char>(),sizeof(uchar)*mimg.numel());
	const uchar* zptr = &(z[0]);

	return zptr;
}

torch::Tensor  CvMatToTensorAfterFlip(std::string file, std::vector<int> img_size, double fP, int flip_axis) {
	cv::Mat image = cv::imread(file.c_str());
	/*
	 *  flip Code is set to 0, which flips around the x-axis. If flipCode is set greater than zero (e.g., +1),
	 *  the image will be flipped around the yaxis, and if set to a negative value (e.g., -1), the image will
	 *  be flipped about both axes.
	 */
	cv::Mat dst;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0, 1);
	if( dis(gen) > fP ) {
		cv::flip(image, dst, flip_axis);
	} else {
		dst = image;
	}

	// ----------------------------------------------------------
	// opencv BGR format change to RGB
	// ----------------------------------------------------------
	cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
	if( img_size.size() > 0 ) cv::resize(dst, dst, cv::Size(img_size[0], img_size[1]));

	torch::Tensor imgT = torch::from_blob(dst.data,
						{dst.rows, dst.cols, dst.channels()}, at::TensorOptions(torch::kByte)).clone(); // Channels x Height x Width

	imgT = imgT.permute({2, 0, 1}).to(torch::kFloat).div_(255.0);

	return imgT;
}

torch::Tensor  CvMatToTensorChangeBrightness(std::string file, std::vector<int> img_size, double alpha, double beta) {
	cv::Mat image = cv::imread(file.c_str());
	/*
	 *  flip Code is set to 0, which flips around the x-axis. If flipCode is set greater than zero (e.g., +1),
	 *  the image will be flipped around the yaxis, and if set to a negative value (e.g., -1), the image will
	 *  be flipped about both axes.
	 */
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0, 1);

	cv::Mat new_image = cv::Mat::zeros( image.size(), image.type() );
	// randomly change the brightness] of the image to a value between 50% (1−0.5)
	// and 50% (0.5 - 1) of the original image
	for( int y = 0; y < image.rows; y++ ) {
		for( int x = 0; x < image.cols; x++ ) {
		    for( int c = 0; c < image.channels(); c++ ) {
		        beta = dis(gen);
		        new_image.at<cv::Vec3b>(y,x)[c] =
		        			            cv::saturate_cast<uchar>( alpha*image.at<cv::Vec3b>(y,x)[c] + beta );
		    }
		}
	}

	// ----------------------------------------------------------
	// opencv BGR format change to RGB
	// ----------------------------------------------------------
	cv::cvtColor(new_image, new_image, cv::COLOR_BGR2RGB);
	if( img_size.size() > 0 ) cv::resize(new_image, new_image, cv::Size(img_size[0], img_size[1]));

	auto imgT = torch::from_blob(new_image.data,
							{new_image.rows, new_image.cols, new_image.channels()}, at::TensorOptions(torch::kByte)).clone(); // Channels x Height x Width

	imgT = imgT.permute({2, 0, 1}).to(torch::kFloat).div_(255.0);

	return imgT;
}

torch::Tensor  CvMatToTensorChangeHue(std::string file, std::vector<int> img_size, int hue) {
	cv::Mat image = cv::imread(file.c_str());
	/*
	 *  flip Code is set to 0, which flips around the x-axis. If flipCode is set greater than zero (e.g., +1),
	 *  the image will be flipped around the yaxis, and if set to a negative value (e.g., -1), the image will
	 *  be flipped about both axes.
	 */
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0, 1);

	cv::Mat hsv = image.clone();
	cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

	std::vector<cv::Mat> hsv_vec;
	cv::split(hsv, hsv_vec);
	cv::Mat &H = hsv_vec[0];	// hue
	cv::Mat &S = hsv_vec[1];	// saturation
	cv::Mat &V = hsv_vec[2];	// brightness

	if( dis(gen) > 0.5 ) {
		//image = (V > 10); 	// non-zero pixels in the original image
		hsv_vec[0] = hue; 		// H is between 0-180 in OpenCV
	}
	cv::merge(hsv_vec, hsv);
	cv::Mat new_image;
	cv::cvtColor(hsv, new_image, cv::COLOR_HSV2BGR);

	// ----------------------------------------------------------
	// opencv BGR format change to RGB
	// ----------------------------------------------------------
	cv::cvtColor(new_image, new_image, cv::COLOR_BGR2RGB);
	if( img_size.size() > 0 ) cv::resize(new_image, new_image, cv::Size(img_size[0], img_size[1]));

	auto imgT = torch::from_blob(new_image.data,
							{new_image.rows, new_image.cols, new_image.channels()}, at::TensorOptions(torch::kByte)).clone(); // Channels x Height x Width

	imgT = imgT.permute({2, 0, 1}).to(torch::kFloat).div_(255.0);

	return imgT;
}


