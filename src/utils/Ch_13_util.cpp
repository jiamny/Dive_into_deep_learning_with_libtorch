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

cv::Mat TensorToCvMat(torch::Tensor img, bool is_float, bool  toBGR) {

	if( is_float ) {
		float maxV = torch::max(img).data().item<float>();
		if( maxV > 1.0 ) img.div_(maxV);
	}

	torch::Tensor rev_tensor;
	if( is_float ) {
		auto data_out = img.contiguous().detach().clone();
		rev_tensor = data_out.mul(255).to(torch::kByte).permute({1, 2, 0});
	} else {
		auto data_out = img.contiguous().detach().clone();
		rev_tensor = data_out.to(torch::kByte).permute({1, 2, 0});
	}

	//std::cout << "rev_tensor: " << rev_tensor.sizes() << '\n';
	// shape of tensor
	int64_t height = rev_tensor.size(0);
	int64_t width = rev_tensor.size(1);

	// Mat takes data form like {0,0,255,0,0,255,...} ({B,G,R,B,G,R,...})
	// so we must reshape tensor, otherwise we get a 3x3 grid
	auto tensor = rev_tensor.reshape({width * height * rev_tensor.size(2)});

	// CV_8UC3 is an 8-bit unsigned integer matrix/image with 3 channels
	cv::Mat rev_rgb_mat(cv::Size(width, height), CV_8UC3, tensor.data_ptr());
	cv::Mat rev_bgr_mat = rev_rgb_mat.clone();

	if( toBGR )
		cv::cvtColor(rev_bgr_mat, rev_bgr_mat, cv::COLOR_RGB2BGR);

	return rev_bgr_mat;
}


torch::Tensor  CvMatToTensor(std::string imgf, std::vector<int> img_size) {
	auto image = cv::imread(imgf.c_str());

	// ----------------------------------------------------------
	// opencv BGR format change to RGB
	// ----------------------------------------------------------
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

	if( img_size.size() > 0 ) {
		// ---------------------------------------------------------------
		// opencv resize the Size() - should be (width/cols x height/rows)
		// ---------------------------------------------------------------
		cv::resize(image, image, cv::Size(img_size[0], img_size[1]));
	}

	//std::cout << "r: " << image.rows << " c: " << image.cols << '\n';

	torch::Tensor imgT = torch::from_blob(image.data,
						{image.rows, image.cols, image.channels()}, at::TensorOptions(torch::kByte)).clone(); // Channels x Height x Width

	imgT = imgT.permute({2, 0, 1}).to(torch::kFloat).div_(255.0);

	return imgT;
}

torch::Tensor  CvMatToTensor2(cv::Mat img, std::vector<int> img_size, bool toRGB) {
	// ----------------------------------------------------------
	// opencv BGR format change to RGB
	// ----------------------------------------------------------
	if( toRGB )
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

	if( img_size.size() > 0 ) {
		// ---------------------------------------------------------------
		// opencv resize the Size() - should be (width/cols x height/rows)
		// ---------------------------------------------------------------
		cv::resize(img, img, cv::Size(img_size[0], img_size[1]));
	}

	torch::Tensor imgT = torch::from_blob(img.data,
						{img.rows, img.cols, img.channels()}, at::TensorOptions(torch::kByte)).clone(); // Channels x Height x Width

	imgT = imgT.permute({2, 0, 1}).to(torch::kFloat).div_(255.0);

	return imgT;
}

std::vector<uint8_t> tensorToMatrix4Matplotlib(torch::Tensor img, bool is_float, bool need_permute) {
	// OpenCV is BGR, Pillow is RGB
	torch::Tensor data_out = img.contiguous().detach().clone();
	torch::Tensor rev_tensor;
	if( is_float ) {
		if( need_permute )
			rev_tensor = data_out.mul(255).to(torch::kByte).permute({1, 2, 0});
		else
			rev_tensor = data_out.mul(255).to(torch::kByte);
	} else {
		if( need_permute )
			rev_tensor = data_out.to(torch::kByte).permute({1, 2, 0});
		else
			rev_tensor = data_out.to(torch::kByte);
	}

	//std::vector<uint8_t> z(mimg.data_ptr<uint8_t>(), mimg.data_ptr<uint8_t>() + sizeof(uint8_t)*mimg.numel());
	//uint8_t* zptr = &(z[0]);
	// shape of tensor
	int64_t height = rev_tensor.size(0);
	int64_t width = rev_tensor.size(1);

	// Mat takes data form like {0,0,255,0,0,255,...} ({B,G,R,B,G,R,...})
	// so we must reshape tensor, otherwise we get a 3x3 grid
	auto tensor = rev_tensor.reshape({width * height * rev_tensor.size(2)});

	// CV_8UC3 is an 8-bit unsigned integer matrix/image with 3 channels
	cv::Mat rev_rgb_mat(cv::Size(width, height), CV_8UC3, tensor.data_ptr());

	std::vector<uint8_t> z;
	if(rev_rgb_mat.isContinuous()) {
	  // z.assign(mat.datastart, mat.dataend); // <- has problems for sub-matrix like mat = big_mat.row(i)
	  z.assign(rev_rgb_mat.data, rev_rgb_mat.data + rev_rgb_mat.total()*rev_rgb_mat.channels());
	} else {
	  for (int i = 0; i < rev_rgb_mat.rows; ++i) {
	    z.insert(z.end(), rev_rgb_mat.ptr<uchar>(i), rev_rgb_mat.ptr<uchar>(i)+rev_rgb_mat.cols*rev_rgb_mat.channels());
	  }
	}

	return z;
}

torch::Tensor deNormalizeTensor(torch::Tensor imgT, std::vector<float> mean_, std::vector<float> std_) {

	torch::Tensor mean = torch::from_blob((float *)mean_.data(),
				{(long int)mean_.size(), 1, 1}, at::TensorOptions(torch::kFloat)).clone();  	// mean{C,1,1}
	torch::Tensor std = torch::from_blob((float *)std_.data(),
				{(long int)std_.size(), 1, 1}, at::TensorOptions(torch::kFloat)).clone();		// std{C,1,1}

	long int channels = imgT.size(0);

	torch::Tensor meanF = mean;
	if(channels < meanF.size(0)){
		meanF = meanF.split(/*split_size=*/channels, /*dim=*/0).at(0);  // meanF{*,1,1} ===> {C,1,1}
	}

	torch::Tensor stdF = std;
	if(channels < stdF.size(0)){
		stdF = stdF.split(/*split_size=*/channels, /*dim=*/0).at(0);	// stdF{*,1,1} ===> {C,1,1}
	}

	torch::Tensor data_out_src = imgT * stdF.to(imgT.device()) + meanF.to(imgT.device());  // data_in{C,H,W}, meanF{*,1,1}, stdF{*,1,1} ===> data_out_src{C,H,W}

	return data_out_src.contiguous().detach().clone();
}


torch::Tensor NormalizeTensor(torch::Tensor imgT, std::vector<float> mean_, std::vector<float> std_) {

	torch::Tensor mean = torch::from_blob((float *)mean_.data(),
				{(long int)mean_.size(), 1, 1}, at::TensorOptions(torch::kFloat)).clone();  	// mean{C,1,1}
	torch::Tensor std = torch::from_blob((float *)std_.data(),
				{(long int)std_.size(), 1, 1}, at::TensorOptions(torch::kFloat)).clone();		// std{C,1,1}

	long int channels = imgT.size(0);

	torch::Tensor meanF = mean;
	if(channels < meanF.size(0)){
		meanF = meanF.split(/*split_size=*/channels, /*dim=*/0).at(0);  // meanF{*,1,1} ===> {C,1,1}
	}

	torch::Tensor stdF = std;
	if(channels < stdF.size(0)){
		stdF = stdF.split(/*split_size=*/channels, /*dim=*/0).at(0);	// stdF{*,1,1} ===> {C,1,1}
	}

	torch::Tensor data_out_src = (imgT - meanF.to(imgT.device())) / stdF.to(imgT.device());  // data_in{C,H,W}, meanF{*,1,1}, stdF{*,1,1} ===> data_out_src{C,H,W}

	return data_out_src.contiguous().detach().clone();
}


std::string cvMatType2Str(int type) {
  std::string r = "";

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
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


std::vector<std::pair<std::string, std::string>> read_voc_images(const std::string voc_dir,
		bool is_train, int num_sample, bool shuffle, std::vector<int> cropSize) {
    //Read the banana detection dataset images and labels
	std::vector<std::pair<std::string, std::string>> imgpaths;

	std::string txt_fname = "";
	if( is_train )
		txt_fname = voc_dir + "/ImageSets/Segmentation/train.txt";
	else
		txt_fname = voc_dir + "/ImageSets/Segmentation/val.txt";

    bool rgb = false;
    std::ifstream file;

    file.open(txt_fname, std::ios_base::in);
    // Exit if file not opened successfully
    if( !file.is_open() ) {
    	std::cout << "File not read successfully" << std::endl;
    	std::cout << "Path given: " << txt_fname << std::endl;
    	exit(-1);
    }

    std::string fname;
    int num_images = 0;

	while( std::getline(file, fname) ) {
		//std::cout << fname << '\n';

		std::string imgf = voc_dir + "/JPEGImages/" + fname + ".jpg";
		std::string labf = voc_dir + "/SegmentationClass/" + fname + ".png";
/*
		auto imgT = CvMatToTensor(labf.c_str(), {img_size, img_size});
		auto rev_bgr_mat = TensorToCvMat(imgT);

		cv::imshow("rev_bgr_mat", rev_bgr_mat);
		cv::waitKey(-1);
		cv::destroyAllWindows();
*/
		if( cropSize.size() > 0 ) {
			cv::Mat fimg = cv::imread(imgf.c_str());
			cv::Mat limg = cv::imread(labf.c_str());
			//std::cout << "rows: " << fimg.rows << " cols: " << fimg.cols << '\n';
			if( fimg.cols >= cropSize[0] && fimg.rows >= cropSize[1] ) {
				if( num_sample < 1) {
					imgpaths.push_back(std::make_pair(imgf, labf));
				} else {
					if( num_images < num_sample ) {
						imgpaths.push_back(std::make_pair(imgf, labf));
						num_images++;
					}
				}
			}
		} else {
			if( num_sample < 1) {
				imgpaths.push_back(std::make_pair(imgf, labf));
			} else {
				if( num_images < num_sample ) {
					imgpaths.push_back(std::make_pair(imgf, labf));
					num_images++;
				}
			}
		}
	}

	if( shuffle ) {
		auto rng = std::default_random_engine {};
		std::shuffle(std::begin(imgpaths), std::end(imgpaths), rng);
	}

	std::cout << "read: " << imgpaths.size() << " examples\n";
	return imgpaths;
}

std::unordered_map<std::string, int> voc_colormap2label() {
    // Build the mapping from RGB to class indices for VOC labels.
	// Defined in :numref:`sec_semantic_segmentation`"""

	/*
    auto colormap2label = torch::zeros(256*256*256).to(torch::kLong);
    for(long i = 0; i < VOC_COLORMAP.size(); i++ ) {
    	std::vector<long> colormap = VOC_COLORMAP[i];
        colormap2label.index_put_({(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]}, i);
    }
    //std::cout << "colormap2label: \n" << torch::max(colormap2label) << std::endl;
    return colormap2label;
    */

	std::unordered_map<std::string, int> HSH_MAPS;

	for(int j = 0; j < VOC_COLORMAP.size(); j++) {
			std::string ss = std::to_string(VOC_COLORMAP[j][0]) +
								std::to_string(VOC_COLORMAP[j][1]) +
								std::to_string(VOC_COLORMAP[j][2]);
			HSH_MAPS.insert({ss, j});
	}
	return HSH_MAPS;
}


torch::Tensor voc_label_indices(torch::Tensor colormap, std::unordered_map<std::string, int> HSH_MAPS) {
    // Map any RGB values in VOC labels to their class indices.
	// Defined in :numref:`sec_semantic_segmentation`"""

	/*
    colormap = colormap.permute({1, 2, 0}).to(torch::kLong).clone();

    //std::cout << "colormap.max: " << torch::max(colormap) << '\n';
    // idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    auto idx = ((colormap.index({Slice(), Slice(), 0}) * 256 + colormap.index({Slice(), Slice(), 1})) * 256
           + colormap.index({Slice(), Slice(), 2}));
    //std::cout << "idx: " << idx.sizes() << '\n';
    //std::cout << "idx.max: " << torch::max(idx) << '\n';
    for( long r = 0; r < idx.size(0); r++ ) {
    	for( long c = 0; c < idx.size(1); c++ ) {
    		auto i = idx.index({r,c}).data().item<long>();
    		//std::cout << i << " " << colormap2label[i].data().item<long>() << "; ";
    		idx.index_put_({r,c}, colormap2label[i].data().item<long>());
    	}
    	//std::cout << '\n';
    }

    return idx.detach().clone(); //colormap2label[idx].clone();
    */
	colormap = colormap.permute({1, 2, 0}).to(torch::kByte).clone();

	torch::Tensor index = torch::zeros({colormap.size(0), colormap.size(1)}, at::TensorOptions(torch::kByte));
	for( int r = 0; r < colormap.size(0); r++ ) {
		for(int c = 0; c < colormap.size(1); c++ ) {
			auto t = colormap.index({r, c, Slice()});
			std::string ss = std::to_string(t[0].data().item<int>()) +
					std::to_string(t[1].data().item<int>()) +
					std::to_string(t[2].data().item<int>());
			  auto it = HSH_MAPS.find(ss);
			  if (it != HSH_MAPS.end()) {
				  index.index_put_({r, c}, HSH_MAPS[ss]);
			  }
		}
	}
	return index.detach().clone();
}

std::pair<torch::Tensor, torch::Tensor> voc_rand_crop(torch::Tensor feature, torch::Tensor label,
		int height, int width, std::vector<float> mean_, std::vector<float> std_, bool toRGB) {

	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_int_distribution<std::mt19937::result_type> Hdist(0, feature.size(1) - height);
	std::uniform_int_distribution<std::mt19937::result_type> Wdist(0, feature.size(2) - width);
	int H = Hdist(rng), W = Wdist(rng);

	feature = deNormalizeTensor(feature, mean_, std_);
	cv::Mat fmat = TensorToCvMat(feature);						// to BGR format
	cv::Mat lmat = TensorToCvMat(label, false);					// to BGR format

	cv::Mat fimage = fmat(cv::Range(H, H + height), cv::Range(W, W + width));
	cv::Mat limage = lmat(cv::Range(H, H + height), cv::Range(W, W + width));

	torch::Tensor fimg = CvMatToTensor2(fimage.clone(), {}, toRGB); // to RGB format
	auto ftsr = NormalizeTensor(fimg.clone(), mean_, std_);
	torch::Tensor ltsr = CvMatToTensor2(limage.clone(), {}, toRGB); // to RGB format
	ltsr = ltsr.mul(255).to(torch::kByte).clone();

	return std::make_pair(ftsr,ltsr);
}

torch::Tensor decode_segmap(torch::Tensor pred, int nc) {

	auto r = torch::zeros_like(pred).to(torch::kByte);
	auto g = torch::zeros_like(pred).to(torch::kByte);
	auto b = torch::zeros_like(pred).to(torch::kByte);

	for( int l = 0; l < nc; l++ ) {
		std::vector<int> clr = VOC_COLORMAP[l];
		//std::cout << clr << '\n';
		auto idx = {pred == l};
		//std::cout << idx.size() << '\n';

		if( idx.size() > 0 ) {
			r.index_put_({pred == l}, clr[0]);
			g.index_put_({pred == l}, clr[1]);
			b.index_put_({pred == l}, clr[2]);
		}
	}

	auto imgT = torch::stack({r, g, b}, 2).to(torch::kByte);
	std::cout << imgT.sizes() << '\n';
	return imgT;
}

