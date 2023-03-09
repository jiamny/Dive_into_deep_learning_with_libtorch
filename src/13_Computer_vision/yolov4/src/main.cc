#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "darknet.h"
#include "coco_names.h"
//#include "../../../utils.h"
#include "../../../TempHelpFunctions.hpp"

void setLabel(cv::Mat& im, std::string label, cv::Scalar text_color, cv::Scalar text_bk_color, const cv::Point& pt) {
    int fontface = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.4;
    int thickness = 1;
    int baseline = 0;

    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::rectangle(im, pt + cv::Point(0, baseline), pt + cv::Point(text.width + 5, - text.height), text_bk_color, cv::FILLED);
    cv::putText(im, label, pt, fontface, scale, text_color, thickness, cv::LINE_AA);
}

int main(int argc, char* argv[]) {
  std::cout << "hello\n";

//  if (argc != 4) {
//    std::cerr << "usage: yolov4 <cfg_path>, <weight_path> <image path>\n";
//    return -1;
//  }
  std::string cfg_path    = "./src/13_Computer_vision/yolov4/yolov4.cfg";
  std::string weight_path = "./src/13_Computer_vision/yolov4/yolov4.weights";
  std::string image_path  = "./data/dogbike.jpg";

  torch::DeviceType device_type = torch::kCPU;

  torch::Device device(device_type);
  std::string cfg_file = cfg_path;
  if (argc == 4) cfg_file = argv[1];

  Darknet net(cfg_file.c_str(), &device);
  int input_image_size = net.get_input_size();

  std::cout << "loading weight ..." << std::endl;

  if (argc == 4) weight_path = argv[2];
  net.load_darknet_weights(weight_path.c_str());
  std::cout << "weight loaded ..." << std::endl;

  cv::Mat origin_image, resized_image;
  net.to(device);
  torch::NoGradGuard no_grad;
  net.eval();

  if (argc == 4) image_path = argv[3];
  origin_image = cv::imread(image_path);

  cv::cvtColor(origin_image, resized_image, cv::COLOR_BGR2RGB);
  cv::resize(resized_image, resized_image, { input_image_size , input_image_size });

  auto img_tensor = torch::from_blob(resized_image.data, { resized_image.rows, resized_image.cols, 3 },
		  	  	  	  	  	  	  	  at::TensorOptions(torch::kByte));

  img_tensor = img_tensor.permute({ 2, 0, 1 }).unsqueeze(0).to(torch::kFloat) / 255.0;

  auto start = std::chrono::high_resolution_clock::now();
  std::cout << "img_tensor:\n" << img_tensor.sizes() << '\n';

  auto result = net.predict(img_tensor, coco_class_names.size(), 0.6, 0.4);
  std::cout << "result:\n" << result << '\n';

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); 
  std::cout<<  "time cost: " << duration.count()<< " ms\n";

  if (result.dim() == 1) {
    std::cout << "no object found" << std::endl;
  } else {
    int obj_num = result.size(0);

    std::cout << obj_num << " objects found" << std::endl;

    cv::cvtColor(resized_image, resized_image, cv::COLOR_RGB2BGR);

    float w_scale = float(origin_image.cols) / input_image_size;
    float h_scale = float(origin_image.rows) / input_image_size;

    result.select(1, 1).mul_(w_scale);
    result.select(1, 2).mul_(h_scale);
    result.select(1, 3).mul_(w_scale);
    result.select(1, 4).mul_(h_scale);

    auto result_data = result.accessor<float, 2>();
 
    for (int i = 0; i < result.size(0); i++) {

      cv::rectangle(origin_image, cv::Point(result_data[i][1], result_data[i][2]),
    		  	  	  cv::Point(result_data[i][3], result_data[i][4]), cv::Scalar(0, 0, 255), 1, 1, 0);

      int clas_id = static_cast<size_t>(result_data[i][7]);
      float score = result_data[i][6];
      std::string text = coco_class_names[clas_id] + "-" + to_string_with_precision(score, 3);

      setLabel(origin_image, text, cv::Scalar(255, 255, 255), cv::Scalar(0, 0, 255),
    		   cv::Point(result_data[i][1], result_data[i][2]- 4));
    }

    //cv::imwrite("det_result.jpg", origin_image);
    cv::imshow("det_result", origin_image);
    cv::waitKey(0);
    cv::destroyAllWindows();
  }
  std::cout << "Done" << std::endl;
}
