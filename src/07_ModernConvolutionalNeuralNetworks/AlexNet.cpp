
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../utils.h"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

using Options = torch::nn::Conv2dOptions;

using Data = std::vector<std::pair<std::string, long>>;
using Example = torch::data::Example<>;

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {

 public:
  CustomDataset(const Data& dt, std::string fileDir, int imgSize) {
	  data=dt;
	  datasetPath=fileDir;
	  image_size=imgSize;
  }

  Example get(size_t index) {
    std::string path = datasetPath + data[index].first;
//    auto tlabel = torch::from_blob(&data[index].second, {1}, torch::kLong);
//    std::cout << "path = " << path << " label = " << tlabel << "\n";
//    auto mat = cv::imread(path.c_str(), 1);
    cv::Mat mat = cv::imread(path.c_str(), cv::IMREAD_COLOR);

    if(! mat.empty() ) {
//    	cv::namedWindow("Original Image");
//    	cv::imshow("Original Image",mat);
//    	cv::waitKey(0);
//    	std::cout << "ok!\n";

    	// ----------------------------------------------------------
    	// opencv BGR format change to RGB
    	// ----------------------------------------------------------
    	cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);

        int h = image_size, w = image_size;
        int im_h = mat.rows, im_w = mat.cols, chs = mat.channels();
        float res_aspect_ratio = w*1.0/h;
        float input_aspect_ratio = im_w*1.0/im_h;

        int dif = im_w;
        if( im_h > im_w ) int dif = im_h;

        int interpolation = cv::INTER_CUBIC;
        if( dif > static_cast<int>((h+w)*1.0/2) ) interpolation = cv::INTER_AREA;

        cv::Mat Y;
        if( input_aspect_ratio != res_aspect_ratio ) {
            if( input_aspect_ratio > res_aspect_ratio ) {
                int im_w_r = static_cast<int>(input_aspect_ratio*h);
                int im_h_r = h;
                cv::resize(mat, mat, cv::Size(im_w_r, im_h_r), interpolation);
                int x1 = static_cast<int>((im_w_r - w)/2);
                int x2 = x1 + w;
                mat(cv::Rect(x1, 0, w, im_h_r)).copyTo(Y);
            }

            if( input_aspect_ratio < res_aspect_ratio ) {
                int im_w_r = w;
                int im_h_r = static_cast<int>(w/input_aspect_ratio);
                cv::resize(mat, mat, cv::Size(im_w_r , im_h_r), interpolation);
                int y1 = static_cast<int>((im_h_r - h)/2);
                int y2 = y1 + h;
                mat(cv::Rect(0, y1, im_w_r, h)).copyTo(Y); // startX,startY,cols,rows
            }
        } else {
        	 cv::resize(mat, Y, cv::Size(w, h), interpolation);
        }

        int label = data[index].second;

        torch::Tensor img_tensor = torch::from_blob(Y.data, {  Y.channels(), Y.rows, Y.cols }, torch::kByte); // Channels x Height x Width
        torch::Tensor label_tensor = torch::full({1}, label);

    	return {img_tensor.clone().to(torch::kFloat32).div_(255.0), label_tensor.clone().to(torch::kInt64)};
    } else {

    	torch::data::Example<> EE;
    	return(EE);
    }
  }

  torch::optional<size_t> size() const {
    return data.size();
  }

 private:
  Data data;
  std::string datasetPath;
  int image_size;
};

std::pair<Data, Data> readInfo( std::string infoFilePath ) {
  Data train, test;

  std::ifstream stream( infoFilePath.c_str());
  assert(stream.is_open());

  long label;
  std::string path, type;

  while (true) {
    stream >> path >> label >> type;
//    std::cout << path << " " << label << " " << type << std::endl;
    if (type == "train")
      train.push_back(std::make_pair(path, label));
    else if (type == "test")
      test.push_back(std::make_pair(path, label));
    else
      assert(false);

    if (stream.eof())
      break;
  }

  std::random_shuffle(train.begin(), train.end());
  std::random_shuffle(test.begin(), test.end());
  return std::make_pair(train, test);
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	std::string datasetPath = "./data/Caltech_101/";
	std::string infoFilePath = "./data/Caltech_101_info.txt";

	int64_t imgSize = 224;
	int64_t train_batch_size = 128;
	int64_t test_batch_size = 200;
	size_t iterations = 20;
	size_t log_interval = 20;

	auto data = readInfo( infoFilePath );

	auto train_set = CustomDataset(data.first, datasetPath, imgSize).map(torch::data::transforms::Stack<>());
	auto train_size = train_set.size().value();

	std::cout << "train_size = " << train_size << std::endl;
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
	          	  	  	  	  	  	  	  	  	  	  std::move(train_set),
													  torch::data::DataLoaderOptions()
													  .batch_size(train_batch_size)
													  .workers(2)
													  .enforce_ordering(false));
	auto test_set =
	      CustomDataset(data.second, datasetPath, imgSize).map(torch::data::transforms::Stack<>());
	std::cout << test_set.size().value() << std::endl;

	auto test_size = test_set.size().value();
	std::cout << "test_size = " << test_size << std::endl;

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
	                                                 std::move(test_set),
													 torch::data::DataLoaderOptions()
													 .batch_size(test_batch_size)
													 .workers(2)
													 .enforce_ordering(false));

	// Deep Convolutional Neural Networks (AlexNet)
	auto net = torch::nn::Sequential(
		//# Here, we use a larger 11 x 11 window to capture objects. At the same
    	//# time, we use a stride of 4 to greatly reduce the height and width of the
    	//# output. Here, the number of output channels is much larger than that in
    	//# LeNet
	    torch::nn::Conv2d(Options(3, 96, 11).stride(4).padding(1)), torch::nn::ReLU(),
	    torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2)),
		//# Make the convolution window smaller, set padding to 2 for consistent
    	//# height and width across the input and output, and increase the number of
    	//# output channels
		torch::nn::Conv2d(Options(96, 256, 5).padding(2)), torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2)),
	    //# Use three successive convolutional layers and a smaller convolution
	    //# window. Except for the final convolutional layer, the number of output
	    //# channels is further increased. Pooling layers are not used to reduce the
	    //# height and width of input after the first two convolutional layers
		torch::nn::Conv2d(Options(256, 384, 3).padding(1)), torch::nn::ReLU(),
		torch::nn::Conv2d(Options(384, 384, 3).padding(1)), torch::nn::ReLU(),
		torch::nn::Conv2d(Options(384, 256, 3).padding(1)), torch::nn::ReLU(),
		torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2)), torch::nn::Flatten(),
	    //# Here, the number of outputs of the fully-connected layer is several
	    //# times larger than that in LeNet. Use the dropout layer to mitigate
	    //# overfitting
	    torch::nn::Linear(6400, 4096), torch::nn::ReLU(), torch::nn::Dropout(0.5),
	    torch::nn::Linear(4096, 4096), torch::nn::ReLU(), torch::nn::Dropout(0.5),
	    //# Output layer. Since we are using Fashion-MNIST, the number of classes is
	    //# 10, instead of 1000 as in the paper
	    torch::nn::Linear(4096, 102));

	// make sure that its operations line up with what we expect from
	//auto X = torch::randn({1, 3, imgSize, imgSize}).to(torch::kFloat32);
	//std::cout << net->forward(X) << std::endl;


	std::vector<float> train_loss;
	std::vector<float> train_acc;
	std::vector<float> test_acc;
	std::vector<float> xx;
/*
	auto batch = *train_loader->begin();
	auto data  = batch.data.to(device);
	auto y  = batch.target.to(device).flatten(0, -1);
	std::cout << "y: " << y << std::endl;

	auto y_hat = net->forward(data);
	std::cout << "y_hat: " << y_hat << std::endl;
	auto l = loss(y_hat, y);
	std::cout << l.item<float>() * data.size(0) << std::endl;
	std::cout << accuracy( y_hat, y) << std::endl;
	optimizer.zero_grad();
	l.backward();
	optimizer.step();
*/
	net->to(device);

	auto optimizer = torch::optim::Adam(net->parameters(), 0.001);

	auto loss_fn = torch::nn::CrossEntropyLoss();

	for (size_t i = 0; i < iterations; ++i) {
	    //train(network, *train_loader, optimizer, loss_fn, i + 1, train_size);
		size_t index = 0;
		net->train();
		float Loss = 0, Acc = 0;
		std::cout << "Start train ...\n";

		for (auto& batch : *train_loader) {
		    auto data = batch.data.to(device);
		    auto targets = batch.target.to(device).view({-1});

		    if( index == 0 ) std::cout << data.sizes() << std::endl;

		    auto output = net->forward(data);
		    auto loss = loss_fn(output, targets);
		    //assert(!std::isnan(loss.template item<float>()));
		    auto acc = output.argmax(1).eq(targets).sum();

		    optimizer.zero_grad();
		    loss.backward();
		    optimizer.step();

		    Loss += loss.item<float>() * data.size(0);
		    Acc += acc.template item<float>();

		    if(index++ % log_interval == 0) {
		    	auto end = std::min(train_size, (index + 1) * train_batch_size);

		    	std::cout << "Train Epoch: " << i << " " << end << "/" << train_size
		                << "\tLoss: " << Loss / end << "\tAcc: " << Acc / end
		                << std::endl;
		    }
	    }

		train_loss.push_back(Loss / train_size);
		train_acc.push_back(Acc / train_size );

	    std::cout << std::endl;
	    //test(network, *test_loader, loss_fn, test_size);
	    index = 0;
	    net->eval();
	    torch::NoGradGuard no_grad;
	    Loss = 0;
	    Acc = 0;

	    for (const auto& batch : *test_loader) {
	    	auto data = batch.data.to(device);
	        auto targets = batch.target.to(device).view({-1});

	        auto output = net->forward(data);
	        auto loss = loss_fn(output, targets);
	        //assert(!std::isnan(loss.template item<float>()));
	        auto acc = output.argmax(1).eq(targets).sum();

	        Loss += loss.item<float>() * data.size(0);
	        Acc += acc.template item<float>();
	    }

	    if (index++ % log_interval == 0) {
	        std::cout << "Test Loss: " << Loss / test_size
	                  << "\tAcc: " << Acc / test_size << std::endl;
	    }
	    test_acc.push_back(Acc / test_size);
	    xx.push_back(i*1.0);

	    std::cout << std::endl;
	}
	plt::figure_size(800, 600);
	plt::subplot(1, 1, 1);
	plt::ylim(0.0, 0.9);
	plt::named_plot("Train loss", xx, train_loss, "b");
	plt::named_plot("Train acc", xx, train_acc, "g--");
	plt::named_plot("Valid acc", xx, test_acc, "r-.");
	plt::ylabel("loss; acc");
	plt::xlabel("epoch");
	plt::legend();
	plt::show();

	std::cout << "Done!\n";
	return 0;
}




