#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdint.h>
#include <torch/torch.h>
#include <torch/nn.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Parameters {
  int image_size = 224;
  size_t train_batch_size = 128;
  size_t test_batch_size = 200;
  size_t iterations = 10;
  size_t log_interval = 20;
  // path must end in delimiter
  std::string datasetPath = "./data/Caltech_101/";
  std::string infoFilePath = "./data/Caltech_101_info.txt";
  torch::DeviceType device = torch::kCPU;
};

static Parameters parameters;

using Options = torch::nn::Conv2dOptions;

using Data = std::vector<std::pair<std::string, long>>;
using Example = torch::data::Example<>;

class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {

  Data data;

 public:
  CustomDataset(const Data& data) : data(data) {}

  Example get(size_t index) {
    std::string path = parameters.datasetPath + data[index].first;
//    auto tlabel = torch::from_blob(&data[index].second, {1}, torch::kLong);
//    std::cout << "path = " << path << " label = " << tlabel << "\n";
    auto mat = cv::imread(path.c_str(), 1);

    if(! mat.empty() ) {
//    	cv::namedWindow("Original Image");
//    	cv::imshow("Original Image",mat);
//    	cv::waitKey(0);
//    	std::cout << "ok!\n";

    	cv::resize(mat, mat, cv::Size(parameters.image_size, parameters.image_size));
/*
    	std::vector<cv::Mat> channels(3);
    	cv::split(mat, channels);
    	auto R = torch::from_blob(
    	        channels[2].ptr(),
    	        {options.image_size, options.image_size},
    	        torch::kUInt8);
    	auto G = torch::from_blob(
    	        channels[1].ptr(),
    	        {options.image_size, options.image_size},
    	        torch::kUInt8);
    	auto B = torch::from_blob(
    	        channels[0].ptr(),
    	        {options.image_size, options.image_size},
    	        torch::kUInt8);

    	auto tdata = torch::cat({R, G, B})
    	                     .view({3, options.image_size, options.image_size})
    	                     .to(torch::kFloat);
    	//tdata.div_(255.0);
*/
        // Convert the image and label to a tensor.
        // Here we need to clone the data, as from_blob does not change the ownership of the underlying memory,
        // which, therefore, still belongs to OpenCV. If we did not clone the data at this point, the memory
        // would be deallocated after leaving the scope of this get method, which results in undefined behavior.
    	auto tdata = torch::from_blob(mat.data, { mat.rows, mat.cols, mat.channels() }, torch::kByte).clone();
    	tdata = tdata.permute({ 2, 0, 1 }).to(torch::kFloat); // Channels x Height x Width

    	auto tlabel = torch::from_blob(&data[index].second, {1}, torch::kInt64);
    	//return {tdata.div_(255.0), tlabel};
    	return {tdata, tlabel};
    } else {

    	torch::data::Example<> EE;
    	return(EE);
    }
  }

  torch::optional<size_t> size() const {
    return data.size();
  }
};

std::pair<Data, Data> readInfo() {
  Data train, test;

  std::ifstream stream(parameters.infoFilePath);
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
  torch::manual_seed(1);

  if (torch::cuda::is_available())
	  parameters.device = torch::kCUDA;
  std::cout << "Running on: "
            << (parameters.device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

  auto data = readInfo();

  auto train_set = CustomDataset(data.first).map(torch::data::transforms::Stack<>());
  auto train_size = train_set.size().value();

  std::cout << "train_size = " << train_size << std::endl;
  auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          	  	  	  	  	  	  	  	  	  	  std::move(train_set),
												  parameters.train_batch_size);
  auto test_set =
      CustomDataset(data.second).map(torch::data::transforms::Stack<>());
  std::cout << test_set.size().value() << std::endl;

  auto test_size = test_set.size().value();
  std::cout << "test_size = " << test_size << std::endl;

  auto test_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(test_set), parameters.test_batch_size);


	// Deep Convolutional Neural Networks (AlexNet)
	auto network = torch::nn::Sequential(
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

  network->to(parameters.device);

  auto optimizer = torch::optim::Adam(network->parameters(), 0.001);
  //torch::optim::SGD optimizer(
  //    network->parameters(), torch::optim::SGDOptions(0.001).momentum(0.5));

  auto loss_fn = torch::nn::CrossEntropyLoss();

  for (size_t i = 0; i < parameters.iterations; ++i) {
      //train(network, *train_loader, optimizer, loss_fn, i + 1, train_size);
	    size_t index = 0;
	    network->train();
	    float Loss = 0, Acc = 0;
	    std::cout << "Start train ...\n";

	    for (auto& batch : *train_loader) {
	    	auto data = batch.data.to(parameters.device);
	    	auto targets = batch.target.to(parameters.device).view({-1});

	    	if( index == 0 ) std::cout << data.sizes() << std::endl;

	    	auto output = network->forward(data);
	    	auto loss = loss_fn(output, targets);
	    	//assert(!std::isnan(loss.template item<float>()));
	    	auto acc = output.argmax(1).eq(targets).sum();

	    	optimizer.zero_grad();
	    	loss.backward();
	    	optimizer.step();

	    	Loss += loss.item<float>();
	    	Acc += acc.template item<float>();

	    	if(index++ % parameters.log_interval == 0) {
	    		auto end = std::min(train_size, (index + 1) * parameters.train_batch_size);

	    		std::cout << "Train Epoch: " << i << " " << end << "/" << train_size
	                << "\tLoss: " << Loss / end << "\tAcc: " << Acc / end
	                << std::endl;
	    	}
	///    	std::cout << index << std::endl;
	  }

      std::cout << std::endl;
      //test(network, *test_loader, loss_fn, test_size);
      index = 0;
      network->eval();
      torch::NoGradGuard no_grad;
      Loss = 0;
      Acc = 0;

      for (const auto& batch : *test_loader) {
        auto data = batch.data.to(parameters.device);
        auto targets = batch.target.to(parameters.device).view({-1});

        auto output = network->forward(data);
        auto loss = loss_fn(output, targets);
        //assert(!std::isnan(loss.template item<float>()));
        auto acc = output.argmax(1).eq(targets).sum();

        Loss += loss.item<float>();
        Acc += acc.template item<float>();
      }

      if (index++ % parameters.log_interval == 0)
        std::cout << "Test Loss: " << Loss / test_size
                  << "\tAcc: " << Acc / test_size << std::endl;
      std::cout << std::endl;
  }


//  auto batch = *train_loader->begin();
//  printf("--------------\n");
//  std::cout << batch.data << std::endl;
  //std::cout << batch.target << std::endl;


  std::cout << "Done!\n";
  return 0;
}
