#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>


#include <torch/script.h> // One-stop header.

#include "../utils/Ch_13_util.h"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	std::vector<float> mean_ = {0.485, 0.456, 0.406};
	std::vector<float> std_  = {0.229, 0.224, 0.225};

	bool is_train  = true;
	int batch_size = 32;
	const std::string voc_dir = "./data/VOCdevkit/VOC2012";
	std::vector<int> crop_size = {224,224};

	auto data_set  = read_voc_images(voc_dir, is_train, 0, false, {});

	auto train_set = VOCSegDataset(data_set, crop_size, false).map(torch::data::transforms::Stack<>());

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
				          	  	  	  	  	  	  	  	  	  	  std::move(train_set), batch_size);

	auto batch = *train_loader->begin();
	auto data  = batch.data.to(device);
	auto y     = batch.target.to(device);

	plt::figure_size(1000, 500);
	for(int r = 0; r < 2; r++) {
		for(int c = 0; c < 5; c++) {
			torch::Tensor img;
			if(r == 0) {
				img = data[c].clone().squeeze();
				img = deNormalizeTensor(img, mean_, std_);
			} else {
				img = y[c].clone().squeeze();
			}

			std::vector<uint8_t> z = tensorToMatrix4Matplotlib(img);
			const unsigned char* zptr = &(z[0]);

			plt::subplot2grid(2, 5, r, c, 1, 1);
			plt::imshow(zptr, static_cast<int>(img.size(1)),
					static_cast<int>(img.size(2)), static_cast<int>(img.size(0)));
		}
	}
	plt::show();
	plt::close();

	auto sgimg = CvMatToTensor("./data/2007_000032.png", {});

	sgimg = sgimg.squeeze().mul(255);
	std::cout << "sgimg.sizes: " << sgimg.sizes() << '\n';
	std::cout << "sgimg.max: " << sgimg.max() << '\n';

	auto ssgimg = sgimg.permute({1, 2, 0}).to(torch::kLong).clone();

	auto idx = ((ssgimg.index({Slice(), Slice(), 0}) * 256 + ssgimg.index({Slice(), Slice(), 1})) * 256
	           + ssgimg.index({Slice(), Slice(), 2}));
	std::cout << "idx.sizes: " << idx.sizes() << '\n';
	std::cout << idx.index({Slice(105,115), Slice(130,140)}) << '\n';

	auto lab = voc_label_indices(sgimg.clone(), voc_colormap2label());
	std::cout << lab.index({Slice(105,115), Slice(130,140)}) << '\n';

	// -----------------------------------------
	// Data Preprocessing
	// -----------------------------------------
	int height = 224, width = 224;

	torch::Tensor feature = data[0].clone().squeeze();
	torch::Tensor label = y[0].clone().squeeze();

	plt::figure_size(1000, 500);
	for(int c = 0; c < 5; c++) {

		auto dt = voc_rand_crop(feature.clone(), label.clone(), height, width, mean_, std_);

		auto img = deNormalizeTensor(dt.first.clone(), mean_, std_);
		auto limg = dt.second.clone();

		std::vector<uint8_t> z = tensorToMatrix4Matplotlib(img);
		const unsigned char* zptr = &(z[0]);

		plt::subplot2grid(2, 5, 0, c, 1, 1);
		plt::imshow(zptr, static_cast<int>(img.size(1)),
					static_cast<int>(img.size(2)), static_cast<int>(img.size(0)));

		std::vector<uint8_t> lz = tensorToMatrix4Matplotlib(limg);
		const unsigned char* lzptr = &(lz[0]);
		plt::subplot2grid(2, 5, 1, c, 1, 1);
		plt::imshow(lzptr, static_cast<int>(limg.size(1)),
						static_cast<int>(limg.size(2)), static_cast<int>(limg.size(0)));
	}
	plt::show();
	plt::close();


	batch_size = 32;
	data_set  = read_voc_images(voc_dir, is_train, 0, false, crop_size);
	train_set = VOCSegDataset(data_set, crop_size).map(torch::data::transforms::Stack<>());

	train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
					          	  	  	  	  	  	  	  	  	  	  std::move(train_set), batch_size);

	for(auto& batch_data : *train_loader) {
		auto X = batch.data.to(device);
		auto y = batch.target.to(device);
		std::cout << "X: " << X.sizes() << '\n';
		std::cout << "y: " << y.sizes() << '\n';
		break;
	}

	int num_classes = 21;
	batch_size = 10;
	is_train = true;
	auto test_data   = read_voc_images(voc_dir, is_train, 64, false, crop_size);
	auto test_set = VOCSegDataset(test_data, crop_size).map(torch::data::transforms::Stack<>());

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
							          	  	  	  	  	  	  	  	  	  	  std::move(test_set), batch_size);
/*
	plt::figure_size(1000, 500);

	for( int i = 0; i < 5; i++ ) {
		std::cout << test_data[i].first << '\n';
		std::cout << test_data[i].second << '\n';

		auto imgT = CvMatToTensor(test_data[i].first, {});
		imgT = NormalizeTensor(imgT, mean_, std_);

		auto labT = CvMatToTensor(test_data[i].second, {});
		labT = labT.mul(255).to(torch::kByte).clone();

		auto dt = voc_rand_crop(imgT.clone(), labT.clone(), 224, 224, mean_, std_, true);

		std::cout << "dt.first: " << dt.first.sizes() << '\n'; 		// (3, 320, 480)
		std::cout << "dt.second: " << dt.second.sizes() << '\n'; 	// (3, 320, 480)

		auto colormap = dt.second.clone();
		colormap = colormap.permute({1, 2, 0}).to(torch::kByte).clone();

		std::vector<uint8_t> lz = tensorToMatrix4Matplotlib(colormap, false, false);
		const unsigned char* lzptr = &(lz[0]);
		plt::subplot2grid(2, 5, 0, i, 1, 1);
		plt::title("colormap");
		plt::imshow(lzptr, static_cast<int>(colormap.size(0)),
								  static_cast<int>(colormap.size(1)), static_cast<int>(colormap.size(2)));

		auto coded = voc_label_indices(dt.second.clone(), voc_colormap2label());
		std::cout << "coded: " << coded.index({Slice(105,115), Slice(130,140)}) << '\n';

		auto decoded = decode_segmap(coded, num_classes);

		lz = tensorToMatrix4Matplotlib(decoded, false, false);
		lzptr = &(lz[0]);
		plt::subplot2grid(2, 5, 1, i, 1, 1);
		plt::title("decoded");
		plt::imshow(lzptr, static_cast<int>(decoded.size(0)),
								  static_cast<int>(decoded.size(1)), static_cast<int>(decoded.size(2)));
	}
	plt::show();
	plt::close();
*/

	// ---------------------------------------
	// load pretrained fcn segmentation model
	// ---------------------------------------
	std::string mdlf = "./src/13_Computer_vision/resnet101_trace_model.pt";
	torch::jit::script::Module net;

	try {
	    // Deserialize the ScriptModule from a file using torch::jit::load().
	    net = torch::jit::load(mdlf);
	} catch (const c10::Error& e) {
	    std::cerr << e.backtrace() << "error loading the model\n";
	    return -1;
	}

	net.eval();
/*
	auto X = torch::randn({1,3,224,224});
	std::vector<torch::jit::IValue> input;
	input.push_back(X);
	auto out = net.forward(input).toGenericDict();
	std::cout << out.at(c10::IValue("out")) << '\n';
*/

	batch = *test_loader->begin();
	data  = batch.data.to(device);
	y     = batch.target.to(device);

	std::cout << "data: " << data.sizes() << '\n';
	std::cout << "y: " << y.sizes() << '\n';

	int wct = 10;
	plt::figure_size(200 * wct, 700);

	for( int i = 0; i < wct; i ++ ) {
		auto x = data[i].unsqueeze(0);
		auto lab = y[i].squeeze();

		std::cout <<"x: " << x.sizes() << '\n';
		auto jimg = x.squeeze().clone();
		jimg = deNormalizeTensor(jimg, mean_, std_);
		jimg = jimg.mul(255).permute({1, 2, 0}).to(torch::kByte).clone();
		std::vector<uint8_t> iz = tensorToMatrix4Matplotlib(jimg, false, false);
		const unsigned char* izptr = &(iz[0]);
		plt::subplot2grid(3, wct, 0, i, 1, 1);
		plt::title("Image");
		plt::imshow(izptr, static_cast<int>(jimg.size(0)),
								   static_cast<int>(jimg.size(1)), static_cast<int>(jimg.size(2)));

		std::vector<torch::jit::IValue> input;
		input.push_back(x);
		auto out = net.forward(input).toGenericDict();

		auto ot = out.at(c10::IValue("out")).toTensor();
		std::cout << ot.sizes() << '\n';
		auto pred = torch::argmax(ot.squeeze(), 0).detach().to(device);
		std::cout << "pred: " << pred.sizes() << '\n';

		std::cout << "label: " << lab.sizes() << '\n';
		auto t = torch::sum((pred != lab)).data().item<int>();
		std::cout << "t: " << t << '\n';

		auto img = decode_segmap(pred.squeeze(), num_classes);
		std::vector<uint8_t> z = tensorToMatrix4Matplotlib(img.clone(), false, false);
		const unsigned char* zptr = &(z[0]);

		plt::subplot2grid(3, wct, 1, i, 1, 1);
		plt::title("pred");
		plt::imshow(zptr, static_cast<int>(img.size(0)),
						  static_cast<int>(img.size(1)), static_cast<int>(img.size(2)));

		auto limg = decode_segmap(lab.squeeze(), num_classes);
		std::vector<uint8_t> lz = tensorToMatrix4Matplotlib(limg.clone(), false, false);
		const unsigned char* lzptr = &(lz[0]);
		plt::subplot2grid(3, wct, 2, i, 1, 1);
		plt::title("label");
		plt::imshow(lzptr, static_cast<int>(limg.size(0)),
						   static_cast<int>(limg.size(1)), static_cast<int>(limg.size(2)));
	}
	plt::show();
	plt::close();

	std::cout << "Done!\n";
}

