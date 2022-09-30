#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/ch_13_util.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <utility>
#include <tuple>

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;


enum class Layer { CONV64, CONV128, CONV256, CONV512, MAXPOOL };

void initialize_weights(const torch::nn::Module& module) {
    torch::NoGradGuard no_grad;

    if (auto conv2d = module.as<torch::nn::Conv2d>()) {
        torch::nn::init::kaiming_normal_(conv2d->weight, 0.0, torch::kFanOut, torch::kReLU);
    } else if (auto bn2d = module.as<torch::nn::BatchNorm2d>()) {
        torch::nn::init::constant_(bn2d->weight, 1);
        torch::nn::init::constant_(bn2d->bias, 0);
    } else if (auto linear = module.as<torch::nn::Linear>()) {
        torch::nn::init::normal_(linear->weight, 0, 0.01);
        torch::nn::init::constant_(linear->bias, 0);
    }
}

struct VGGNetImpl : public torch::nn::Module {
 public:
    VGGNetImpl(const std::vector<Layer>& config, const std::vector<size_t>& selected, const std::vector<size_t>& style,
        bool batch_norm, const std::string& scriptmodule_file_path) {

    	layers = make_layers(config, batch_norm);
    	std::cout << "L-size: " << layers.get()->size() << "\n";
    	set_content_layer_idxs(selected);
    	set_style_layer_idxs(style);

        register_module("layers", layers);

        if (scriptmodule_file_path.empty()) {
            layers->apply(initialize_weights);
        } else {
            torch::load(layers, scriptmodule_file_path);
        }
        //std::cout << "Load...done!\n";
    }

    std::vector<torch::Tensor> forward(torch::Tensor x, bool is_style) {
        std::vector<torch::Tensor> tensors;

        std::vector<size_t> slt_idxs;
        if( is_style )
        	slt_idxs = get_style_layer_idxs();
        else
        	slt_idxs = get_content_layer_idxs();

        size_t layer_id = 0;
        for (auto m : *layers) {
            x = m.forward<>(x);

            //if (selected_layer_idxs_.find(layer_id) != selected_layer_idxs_.end()) {
            //    tensors.push_back(x);
            //}
            if( std::find(slt_idxs.begin(),
            		slt_idxs.end(), layer_id) != slt_idxs.end() ) {
            	tensors.push_back(x);
            }

            ++layer_id;
        }

        return tensors;
    }

    void set_content_layer_idxs(const std::vector<size_t>& idxs) { content_layers_idxs_ = idxs; }
    std::vector<size_t> get_content_layer_idxs() const { return content_layers_idxs_; }
    void set_style_layer_idxs(const std::vector<size_t>& idxs) { style_layers_idxs_ = idxs; }
    std::vector<size_t> get_style_layer_idxs() const { return style_layers_idxs_; }

    size_t get_num_layers() const { return layers->size(); }
 private:
    torch::nn::Sequential make_layers(const std::vector<Layer>& config, bool batch_norm) {
        torch::nn::Sequential layers;
        int64_t in_channels = 3;

        for (auto layer_type : config) {
            if (layer_type == Layer::MAXPOOL) {
                layers->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
            } else {
                int64_t out_channels = 0;

                switch (layer_type) {
                    case Layer::CONV64:
                        out_channels = 64;
                        break;
                    case Layer::CONV128:
                        out_channels = 128;
                        break;
                    case Layer::CONV256:
                        out_channels = 256;
                        break;
                    case Layer::CONV512:
                        out_channels = 512;
                        break;
                    default:
                        throw std::runtime_error("Invalid layer type.");
                }

                layers->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)));

                if (batch_norm) {
                    layers->push_back(torch::nn::BatchNorm2d(out_channels));
                }

                layers->push_back(torch::nn::ReLU());

                in_channels = out_channels;
            }
        }

        return layers;
    }

    torch::nn::Sequential layers;
    std::vector<size_t> content_layers_idxs_;
    std::vector<size_t> style_layers_idxs_;
};

TORCH_MODULE(VGGNet);

// --------------------------------
// Extracting Features
// --------------------------------
torch::Tensor preprocess(torch::Tensor imgT,  std::vector<float> mean_, std::vector<float> std_) {
	auto rimg = NormalizeTensor(imgT, mean_, std_);
	return rimg.unsqueeze(0);
}


torch::Tensor postprocess(torch::Tensor img, torch::Device device,
						  std::vector<float> mean_, std::vector<float> std_) {

	img.requires_grad_(false);		// turn off grad

	auto imgT = img[0].to(device);
    imgT = deNormalizeTensor(imgT, mean_, std_);
    imgT = torch::clamp(imgT, 0, 1);

    return imgT.clone();
}

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> extract_features(
										VGGNet& net, torch::Tensor& x) {
	std::vector<torch::Tensor> contents = net->forward(x, false);
	std::vector<torch::Tensor> styles   = net->forward(x, true);

    return std::make_pair(contents, styles);
}

std::pair<torch::Tensor, std::vector<torch::Tensor>> get_contents(VGGNet& net, torch::Tensor& content_X) {
    auto contents_Y = extract_features(net, content_X).first;
    return std::make_pair(content_X, contents_Y);
}

std::pair<torch::Tensor, std::vector<torch::Tensor>> get_styles(VGGNet& net, torch::Tensor& style_X) {
    auto styles_Y = extract_features(net, style_X).second;
    return std::make_pair(style_X, styles_Y);
}

// --------------------------------
// Defining the Loss Function
// --------------------------------
torch::Tensor content_loss(torch::Tensor Y_hat, torch::Tensor Y) {
    // We detach the target content from the tree used to dynamically compute
    // the gradient: this is a stated value, not a variable. Otherwise the loss
    // will throw an error.
    return torch::square(Y_hat - Y.detach()).mean();
}

torch::Tensor gram(torch::Tensor X) {
    int64_t num_channels = X.sizes()[1];
    int64_t n = static_cast<int64_t>(X.numel() * 1.0 / X.sizes()[1]);

    X = X.reshape({num_channels, n});
    return torch::matmul(X, X.t()) / (num_channels * n);
}

torch::Tensor style_loss(torch::Tensor Y_hat, torch::Tensor gram_Y) {
    return torch::square(gram(Y_hat) - gram_Y.detach()).mean();
}

torch::Tensor tv_loss(torch::Tensor Y_hat) {
    return 0.5 * (torch::abs(Y_hat.index({Slice(), Slice(), Slice(1, None), Slice()})
    					- Y_hat.index({Slice(), Slice(), Slice(None,-1), Slice()})).mean()
					+
                  torch::abs(Y_hat.index({Slice(), Slice(), Slice(), Slice(1, None)})
                		  - Y_hat.index({Slice(), Slice(), Slice(), Slice(None, -1)})).mean());
}

// --------------------------
// Loss Function
// --------------------------

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, torch::Tensor, torch::Tensor>
	compute_loss(torch::Tensor& X, std::vector<torch::Tensor>  contents_Y_hat,
		std::vector<torch::Tensor> styles_Y_hat, std::vector<torch::Tensor> contents_Y,
		std::vector<torch::Tensor>  styles_Y_gram) {

	int64_t content_weight = 1, style_weight = 1e4, tv_weight = 10;
	std::vector<torch::Tensor> contents_l, styles_l;

    // Calculate the content, style, and total variance losses respectively
    //contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
    //    contents_Y_hat, contents_Y)]

	for(int i = 0; i < contents_Y_hat.size(); i++)
		contents_l.push_back(content_loss(contents_Y_hat[i], contents_Y[i]) * content_weight);

    //styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
    //    styles_Y_hat, styles_Y_gram)]

	for(int i = 0; i < styles_Y_hat.size(); i++)
		styles_l.push_back(style_loss(styles_Y_hat[i], styles_Y_gram[i]) * style_weight);

    torch::Tensor tv_l = tv_loss(X) * tv_weight;
    // Add up all the losses
    torch::Tensor l = tv_l.clone();

    //l = sum(styles_l + contents_l + [tv_l])
    for(auto& t : contents_l)
    	l = l + torch::sum(t);

    for(auto& t : styles_l)
        l = l + torch::sum(t);

    return std::make_tuple(contents_l, styles_l, tv_l, l);
}

class SynthesizedImage : public torch::nn::Module {
public:
	torch::Tensor weight;
	SynthesizedImage(c10::IntArrayRef img_shape) {
        weight = torch::rand(img_shape);
        register_parameter("weight", weight, true); // To register a parameter (or tensor which requires gradients) to a module
	}

    torch::Tensor forward() {
        return weight;
    }
};


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	std::vector<int> image_size =  {450, 300};

	auto cimg = CvMatToTensor("./data/rainier.jpg", image_size);

	std::vector<uint8_t> z = tensorToMatrix4Matplotlib(cimg);
	const unsigned char* zptr1 = &(z[0]);

	auto simg = CvMatToTensor("./data/autumn-oak.jpg", image_size);
	std::cout << "simg: " << simg.sizes() << '\n';
	std::vector<uint8_t> z2 = tensorToMatrix4Matplotlib(simg);
	const unsigned char* zptr2 = &(z2[0]);

	// ------------------------------------------
	// Preprocessing and Postprocessing
	// ------------------------------------------
	std::vector<float> rgb_mean = {0.485, 0.456, 0.406};
	std::vector<float> rgb_std = {0.229, 0.224, 0.225};

	// ------------------------------------------
	// Extracting Features
	// ------------------------------------------

	//torch::jit::script::Module net = torch::jit::load("./src/13_Computer_vision/vgg19_neural_style.pt");
	const std::string vgg19_layers_scriptmodule_path = "./src/13_Computer_vision/vgg19_neural_style.pt";

	const std::vector<size_t> style_layers = {0, 5, 10, 19, 28}, content_layers = {25};

	const std::vector<Layer> config = { Layer::CONV64, Layer::CONV64, Layer::MAXPOOL, Layer::CONV128, Layer::CONV128,
			Layer::MAXPOOL, Layer::CONV256, Layer::CONV256, Layer::CONV256, Layer::CONV256,
			Layer::MAXPOOL, Layer::CONV512, Layer::CONV512, Layer::CONV512, Layer::CONV512,
			Layer::MAXPOOL, Layer::CONV512, Layer::CONV512, Layer::CONV512, Layer::CONV512,
			Layer::MAXPOOL};

	// Model
	VGGNet vgg19(config, content_layers, style_layers, false, vgg19_layers_scriptmodule_path);

	auto content_img = preprocess(cimg, rgb_mean, rgb_std).to(device);
	auto style_img   = preprocess(simg, rgb_mean, rgb_std).to(device);

	// auto t = torch::randn({3, 450, 300}).to(torch::kFloat32);
	// std::cout << torch::sum(t) << '\n';

	std::pair<torch::Tensor, std::vector<torch::Tensor>> cnt = get_contents(vgg19, content_img);
	torch::Tensor contents_X = cnt.first;
	std::vector<torch::Tensor> contents_Y = cnt.second;
	std::vector<torch::Tensor> styles_Y = get_styles(vgg19, style_img).second;

	// get_inits
	SynthesizedImage gen_img(contents_X.sizes());
	gen_img.to(device);

	gen_img.weight.data().copy_(contents_X.data());

	std::cout << gen_img.weight.sizes() << '\n';

	torch::optim::Adam trainer(gen_img.parameters(), torch::optim::AdamOptions(0.3));

	//styles_Y_gram = [gram(Y) for Y in styles_Y]
	std::vector<torch::Tensor> styles_Y_gram;
	for(auto& Y : styles_Y)
		styles_Y_gram.push_back(gram(Y));

	torch::Tensor X = gen_img.forward();
	//    scheduler = torch::optim::LRScheduler(trainer, lr_decay_epoch, 0.8)

	// Training
	int64_t num_epoches = 200;

	std::vector<float> nepoch, c_loss, s_loss, tv_loss;

	for( int64_t epoch = 1; epoch <= num_epoches; epoch++ ){
        trainer.zero_grad();
        auto features = extract_features(vgg19, X);
        auto contents_Y_hat = features.first;
		auto styles_Y_hat   = features.second;
		auto losses = compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram);
        auto contents_l = std::get<0>(losses);
        auto styles_l   = std::get<1>(losses);
        auto tv_l = std::get<2>(losses);
        auto l    = std::get<3>(losses);
        l.backward();
        trainer.step();
        //scheduler.step()
        if(epoch % 10 == 0 ) {
        	std::cout << "content_loss: " << torch::sum(contents_l[0]).data().item<float>()
					  << ", " << "style_loss: " << torch::sum(styles_l[0]).data().item<float>()
        			  << ", " << "tv_loss: " << tv_l.data().item<float>() << '\n';

        	nepoch.push_back(epoch*1.0);
        	c_loss.push_back(torch::sum(contents_l[0]).data().item<float>());
        	s_loss.push_back(torch::sum(styles_l[0]).data().item<float>());
        	tv_loss.push_back(tv_l.data().item<float>());
        }
	}

	//auto X = preprocess(cimg, rgb_mean, rgb_std).to(device);

	auto gimg = postprocess(X, device, rgb_mean, rgb_std);
	//std::cout << "gimg: " << gimg.sizes() << '\n';

	std::vector<uint8_t> z3 = tensorToMatrix4Matplotlib(gimg);
	const unsigned char* zptr3 = &(z3[0]);

	plt::figure_size(1000, 780);
	plt::subplot2grid(2, 2, 0, 0, 1, 1);
	plt::title("Content image");
	plt::imshow(zptr1, static_cast<int>(cimg.size(1)),
							static_cast<int>(cimg.size(2)), static_cast<int>(cimg.size(0)));

	plt::subplot2grid(2, 2, 0, 1, 1, 1);
	plt::title("Style image");
	plt::imshow(zptr2, static_cast<int>(simg.size(1)),
								static_cast<int>(simg.size(2)), static_cast<int>(simg.size(0)));

	plt::subplot2grid(2, 2, 1, 0, 1, 1);
	plt::title("NeuralStyle image");
	plt::imshow(zptr3, static_cast<int>(gimg.size(1)),
									static_cast<int>(gimg.size(2)), static_cast<int>(gimg.size(0)));
	plt::subplot2grid(2, 2, 1, 1, 1, 1);
	plt::named_plot("Content", nepoch, c_loss, "b");
	plt::named_plot("Style", nepoch, s_loss, "g--");
	plt::named_plot("TV", nepoch, tv_loss, "r-.");
	plt::ylabel("loss");
	plt::xlabel("epoch");
	plt::legend();
	plt::show();
	plt::close();

	std::cout << "Done!\n";
}

