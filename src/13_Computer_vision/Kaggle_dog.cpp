#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <algorithm>
#include <random>

#include "../utils/Ch_13_util.h"

#include "../utils/transforms.hpp"              // transforms_Compose
#include "../utils/datasets.hpp"                // datasets::ImageFolderClassesWithPaths
#include "../utils/dataloader.hpp"              // DataLoader::ImageFolderClassesWithPaths

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;


struct DogNetImpl : public torch::nn::Module {
	torch::nn::Sequential classifier{nullptr};
	torch::jit::script::Module features;

	DogNetImpl(torch::jit::script::Module& net, torch::nn::Sequential& cls) {
		features = net;
		classifier = cls;
		register_module("classifier", classifier);
	}

	torch::Tensor forward(torch::Tensor x) {
;
		// convert tensor to IValue
		std::vector<torch::jit::IValue> input;
		input.push_back(x);
		x = features.forward(input).toTensor();

		return classifier->forward(x);
	}
};
TORCH_MODULE(DogNet);


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	std::string mdlf = "./src/13_Computer_vision/resnet34_turn_off_grad.pt";
	torch::jit::script::Module net;

	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		net = torch::jit::load(mdlf);
	} catch (const c10::Error& e) {
		std::cerr << e.backtrace() << "error loading the model\n";
		return -1;
	}

	// Freeze parameters of feature layers
    // for(auto& param : net.named_parameters(true) )
    //    param..requires_grad_(false);

	// Define a new output network (there are 120 output categories)
	auto output_new = torch::nn::Sequential(torch::nn::Linear(1000, 256),
                                        torch::nn::ReLU(),
                                        torch::nn::Linear(256, 120));


	size_t img_size = 256;
	size_t batch_size = 32;
	const size_t class_num = 10;
	const size_t valid_batch_size = 1;
	std::vector<std::string> class_names = {
			"affenpinscher", "afghan_hound", "african_hunting_dog", "airedale", "american_staffordshire_terrier", "appenzeller",
	"australian_terrier", "basenji", "basset", "beagle", "bedlington_terrier", "bernese_mountain_dog", "black-and-tan_coonhound",
	"blenheim_spaniel", "bloodhound", "bluetick", "border_collie", "border_terrier", "borzoi", "boston_bull", "bouvier_des_flandres",
	"boxer", "brabancon_griffon", "briard", "brittany_spaniel", "bull_mastiff", "cairn", "cardigan", "chesapeake_bay_retriever",
	"chihuahua", "chow", "clumber", "cocker_spaniel", "collie", "dandie_dinmont", "dhole", "dingo", "doberman", "english_foxhound",
	"english_setter", "english_springer", "entlebucher", "eskimo_dog", "flat-coated_retriever", "french_bulldog", "german_shepherd",
	"german_short-haired_pointer", "giant_schnauzer", "golden_retriever", "gordon_setter", "great_dane", "greater_swiss_mountain_dog",
	"great_pyrenees", "groenendael", "ibizan_hound", "irish_setter", "irish_terrier", "irish_water_spaniel", "irish_wolfhound",
	"italian_greyhound", "japanese_spaniel", "keeshond", "kelpie", "kerry_blue_terrier", "komondor", "kuvasz", "labrador_retriever",
	"lakeland_terrier", "leonberg", "lhasa", "malamute", "malinois", "maltese_dog", "mexican_hairless", "miniature_pinscher",
	"miniature_poodle", "miniature_schnauzer", "newfoundland", "norfolk_terrier", "norwegian_elkhound", "norwich_terrier",
	"old_english_sheepdog", "otterhound","papillon", "pekinese", "pembroke", "pomeranian", "pug", "redbone", "rhodesian_ridgeback",
	"rottweiler", "saint_bernard", "saluki", "samoyed", "schipperke", "scotch_terrier", "scottish_deerhound", "sealyham_terrier",
	"shetland_sheepdog", "shih-tzu", "siberian_husky", "silky_terrier", "soft-coated_wheaten_terrier", "staffordshire_bullterrier",
	"standard_poodle", "standard_schnauzer", "sussex_spaniel", "tibetan_mastiff", "tibetan_terrier", "toy_poodle", "toy_terrier",
	"vizsla", "walker_hound", "weimaraner", "welsh_springer_spaniel", "west_highland_white_terrier", "whippet",
	"wire-haired_fox_terrier", "yorkshire_terrier"};

	constexpr bool train_shuffle = true;    // whether to shuffle the training dataset
	constexpr size_t train_workers = 2;  	// the number of workers to retrieve data from the training dataset
	constexpr bool valid_shuffle = true;    // whether to shuffle the validation dataset
	constexpr size_t valid_workers = 2;     // the number of workers to retrieve data from the validation dataset

    // Set Transforms
    std::vector<transforms_Compose> transform {
        transforms_Resize(cv::Size(img_size, img_size), cv::INTER_LINEAR),        // {IH,IW,C} ===method{OW,OH}===> {OH,OW,C}
        transforms_ToTensor(),                                                     // Mat Image [0,255] or [0,65535] ===> Tensor Image [0,1]
		transforms_Normalize(std::vector<float>{0.485, 0.456, 0.406}, std::vector<float>{0.229, 0.224, 0.225})  // Pixel Value Normalization for ImageNet
    };

	std::string dataroot = "./data/kaggle_dog_tiny/train_valid_test/train";
    std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>> mini_batch;
    torch::Tensor loss, image, label, output;
    datasets::ImageFolderClassesWithPaths dataset, valid_dataset, test_dataset;      		// dataset;
    DataLoader::ImageFolderClassesWithPaths dataloader, valid_dataloader, test_dataloader; 	// dataloader;

    // Get Dataset
    dataset = datasets::ImageFolderClassesWithPaths(dataroot, transform, class_names);
    dataloader = DataLoader::ImageFolderClassesWithPaths(dataset, batch_size, /*shuffle_=*/train_shuffle, /*num_workers_=*/train_workers);

	std::cout << "total training images : " << dataset.size() << std::endl;

    std::string valid_dataroot = "./data/kaggle_dog_tiny/train_valid_test/valid";
    valid_dataset = datasets::ImageFolderClassesWithPaths(valid_dataroot, transform, class_names);
    valid_dataloader = DataLoader::ImageFolderClassesWithPaths(valid_dataset, valid_batch_size, /*shuffle_=*/valid_shuffle, /*num_workers_=*/valid_workers);

    std::cout << "total validation images : " << valid_dataset.size() << std::endl;

    bool valid = true;
    bool test  = true;
    bool vobose = false;
    float lr = 1e-4, wd = 1e-4;

    DogNet model(net, output_new);

    //auto X = torch::rand({1, 3, 256, 256}).to(device);
    //auto out = model->forward(X);
    //std::cout << out << '\n';

    const unsigned step_size = 2;
    const double gamma = 0.9;

    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(lr).momentum(0.9).weight_decay(wd));
    auto scheduler = torch::optim::StepLR(optimizer, step_size, gamma);

    auto criterion = torch::nn::NLLLoss(torch::nn::NLLLossOptions().ignore_index(-100).reduction(torch::kMean));
    //auto criterion = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().reduction(torch::kNone));

    model->to(device);

    size_t epoch;
    size_t total_iter = dataloader.get_count_max();
    size_t start_epoch, total_epoch;
    start_epoch = 1;
    total_iter = dataloader.get_count_max();
    total_epoch = 50;
    bool first = true;
    std::vector<float> train_loss_ave;
    std::vector<float> train_epochs;
    std::vector<float> valid_loss_ave;

    for (epoch = start_epoch; epoch <= total_epoch; epoch++) {
       	model->train();
       	std::cout << "--------------- Training --------------------\n";
       	first = true;
       	float loss_sum = 0.0;
       	size_t total_match = 0, total_counter = 0;
       	torch::Tensor responses;
       	while (dataloader(mini_batch)) {

       		image = std::get<0>(mini_batch).to(device);
       		label = std::get<1>(mini_batch).to(device);
       		size_t mini_batch_size = image.size(0);

       		if( first && vobose ) {
       			for(size_t i = 0; i < label.size(0); i++)
       				std::cout << label[i].item<int64_t>() << " ";
       			std::cout << "\n";
       			first = false;
       		}

       		image = std::get<0>(mini_batch).to(device);
       		label = std::get<1>(mini_batch).to(device);
       		output = model->forward(image);

       		auto out = torch::nn::functional::log_softmax(output, 1);
       		//std::cout << output.sizes() << "\n" << out.sizes() << std::endl;
       		loss = criterion(out, label); //torch::mse_loss(out, label)

       		optimizer.zero_grad();
       		loss.backward();
       		optimizer.step();

       		loss_sum += loss.item<float>();
        }

        train_loss_ave.push_back(loss_sum/total_iter);
        train_epochs.push_back(epoch*1.0);
       	std::cout << "epoch: " << epoch << "/"  << total_epoch << ", avg_loss: "
       			  << (loss_sum/total_iter) << std::endl;

       	// ---------------------------------
       	// validation
       	// ---------------------------------
       	if( valid ) {
       		std::cout << "--------------- validation --------------------\n";
       		torch::NoGradGuard nograd;

       		model->eval();
       		size_t iteration = 0;
       		float total_loss = 0.0;
       		total_match = 0;
       		total_counter = 0;
       		torch::Tensor responses;
       		first = true;
       		while (valid_dataloader(mini_batch)){

       			image = std::get<0>(mini_batch).to(device);
       			label = std::get<1>(mini_batch).to(device);
       			size_t mini_batch_size = image.size(0);

       			if( first && vobose ) {
       				for(size_t i = 0; i < label.size(0); i++)
       					std::cout << label[i].item<int64_t>() << " ";
       				std::cout << "\n";
       				first = false;
       			}

       			// evaluate_loss
       			output = model->forward(image);
       			auto out = torch::nn::functional::log_softmax(output, 1);
       			loss = criterion(out, label);

       			total_loss += loss.item<float>();
       			iteration++;
       		}

       		// Optimizer scheduler
       		scheduler.step();

       		// Calculate Average Loss
       		float ave_loss = total_loss / (float)iteration;
       		valid_loss_ave.push_back(ave_loss);
       		std::cout << "Valid loss: " << ave_loss << "\n\n";
       	}
    }

   	plt::figure_size(600, 500);
   	plt::named_plot("Train loss", train_epochs, train_loss_ave, "b");
   	plt::named_plot("Valid loss", train_epochs, valid_loss_ave, "m-.");
   	plt::xlabel("epoch");
   	plt::legend();
   	plt::show();


	std::cout << "Done!\n";
}

