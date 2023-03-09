#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/ch_13_util.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "../TempHelpFunctions.hpp"

#include <matplot/matplot.h>
using namespace matplot;

// ----------------------------------------------------
// Class Prediction Layer
torch::nn::Sequential cls_predictor(int64_t num_inputs, int64_t num_anchors, int64_t num_classes) {
    return torch::nn::Sequential(
    		torch::nn::Conv2d(torch::nn::Conv2dOptions(num_inputs, num_anchors * (num_classes + 1), 3).padding(1)));
}

// ----------------------------------------------------
// Bounding Box Prediction Layer)
torch::nn::Sequential bbox_predictor(int64_t num_inputs, int64_t num_anchors) {
    return torch::nn::Sequential(
    		torch::nn::Conv2d(torch::nn::Conv2dOptions(num_inputs, num_anchors * 4, 3).padding(1)));
}

// ---------------------------------------------------
// Concatenating Predictions for Multiple Scales
torch::Tensor forward(torch::Tensor x, std::vector<torch::nn::Sequential> block) {
	for(auto& b : block)
		x = b->forward(x);
    return x;
}

torch::Tensor flatten_pred(torch::Tensor& pred) {
    return torch::flatten(pred.permute({0, 2, 3, 1}), 1);
}

torch::Tensor concat_preds(std::vector<torch::Tensor> preds) {
	std::vector<torch::Tensor> V;
	for(auto& p : preds)
		V.push_back(flatten_pred(p));

	torch::TensorList z = torch::TensorList(V);

    return torch::cat(z, 1);
}

// -------------------------------------------------
// Downsampling Block
torch::nn::Sequential down_sample_blk(int64_t in_channels, int64_t out_channels) {
	torch::nn::Sequential blk;
    for(int i = 0; i < 2; i++) {
        blk->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)));
        blk->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels)));
        blk->push_back(torch::nn::ReLU());
        in_channels = out_channels;
    }
    blk->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)));
    return blk;
}


// -----------------------------------------------
// Base Network Block
std::vector<torch::nn::Sequential>  base_net() {
	std::vector<torch::nn::Sequential> blk;
    std::vector<int64_t> num_filters = {3, 16, 32, 64};
    for(int i = 0; i < (num_filters.size() - 1); i++) {
        blk.push_back(down_sample_blk(num_filters[i], num_filters[i+1]));
    }
    return blk;
}

// -----------------------------------------------
//  The Complete Model

std::vector<torch::nn::Sequential> get_blk(int i) {
	std::vector<torch::nn::Sequential> blk;
    if( i == 0 ) {
        blk = base_net();
    } else if( i == 1 ) {
        blk.push_back(down_sample_blk(64, 128));
    } else if( i == 4 ) {
        blk.push_back( torch::nn::Sequential(
        		torch::nn::AdaptiveMaxPool2d(torch::nn::AdaptiveMaxPool2dOptions({1,1}))));
    } else {
        blk.push_back(down_sample_blk(128, 128));
    }
    return blk;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> blk_forward(
		torch::Tensor X,  std::vector<torch::nn::Sequential> blk, std::vector<float> sizes, std::vector<float> ratios,
		torch::nn::Sequential cls_predictor, torch::nn::Sequential bbox_predictor) {

    for(auto& b : blk)
    	X = b->forward(X);
    torch::Tensor anchors = multibox_prior(X, sizes, ratios);
    torch::Tensor cls_preds = cls_predictor->forward(X);
	torch::Tensor bbox_preds = bbox_predictor->forward(X);
    return std::make_tuple(X, anchors, cls_preds, bbox_preds);
}


struct TinySSDImpl : public torch::nn::Module {
	int64_t num_classes;
	std::vector<std::vector<float>> sizes, ratios;
	std::vector<std::vector<torch::nn::Sequential>> blk;
	std::vector<torch::nn::Sequential> cls, bbox;
	TinySSDImpl( int64_t num_classes, int64_t num_anchors,
			std::vector<std::vector<float>> sizes, std::vector<std::vector<float>> ratios ) {
        this->num_classes = num_classes;
        this->sizes = sizes;
        this->ratios = ratios;
        std::vector<int64_t> idx_to_in_channels = {64, 128, 128, 128, 128};
        for(int i = 0; i < 5; i++) {
            // Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            //setattr(self, f'blk_{i}', get_blk(i))
            //setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
            //                                        num_anchors, num_classes))
            //setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
            //                                          num_anchors))
        	//register_module("blk_" + i, get_blk(i));

        	std::vector<torch::nn::Sequential> bk = get_blk(i);
        	for(int j = 0; j < bk.size(); j++) {
        		register_module("blk_" + std::to_string(i) + "_" + std::to_string(j), bk[j]);
        	}
        	blk.push_back( bk );

        	torch::nn::Sequential c = cls_predictor(idx_to_in_channels[i], num_anchors, num_classes);
			register_module("cls_" + std::to_string(i), c);
        	cls.push_back( c );

        	torch::nn::Sequential bb = bbox_predictor(idx_to_in_channels[i], num_anchors);
        	register_module("bbox_" + std::to_string(i), bb);
        	bbox.push_back( bb );
        }
	}

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor X) {
        std::vector<torch::Tensor> anchors(5), cls_preds(5), bbox_preds(5);
        for(int i = 0; i <5; i++) {
            // Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            std::tie(X, anchors[i], cls_preds[i], bbox_preds[i]) = blk_forward(
                X, blk[i], sizes[i], ratios[i], cls[i], bbox[i]);
        }
		torch::Tensor anchor = torch::cat(torch::TensorList(anchors), 1);
		torch::Tensor cls_pred = concat_preds(cls_preds);
        cls_pred = cls_pred.reshape({cls_pred.sizes()[0], -1, num_classes + 1});
		torch::Tensor bbox_pred = concat_preds(bbox_preds);
		return std::make_tuple(anchor, cls_pred, bbox_pred);
    }
};
TORCH_MODULE(TinySSD);

torch::Tensor calc_loss(torch::nn::CrossEntropyLoss cls_loss, torch::nn::L1Loss bbox_loss,
						torch::Tensor cls_preds, torch::Tensor cls_labels,
						torch::Tensor bbox_preds, torch::Tensor bbox_labels, torch::Tensor bbox_masks) {

//	static auto cls_loss = torch::nn::CrossEntropyLoss(
//											torch::nn::CrossEntropyLossOptions().reduction(torch::kNone));

//	static auto bbox_loss = torch::nn::L1Loss(torch::nn::L1LossOptions().reduction(torch::kNone));

    int64_t batch_size = cls_preds.size(0);
    int64_t num_classes = cls_preds.size(2);

    torch::Tensor cls = cls_loss(cls_preds.reshape({-1, num_classes}),
                   	    cls_labels.reshape(-1)).reshape({batch_size, -1}).mean(1);

    torch::Tensor bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(1);

    return cls.add_(bbox);
}

float cls_eval(torch::Tensor cls_preds, torch::Tensor cls_labels) {
    // Because the class prediction results are on the final dimension,
    // `argmax` needs to specify this dimension
	// std::cout << "------cls_eval "<< '\n';
	auto t = cls_preds.argmax(-1).to(cls_labels.dtype());
	auto eq = (t == cls_labels);
    return static_cast<float>(torch::sum(eq).data().item<int>());
}

float bbox_eval(torch::Tensor bbox_preds, torch::Tensor bbox_labels, torch::Tensor bbox_masks) {
    return static_cast<float>((torch::abs((bbox_labels - bbox_preds) * bbox_masks)).sum().data().item<int>());
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	torch::Device device(torch::kCPU);
//	auto cuda_available = torch::cuda::is_available();
//	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
//	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);
	auto x = torch::tensor({{1, 0}, {1, 1}});
	auto y = torch::tensor({{2, 0}, {0, 1}});
	auto z = (x == y);
	std::cout << z << "\n" << torch::sum(z) << '\n';
	std::cout << "x * y" << "\n" << (x * y) << '\n';


	/*
	 * In the following example, we construct feature maps at two different scales, Y1 and Y2,
	 * for the same minibatch, where the height and width of Y2 are half of those of Y1. Let's take class
	 * prediction as an example. Suppose that 5 and 3 anchor boxes are generated for every unit in Y1 and Y2,
	 * respectively. Suppose further that the number of object classes is 10. For feature maps Y1 and Y2
	 * the numbers of channels in the class prediction outputs are 5×(10+1)=55 and 3×(10+1)=33, respectively,
	 * where either output shape is (batch size, number of channels, height, width).
	 */

	auto Y1 = forward(torch::zeros({2, 8, 20, 20}), {cls_predictor(8, 5, 10)});
	auto Y2 = forward(torch::zeros({2, 16, 10, 10}), {cls_predictor(16, 3, 10)});
	std::cout << "Y1:\n" << Y1.sizes() << "\nY2:\n" <<  Y2.sizes() << '\n';

	std::cout << "concat_preds:\n" << concat_preds({Y1, Y2}).sizes() << '\n';

	// Downsampling Block
	std::cout << forward(torch::zeros({2, 3, 20, 20}), {down_sample_blk(3, 10)}).sizes() << '\n';

	std::vector<std::vector<float>> sizes = {{0.2, 0.272}, {0.37, 0.447}, {0.54, 0.619}, {0.71, 0.79},
	         {0.88, 0.961}};
	std::vector<std::vector<float>> ratios = {{1., 2., 0.5}, {1., 2., 0.5}, {1., 2., 0.5}, {1., 2., 0.5}, {1., 2., 0.5}};
	int64_t num_anchors = sizes[0].size() + ratios[0].size() - 1;

	TinySSD tnet(1, num_anchors, sizes, ratios);
	auto X = torch::zeros({32, 3, 256, 256});
	torch::Tensor anchors, cls_preds, bbox_preds;

	std::tie(anchors, cls_preds, bbox_preds) = tnet->forward(X);

	std::cout << "output anchors:\n" <<  anchors.sizes() << '\n';
	std::cout << "output class preds:\n" << cls_preds.sizes() << '\n';
	std::cout << "output bbox preds:\n" <<  bbox_preds.sizes() << '\n';

	// --------------------------------------
	// Training
	// --------------------------------------
	TinySSD net(1, num_anchors, sizes, ratios);

	torch::optim::SGD trainer = torch::optim::SGD(net->parameters(),
								torch::optim::SGDOptions(0.2).weight_decay(5e-4));

	auto cls_loss = torch::nn::CrossEntropyLoss(
											torch::nn::CrossEntropyLossOptions().reduction(torch::kNone));

	auto bbox_loss = torch::nn::L1Loss(torch::nn::L1LossOptions().reduction(torch::kNone));

	const std::string data_dir = "./data/banana-detection";
	bool is_train = true;
	int imgSize = 256;
	int batch_size = 32;

	auto data_targets = load_bananas_img_data(data_dir, is_train, imgSize);

	auto train_set = BananasDataset(data_targets, imgSize).map(torch::data::transforms::Stack<>());

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
			          	  	  	  	  	  	  	  	  	  	  std::move(train_set), batch_size);
/*
	auto batch = *train_loader->begin();
	auto data  = batch.data.to(device);
	auto a     = batch.target.to(device);
	std::cout << "data: " << data.sizes() << std::endl;
	std::cout << "a: " << a.sizes() << std::endl;
*/
	torch::Tensor bbox_labels, bbox_masks, cls_labels;
	std::vector<float> cls_errs, bbox_maes, epoch_num;

	int64_t num_epochs = 10;

	net->to(device);

	for(int64_t epoch = 1; epoch <= num_epochs; epoch++) {
	    // Sum of training accuracy, no. of examples in sum of training accuracy,
	    // Sum of absolute error, no. of examples in sum of absolute error
	    //metric = d2l.Accumulator(4)
	    net->train();
	    torch::AutoGradMode enable_grad(true);

	    float tlt_cls_error = 0.0, tlt_bbox_mae = 0.0;
	    int64_t tlt_cls_num = 0, tlt_bbox_num = 0;

	    for(auto& batch : *train_loader ) {
	        //timer.start()

	        torch::Tensor X = batch.data.to(device);
	        torch::Tensor Y = batch.target.to(device);

	        // std::cout << "---------------------------------------\nY: " <<  Y[0] << '\n';
	        // Generate multiscale anchor boxes and predict their classes and offsets

	        std::tie(anchors, cls_preds, bbox_preds) = net->forward(X);

	        //std::cout << "anchors: " << anchors[0].index({0, Slice()}) << '\n';

	        // Label the classes and offsets of these anchor boxes
	        std::tie(bbox_labels, bbox_masks, cls_labels) = multibox_target(anchors, Y);

	        // Calculate the loss function using the predicted and labeled values
	        // of the classes and offsets
	        torch::Tensor l = calc_loss(cls_loss, bbox_loss, cls_preds, cls_labels, bbox_preds, bbox_labels,
	                      bbox_masks);

	        trainer.zero_grad();
	        l.mean().backward();
	        trainer.step();

	        tlt_cls_error += cls_eval(cls_preds, cls_labels);
	        tlt_cls_num   += cls_labels.numel();
	        tlt_bbox_mae  += bbox_eval(bbox_preds, bbox_labels, bbox_masks);
	        tlt_bbox_num  += bbox_labels.numel();
	    }

	    float cls_err  = 1 - tlt_cls_error / tlt_cls_num;
		float bbox_mae =  tlt_bbox_mae / tlt_bbox_num;
	    //animator.add(epoch + 1, (cls_err, bbox_mae))
		std::cout << "Epoch: " << epoch << ", class err: " << cls_err << ", bbox mae: " << bbox_mae << '\n';
		cls_errs.push_back(cls_err);
		bbox_maes.push_back(bbox_mae);
		epoch_num.push_back(epoch*1.0);
	}

	auto rlt = readImg("./data/banana.jpg", {imgSize, imgSize});
	cv::Mat img = rlt.first;
	torch::Tensor imgT = rlt.second;

	net->eval();
	torch::NoGradGuard no_grad;

	std::tie(anchors, cls_preds, bbox_preds)= net->forward(imgT.to(device));

	torch::Tensor cls_probs = torch::nn::functional::softmax(cls_preds, 2).permute({0, 2, 1});
	torch::Tensor output = multibox_detection(cls_probs, bbox_preds, anchors, 0.95, 0.1);
	output.squeeze_();
	auto prd = output.cpu();

	std::cout << prd.sizes() << '\n';
	std::cout << prd.index({Slice(0, 20), Slice()}) << '\n';

	float threshold = 0.98;

	for(int64_t i = 0; i < prd.size(0); i++ ) {
		torch::Tensor row = prd[i];
		float score = row[1].data().item<float>();
	    if( score < threshold )
	        continue;

	    int h = img.rows,  w = img.cols;

		torch::Tensor bbox_scale = torch::tensor({w, h, w, h});
		torch::Tensor tbox = row.index({Slice(2, None)}) * bbox_scale;
		tbox.to(device);

		show_bboxes(img, tbox.unsqueeze(0), {to_string_with_precision(row[1].item<float>(), 3)}, {});
	}

	torch::Tensor gimg = CvMatToTensor2(img, {}, true);
	std::vector<std::vector<std::vector<unsigned char>>> M = tensorToMatrix4MatplotPP(gimg.clone());

	auto f = figure(true);
	f->width(f->width() * 2);
	f->height(f->height() * 2);
	f->x_position(0);
	f->y_position(0);

	matplot::subplot(1, 2, 0);
	matplot::legend();
	matplot::hold(true);
	matplot::ylim({0., 0.01});
	matplot::plot(epoch_num, cls_errs, "b")->line_width(2)
				.display_name("class error");
	matplot::plot(epoch_num, bbox_maes, "m--")->line_width(2)
					.display_name("bbox mae");
	matplot::hold( false);
	matplot::xlabel("epoch");
	f->draw();

	matplot::subplot(1, 2, 1);
	matplot::imshow(M);
	matplot::title("SSD image");
	f->draw();
	matplot::show();


	std::cout << "Done!\n";
}



