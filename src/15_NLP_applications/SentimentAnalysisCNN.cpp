#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>

#include "../utils/ch_15_util.h"
#include "../TempHelpFunctions.hpp"

#include <matplot/matplot.h>
using namespace matplot;

torch::Tensor corr1d(torch::Tensor X, torch::Tensor K) {
    int w = K.size(0);
    auto Y = torch::zeros(X.size(0) - w + 1);
    for(int i = 0; i < Y.size(0); i++) {
        Y.index_put_({i}, (X.index({Slice(i, i + w)}) * K).sum());
    }
    return Y;
}


torch::Tensor corr1d_multi_in(torch::Tensor X, torch::Tensor K) {
    // First, iterate through the 0th dimension (channel dimension) of `X` and
    // `K`. Then, add them together
    // return sum(corr1d(x, k) for x, k in zip(X, K))
	int w = X.size(1);
	int wk = K.size(1);
	auto Y = torch::zeros(X.size(1)-1);

	for( int i = 0; i < (w - wk + 1); i++ ) {
		auto x = X.index({Slice(), Slice(i, i+wk)});
		auto y = x * K;

		Y.index_put_({i}, y.sum());
	}
	return Y;
}

// Defining the Model

struct TextCNNImpl : public torch::nn::Module {
	torch::nn::Embedding embedding{nullptr};
	torch::nn::Embedding constant_embedding{nullptr};
	torch::nn::Dropout dropout{nullptr};
	torch::nn::Linear decoder{nullptr};
	torch::nn::AdaptiveAvgPool1d pool{nullptr};
	torch::nn::ReLU relu;
	//torch::nn::ModuleList convs;
	std::vector<torch::nn::Conv1d> convs;

	TextCNNImpl(int64_t vocab_size, int64_t embed_size, std::vector<int64_t> kernel_sizes, std::vector<int64_t> num_channels) {

        embedding = torch::nn::Embedding(vocab_size, embed_size);
        // The embedding layer not to be trained
        constant_embedding = torch::nn::Embedding(vocab_size, embed_size);
        dropout = torch::nn::Dropout(0.5);
        decoder = torch::nn::Linear(vector_sum(num_channels), 2);
        // The max-over-time pooling layer has no parameters, so this instance
        // can be shared
        pool = torch::nn::AdaptiveAvgPool1d(1);
        relu = torch::nn::ReLU();

        // Create multiple one-dimensional convolutional layers
        //convs = torch::nn::ModuleList();
        //for c, k in zip(num_channels, kernel_sizes):
        for( int i = 0; i < num_channels.size(); i++ )
            convs.push_back( torch::nn::Conv1d(
            		torch::nn::Conv1dOptions(2 * embed_size, num_channels[i], kernel_sizes[i])));

        register_module("embedding", embedding);
        register_module("constant_embedding", constant_embedding);
        register_module("decoder", decoder);

	}

    torch::Tensor forward(torch::Tensor inputs) {
        // Concatenate two embedding layer outputs with shape (batch size, no.
        // of tokens, token vector dimension) along vectors
        auto embeddings = torch::cat({
            embedding->forward(inputs), constant_embedding->forward(inputs)}, 2); // dim=2
        // Per the input format of one-dimensional convolutional layers,
        // rearrange the tensor so that the second dimension stores channels
        embeddings = embeddings.permute({0, 2, 1});

        // For each one-dimensional convolutional layer, after max-over-time
        // pooling, a tensor of shape (batch size, no. of channels, 1) is
        // obtained. Remove the last dimension and concatenate along channels
        std::vector<torch::Tensor> convT;
        for(auto& conv : convs) {
        	auto T = torch::squeeze(relu->forward(pool->forward(conv->forward(embeddings))), -1);
        	convT.push_back(T.clone());
        }
        auto encoding = torch::cat(convT, 1);
        /*
        encoding = torch.cat([
            torch::squeeze(relu(pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
         */
        auto outputs = decoder->forward(dropout->forward(encoding));

        return outputs;
    }

    void init_weights() {
    	torch::NoGradGuard noGrad;
        for(auto& module : modules(/*include_self=*/false)) {
            if(auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
            	torch::nn::init::xavier_uniform_(M->weight);
            }
            if(auto M = dynamic_cast<torch::nn::Conv1dImpl*>(module.get())) {
                torch::nn::init::xavier_uniform_(M->weight);
            }
        }
    }

};

TORCH_MODULE(TextCNN);


std::string predict_sentiment(TextCNN net, Vocab vocab, std::string sequence, size_t num_steps, torch::Device device) {
    //Predict the sentiment of a text sequence
	std::vector<std::string> seqs;
	seqs.push_back(sequence);
	std::vector<std::string> tks = tokenize(seqs, "word", false);
	auto dt = truncate_pad(vocab[tks], num_steps, vocab["<pad>"]);
	torch::Tensor seq = torch::from_blob(dt.data(), {1, static_cast<long>(num_steps)}, torch::TensorOptions(torch::kLong)).clone();
	seq.to(device);
	net->eval();

	torch::NoGradGuard no_grad;
	auto pred = net->forward(seq.reshape({1, -1}));
	auto label = torch::argmax(pred, 1).data().item<int64_t>();

    return  label == 1 ? "positive" : "negative";
}



int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	torch::Device device(torch::kCPU);
//	auto cuda_available = torch::cuda::is_available();
//	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
//	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	// ---------------------------------------------
	// One-Dimensional Convolutions
	// ---------------------------------------------
	auto X = torch::tensor({0, 1, 2, 3, 4, 5, 6}).to(torch::kFloat32);
	auto K = torch::tensor({1, 2}).to(torch::kFloat32);
	auto Y = corr1d(X, K);
	std::cout << Y.sizes() << '\n';
	std::cout << "corr1d(X, K):\n" << Y << '\n';

	// multiple input channels
	X = torch::tensor({{0, 1, 2, 3, 4, 5, 6},
	              {1, 2, 3, 4, 5, 6, 7},
	              {2, 3, 4, 5, 6, 7, 8}}).to(torch::kFloat32);

	K = torch::tensor({{1, 2}, {3, 4}, {-1, -3}}).to(torch::kFloat32);

	Y = corr1d_multi_in(X, K);
	std::cout << "corr1d_multi_in(X, K):\n" << Y << '\n';

	// -------------------------------------------
	// The textCNN Model
	// -------------------------------------------
	size_t num_steps = 500, embed_size = 100;  // sequence length

	std::string data_dir = "./data/aclImdb";

	auto rlt = load_data_imdb(data_dir, num_steps, 0);
	auto features  = std::get<0>(rlt);
	auto rlab      = std::get<1>(rlt);
	auto tfeatures = std::get<2>(rlt);
	auto trlab     = std::get<3>(rlt);
	Vocab vocab    = std::get<4>(rlt);

	int64_t batch_size = 64;

	//	std::cout << "features: " << features.sizes() << ", rlab: " << rlab.sizes() << '\n';

	auto dataset = LRdataset(std::make_pair(features, rlab)).map(torch::data::transforms::Stack<>());
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		        												std::move(dataset), batch_size);

	auto testset = LRdataset(std::make_pair(tfeatures, trlab)).map(torch::data::transforms::Stack<>());
	auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
			        												std::move(testset), batch_size);

	//vocab.idx_to_token
	std::string embedding_name = "./data/glove.6B.100d/vec.txt";

	TokenEmbedding glove_embedding = TokenEmbedding(embedding_name);
	std::cout << glove_embedding.idx_to_token[1] << '\n';

	auto embeds = glove_embedding[vocab.idx_to_token];
	std::cout << embeds.sizes() << '\n';

	std::vector<int64_t> kernel_sizes = {3, 4, 5};
	std::vector<int64_t> nums_channels = {100, 100, 100};

	std::cout << vector_sum(kernel_sizes) << '\n';

	auto net = TextCNN(vocab.length(), embed_size, kernel_sizes, nums_channels);
	net->init_weights();
	net->to(device);

	net->embedding->weight.data().copy_(embeds);
	net->constant_embedding->weight.data().copy_(embeds);
	net->constant_embedding->weight.requires_grad_(false);

	// ----------------------------------------------------------
	// Training and Evaluating the Model
	// ----------------------------------------------------------
	float lr = 0.01;
	int num_epochs = 3;

	auto trainer = torch::optim::Adam(net->parameters(), lr);
	auto loss = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().reduction(torch::kNone)); //reduction="none"

	net->train();
/*
	auto batch_data = *data_loader->begin();
	auto img_data  = batch_data.data.to(device);
	auto lab_data  = batch_data.target.to(device).squeeze().flatten();
	trainer.zero_grad();
	auto pred = net->forward(img_data);
	std::cout << "pred:\n" << pred.sizes() << "\ny:\n" << lab_data.sizes() << std::endl;
	std::cout << "pred:\n" << pred << "\ny:\n" << lab_data << std::endl;
	auto l = loss(pred, lab_data);
	l.sum().backward();

	trainer.step();
	std::cout << accuracy(pred, lab_data) << '\n';
*/

	std::vector<double> train_loss, train_acc, test_acc;
	std::vector<double> train_epochs;

	for(size_t epoch = 1; epoch <= num_epochs; epoch++) {
		net->train();

		std::cout << "--------------- Training -----------------> " << epoch << " / " << num_epochs << "\n";

		float loss_sum = 0.0;
		int64_t total_corrects = 0, total_samples = 0;
		size_t num_batch = 0;

		for(auto& batch_data : *data_loader) {
			auto ftr_data  = batch_data.data.to(device);
			auto lab_data  = batch_data.target.to(device).squeeze().flatten();
			//std::cout << "img_data: " << img_data.sizes() << " y: " << lab_data.sizes() << std::endl;
			//std::cout << "img_data:\n" << img_data << "\ny:\n" << lab_data << std::endl;

			trainer.zero_grad();
			auto pred = net->forward(ftr_data);
			auto l = loss(pred, lab_data);
			l.sum().backward();

			trainer.step();
			loss_sum += (l.sum().data().item<float>()/ftr_data.size(0));

			total_corrects += accuracy(pred, lab_data);

			total_samples += ftr_data.size(0);
			num_batch++;
			//std::cout << "num_batch: " << num_batch << '\n';
		}
		train_epochs.push_back(epoch*1.0);
		train_loss.push_back((loss_sum*1.0/num_batch));
		train_acc.push_back((total_corrects*1.0/total_samples));
		std::cout << "loss: " << (loss_sum*1.0/num_batch) << ", train acc: " << (total_corrects*1.0/total_samples) << '\n';

		net->eval();
		//torch::NoGradGuard no_grad;
		total_corrects = 0, total_samples = 0;
		for(auto& batch_data : *test_loader) {
			auto ftr_data  = batch_data.data.to(device);
			auto lab_data  = batch_data.target.to(device).squeeze().flatten();

			auto pred = net->forward(ftr_data);
			total_corrects += accuracy(pred, lab_data);
			total_samples += ftr_data.size(0);
		}
		std::cout << "test acc: " << (total_corrects*1.0/total_samples) << '\n';
		test_acc.push_back((total_corrects*1.0/total_samples));
	}

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::plot(ax1, train_epochs, train_loss, "b")->line_width(2)
			.display_name("train loss");
	matplot::plot(ax1, train_epochs, train_acc, "g--")->line_width(2)
			.display_name("train acc");
	//matplot::plot(ax1, train_epochs, test_acc, "r-.")->line_width(2)
	//		.display_name("test acc");
	matplot::hold(ax1, false);
	matplot::legend(ax1);
    matplot::xlabel(ax1, "epoch");
    matplot::show();

	// ----------------------------------------------------------
	//  predict the sentiment of a text sequence using the trained model
	// ----------------------------------------------------------
	std::string sequence = "this movie is so great";
	std::string review = predict_sentiment(net, vocab, sequence,  num_steps, device);
	std::cout << "review: " << review << '\n';

	sequence = "this movie is so bad";
	review = predict_sentiment(net, vocab, sequence,  num_steps, device);
	std::cout << "review: " << review << '\n';

	std::cout << "Done!\n";
	return 0;
}


