#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include <vector>
#include <cstdio>

#include "../utils/ch_15_util.h"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

struct BiRNNImpl : public torch::nn::Module {
	torch::nn::Linear decoder{nullptr};
	torch::nn::LSTM encoder{nullptr};
	torch::nn::Embedding embedding{nullptr};

	BiRNNImpl(int vocab_size, int embed_size, int num_hiddens, int num_layers ) {

        embedding = torch::nn::Embedding(vocab_size, embed_size);
        // Set `bidirectional` to True to get a bidirectional RNN
        encoder = torch::nn::LSTM( torch::nn::LSTMOptions(embed_size, num_hiddens)
        		.num_layers(num_layers)
				.bidirectional(true));
        decoder = torch::nn::Linear(4 * num_hiddens, 2);
        //decoder = torch::nn::Linear(2 * num_hiddens, 2);

        register_module("embedding", embedding);
        register_module("encoder", encoder);
        register_module("decoder", decoder);

        // init_weights
        init_weights();
	}

    torch::Tensor forward(torch::Tensor inputs) {
    	/*
        # The shape of `inputs` is (batch size, no. of time steps). Because
        # LSTM requires its input's first dimension to be the temporal
        # dimension, the input is transposed before obtaining token
        # representations. The output shape is (no. of time steps, batch size,
        # word vector dimension)
        */
        torch::Tensor embeddings = embedding->forward(inputs.transpose(1, 0));
        //std::cout << "embeddings: " << embeddings.sizes() << '\n';
        encoder->flatten_parameters();
		/*
        # Returns hidden states of the last hidden layer at different time
        # steps. The shape of `outputs` is (no. of time steps, batch size,
        # 2 * no. of hidden units)
        */
        //outputs, _ = encoder(embeddings);

        torch::Tensor outputs = std::get<0>(encoder->forward(embeddings));

        /*
        # Concatenate the hidden states of the initial time step and final
        # time step to use as the input of the fully connected layer. Its
        # shape is (batch size, 4 * no. of hidden units)
        */

        //std::cout << "outputs: " << outputs.sizes() << '\n';
        auto encoding = torch::cat({outputs[0], outputs[outputs.size(0)-1]}, 1);
        //std::cout << "outputs[0]: " << outputs[0].sizes() << '\n';
        //std::cout << "outputs[-1]: " << outputs[outputs.size(0)-1].sizes() << '\n';
        //std::cout << "encoding: " << encoding.sizes() << '\n';

        /*
        # Concatenate the hidden states at the initial and final time steps as
        # the input of the fully-connected layer. Its shape is (batch size,
        # 4 * no. of hidden units)
        */
        auto outs = decoder->forward(encoding);
        //std::cout << "outs: " << outs.sizes() << '\n';
        return outs;
    }

    void init_weights() {
    	torch::NoGradGuard noGrad;
        for(auto& module : modules(/*include_self=*/false)) {
        	if(auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
        		torch::nn::init::xavier_normal_(M->weight);
        	}
        }
    }
};

TORCH_MODULE(BiRNN);


std::string predict_sentiment(BiRNN net, Vocab vocab, std::string sequence, size_t num_steps, torch::Device device) {
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
	auto label = torch::argmax(pred, 1);

    return  label.item<long>() == 1 ? "positive" : "negative";
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	size_t num_steps = 500;  // sequence length

	std::string data_dir = "/home/stree/git/Deep_Learning_with_Libtorch/data/aclImdb";

	auto rlt = load_data_imdb(data_dir, num_steps, 0);
	auto features  = std::get<0>(rlt);
	auto rlab      = std::get<1>(rlt);
	auto tfeatures = std::get<2>(rlt);
	auto trlab     = std::get<3>(rlt);
	Vocab vocab    = std::get<4>(rlt);

//	std::cout << features[0] << '\n';
//	std::cout << rlab << '\n';

	int64_t batch_size = 64;

//	std::cout << "features: " << features.sizes() << ", rlab: " << rlab.sizes() << '\n';

	auto dataset = LRdataset(std::make_pair(features, rlab)).map(torch::data::transforms::Stack<>());
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		        												std::move(dataset), batch_size);

	auto testset = LRdataset(std::make_pair(tfeatures, trlab)).map(torch::data::transforms::Stack<>());
	auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
			        												std::move(testset), batch_size);

	int embed_size = 100, num_hiddens = 100, num_layers = 2;
	auto net = BiRNN(vocab.length(), embed_size, num_hiddens, num_layers);
	net->to(device);

	//vocab.idx_to_token
	std::string embedding_name = "./data/glove.6B.100d/vec.txt";

	TokenEmbedding glove_embedding = TokenEmbedding(embedding_name);
	std::cout << glove_embedding.idx_to_token[1] << '\n';

	auto embeds = glove_embedding[vocab.idx_to_token];
	std::cout << embeds.sizes() << '\n';

	// We use these pretrained word vectors to represent tokens in the reviews
	// and will not update these vectors during training.
	net->embedding->weight.data().copy_(embeds);
	net->embedding->weight.requires_grad_(false);

	// ----------------------------------------------------------
	// Training and Evaluating the Model
	// ----------------------------------------------------------
	float lr = 0.01;
	int num_epochs = 30;

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
			//std::cout << "ftr_data: " << ftr_data.sizes() << " y: " << lab_data.sizes() << std::endl;
			//std::cout << "ftr_data:\n" << ftr_data << "\ny:\n" << lab_data << std::endl;

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

	plt::figure_size(800, 600);
	plt::named_plot("train loss", train_epochs, train_loss, "b");
	plt::named_plot("train acc", train_epochs, train_acc, "g--");
	plt::named_plot("test acc", train_epochs, test_acc, "r-.");
	plt::xlabel("epoch");
	plt::legend();
	plt::show();
	plt::close();

	// ----------------------------------------------------------
	//  predict the sentiment of a text sequence using the trained model
	// ----------------------------------------------------------
	std::string sequence = "this movie is so great";
	std::string review = predict_sentiment(net, vocab, sequence,  num_steps, device);
	std::cout << "\nreview: " << review << '\n';

	sequence = "this movie is so bad";
	review = predict_sentiment(net, vocab, sequence,  num_steps, device);
	std::cout << "review: " << review << '\n';

	// ----------------------------------------------------------
	// export model
	// ----------------------------------------------------------

	torch::save(net, "./src/15_NLP_applications/text_rnn_model.pt");

	std::cout << "Done!\n";

	return 0;
}




