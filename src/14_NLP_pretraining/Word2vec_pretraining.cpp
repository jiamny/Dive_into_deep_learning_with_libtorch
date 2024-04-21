
#include "../utils/ch_14_util.h"
#include <matplot/matplot.h>
using namespace matplot;

template<typename T1, typename T2>
torch::Tensor skip_gram(torch::Tensor center, torch::Tensor contexts_and_negatives,
						T1 embed_v, T2 embed_u) {
	torch::Tensor v = embed_v->forward(center);
	torch::Tensor u = embed_u->forward(contexts_and_negatives);
	torch::Tensor pred = torch::bmm(v, u.permute({0, 2, 1}));
	return pred;
}

struct SigmoidBCELossImpl : public torch::nn::Module {
    // Binary cross-entropy loss with masking
	SigmoidBCELossImpl(void) {};

    torch::Tensor forward(torch::Tensor inputs, torch::Tensor target, torch::Tensor mask=torch::empty(0)) {

    	torch::Tensor out = torch::nn::functional::binary_cross_entropy_with_logits(
            inputs, target, torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions()
			.weight(mask).reduction(torch::kNone));

    	 c10::OptionalArrayRef<long int> dim = {1};
        return torch::mean(out, dim); //mean(dim=1)
    }
};
TORCH_MODULE(SigmoidBCELoss);

double sigmd(double x) {
    return -1 * std::log(1 / (1 + std::exp(-1 * x)));
}

struct Word2VecImpl : public torch::nn::Module {

	torch::nn::Embedding embed_v{nullptr}, embed_u{nullptr};

	Word2VecImpl(int64_t num_embeddings, int64_t embedding_dim) {

		embed_v = torch::nn::Embedding(torch::nn::EmbeddingOptions(num_embeddings, embedding_dim));
		embed_u = torch::nn::Embedding(torch::nn::EmbeddingOptions(num_embeddings, embedding_dim));
		register_module("embed_v", embed_v);
		register_module("embed_u", embed_u);

		// init_weights
		init_weights();
	}

	void init_weights() {
		torch::NoGradGuard noGrad;
		for( auto& module : modules(false) ) {
			if( auto M = dynamic_cast<torch::nn::EmbeddingImpl*>(module.get())) //type(module) == nn.Embedding )
				torch::nn::init::xavier_uniform_(M->weight);
		}
	}
};
TORCH_MODULE(Word2Vec);


int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	torch::Device device = torch::Device(torch::kCPU);

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "// The Skip-Gram Model\n";
	std::cout << "// Embedding Layer\n";
	std::cout << "// -----------------------------------------------------------------\n";

    auto embed = torch::nn::Embedding(torch::nn::EmbeddingOptions(20, 4));
	std::cout << "Parameter embedding_weight: " << embed->weight.sizes() << "dtype=" << embed->weight.dtype() << '\n';

	torch::Tensor x = torch::tensor({{1, 2, 3}, {4, 5, 6}});
	std::cout << "embed->forward(x): " << embed->forward(x) << '\n';

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "// Defining the Forward Propagation\n";
	std::cout << "// -----------------------------------------------------------------\n";
	auto pred = skip_gram(torch::ones({2, 1}).to(torch::kLong),
						  torch::ones({2, 4}).to(torch::kLong), embed, embed);

	std::cout << "pred: " << pred.sizes() << '\n';

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "// Training\n";
	std::cout << "// Binary Cross-Entropy Loss\n";
	std::cout << "// -----------------------------------------------------------------\n";

	auto loss = SigmoidBCELoss();

	torch::Tensor p = torch::tensor({{1.1, -2.2, 3.3, -4.4}, {1.1, -2.2, 3.3, -4.4}});
	torch::Tensor lab = torch::tensor({{1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}});
	torch::Tensor msk = torch::tensor({{1, 1, 1, 1}, {1, 1, 0, 0}});
	c10::OptionalArrayRef<long int> dim = {1};
	std::cout << "loss: " << (loss(p, lab, msk) * msk.size(1) / msk.sum(dim)) << '\n'; // axis=1

	double sig1 = (sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4;
	double sig2 = (sigmd(-1.1) + sigmd(-2.2)) / 2;
	std::cout << "sig1: " << sig1 << " sig2: " << sig2 << '\n';

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "// Defining the Training Loop\n";
	std::cout << "// -----------------------------------------------------------------\n";

	// Initializing Model Parameters
	int64_t batch_size = 512, max_window_size = 5, num_noise_words = 5;
	const std::string file_dir = "./data/ptb";
	Vocab vocab;
	torch::Tensor contexts_negatives_ts, masks_ts, labels_ts, centers_ts;

	std::tie(contexts_negatives_ts, masks_ts, labels_ts, centers_ts, vocab) = load_data_ptb(
			file_dir, batch_size, max_window_size, num_noise_words, 0);

	contexts_negatives_ts = contexts_negatives_ts.to(device);
	masks_ts = masks_ts.to(device);
	labels_ts = labels_ts.to(device);

	float lr = 0.002;
	int64_t num_epochs = 5;
	int64_t embed_size = 100;

	//init_weights(net);
	auto net = Word2Vec(6719, embed_size); //vocab.length(), embed_size);

	net->to(device);
	auto optimizer = torch::optim::Adam(net->parameters(), torch::optim::AdamOptions(lr));

	//std::cout << "net->nets[0] name: " << net->embed_v->weight.sizes()
	//		  << " net->nets[1]: \t" <<  net->embed_u->weight << std::endl;


	auto dataset = PTBDataset(centers_ts).map(torch::data::transforms::Stack<>());
	auto train_iter = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
			        					std::move(dataset),
										torch::data::DataLoaderOptions().batch_size(batch_size).drop_last(true));

	std::vector<double> train_loss;
	std::vector<double> train_epochs;

	for(int64_t epoch = 0; epoch < num_epochs; epoch++) {
		net->train();
		float loss_sum = 0.0;
		size_t num_batch = 0;

		for(auto& batch_data : *train_iter) {
			torch::Tensor centers  = batch_data.data.to(device);
			torch::Tensor tidx  = batch_data.target.squeeze().to(device);

			torch::Tensor ctx_neg_ts = torch::index_select(contexts_negatives_ts, 0, tidx);
		    torch::Tensor msk_ts     = torch::index_select(masks_ts, 0, tidx);
		    torch::Tensor lbls       = torch::index_select(labels_ts, 0, tidx);
/*
		    std::cout << "contexts_negatives_ts: " << ctx_neg_ts.sizes()
		    		  << " masks_ts: " << msk_ts.sizes()
					  << " labels_ts: " << lbls.sizes()
					  << " centers_ts: " << centers.sizes()<< '\n';
*/
		    optimizer.zero_grad();
			auto pred = skip_gram(centers, ctx_neg_ts, net->embed_v, net->embed_u);
			auto ppred = pred.reshape(lbls.sizes()).to(torch::kFloat32);
			auto ll = loss(ppred, lbls.to(torch::kFloat32), msk_ts);
			c10::OptionalArrayRef<long int> dm = {1};
			auto l = (ll * msk_ts.size(1) / msk_ts.sum(dm));

			l.sum().backward();
			optimizer.step();

			loss_sum += (l.sum().data().item<float>()/msk_ts.size(0));
			num_batch++;
		}
		train_epochs.push_back((epoch + 1)*1.0);
		train_loss.push_back((loss_sum*1.0/num_batch));
		std::cout << "epoch: " << (epoch + 1) << " avg_loss: " << (loss_sum*1.0/num_batch) << '\n';
	}

	plot(train_epochs, train_loss, "-o");
	xlabel("epoch");
	ylabel("loss");
	show();

	std::cout << "Done\n";
	return 0;
}

