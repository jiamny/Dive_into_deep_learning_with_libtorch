#include <torch/utils.h>
#include "../utils/ch_15_util.h"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

torch::nn::Sequential mlp(size_t num_inputs, size_t num_hiddens, bool flatten) {
	torch::nn::Sequential net{nullptr};

    if( flatten ) {
    	net = torch::nn::Sequential(
    		torch::nn::Dropout(0.2),
    		torch::nn::Linear(num_inputs, num_hiddens),
			torch::nn::ReLU(),
			torch::nn::Flatten()
    	);

        //net.append(nn.Flatten(start_dim=1))
    } else {
    	net = torch::nn::Sequential(
    	    torch::nn::Dropout(0.2),
    		torch::nn::Linear(num_inputs, num_hiddens),
    		torch::nn::ReLU()
    	);
    }
    /*
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)
    */
    return net;
}


class Attend : public torch::nn::Module {
public:
	Attend(){};
	Attend(size_t num_inputs, size_t num_hiddens) {
        this->f = mlp(num_inputs, num_hiddens, false);
	}

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor A, torch::Tensor B) {
        // Shape of `A`/`B`: (`batch_size`, no. of tokens in sequence A/B,
        // `embed_size`)
        // Shape of `f_A`/`f_B`: (`batch_size`, no. of tokens in sequence A/B,
        // `num_hiddens`)
        auto f_A = f->forward(A);
        auto f_B = f->forward(B);

        // Shape of `e`: (`batch_size`, no. of tokens in sequence A,
        // no. of tokens in sequence B)
        auto e = torch::bmm(f_A, f_B.permute({0, 2, 1}));

        // Shape of `beta`: (`batch_size`, no. of tokens in sequence A,
        // `embed_size`), where sequence B is softly aligned with each token
        // (axis 1 of `beta`) in sequence A
        auto beta = torch::bmm(torch::softmax(e, -1), B);

        // Shape of `alpha`: (`batch_size`, no. of tokens in sequence B,
        // `embed_size`), where sequence A is softly aligned with each token
        // (axis 1 of `alpha`) in sequence B
        auto alpha = torch::bmm(torch::softmax(e.permute({0, 2, 1}), -1), A);

        return std::make_pair(beta, alpha);
    }
private:
	torch::nn::Sequential f{nullptr};
};


class Compare : public torch::nn::Module {
public:
	Compare(){};
    Compare(size_t num_inputs, size_t num_hiddens) {
        this->g = mlp(num_inputs, num_hiddens, false);
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor A, torch::Tensor B, torch::Tensor beta, torch::Tensor alpha) {
        auto V_A = g->forward(torch::cat({A, beta}, 2));
        auto V_B = g->forward(torch::cat({B, alpha}, 2));

        return std::make_pair(V_A, V_B);
    }
private:
	torch::nn::Sequential g{nullptr};
};


class Aggregate : public torch::nn::Module {
public:
	Aggregate(){};
	Aggregate(size_t num_inputs, size_t num_hiddens, size_t num_outputs) {
        this->h = mlp(num_inputs, num_hiddens, true);
        this->linear = torch::nn::Linear(num_hiddens, num_outputs);
	}

	torch::Tensor forward(torch::Tensor V_A, torch::Tensor V_B) {
        // Sum up both sets of comparison vectors
        V_A = V_A.sum(1);
        V_B = V_B.sum(1);

        // Feed the concatenation of both summarization results into an MLP
        auto Y_hat = linear->forward(h->forward(torch::cat({V_A, V_B}, 1)));
        return Y_hat;
	}

private:
	torch::nn::Sequential h{nullptr};
	torch::nn::Linear linear{nullptr};
};


class DecomposableAttention : public torch::nn::Module {
public:
	torch::nn::Embedding embedding{nullptr};
	DecomposableAttention(Vocab vocab, size_t embed_size, size_t num_hiddens, size_t num_inputs_attend,
                 size_t num_inputs_compare, size_t num_inputs_agg) {
        this->embedding = torch::nn::Embedding(vocab.length(), embed_size);
        this->attend = Attend(num_inputs_attend, num_hiddens);
        this->compare = Compare(num_inputs_compare, num_hiddens);
        // There are 3 possible outputs: entailment, contradiction, and neutral
        this->aggregate = Aggregate(num_inputs_agg, num_hiddens, 3);
	}

	torch::Tensor forward(std::pair<torch::Tensor, torch::Tensor> X) {
        auto premises = X.first;
        auto hypotheses = X.second;
        auto A = embedding->forward(premises);
        auto B = embedding->forward(hypotheses);
        auto rlt = attend.forward(A, B);
        auto beta = rlt.first, alpha = rlt.second;
        auto V = compare.forward(A, B, beta, alpha);
        auto Y_hat = aggregate.forward(V.first, V.second);
	    return Y_hat;
	}
private:
	Attend attend;
	Compare compare;
	Aggregate aggregate;
};


std::string predict_snli(DecomposableAttention& net, Vocab vocab, std::vector<std::string> premise_dt,
						 std::vector<std::string> hypothesis_dt, size_t num_steps, torch::Device device) {
	// Predict the logical relationship between the premise and hypothesis."""
    net.eval();

    std::vector<torch::Tensor> ptensors;

    for( int i = 0; i < premise_dt.size(); i++ ) {
        auto dt = truncate_pad(vocab[count_num_tokens(premise_dt[i]).first], num_steps, vocab["<pad>"]);

        auto TT = torch::from_blob(dt.data(), {1, static_cast<long>(num_steps)}, torch::TensorOptions(torch::kLong)).clone();

        ptensors.push_back(TT);
    }

    torch::Tensor premise = torch::concat(ptensors, 0).to(torch::kLong);

    std::vector<torch::Tensor> htensors;

    for( int i = 0; i < hypothesis_dt.size(); i++ ) {
        auto dt = truncate_pad(vocab[count_num_tokens(hypothesis_dt[i]).first], num_steps, vocab["<pad>"]);

        auto TT = torch::from_blob(dt.data(), {1, static_cast<long>(num_steps)}, torch::TensorOptions(torch::kLong)).clone();

        htensors.push_back(TT);
    }

    torch::Tensor hypothesis = torch::concat(htensors, 0).to(torch::kLong);

    torch::Tensor label = torch::argmax(net.forward(std::make_pair(premise.reshape({1, -1}),hypothesis.reshape({1, -1}))), 1);
    if(label.data().item<long>() == 0)
    	return "entailment";
    else if(label.data().item<long>() == 1)
    	return "contradiction";
    else
    	return "neutral";
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	// ----------------------------------------------------------
	// Read the SNLI dataset into premises, hypotheses, and labels.
	// ----------------------------------------------------------
	int batch_size = 256, num_steps = 50;
	const std::string data_dir = "./data/snli_1.0";
	bool is_train = true;

	auto train_data = read_snli(data_dir, is_train, 12000);
	auto test_data = read_snli(data_dir, false, 2000);

	float min_freq = 5.0f;
	std::vector<std::string> reserved_tokens;
	reserved_tokens.push_back("<pad>");

	Vocab vocab;
	vocab = get_snil_vocab( train_data, min_freq, reserved_tokens);
	std::cout << "vocab.length: " << vocab.length() << '\n';

	auto dataset = SNLIDataset(train_data, num_steps, vocab).map(torch::data::transforms::Stack<>());
	auto train_iter = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
			        									std::move(dataset),
														torch::data::DataLoaderOptions().batch_size(batch_size).drop_last(true));


/*
	for(auto& dt : *train_iter ) {
	    auto data = dt.data;
	    auto target = dt.target;
	    std::cout << "data.sizes: " << data.sizes() << '\n';
	    std::cout << "target.sizes: " << target.sizes() << '\n';
	    std::cout << "data[:,0:50]: " << data.index({Slice(), Slice(0, num_steps)}).sizes() << '\n';
	    std::cout << "data[:,50:100]: " << data.index({Slice(), Slice(num_steps, 2*num_steps)}).sizes() << '\n';
	    break;
	}
*/
	// ---------------------------------------------------------
	// Creating the Model
	// ---------------------------------------------------------
	size_t embed_size = 100, num_hiddens = 200;
	auto net = DecomposableAttention(vocab, embed_size, num_hiddens, 100, 200, 400);
	auto glove_embedding = TokenEmbedding("./data/glove.6B.100d/vec.txt");
	auto embeds = glove_embedding[vocab.idx_to_token];
	net.embedding->weight.data().copy_(embeds);

	// ---------------------------------------------------------
	// Training and Evaluating the Model
	// ---------------------------------------------------------
	float lr = 0.01;
	int num_epochs = 100;

	auto trainer = torch::optim::Adam(net.parameters(), lr);
	auto loss = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().reduction(torch::kNone));

	auto testdata = SNLIDataset(test_data, num_steps, vocab).map(torch::data::transforms::Stack<>());
	auto test_iter = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
				        												std::move(testdata), batch_size);

	net.train();
/*
	auto batch_data = *train_iter->begin();
	auto t_data     = batch_data.data.to(device);
	auto img_data   = std::make_pair(t_data.index({Slice(), Slice(0, num_steps)}),
									 t_data.index({Slice(), Slice(num_steps, 2*num_steps)}));
	auto lab_data   = batch_data.target.to(device).squeeze().flatten();
	trainer.zero_grad();
	auto pred = net.forward(img_data);
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
		//net.train();

		std::cout << "--------------- Training -----------------> " << epoch << " / " << num_epochs << "\n";

		float loss_sum = 0.0;
		int64_t total_corrects = 0, total_samples = 0;
		size_t num_batch = 0;

		for(auto& batch_data : *train_iter) {
			torch::Tensor t_data  = batch_data.data.to(device);
			std::pair<torch::Tensor, torch::Tensor> ftr_data   = std::make_pair(t_data.index({Slice(), Slice(0, num_steps)}),
											 t_data.index({Slice(), Slice(num_steps, 2*num_steps)}));
			torch::Tensor lab_data  = batch_data.target.to(device).squeeze().flatten();

			trainer.zero_grad();
			torch::Tensor pred = net.forward(ftr_data);
			auto l = loss(pred, lab_data);
			l.sum().backward();

			trainer.step();
			loss_sum += (l.sum().data().item<float>()/t_data.size(0));

			total_corrects += accuracy(pred, lab_data);

			total_samples += t_data.size(0);
			num_batch++;
			//std::cout << "num_batch: " << num_batch << '\n';
		}
		train_epochs.push_back(epoch*1.0);
		train_loss.push_back((loss_sum*1.0/num_batch));
		train_acc.push_back((total_corrects*1.0/total_samples));
		std::cout << "loss: " << (loss_sum*1.0/num_batch) << ", train acc: " << (total_corrects*1.0/total_samples) << '\n';

		//net.eval();
		torch::NoGradGuard no_grad;
		total_corrects = 0, total_samples = 0;
		for(auto& b_data : *test_iter) {
			torch::Tensor t_data  = b_data.data.to(device);
			std::pair<torch::Tensor, torch::Tensor> ftr_data   = std::make_pair(t_data.index({Slice(), Slice(0, num_steps)}),
											 t_data.index({Slice(), Slice(num_steps, 2*num_steps)}));
			auto lab_data  = b_data.target.to(device).squeeze().flatten();

			auto pred = net.forward(ftr_data);
			total_corrects += accuracy(pred, lab_data);
			total_samples += t_data.size(0);
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
	//  Using the Model
	// ----------------------------------------------------------
	auto pred = predict_snli(net, vocab, {"he", "is", "good", "."}, {"he", "is", "bad", "."}, num_steps, device);
	std::cout << "pred: " << pred << '\n';

	std::cout << "Done!\n";
}
