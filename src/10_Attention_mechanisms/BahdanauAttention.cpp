#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../utils/ch_10_util.h"
#include "../utils.h"

// ----------------------------------------------------------
// Prediction
// To predict the output sequence token by token,
// at each decoder time step the predicted token from
// the previous time step is fed into the decoder as an input.
// -----------------------------------------------------------

template<typename T>
std::string predict_seq2seq(T net, std::string src_sentence, Vocab src_vocab, Vocab tgt_vocab, int64_t num_steps,
					torch::Device device, bool save_attention_weights=false) {

	auto options = torch::TensorOptions().dtype(torch::kLong);
    // Predict for sequence to sequence.
	net->eval();
    std::vector<std::string> etks = tokenize({src_sentence}, "word", false);

	std::vector<int64_t> vec;
	std::vector<int64_t> a = src_vocab[etks];
	a.push_back(src_vocab["<eos>"]);
	auto c = truncate_pad( a, num_steps, src_vocab["<pad>"]);
	for(auto i : c)
		vec.push_back(i);

	torch::Tensor src_ar = (torch::from_blob(vec.data(), {1, num_steps}, options)).clone();
	src_ar = src_ar.to(torch::kLong);
	torch::Tensor src_val_len = (src_ar != src_vocab["<pad>"]).to(torch::kLong).sum(1);


	torch::Tensor empty;
	std::tuple<torch::Tensor, torch::Tensor> enc_outputs = net->encoder->forward(src_ar);

	torch::Tensor prd;
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> dec_state = net->decoder->init_state(enc_outputs, empty);
	std::tie(prd, dec_state) = net->decoder->forward(src_ar, dec_state);
	prd = prd.argmax(2);

	auto r_ptr = prd.data_ptr<int64_t>();
	std::vector<int64_t> idx{r_ptr, r_ptr + prd.numel()};
//	std::for_each(std::begin(idx), std::end(idx), [](const auto & element) { std::cout << element << " "; });
//	std::cout << std::endl;

	int64_t L = src_val_len.data().item<int64_t>() - 1;
	std::string out = "";
	for( int64_t i = 0; i < L; i++ ) {
		std::string tk = tgt_vocab.idx_to_token[idx[i]];
		if( out == "" )
			out = tk;
		else
			if( tk != "<eos>" ) out += (" " + tk);
	}
    return out;
}

// ----------------------------------------------------
// implement the RNN decoder with Bahdanau attention
// ----------------------------------------------------
struct Seq2SeqAttentionDecoderImpl : public torch::nn::Module  {
	AdditiveAttention attention{nullptr};
	torch::nn::Embedding embedding{nullptr};
	torch::nn::GRU rnn{nullptr};
	torch::nn::Linear dense{nullptr};
	std::vector<torch::Tensor> _attention_weights;

	Seq2SeqAttentionDecoderImpl(int64_t vocab_size, int64_t embed_size, int64_t num_hiddens, int64_t num_layers, float dropout=0) {
        attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout);
        embedding = torch::nn::Embedding(vocab_size, embed_size);
        rnn = torch::nn::GRU( torch::nn::GRUOptions(embed_size + num_hiddens, num_hiddens).num_layers(num_layers).dropout(dropout));
        		//embed_size + num_hiddens, num_hiddens, num_layers, dropout);
        dense = torch::nn::Linear(num_hiddens, vocab_size);
        register_module("attention", attention);
        register_module("embedding", embedding);
        register_module("rnn", rnn);
        register_module("dense", dense);
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>  init_state(std::tuple<torch::Tensor, torch::Tensor> enc_outputs,
			torch::Tensor enc_valid_lens) {
        // Shape of `outputs`: (`num_steps`, `batch_size`, `num_hiddens`).
        // Shape of `hidden_state[0]`: (`num_layers`, `batch_size`, `num_hiddens`)
		torch::Tensor outputs, hidden_state;
        std::tie(outputs, hidden_state) = enc_outputs;
        return {outputs.permute({1, 0, 2}), hidden_state, enc_valid_lens};
    }

    std::pair<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> forward(torch::Tensor X,
    															std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> state) {
        // Shape of `enc_outputs`: (`batch_size`, `num_steps`, `num_hiddens`).
        // Shape of `hidden_state[0]`: (`num_layers`, `batch_size`, `num_hiddens`)
		torch::Tensor enc_outputs, hidden_state, enc_valid_lens;
        std::tie(enc_outputs, hidden_state, enc_valid_lens) = state;
        // Shape of the output `X`: (`num_steps`, `batch_size`, `embed_size`)
        X = embedding->forward(X).permute({1, 0, 2});
        _attention_weights.clear();
        std::vector<torch::Tensor> outputs;
        //std::cout << "X---1: " << X.sizes() << "\n";
        //std::cout << "enc_outputs---1: " << enc_outputs.sizes() << "\n";
        //std::cout << "hidden_state---1: " << hidden_state.sizes() << "\n";

        for( int i = 0; i < X.size(0); i++ ) {
        	auto x = X[i].clone();
        	//std::cout << "x---2: " << x.sizes() << "\n";
            // Shape of `query`: (`batch_size`, 1, `num_hiddens`)
        	//std::cout << "hidden_state[-1]---2: " << hidden_state[-1].sizes() << "\n";
            auto query = torch::unsqueeze(hidden_state[-1], /*dim=*/1);
            //std::cout << "query---2: " << query.sizes() << "\n";

            //std::cout << "enc_outputs---2: " << enc_outputs.sizes() << "\n";
            // Shape of `context`: (`batch_size`, 1, `num_hiddens`)
            auto context = attention->forward(query, enc_outputs, enc_outputs, enc_valid_lens);
            //std::cout << "context---2: " << context.sizes() << "\n";

            // Concatenate on the feature dimension
            x = torch::cat({context, torch::unsqueeze(x, 1)}, -1);
            //std::cout << "x_cat---2: " << x.sizes() << "\n";
            // Reshape `x` as (1, `batch_size`, `embed_size` + `num_hiddens`)
            torch::Tensor out;
            std::tie(out, hidden_state) = rnn->forward(x.permute({1, 0, 2}), hidden_state);
            //std::cout << "out---2: " << out.sizes() << "\n";
            outputs.push_back(out);
            _attention_weights.push_back(attention->attention_weights);
        }
        // After fully-connected layer transformation, shape of `outputs`:
        // (`num_steps`, `batch_size`, `vocab_size`)
        torch::Tensor output = dense->forward(torch::cat(outputs, 0));
        //std::cout << "output---2: " << output.sizes() << "\n";
        //std::cout << "output.permute({1, 0, 2})---2: " << output.permute({1, 0, 2}).sizes() << "\n";
        //return {torch::empty({0}), std::make_tuple(enc_outputs, hidden_state,enc_valid_lens)};
        return {output.permute({1, 0, 2}), std::make_tuple(enc_outputs, hidden_state, enc_valid_lens)};
    }

    std::vector<torch::Tensor> attention_weights() {
        return _attention_weights;
    }
};

TORCH_MODULE(Seq2SeqAttentionDecoder);

// EcoderDecoder
struct EncoderDecoderImpl : public torch::nn::Module {
    //The base class for the encoder-decoder architecture.
    //Defined in :numref:`sec_encoder-decoder`
	Seq2SeqAttentionDecoder decoder{nullptr};
	Seq2SeqEncoder encoder{nullptr};

	EncoderDecoderImpl(Seq2SeqEncoder encoder, Seq2SeqAttentionDecoder decoder) {
        this->encoder = encoder;
        this->decoder = decoder;
        register_module("decoder", decoder);
        register_module("encoder", encoder);
	}

	std::pair<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> forward(torch::Tensor enc_X,
													   torch::Tensor dec_X, torch::Tensor val_lens=torch::empty({0})) {
		// input torch::Tensor
		std::tuple<torch::Tensor, torch::Tensor> enc_outputs = this->encoder->forward(enc_X);
		// input std::tuple<torch::Tensor, torch::Tensor> and torch::Tensor
		std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> dec_state = this->decoder->init_state(enc_outputs, val_lens);
		// torch::Tensor and std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
        return this->decoder->forward(dec_X, dec_state);
	}

};

TORCH_MODULE(EncoderDecoder);

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	// test the implemented decoder with Bahdanau attention using a minibatch of 4 sequence inputs of 7 time steps.
	auto encoder = Seq2SeqEncoder(10, 8, 16, 2);
	encoder->to(device);
	encoder->eval();

	auto decoder = Seq2SeqAttentionDecoder(10, 8, 16, 2);
	decoder->to(device);
	decoder->eval();

	auto X = torch::zeros({4, 7}).to(torch::kLong); 	// (`batch_size`, `num_steps`)
	std::tuple<torch::Tensor, torch::Tensor> enc_outputs = encoder->forward(X);
	std::cout << std::get<0>(enc_outputs).sizes() << std::endl;
	std::cout << std::get<1>(enc_outputs).sizes() << std::endl;

	torch::Tensor val_lens; // ! defined()
	torch::Tensor val_lens2 = torch::empty({0});
	std::cout << "defined: " << val_lens.defined() << ", numel(): " << val_lens2.numel() << "\n";

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> state = decoder->init_state(enc_outputs, val_lens);

	std::cout << std::tuple_size<decltype(state)>::value << std::endl;
	std::cout << std::get<0>(state).sizes() << std::endl;
	std::cout << std::get<1>(state).sizes() << std::endl;
	std::cout << std::get<2>(state).numel() << std::endl;


	torch::Tensor output;
	std::tie(output, state) = decoder->forward(X, state);
	std::cout << output.sizes() << std::endl;

	// ------------------------------------------
	// Training
	// ------------------------------------------

	int64_t embed_size = 32, num_hiddens = 32, num_layers = 2;
	float dropout = 0.1, lr = 0.005;

	int64_t batch_size = 64, num_steps = 10,  num_epochs = 500;

	std::string filename = "./data/fra-eng/fra.txt";

	Vocab src_vocab, tgt_vocab;
	torch::Tensor src_array, src_valid_len, tgt_array, tgt_valid_len;

	std::tie(src_array, src_valid_len, tgt_array, tgt_valid_len, src_vocab, tgt_vocab) = load_data_nmt(filename, num_steps, 600);

	// define model
	auto enc = Seq2SeqEncoder( src_vocab.length(), embed_size, num_hiddens, num_layers, dropout );
	auto dec = Seq2SeqAttentionDecoder( tgt_vocab.length(), embed_size, num_hiddens, num_layers, dropout );

	auto net = EncoderDecoder(enc, dec);
	net->to(device);

	net->apply(xavier_init_weights);

	//auto optimizer = torch::optim::Adam(net->parameters(), lr);
	torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(lr).betas({0.5, 0.999}));

	auto loss_fn = MaskedSoftmaxCELoss();
	net->train(true);

	std::vector<float> epochs, plsum;
	std::vector<int64_t> wtks;
	torch::Tensor Y_hat;
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> stat;

	for( int64_t epoch = 0;  epoch < num_epochs; epoch++ ) {
		// get shuffled batch data indices
		std::list<torch::Tensor> idx_iter = data_index_iter(src_array.size(0), batch_size, true);

		float t_loss = 0.0;
		int64_t cnt = 0;
		int64_t n_wtks = 0;

		for(auto& batch_idx : idx_iter) {

	        optimizer.zero_grad();

	        auto X = src_array.index_select(0, batch_idx);
	        auto X_valid_len = src_valid_len.index_select(0, batch_idx);
	        auto Y = tgt_array.index_select(0, batch_idx);
	        auto Y_valid_len = tgt_valid_len.index_select(0, batch_idx);
	        X.to(device);
	        X_valid_len.to(device);
	        Y.to(device);
	        Y_valid_len.to(device);

	        std::vector<int64_t> tmp;
	        for(int i = 0; i < Y.size(0); i++)
	        	tmp.push_back(tgt_vocab["<bos>"]);

	        auto options = torch::TensorOptions().dtype(torch::kLong);
	        torch::Tensor bos = torch::from_blob(tmp.data(), {1, Y.size(0)}, options).clone();
	        bos = bos.to(device).reshape({-1, 1});

	        torch::Tensor dec_input = torch::cat({bos, Y.index({Slice(), Slice(None, -1)})}, 1); // Y[:, :-1]

	        std::pair<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> out = net->forward(X, dec_input); // X_valid_len

	        Y_hat = out.first;
			stat = out.second;

	        auto l = loss_fn.forward(Y_hat, Y, Y_valid_len);
	        l.sum().backward(); 	// Make the loss scalar for `backward`

	        //"""Clip the gradient."""
	        torch::nn::utils::clip_grad_norm_(net->parameters(), 1.0);
	        auto num_tokens = Y_valid_len.sum();

	        optimizer.step();
	        torch::NoGradGuard no_grad;
	        //torch::no_grad();
	        t_loss += ((l.sum()).item<float>()/Y.size(0));
	       	n_wtks += num_tokens.item<long>();
	       	cnt++;
	    }

	    if(epoch % 10 == 0) {
	    	std::cout << "loss: " << (t_loss/cnt) << std::endl;
	    	plsum.push_back((t_loss/cnt));
	    	wtks.push_back(static_cast<int64_t>(n_wtks/cnt));
	    	epochs.push_back(1.0*epoch);
	    }
	}

	plt::figure_size(800, 600);
	plt::named_plot("train", epochs, plsum, "b");
	plt::legend();
	plt::xlabel("epoch");
	plt::ylabel("loss");
	plt::show();
	plt::close();

	printf("\n\n");
	// Prediction
	std::vector<std::string> engs = {"go .", "i lost .", "he\'s calm .", "i\'m home ."};
	std::vector<std::string> fras = {"va !", "j\'ai perdu .", "il est calme .", "je suis chez moi ."};

	for( int t = 0; t < engs.size(); t++ ) {
		std::string translation = predict_seq2seq(net, engs[t], src_vocab, tgt_vocab, num_steps, device);
		std::cout << "translation: " << translation << "\n";
		std::cout << "target: " << fras[t] << "\n";

		auto score = bleu(translation, fras[t],  2);
		std::cout << "Bleu score: " << score << "\n";
	}

	std::cout << "Done!\n";
	return 0;
}



