#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <vector>
#include <cmath>

#include "../utils/ch_10_util.h"
#include "../utils.h"

struct PositionWiseFFNImpl : public torch::nn::Module {
	torch::nn::Linear dense1{nullptr}, dense2{nullptr};
	torch::nn::ReLU relu{nullptr};
    //Positionwise feed-forward network.
	PositionWiseFFNImpl( int64_t ffn_num_input, int64_t ffn_num_hiddens, int64_t ffn_num_outputs ) {
        //super(PositionWiseFFN, self).__init__(**kwargs)
        dense1 = torch::nn::Linear(ffn_num_input, ffn_num_hiddens);
        relu = torch::nn::ReLU();
        dense2 = torch::nn::Linear(ffn_num_hiddens, ffn_num_outputs);
        register_module("dense1", dense1);
        register_module("relu", relu);
        register_module("dense2", dense2);
	}

	torch::Tensor forward(torch::Tensor X) {
        return dense2->forward(relu->forward(dense1->forward(X)));
	}
};
TORCH_MODULE(PositionWiseFFN);

// Now we can implement the AddNorm class [using a residual connection followed
// by layer normalization]. Dropout is also applied for regularization.
struct AddNormImpl : public torch::nn::Module {
	torch::nn::Dropout drpout{nullptr};
	torch::nn::LayerNorm ln{nullptr};

    //Residual connection followed by layer normalization.
	AddNormImpl(std::vector<int64_t> normalized_shape, float dropout) {
        drpout = torch::nn::Dropout(dropout);
        ln = torch::nn::LayerNorm(torch::nn::LayerNormOptions(normalized_shape));
        register_module("drpout", drpout);
        register_module("ln", ln);
	}

    torch::Tensor forward(torch::Tensor X, torch::Tensor Y) {
        return ln->forward(drpout->forward(Y) + X);
    }
};

TORCH_MODULE(AddNorm);

// Encoder
struct EncoderBlockImpl : public torch::nn::Module {
	AddNorm addnorm1{nullptr}, addnorm2{nullptr};
	PositionWiseFFN ffn{nullptr};
	MultiHeadAttention attention{nullptr};

    //Transformer encoder block.
	EncoderBlockImpl(int64_t key_size, int64_t query_size, int64_t value_size, int64_t num_hiddens,
			std::vector<int64_t>  norm_shape, int64_t  ffn_num_input, int64_t  ffn_num_hiddens, int64_t num_heads,
                 float dropout, bool use_bias=false) {

        attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias);

        addnorm1 = AddNorm(norm_shape, dropout);

        ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens);

        addnorm2 = AddNorm(norm_shape, dropout);

        register_module("attention", attention);
        register_module("ffn", ffn);
        register_module("addnorm1", addnorm1);
        register_module("addnorm2", addnorm2);

	}

    torch::Tensor forward(torch::Tensor  X, torch::Tensor  valid_lens) {
        auto Y = addnorm1->forward(X, attention->forward(X, X, X, valid_lens));
        return addnorm2->forward(Y, ffn->forward(Y));
    }
};
TORCH_MODULE(EncoderBlock);

// Transformer encoder
struct TransformerEncoderImpl : public torch::nn::Module {
	int num_hiddens;
	std::vector<EncoderBlock> blks;
	torch::nn::Embedding embedding{nullptr};
	PositionalEncoding pos_encoding{nullptr};
	std::vector<torch::Tensor> attention_weights;

    //Transformer encoder
	TransformerEncoderImpl( int64_t vocab_size, int64_t key_size, int64_t query_size, int64_t value_size,
			int64_t n_hiddens, std::vector<int64_t> norm_shape, int64_t ffn_num_input, int64_t ffn_num_hiddens,
			int64_t num_heads, int64_t num_layers, float dropout, bool use_bias=false) {

        num_hiddens = n_hiddens;
        embedding = torch::nn::Embedding(vocab_size, num_hiddens);
        pos_encoding = PositionalEncoding(num_hiddens, dropout);

        register_module("embedding", embedding);
        register_module("pos_encoding", pos_encoding);

        for(int i = 0; i < num_layers; i++) {
            //blks.add_module("block"+str(i),
             auto blk = EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias);

             register_module("EncoderBlock_"+ std::to_string(i), blk);
             blks.push_back(blk);
             //blks[i].replace_module("block"+ std::to_string(i), *blk);
        }
	}

    torch::Tensor forward(torch::Tensor X, torch::Tensor valid_lens) {
        // Since positional encoding values are between -1 and 1, the embedding
        // values are multiplied by the square root of the embedding dimension
        // to rescale before they are summed up
        X = pos_encoding->forward(embedding->forward(X) * std::sqrt(num_hiddens));
        //attention_weights = [None] * len(self.blks)
        attention_weights.clear();
        attention_weights.resize(blks.size());
        //for i, blk in enumerate(self.blks):
        int i = 0;
        for(auto& blk : blks) {
            X = blk->forward(X, valid_lens);
            attention_weights[i] = blk->attention->attention->attention_weights;
            i++;
        }
        return X;
    }
};

TORCH_MODULE(TransformerEncoder);

// the transformer decoder is composed of multiple identical layers
struct DecoderBlockImpl : public torch::nn::Module {
    // The `i`-th block in the decoder
	int64_t i;
	MultiHeadAttention attention1{nullptr}, attention2{nullptr};
	AddNorm addnorm1{nullptr}, addnorm2{nullptr}, addnorm3{nullptr};
	PositionWiseFFN ffn{nullptr};

	DecoderBlockImpl(int64_t key_size, int64_t query_size, int64_t value_size, int64_t num_hiddens,
				 std::vector<int64_t> norm_shape, int64_t ffn_num_input, int64_t ffn_num_hiddens, int64_t num_heads,
                 float dropout, int64_t ith_block) {

        i = ith_block;
        attention1 = MultiHeadAttention( key_size, query_size, value_size, num_hiddens, num_heads, dropout);
        addnorm1 = AddNorm(norm_shape, dropout);
        attention2 = MultiHeadAttention( key_size, query_size, value_size, num_hiddens, num_heads, dropout);
        addnorm2 = AddNorm(norm_shape, dropout);
        ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens);
        addnorm3 = AddNorm(norm_shape, dropout);

        register_module("attention1", attention1);
        register_module("attention2", attention2);
        register_module("addnorm1", addnorm1);
        register_module("addnorm2", addnorm2);
        register_module("addnorm3", addnorm3);
        register_module("ffn", ffn);
	}

	std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>>> forward(
			torch::Tensor X, std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>> state) {
        auto enc_outputs = std::get<0>(state);
    	auto enc_valid_lens = std::get<1>(state);
    	torch::Tensor key_values;

        // During training, all the tokens of any output sequence are processed
        // at the same time, so `state[2][self.i]` is `None` as initialized.
        // When decoding any output sequence token by token during prediction,
        // `state[2][self.i]` contains representations of the decoded output at
        // the `i`-th block up to the current time step
        if( ! (std::get<2>(state)[i]).defined() || (std::get<2>(state)[i]).numel() == 0 )
            key_values = X;
        else
            key_values = torch::cat({std::get<2>(state)[i], X}, 1); // axis=

        torch::Tensor dec_valid_lens;
        std::get<2>(state)[i] = key_values;
        if( this->is_training() ) {
            int64_t batch_size = X.size(0), num_steps = X.size(1);
            // Shape of `dec_valid_lens`: (`batch_size`, `num_steps`), where
            // every row is [1, 2, ..., `num_steps`]
            dec_valid_lens = torch::arange(1, num_steps + 1).to(X.device()).repeat({batch_size, 1});
        }
        // else
        // dec_valid_lens = None

        // Self-attention
        auto X2 = attention1->forward(X, key_values, key_values, dec_valid_lens);
        auto Y =  addnorm1->forward(X, X2);
        // Encoder-decoder attention. Shape of `enc_outputs`:
        // (`batch_size`, `num_steps`, `num_hiddens`)
        auto Y2 = attention2->forward(Y, enc_outputs, enc_outputs, enc_valid_lens);
        auto Z =  addnorm2->forward(Y, Y2);
        return std::make_tuple(addnorm3->forward(Z, ffn->forward(Z)), state);
    }
};

TORCH_MODULE(DecoderBlock);

// construct the entire transformer decoder
struct TransformerDecoderImpl : torch::nn::Module {
	int64_t num_hiddens, num_layers;
	torch::nn::Embedding embedding{nullptr};
	PositionalEncoding pos_encoding{nullptr};
	std::vector<DecoderBlock> blks;
	torch::nn::Linear dense{nullptr};
	std::vector<std::vector<torch::Tensor>> _attention_weights;

	TransformerDecoderImpl(int64_t vocab_size, int64_t key_size, int64_t query_size, int64_t value_size,
							int64_t n_hiddens, std::vector<int64_t> norm_shape, int64_t ffn_num_input,
							int64_t ffn_num_hiddens, int64_t num_heads, int64_t n_layers, float dropout) {

        num_hiddens = n_hiddens;
        num_layers = n_layers;
        embedding = torch::nn::Embedding(vocab_size, num_hiddens);
        pos_encoding = PositionalEncoding(num_hiddens, dropout);

        register_module("embedding", embedding);
        register_module("pos_encoding", pos_encoding);

        for(int64_t i = 0; i < num_layers; i++ ) {
            //self.blks.add_module("block"+str(i),
            auto blk = DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i);
            register_module("DecoderBlock_"+ std::to_string(i), blk);
            blks.push_back(blk);
        }
        dense = torch::nn::Linear(num_hiddens, vocab_size);
        register_module("dense", dense);

        std::vector<torch::Tensor> wgt;
        _attention_weights.push_back(wgt);
        _attention_weights.push_back(wgt);
	}

	std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>> init_state(
			torch::Tensor enc_outputs, torch::Tensor enc_valid_lens) {
		std::vector<torch::Tensor> state;
		for( int64_t i = 0; i < num_layers; i++ ) {
			torch::Tensor st;
			state.push_back(st);
		}
        return std::make_tuple(enc_outputs, enc_valid_lens, state);
    }

	std::pair<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>>> forward(
			torch::Tensor X, std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>> state) {

        X = pos_encoding->forward(embedding->forward(X) * std::sqrt(num_hiddens));

        //_attention_weights = [[None] * len(self.blks) for _ in range (2)]
        _attention_weights[0].clear();
        _attention_weights[1].clear();
        torch::Tensor tmp;
        for(int64_t i = 0; i < blks.size(); i++) {
        	_attention_weights[0].push_back(tmp);
        	_attention_weights[1].push_back(tmp);
        }

        //for i, blk in enumerate(blks) {
        for( int64_t i = 0; i < blks.size(); i++ ) {
        	auto blk = blks[i];
            std::tie( X, state) = blk->forward(X, state);
            // Decoder self-attention weights
            _attention_weights[0][i] = blk->attention1->attention->attention_weights;
            // Encoder-decoder attention weights
            _attention_weights[1][i] = blk->attention2->attention->attention_weights;
        }
        return {dense->forward(X), state};
    }

	std::vector<std::vector<torch::Tensor>> attention_weights() {
        return _attention_weights;
    }
};
TORCH_MODULE(TransformerDecoder);


// EcoderDecoder
struct EncoderDecoderImpl : public torch::nn::Module {
    //The base class for the encoder-decoder architecture.
    //Defined in :numref:`sec_encoder-decoder`
	TransformerDecoder decoder{nullptr};
	TransformerEncoder encoder{nullptr};

	EncoderDecoderImpl(TransformerEncoder encoder, TransformerDecoder decoder) {
        this->encoder = encoder;
        this->decoder = decoder;
        register_module("decoder", decoder);
        register_module("encoder", encoder);
	}

	std::pair<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>>> forward(torch::Tensor enc_X,
													   torch::Tensor dec_X, torch::Tensor val_lens=torch::empty({0})) {
		// input torch::Tensor
		torch::Tensor enc_outputs = this->encoder->forward(enc_X, val_lens);
		// torch::Tensor and torch::Tensor
		std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>> dec_state = this->decoder->init_state(enc_outputs, val_lens);
		// torch::Tensor and std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
        return this->decoder->forward(dec_X, dec_state);
	}

};

TORCH_MODULE(EncoderDecoder);

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
	torch::Tensor enc_outputs = net->encoder->forward(src_ar, empty);

	torch::Tensor prd;
	std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>> dec_state = net->decoder->init_state(enc_outputs, empty);
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

// tensor to matrix
std::vector<float> tensor2vector(torch::Tensor tsr) {
	int nrows = tsr.size(0), ncols = tsr.size(1);

	std::vector<float> z(ncols * nrows);
	for( int j=0; j<nrows; ++j ) {
		for( int i=0; i<ncols; ++i ) {
			z.at(ncols * j + i) = (tsr.index({j, i})).item<float>();
		}
	}
	return z;
}



int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	// The following example shows that [the innermost dimension of a tensor changes] to the number of outputs
	// in the positionwise feed-forward network. Since the same MLP transforms at all the positions, when the
	// inputs at all these positions are the same, their outputs are also identical.

	auto ffn = PositionWiseFFN(4, 4, 8);
	ffn->to(device);
	ffn->eval();

	std::cout << ffn->forward(torch::ones({2, 3, 4}))[0] << "\n";

	// ----------------------------------------------------------
	// Residual Connection and Layer Normalization
	// ----------------------------------------------------------
	std::cout << "Residual Connection and Layer Normalization\n";
	auto ln = torch::nn::LayerNorm(torch::nn::LayerNormOptions({2,2}));
	auto bn = torch::nn::BatchNorm1d(2);
	auto X = torch::tensor({{1, 2}, {2, 3}}, torch::kFloat);

	// Compute mean and variance from `X` in the training mode
	std::cout << "layer norm:" << X.sizes() << "\nbatch norm:" << bn->forward(X) << "\n";

	// The residual connection requires that the two inputs are of the same shape so that
	// [the output tensor also has the same shape after the addition operation

	std::vector<int64_t> normalized_shape({3, 4});
	auto add_norm = AddNorm(normalized_shape, 0.5); // Normalized_shape is input.size()[1:]
	add_norm->to(device);
	add_norm->eval();
	std::cout << "add_norm: " << add_norm->forward(torch::ones({2, 3, 4}), torch::ones({2, 3, 4})).sizes() << std::endl;

	// As we can see, any layer in the transformer encoder does not change the shape of its input.
	X = torch::ones({2, 100, 24});
	auto valid_lens = torch::tensor({3, 2});

	std::vector<int64_t> norm_shape({100, 24});
	auto encoder_blk = EncoderBlock(24, 24, 24, 24, norm_shape, 24, 48, 8, 0.5);
	encoder_blk->to(device);
	encoder_blk->eval();
	std::cout << "encoder_blk: " << encoder_blk->forward(X, valid_lens).sizes() << std::endl;

	// Below we specify hyperparameters to [create a two-layer transformer encoder]. The shape of the transformer
	// encoder output is (batch size, number of time steps, num_hiddens).
	auto encoder = TransformerEncoder(200, 24, 24, 24, 24, norm_shape, 24, 48, 8, 2, 0.5);
	encoder->to(device);
	encoder->eval();
	std::cout << "encoder: " << encoder->forward(torch::ones({2, 100}).to(torch::kLong), valid_lens).sizes() << std::endl;

	// the feature dimension (num_hiddens) of the decoder is the same as that of the encoder
	auto decoder_blk = DecoderBlock(24, 24, 24, 24, norm_shape, 24, 48, 8, 0.5, 0);
	decoder_blk->eval();
	X = torch::ones({2, 100, 24});
	std::vector<torch::Tensor> tmp;
	torch::Tensor expt;
	tmp.push_back(expt);
	auto state = std::make_tuple(encoder_blk->forward(X, valid_lens), valid_lens, tmp);
	std::cout << std::get<0>(decoder_blk->forward(X, state)).sizes() << std::endl;

	// ----------------------------------------
	// Training
	// ----------------------------------------

	int64_t embed_size = 32, num_hiddens = 32, num_layers = 2;
	float dropout = 0.1, lr = 0.005;

	int64_t batch_size = 64, num_steps = 10,  num_epochs = 200;
	int64_t	ffn_num_input = 32, ffn_num_hiddens = 64, num_heads = 4;
	int64_t	key_size = 32, query_size = 32, value_size = 32;

	std::string filename = "./data/fra-eng/fra.txt";

	Vocab src_vocab, tgt_vocab;
	torch::Tensor src_array, src_valid_len, tgt_array, tgt_valid_len;

	std::tie(src_array, src_valid_len, tgt_array, tgt_valid_len, src_vocab, tgt_vocab) = load_data_nmt(filename, num_steps, 600);

	std::vector<int64_t> tnorm_shape({32});

	auto tencoder = TransformerEncoder(
	    src_vocab.length(), key_size, query_size, value_size, num_hiddens,
	    tnorm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
	    num_layers, dropout);

	auto tdecoder = TransformerDecoder(
	    tgt_vocab.length(), key_size, query_size, value_size, num_hiddens,
	    tnorm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
	    num_layers, dropout);

	// define model
	auto net = EncoderDecoder(tencoder, tdecoder);
	net->to(device);

	net->apply(xavier_init_weights);

	//auto optimizer = torch::optim::Adam(net->parameters(), lr);
	torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(lr).betas({0.5, 0.999}));

	auto loss_fn = MaskedSoftmaxCELoss();
	net->train(true);

	std::vector<float> epochs, plsum;
	std::vector<int64_t> wtks;
	torch::Tensor Y_hat;
	std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>> stat;

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

	        std::pair<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>>> out = net->forward(X, dec_input); // X_valid_len

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

	// --------------------------------------------------
	// visualize the transformer attention weights
	// --------------------------------------------------
	using torch::indexing::Slice;
	using torch::indexing::None;
	using torch::indexing::Ellipsis;

	auto enc_attention_weights = torch::cat(net->encoder->attention_weights, 0).reshape({num_layers, num_heads,-1, num_steps});
	std::cout << "enc_attention_weights.sizes(): " << enc_attention_weights.sizes() << "\n";

	plt::figure_size(1200, 750);
	PyObject* mat;
	for(int i = 0; i < enc_attention_weights.size(0); i++) {
		torch::Tensor layerH = enc_attention_weights.index({i, Slice(), Slice(), Slice()});
		layerH = layerH.squeeze();

		for( int c = 0; c < layerH.size(0); c++ ) {
			torch::Tensor tsr = layerH.index({c, Slice(), Slice()});
			tsr = tsr.squeeze();

			std::vector<float> z = tensor2vector( tsr );
			const float* zptr = &(z[0]);
			const int colors = 1;
			int cnt = (i*layerH.size(0) + c) + 1;
			std::cout << "cnt: " << cnt << "\n";
			plt::subplot(enc_attention_weights.size(0), layerH.size(0), cnt);
			plt::title(("Head " + std::to_string(c)).c_str());
			plt::imshow(zptr, tsr.size(0), tsr.size(1), colors, {}, &mat);
			if( c == 0 )
				plt::ylabel("Query positions");
			if( i == 1 )
				plt::xlabel("Key positions");

			plt::colorbar(mat);
		}
	}

	plt::show();
    plt::close();
    Py_DECREF(mat);

	std::cout << "Done!\n";
	return 0;
}




