#include <torch/utils.h>
#include "../utils/ch_8_9_util.h"
#include "../utils/ch_15_util.h"

class PositionWiseFFN : public torch::nn::Module {
    //The positionwise feed-forward network.

	PositionWiseFFN(ffn_num_hiddens, ffn_num_outputs) {
        dense1 = nn.LazyLinear(ffn_num_hiddens)
        relu = torch::nn::ReLU();
        dense2 = nn.LazyLinear(ffn_num_outputs);
	}

	torch::Tensor forward(torch::Tensor X) {
        return dense2(relu(dense1(X)));
	}
};

class AddNorm : torch::nn::Module {
    //The residual connection followed by layer normalization.

	AddNorm(int64_t norm_shape, double dropout) {
        dropout = torch::nn::Dropout(torch::nn::DropoutOptions(dropout));
        ln = torch::nn::LayerNorm(torch::nn::LayerNormOptions({norm_shape, norm_shape}));
	}

	torch::Tensor forward(torch::Tensor X, torch::Tensor Y) {
        return ln->forward(dropout->forward(Y) + X);
    }
private:
	torch::nn::Dropout dropout{nullptr};
	torch::nn::LayerNorm ln{nullptr};
};


class TransformerEncoderBlock : public torch::nn::Module {
    //The Transformer encoder block.

	TransformerEncoderBlock(int64_t num_hiddens, int64_t ffn_num_hiddens, int64_t num_heads,
			double dropout, bool use_bias=false) {

        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias);
        self.addnorm1 = AddNorm(num_hiddens, dropout);
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens);
        self.addnorm2 = AddNorm(num_hiddens, dropout);
	}

	torch::Tensor forward(torch::Tensor X, valid_lens) {
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens));
        return self.addnorm2(Y, self.ffn(Y));
    }
};

class BERTEncoder : public torch::nn::Module {
    //BERT encoder.
	BERTEncoder(int64_t vocab_size, int64_t num_hiddens, int64_t  ffn_num_hiddens,
			int64_t  num_heads, int64_t  num_blks, double dropout, int64_t  max_len=1000) {

        token_embedding = torch::nn::Embedding(vocab_size, num_hiddens);
        segment_embedding = torch::nn::Embedding(2, num_hiddens);
        //blks = torch::nn::Sequential()
        for(int64_t  i = 0; i < num_blks; i++ ) {
            //self.blks.add_module(f"{i}", d2l.TransformerEncoderBlock(
            //    num_hiddens, ffn_num_hiddens, num_heads, dropout, True))
        }
        // In BERT, positional embeddings are learnable, thus we create a
        // parameter of positional embeddings that are long enough
        pos_embedding = torch::nn::Parameter(torch::randn({1, max_len,
                                                      num_hiddens}));
	}

    torch::Tensor forward(tokens, segments, valid_lens) {
        // Shape of `X` remains unchanged in the following code snippet:
        // (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X;
    }
private:
    torch::nn::Embedding oken_embedding{nullptr}, segment_embedding{nullptr};
    torch::nn::Sequential blks;
};


class BERTModel : public torch::nn::Module {
    //The BERT model.

	BERTModel(int64_t vocab_size, int64_t num_hiddens, int64_t  ffn_num_hiddens,
			int64_t  num_heads, int64_t  num_blks, double dropout, int64_t  max_len=1000) {

        encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_blks, dropout,
                                   max_len);
        hidden = torch::nn::Sequential(nn.LazyLinear(num_hiddens),
                                    nn.Tanh())
        mlm = MaskLM(vocab_size, num_hiddens)
        nsp = NextSentencePred()
	}

	std::tuple<torch::Tensor> forward( tokens, segments, valid_lens=None, pred_positions=None) {
        encoded_X = encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        // The hidden layer of the MLP classifier for next sentence prediction.
        // 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return std::make_tuple( encoded_X, mlm_Y_hat, nsp_Y_hat);
	}
private:
	BERTEncoder encoder;
};


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	torch::Device device(torch::kCPU);
	//auto cuda_available = torch::cuda::is_available();
	//torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	//std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	Vocab vocab;

	std::string file_name = "./data/bert.small.torch/vocab.json";

	std::string line;
	std::ifstream fL;
	std::vector<std::string> tks;

	fL.open(file_name.c_str());
	std::cout << fL.is_open() << '\n';

	std::string delimiter = ", ";
	const char * pStr1 = NULL;
	const char * pStr2 = NULL;
	pStr1 = "[";
	pStr2 = "]";

	if( fL.is_open() ) {
		std::getline(fL, line);
		size_t rpos = line.find(pStr1);

		if(rpos >= 0 )
			line.replace(rpos, 1, "");

		rpos = line.find(pStr2);
		if(rpos >= 0 )
			line.replace(rpos, 1, "");

		size_t pos = 0;

		while ((pos = line.find(delimiter)) != std::string::npos) {
			std::string s = line.substr(0, pos);
			tks.push_back(s);
			line.erase(0, pos + delimiter.length());
			std::cout << s << '\n';
		}
		std::string s = line.substr(0, pos);
		tks.push_back(s);
		std::cout << s << '\n';
	}

	fL.clear();
	fL.close();

	std::vector<std::pair<std::string, int64_t>> counter = count_corpus( tks );
	std::vector<std::string> rv(0);
	vocab = Vocab(counter, 0.0, rv);

	int num_hiddens=256, ffn_num_hiddens=512, num_heads=4, num_blks=2, dropout=0.1, max_len=512;


	std::cout << "Done!\n";
}


