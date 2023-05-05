#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "../utils/ch_14_util.h"
#include "../TempHelpFunctions.hpp"

/*
class TransformerEncoderBlock : public torch::nn::Module {
    //The Transformer encoder block.

	TransformerEncoderBlock(int64_t num_hiddens, int64_t ffn_num_hiddens, int64_t num_heads,
			double dropout, bool use_bias=false) {

        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias);
        addnorm1 = AddNorm(num_hiddens, dropout);
        ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens);
        addnorm2 = AddNorm(num_hiddens, dropout);
	}

	torch::Tensor forward(torch::Tensor X, valid_lens) {
        Y = addnorm1(X, self.attention(X, X, X, valid_lens));
        return addnorm2(Y, self.ffn(Y));
    }
};

class BERTEncoder : public torch::nn::Module {
    //"""BERT encoder."""
	BERTEncoder(int64_t vocab_size, int64_t num_hiddens, int64_t  ffn_num_hiddens,
				int64_t  num_heads, int64_t  num_blks, double dropout, int64_t  max_len=1000) {

	        token_embedding = torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, num_hiddens));
	        segment_embedding = torch::nn::Embedding(torch::nn::EmbeddingOptions(2, num_hiddens));
	        //blks = torch::nn::Sequential()
	        for(int64_t  i = 0; i < num_blks; i++ ) {
	            //self.blks.add_module(f"{i}", d2l.TransformerEncoderBlock(
	            //    num_hiddens, ffn_num_hiddens, num_heads, dropout, True))
	        }
	        // In BERT, positional embeddings are learnable, thus we create a
	        // parameter of positional embeddings that are long enough
	        pos_embedding =torch::randn({1, max_len, num_hiddens});
		}

	    torch::Tensor forward(torch::Tensor tokens, torch::Tensor segments, int64_t valid_lens) {
	        // Shape of `X` remains unchanged in the following code snippet:
	        // (batch size, max sequence length, `num_hiddens`)
	    	torch::Tensor X = token_embedding->forward(tokens) + segment_embedding->forward(segments);
	        X = X + pos_embedding[:, :X.shape[1], :]
	        for blk in self.blks:
	            X = blk(X, valid_lens)
	        return X;
	    }
	private:
	    torch::nn::Embedding token_embedding{nullptr}, segment_embedding{nullptr};
	    torch::nn::Sequential blks;
	    torch::Tensor pos_embedding;
};
*/

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	torch::Device device(torch::kCPU);
	//auto cuda_available = torch::cuda::is_available();
	//torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	//std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	const std::string data_dir = "./data/wikitext-2";
	std::vector<std::vector<std::vector<std::string>>> data = _read_wiki(data_dir, 1000);

	std::cout << data.size() << '\n';
	std::vector<std::vector<std::string>> lines = data[0];
	for(auto& s : lines)
		printVector(s);




	std::cout << "Done!\n";
}



