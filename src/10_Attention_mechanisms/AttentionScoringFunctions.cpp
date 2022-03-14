#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
//#include <math>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "../utils/ch_10_util.h"

// ---------------------------------------------
// Additive Attention]
// ---------------------------------------------
struct AdditiveAttention : public torch::nn::Module {
    //Additive attention
    AdditiveAttention(int64_t key_size, int64_t query_size, int64_t num_hiddens, float dropout) {
        W_k = torch::nn::Linear(torch::nn::LinearOptions(key_size, num_hiddens).bias(false));
        W_q = torch::nn::Linear(torch::nn::LinearOptions(query_size, num_hiddens).bias(false));
        W_v = torch::nn::Linear(torch::nn::LinearOptions(num_hiddens, 1).bias(false));
        dpout = torch::nn::Dropout(dropout);
    }

    torch::Tensor forward(torch::Tensor queries, torch::Tensor keys, torch::Tensor values, torch::Tensor valid_lens) {
        queries = W_q->forward(queries);
		keys = W_k->forward(keys);
        // After dimension expansion, shape of `queries`: (`batch_size`, no. of
        // queries, 1, `num_hiddens`) and shape of `keys`: (`batch_size`, 1,
        // no. of key-value pairs, `num_hiddens`). Sum them up with broadcasting

		std::cout << "queries: " << queries.sizes() << "\n";
		std::cout << "keys: " << keys.sizes() << "\n";
        auto features = queries.unsqueeze(2) + keys.unsqueeze(1);
        features = torch::tanh(features);

        std::cout << "features: " << features.sizes() << "\n";
		// There is only one output of `self.w_v`, so we remove the last
        // one-dimensional entry from the shape. Shape of `scores`:
        // (`batch_size`, no. of queries, no. of key-value pairs)
        auto scores = W_v->forward(features).squeeze(-1); //squeeze() 不加参数的，把所有为1的维度都压缩
        std::cout << "scores: " << scores << "\n";
        std::cout << "valid_lens: " << valid_lens << "\n";

        attention_weights = masked_softmax(scores, valid_lens);
        std::cout << "attention_weights: " << attention_weights.sizes() << "\n";

        // Shape of `values`: (`batch_size`, no. of key-value pairs, value dimension)
        return torch::bmm(dpout->forward(attention_weights), values);
    }
    torch::nn::Linear W_k{nullptr}, W_q{nullptr}, W_v{nullptr};
    torch::nn::Dropout dpout{nullptr};
    torch::Tensor attention_weights;
};


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(1000);

	// To [demonstrate how masked_softmax function works], consider a minibatch of two 2×4 matrix examples,
	// where the valid lengths for these two examples are two and three, respectively. As a result of the masked
	// softmax operation, values beyond the valid lengths are all masked as zero.
	torch::Tensor X = torch::rand({2, 2, 4}).to(device);
	torch::Tensor y = torch::tensor({2, 3}).to(device);
	std::cout << masked_softmax(X, y) << "\n";

/*
	X = torch::tensor({-0.1325, -0.1325, -0.1325, -0.1325, -0.1325, -0.1325, -0.1325, -0.1325, -0.1325, -0.1325,
						0.3546,  0.3546,  0.3546, 0.3546,  0.3546,  0.3546,  0.3546,  0.3546,  0.3546,  0.3546});
	X = X.reshape({2,1,10});
	auto shape = X.sizes();
	std::cout << "shape:\n" << shape << "\n";
	auto Z = X.reshape({-1, shape[shape.size() - 1]});
	std::cout << "Z:\n" << Z << "\n";

	if( y.dim() == 1) {
		y = torch::repeat_interleave(y, shape[shape.size() - 2]);
	} else {
	   y = y.reshape(-1);
	}
	std::cout << "y:\n" << y << "\n";
//	auto W = sequence_mask(Z, y, -1e6);

    int64_t maxlen = Z.size(1);
    auto mask = torch::arange((maxlen),
    torch::TensorOptions().dtype(torch::kFloat32).device(Z.device())).index({None, Slice()}) < y.index({Slice(), None});
    std::cout << "mask:\n" << mask << std::endl;
    auto W = Z.index_put_({torch::ones_like(mask) ^ mask}, -1e6);

	std::cout << "W:\n" << W << "\n";

*/
	// demonstrate the above AdditiveAttention class
	auto queries = torch::normal(0, 1, {2, 1, 20});
	auto keys = torch::ones({2, 10, 2});

	// The two value matrices in the `values` minibatch are identical
	auto values = torch::arange(40, torch::TensorOptions(torch::kFloat32)).reshape({1, 10, 4}).repeat({2, 1, 1});

	auto valid_lens = torch::tensor({2, 6});

	auto attention = AdditiveAttention(2, 20, 8, 0.1);

	attention.eval();
	auto AA = attention.forward(queries, keys, values, valid_lens);
	std::cout << "demonstrate AdditiveAttention class:\n" << AA << std::endl;

	// demonstrate the above DotProductAttention class
	queries = torch::normal(0, 1, {2, 1, 2});
	auto dattention = DotProductAttention(0.5);
	dattention.eval();

    std::cout << "queries:\n" << queries.sizes() << std::endl;
    std::cout << "keys:\n" << keys.sizes() << std::endl;
    std::cout << "values:\n" << values.sizes() << std::endl;
    std::cout << "valid_lens:\n" << valid_lens.sizes() << std::endl;

	auto DA = dattention.forward(queries, keys, values, valid_lens);
	std::cout << "demonstrate DotProductAttention class:\n" << DA << std::endl;

	auto tsr = dattention.attention_weights.reshape({1, 1, 2, 10});
	std::cout << tsr.squeeze() << "\n";

	plot_heatmap(tsr.squeeze(), "keys", "Queries");

	std::cout << "Done!\n";
	return 0;
}




