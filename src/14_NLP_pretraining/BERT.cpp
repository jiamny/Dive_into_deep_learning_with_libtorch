#include <unistd.h>
#include <iomanip>
#include <cmath>
#include <torch/utils.h>
#include <torch/torch.h>
#include "../utils/ch_14_util.h"


/*
torch::Tensor sequence_mask(torch::Tensor X, torch::Tensor valid_len, double value=0.) {
    //在序列中屏蔽不相关的项
    int maxlen = X.sizes()[1];
    auto mask = valid_len.index({Slice(),Slice(None)}).less_equal(torch::arange({maxlen}, torch::kFloat32).index({Slice(None), Slice()}));
    		//[None, :] < valid_len[:, None]
    //X[~mask] = value;
    X.masked_fill(mask, value);
    return X;
}

torch::Tensor  masked_softmax(torch::Tensor X, torch::Tensor valid_lens=torch::empty(0)) {
    //通过在最后一个轴上掩蔽元素来执行 softmax 操作
    if(valid_lens.numel() == 0) {
        return F::softmax(X, F::SoftmaxFuncOptions(-1));
    } else {
        auto shape = X.sizes();
        if(valid_lens.dim() == 1) {
           // valid_lens = d2l.repeat(valid_lens, shape[1]);
        } else {
            valid_lens = valid_lens.reshape(-1);
        }
        X = sequence_mask(X.reshape({-1, shape[-1]}), valid_lens,-1e6);

        return F::softmax(X.reshape(shape), F::SoftmaxFuncOptions(-1));
        //return nn.Softmax(-1)(X.reshape(shape))
    }
}
*/


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	//torch::Device device(torch::kCPU);
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Running on GPU." : "Running on CPU.") << '\n';

	torch::manual_seed(123);
/*
	const std::string data_dir = "./data/wikitext-2";
	std::vector<std::vector<std::vector<std::string>>> data = _read_wiki(data_dir, 1000);

	std::cout << data.size() << '\n';
	std::vector<std::vector<std::string>> lines = data[0];
	for(auto& s : lines)
		printVector(s);
*/
	int64_t vocab_size = 10000, num_hiddens = 768, ffn_num_hiddens = 1024, num_heads = 4;
	std::vector<int64_t> norm_shape = {768};
	int64_t ffn_num_input = 768, num_layers = 2;
	double dropout =  0.2;
	auto encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
	                      ffn_num_hiddens, num_heads, num_layers, dropout);


	torch::Tensor tokens = torch::randint(0, vocab_size, {2, 8});
	torch::Tensor segments = torch::tensor({{0, 0, 0, 0, 1, 1, 1, 1}, {0, 0, 0, 1, 1, 1, 1, 1}});
	auto encoded_X = encoder(tokens, segments, torch::empty(0));

	std::cout << encoded_X.sizes() << '\n';

	// 预训练任务
	// 掩蔽语言模型（Masked Language Modeling）
	auto mlm = MaskLM(vocab_size, num_hiddens);
	torch::Tensor mlm_positions = torch::tensor({{1, 5, 2}, {6, 1, 5}});
	auto mlm_Y_hat = mlm->forward(encoded_X, mlm_positions);
	std::cout << "mlm_Y_hat.shape: " << mlm_Y_hat.sizes() << '\n';


	torch::Tensor mlm_Y = torch::tensor({{7, 8, 9}, {10, 20, 30}}, torch::kLong);
	auto loss = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().reduction(torch::kNone));
	torch::Tensor mlm_l = loss(mlm_Y_hat.reshape({-1, vocab_size}), mlm_Y.reshape(-1));
	std::cout << "mlm_l.shape: " << mlm_l.sizes() << '\n';

	// 下一句预测（Next Sentence Prediction）
	encoded_X = torch::flatten(encoded_X, 1); // start dim = 1
	std::cout << "encoded_X: " << encoded_X.sizes() << '\n';
	// NSP的输入形状:(batchsize，num_hiddens)
	std::cout << "encoded_X.size(-1): " << encoded_X.size(-1) << '\n';

	auto nsp = NextSentencePred(encoded_X.size(-1));
	torch::Tensor nsp_Y_hat = nsp->forward(encoded_X);
	std::cout << "nsp_Y_hat.shape: " << nsp_Y_hat.sizes() << '\n';

	torch::Tensor nsp_y = torch::tensor({0, 1}, torch::kLong);
	auto nsp_l = loss(nsp_Y_hat, nsp_y);
	std::cout << "nsp_l.shape: " << nsp_l.sizes() << '\n';

	std::cout << "Done!\n";
	return 0;
}



