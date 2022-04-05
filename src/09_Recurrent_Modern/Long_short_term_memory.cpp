#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../utils/ch_8_9_util.h"
#include "../utils.h"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

// ------------------------------------------------
// Implementation from Scratch
// ------------------------------------------------

std::vector<torch::Tensor> get_lstm_params(int64_t vocab_size, int64_t num_hiddens, torch::Device device) {
    int64_t num_inputs = vocab_size;
    int64_t num_outputs = vocab_size;
    std::vector<torch::Tensor> params;
/*
    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))


    W_xi, W_hi, b_i = three()  # Input gate parameters
    W_xf, W_hf, b_f = three()  # Forget gate parameters
    W_xo, W_ho, b_o = three()  # Output gate parameters
    W_xc, W_hc, b_c = three()  # Candidate memory cell parameters
*/
    // Input gate parameters
    torch::Tensor W_xi = torch::randn({num_inputs, num_hiddens}, device)*0.01;
    params.push_back(W_xi);
    torch::Tensor W_hi = torch::randn({num_hiddens, num_hiddens}, device)*0.01;
	params.push_back(W_hi);
	torch::Tensor b_i  = torch::zeros(num_hiddens, device);
	params.push_back(b_i);

	// Forget gate parameters
	torch::Tensor W_xf = torch::randn({num_inputs, num_hiddens}, device)*0.01;
    params.push_back(W_xf);
    torch::Tensor W_hf = torch::randn({num_hiddens, num_hiddens}, device)*0.01;
	params.push_back(W_hf);
	torch::Tensor b_f  = torch::zeros(num_hiddens, device);
	params.push_back(b_f);

	// Output gate parameters
	torch::Tensor W_xo = torch::randn({num_inputs, num_hiddens}, device)*0.01;
    params.push_back(W_xo);
    torch::Tensor W_ho = torch::randn({num_hiddens, num_hiddens}, device)*0.01;
	params.push_back(W_ho);
	torch::Tensor b_o  = torch::zeros(num_hiddens, device);
	params.push_back(b_o);

    // Candidate memory cell parameters
    torch::Tensor W_xc = torch::randn({num_inputs, num_hiddens}, device)*0.01;
    params.push_back(W_xc);
    torch::Tensor W_hc = torch::randn({num_hiddens, num_hiddens}, device)*0.01;
	params.push_back(W_hc);
	torch::Tensor b_c  = torch::zeros(num_hiddens, device);
	params.push_back(b_c);

    // Output layer parameters
	torch::Tensor W_hq = torch::randn({num_hiddens, num_outputs}, device)*0.01; //normal((num_hiddens, num_outputs))
    params.push_back(W_hq);
    torch::Tensor b_q = torch::zeros(num_outputs, device);
    params.push_back(b_q);

    // Attach gradients
    //params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
    //          b_c, W_hq, b_q]

    for(int i = 0; i < params.size(); i++ )
           params[i].requires_grad_(true);

    return params;
}

// ---------------------------------------------------
// Defining the Model
// ---------------------------------------------------
/*
 * In [the initialization function], the hidden state of the LSTM needs to return an additional memory cell with
 * a value of 0 and a shape of (batch size, number of hidden units). Hence we get the following state initialization.
 */
std::tuple<torch::Tensor, torch::Tensor> init_lstm_state(int64_t batch_size, int64_t num_hiddens, torch::Device device) {
    return std::make_tuple(torch::zeros({batch_size, num_hiddens}, device),
            torch::zeros({batch_size, num_hiddens}, device));
}

/*
 * [The actual model] is defined just like what we discussed before: providing three gates and an auxiliary memory cell.
 * Note that only the hidden state is passed to the output layer. The memory cell 𝐂𝑡 does not directly participate in the
 * output computation.
 */
std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> lstm(torch::Tensor inputs,
		std::tuple<torch::Tensor, torch::Tensor> state, std::vector<torch::Tensor>& params) {
    //[W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
    // W_hq, b_q] = params
	torch::Tensor H = std::get<0>(state);
	torch::Tensor C = std::get<1>(state);

    //outputs = []
    std::vector<torch::Tensor> outputs;

    for(int i = 0; i < inputs.size(0); i++ ) {
//std::cout << inputs[i].sizes() << " " << params[0].sizes() << " " << params[1].sizes() << " " << params[2].sizes() << std::endl;
        auto I = torch::sigmoid(torch::mm(inputs[i], params[0]) + torch::mm(H, params[1]) + params[2]);
//std::cout << inputs[i].sizes() << " " << params[3].sizes() << " " << params[4].sizes() << " " << params[5].sizes() << std::endl;
        auto F = torch::sigmoid(torch::mm(inputs[i], params[3]) + torch::mm(H, params[4]) + params[5]);
//std::cout << inputs[i].sizes() << " " << params[6].sizes() << " " << params[7].sizes() << " " << params[8].sizes() << std::endl;
        auto O = torch::sigmoid(torch::mm(inputs[i], params[6]) + torch::mm(H, params[7]) + params[8]);
//std::cout << inputs[i].sizes() << " " << params[9].sizes() << " " << params[10].sizes() << " " << params[11].sizes() << std::endl;
        auto C_tilda = torch::tanh(torch::mm(inputs[i], params[9]) + torch::mm(H, params[10]) + params[11]);
//        std::cout << "***1 " << F.sizes() << " " << C.sizes() << std::endl;
//        std::cout << "***2 " << I.sizes() << " " << C_tilda.sizes() << std::endl;
//        std::cout << "***3 " << (F * C).sizes() << std::endl;
//        std::cout << "***4 " << (I * C_tilda).sizes() << std::endl;
        C = F * C + I * C_tilda;
//        std::cout << C.sizes() << std::endl;
        H = O * torch::tanh(C);
//        std::cout << H.sizes() << " " << params[12].sizes() << " " << params[13].sizes() << std::endl;
        auto Y = torch::mm(H, params[12]) + params[13];
        outputs.push_back(Y);
//        printf("===%i", i);
    }
    return std::make_tuple(torch::cat(outputs, 0), std::make_tuple(H, C));
}

struct RNNModelScratchLstm {
	std::vector<torch::Tensor> params;
	int64_t vocab_size, num_hiddens;

    //A RNN Model implemented from scratch.
	RNNModelScratchLstm(int64_t vocab_sz, int64_t number_hiddens, torch::Device device ) {
        vocab_size = vocab_sz;
        num_hiddens = number_hiddens;
        params = get_lstm_params(vocab_size, num_hiddens, device);
	}

	std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> forward(torch::Tensor X,
																	std::tuple<torch::Tensor, torch::Tensor> state) {
		X = torch::nn::functional::one_hot(X.transpose(0, 1), vocab_size).to(torch::kFloat32);

        return lstm(X, state, params);
	}

	std::tuple<torch::Tensor, torch::Tensor> begin_state(int64_t batch_size, torch::Device device) {
        return init_lstm_state(batch_size, num_hiddens, device);
    }

	std::vector<torch::Tensor> parameters() {
		return params;
	}
};


struct RNNModelLstm : public torch::nn::Module {
    //"""The RNN model."""
	torch::nn::LSTM rnn{nullptr};
	int64_t vocab_size;
	int64_t num_hiddens;
	int64_t num_directions;
	torch::nn::Linear linear{nullptr};

	RNNModelLstm( torch::nn::LSTM rnn_layer, int64_t vocab_size ) {
        rnn = rnn_layer;
        this->vocab_size = vocab_size;
        num_hiddens = rnn.get()->options.hidden_size();
        // If the RNN is bidirectional (to be introduced later),
        // `num_directions` should be 2, else it should be 1.
        if( ! rnn.get()->options.bidirectional() ) {
            num_directions = 1;
            linear = torch::nn::Linear(num_hiddens, vocab_size);
        } else {
            num_directions = 2;
            linear = torch::nn::Linear(num_hiddens * 2, vocab_size);
        }
        register_module("rnn_layer", rnn_layer);
	}

	std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> forward(torch::Tensor inputs,
															std::tuple<torch::Tensor, torch::Tensor> state ) {
        auto X = torch::one_hot(inputs.transpose(0, 1), vocab_size); //(inputs.T.long(), self.vocab_size)
        X = X.to(torch::kFloat32);
        torch::Tensor Y;

        std::tie(Y, state) = rnn->forward(X, state);

        // The fully connected layer will first change the shape of `Y` to
        // (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        // (`num_steps` * `batch_size`, `vocab_size`).
        Y = linear(Y.reshape({-1, Y.size(-1)})); // Y.shape[-1]
		return std::make_tuple(Y, state);
	}

	std::tuple<torch::Tensor, torch::Tensor> begin_state(int64_t batch_size, torch::Device device) {
		// `nn.GRU` takes a tensor as hidden state
		return std::make_tuple(torch::zeros( {num_directions * rnn.get()->options.num_layers(),
		                                batch_size, num_hiddens}, device),
				torch::zeros( {num_directions * rnn.get()->options.num_layers(),
            							batch_size, num_hiddens}, device));
	}
};


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(7);

	int64_t batch_size=32, num_steps = 35;
	int64_t max_tokens = 10000;

	std::vector<std::string> lines = read_time_machine("./data/timemachine.txt");

	//-------------------------------------------------------------
	// split words and extract first token upto max_tokens
	//-------------------------------------------------------------
	std::vector<std::string> ttokens = tokenize(lines, "char", true);
	std::vector<std::string> tokens(&ttokens[0], &ttokens[max_tokens]); // first 10000 tokens

	std::vector<std::pair<std::string, int64_t>> counter = count_corpus( tokens );

	std::vector<std::string> rv(0);
	auto vocab = Vocab(counter, 0.0, rv);

	std::cout << vocab["t"] << "\n";

	// -----------------------------------------------------------
	// Training and Predicting
	// -----------------------------------------------------------
	//=============================================================
	// Let us [check whether the outputs have the correct shapes]
	auto X = torch::arange(10).reshape({2, 5});
	int num_hiddens = 512;

	auto net = RNNModelScratchLstm(vocab.length(), num_hiddens, device);

	std::tuple<torch::Tensor, torch::Tensor> state = net.begin_state(X.size(0), device);

	std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>  rlt = net.forward(X.to(device), state);

	auto Z = std::get<0>(rlt);
	std::tuple<torch::Tensor, torch::Tensor> new_state = std::get<1>(rlt);

	std::cout << "Z: " << Z.sizes() << std::endl;
	//std::cout << Z << std::endl;
	std::cout << "new_state H: " << std::get<0>(new_state).sizes() << std::endl;
	std::cout << "new_state C: " << std::get<1>(new_state).sizes() << std::endl;
	//std::cout << new_state << std::endl;

	//================================================
	// RNNModelScratch
	//================================================
	// Let us [first define the prediction function to generate new characters following the user-provided prefix]
	std::string s = "time traveller ";
	std::vector<char> v(s.begin(), s.end());
	std::vector<std::string> prefix;
	for(int i = 0; i < v.size(); i++ ) {
		std::string tc(1, v[i]);
		prefix.push_back(tc);
	}
	std::string prd = predict_ch9(prefix, 10, net, vocab, device);
	std::cout << prd << std::endl;

	std::vector<int> tokens_ids;
	for( size_t i = 0; i < tokens.size(); i++ )
		tokens_ids.push_back(vocab[tokens[i]]);

	// Training and Predicting
	std::vector<std::pair<torch::Tensor, torch::Tensor>> train_iter = seq_data_iter_random(tokens_ids, batch_size, num_steps);

	int64_t num_epochs = 400;
	float lr = 1.0;
	bool use_random_iter = false;

	auto nett = RNNModelScratchLstm(vocab.length(), num_hiddens, device);

	std::pair<std::vector<double>, std::vector<double>> trlt = train_ch9(nett, train_iter, vocab, device, lr,
			num_epochs, use_random_iter);

	//================================================
	// LSTM concise
	//================================================
	std::vector<std::pair<torch::Tensor, torch::Tensor>> ctrain_iter = seq_data_iter_random(tokens_ids, batch_size, num_steps);

	auto lstm_layer = torch::nn::LSTM(vocab.length(), num_hiddens);
	auto cnet = RNNModelLstm(lstm_layer, vocab.length());
	cnet.to(device);

	std::pair<std::vector<double>, std::vector<double>> ctrlt = train_ch9( cnet, ctrain_iter, vocab, device, lr,
			num_epochs, use_random_iter);

	plt::figure_size(1400, 500);
//	plt::subplot(1, 2, 1);
	plt::subplot2grid(1, 2, 0, 0, 1, 1);
	plt::named_plot("train", trlt.first, trlt.second, "b");
	plt::xlabel("epoch");
	plt::ylabel("perplexity");
	plt::title("RNNModelScratch LSTM");
	plt::legend();

//	plt::subplot(1, 2, 2);
	plt::subplot2grid(1, 2, 0, 1, 1, 1);
	plt::named_plot("train", ctrlt.first, ctrlt.second, "b");
	plt::xlabel("epoch");
	plt::ylabel("perplexity");
	plt::title("RNNModel concise LSTM");
	plt::legend();
	plt::show();
	plt::close();

	std::cout << "Done!\n";
	return 0;
}





