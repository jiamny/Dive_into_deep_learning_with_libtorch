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

// ------------------------------------
// Initializing Model Parameters
// ------------------------------------
/*
 * The next step is to initialize the model parameters. We draw the weights from a Gaussian distribution
 * with standard deviation to be 0.01 and set the bias to 0. The hyperparameter num_hiddens defines the number of
 * hidden units. We instantiate all weights and biases relating to the update gate, the reset gate, the candidate
 * hidden state, and the output layer.
 */

std::vector<torch::Tensor> get_params(int64_t vocab_size, int64_t num_hiddens, torch::Device device) {
    int64_t num_inputs = vocab_size;
    int64_t num_outputs = vocab_size;
    std::vector<torch::Tensor> params;

/*
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))
*/

    // Update gate parameters
    torch::Tensor W_xz = torch::randn({num_inputs, num_hiddens}, device)*0.01;
    params.push_back(W_xz);
    torch::Tensor W_hz = torch::randn({num_hiddens, num_hiddens}, device)*0.01;
    params.push_back(W_hz);
    torch::Tensor b_z =  torch::zeros(num_hiddens, device)*0.01;
    params.push_back(b_z);

    // Reset gate parameters
    torch::Tensor W_xr = torch::randn({num_inputs, num_hiddens}, device)*0.01;
    params.push_back(W_xr);
	torch::Tensor W_hr = torch::randn({num_hiddens, num_hiddens}, device)*0.01;
	params.push_back(W_hr);
	torch::Tensor b_r = torch::zeros(num_hiddens, device)*0.01;
	params.push_back(b_r);

	// Candidate hidden state parameters
	torch::Tensor W_xh = torch::randn({num_inputs, num_hiddens}, device)*0.01;
	params.push_back(W_xh);
	torch::Tensor W_hh = torch::randn({num_hiddens, num_hiddens}, device)*0.01;
	params.push_back(W_hh);
	torch::Tensor b_h = torch::zeros(num_hiddens, device)*0.01;
	params.push_back(b_h);

    // Output layer parameters
	torch::Tensor W_hq = torch::randn({num_hiddens, num_outputs}, device)*0.01;//normal((num_hiddens, num_outputs))
	params.push_back(W_hq);
	torch::Tensor b_q = torch::zeros(num_outputs, device);
	params.push_back(b_q);
    // Attach gradients
    //params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for(int i = 0; i < params.size(); i++ )
        params[i].requires_grad_(true);

    return params;
}

// -------------------------------------------
// Defining the Model
// -------------------------------------------
/*
 * Now we will define [the hidden state initialization function] init_gru_state. Just like the init_rnn_state
 * function defined in :numref:sec_rnn_scratch, this function returns a tensor with a shape (batch size,
 * number of hidden units) whose values are all zeros.
 *
 */
torch::Tensor init_gru_state(int64_t batch_size, int64_t num_hiddens, torch::Device device) {
    return torch::zeros({batch_size, num_hiddens}, device);
}

// Now we are ready to [define the GRU model]. Its structure is the same as that of the basic RNN cell,
// except that the update equations are more complex.
std::tuple<torch::Tensor, torch::Tensor> gru(torch::Tensor inputs,
		torch::Tensor state, std::vector<torch::Tensor>& params) {
    //W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    //H, = state
    //outputs = []

	std::vector<torch::Tensor> outputs;
    //for X in inputs:
	for(int i = 0; i < inputs.size(0); i++ ) {
        auto Z = torch::sigmoid( torch::mm(inputs[i], params[0]) + torch::mm(state, params[1]) + params[2]); // Matrix Multiplication OP @ in Torch matrix multiplication using mm()
        auto R = torch::sigmoid( torch::mm(inputs[i], params[3]) + torch::mm(state, params[4]) + params[5]);
        auto H_tilda = torch::tanh(torch::mm(inputs[i], params[6]) + torch::mm((R * state), params[7]) + params[8]);
        state = Z * state + (1 - Z) * H_tilda;
        auto Y = torch::mm(state, params[9]) + params[10];
        outputs.push_back(Y);
	}
    return std::make_tuple(torch::cat(outputs, 0), state); //(H,)
}

struct RNNModelScratchGru {
	std::vector<torch::Tensor> params;
	int vocab_size, num_hiddens;

    //A RNN Model implemented from scratch.
	RNNModelScratchGru(int vocab_sz, int number_hiddens, torch::Device device ) {
        vocab_size = vocab_sz;
        num_hiddens = number_hiddens;
        params = get_params(vocab_size, num_hiddens, device);
	}

	std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor X, torch::Tensor state) {

		X = torch::nn::functional::one_hot(X.transpose(0, 1), vocab_size).to(torch::kFloat32);

        return gru(X, state, params);
	}

	torch::Tensor  begin_state(int batch_size, torch::Device device) {
        return init_gru_state(batch_size, num_hiddens, device);
    }

	std::vector<torch::Tensor> parameters() {
		return params;
	}
};


struct RNNModelGru : public torch::nn::Module {
    //"""The RNN model."""
	torch::nn::RNN rnn{nullptr};
	int64_t vocab_size;
	int64_t num_hiddens;
	int64_t num_directions;
	torch::nn::Linear linear{nullptr};

	RNNModelGru( torch::nn::RNN rnn_layer, int64_t vocab_size) {

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

	std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor inputs, torch::Tensor state ) {
        auto X = torch::one_hot(inputs.transpose(0, 1), vocab_size); //(inputs.T.long(), self.vocab_size)
        X = X.to(torch::kFloat32);
        torch::Tensor Y;
        //torch::Tensor H = std::get<0>(state);

        std::tie(Y, state) = rnn->forward(X, state);

        // The fully connected layer will first change the shape of `Y` to
        // (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        // (`num_steps` * `batch_size`, `vocab_size`).
        Y = linear(Y.reshape({-1, Y.size(-1)})); // Y.shape[-1]
		return std::make_tuple(Y, state);
	}

	torch::Tensor begin_state(int64_t batch_size, torch::Device device) {
		// `nn.GRU` takes a tensor as hidden state
		return torch::zeros( {num_directions * rnn.get()->options.num_layers(),
		                                batch_size, num_hiddens}, device);
	}
};

//================================================
// Prediction
//================================================
template<typename T>
std::string predict_ch9_gru(std::vector<std::string> prefix, int64_t num_preds, T net, Vocab vocab,
							torch::Device device) {
    //"""Generate new characters following the `prefix`."""
	torch::Tensor state = net.begin_state(1, device);

    std::vector<int64_t> outputs;
    outputs.push_back(vocab[prefix[0]]);

    //outputs = [vocab[prefix[0]]]
    //get_input = lambda: d2l.reshape(d2l.tensor([outputs[-1]], device=device),
    //                                (1, 1))

    for( int i = 1; i < prefix.size(); i ++ ) { //# Warm-up period
    	std::string y = prefix[i];
    	torch::Tensor xx;
    	torch::Tensor H;
    	// get_inpt
    	torch::Tensor X = torch::tensor({{outputs[i-1]}}, device).reshape({1,1});

    	std::tie(xx, state) = net.forward(X, state);

    	outputs.push_back(vocab[prefix[i]]);
    }
    //for y in prefix[1:]:  			//# Warm-up period
    //    _, state = net(get_input(), state)
    //    outputs.append(vocab[y])

    for( int i = 0; i < num_preds; i ++ ) {
    	torch::Tensor y;
    	torch::Tensor H;

    	int j = outputs.size();
    	torch::Tensor X = torch::tensor({{outputs[j-1]}}, device).reshape({1,1});

    	std::tie(y, state) = net.forward(X, state);
    	outputs.push_back(static_cast<int>(y.argmax(1, 0).reshape({1}).item<int>()));
    }
    //for _ in range(num_preds):  	//# Predict `num_preds` steps
    //    y, state = net(get_input(), state)
    //    outputs.append(int(y.argmax(dim=1).reshape(1)))
    std::string pred("");
    for( auto& p : outputs )
    	pred += vocab.idx_to_token[p];

    return pred; // ''.join([vocab.idx_to_token[i] for i in outputs]);
}


template<typename T>
std::pair<double, double> train_epoch_ch9_gru(T& net, std::vector<std::pair<torch::Tensor, torch::Tensor>> train_iter,
		torch::nn::CrossEntropyLoss loss, torch::optim::Optimizer& updater, torch::Device device,
		float lr, bool use_random_iter ) {
	//***********************************************
	//two ways of setting the precision
	std::streamsize ss = std::cout.precision();
	std::cout.precision(15);
	// another way std::setprecision(N)

	double ppx = 0.0;
	int64_t tot_tk = 0;

	precise_timer timer;
	torch::Tensor state = torch::empty({0});

	for( int i = 0; i < train_iter.size(); i++ ) {
	    auto X = train_iter[i].first;
	    auto Y = train_iter[i].second;

	    if( state.numel() == 0 || use_random_iter ) {
	    	state = net.begin_state(X.size(0), device);
	    } else {
	    	state.detach_();
	    }

	    torch::Tensor y = Y.transpose(0, 1).reshape({-1});
	    X.to(device),
	    y.to(device);
	    torch::Tensor y_hat;

	    std::tie(y_hat, state) = net.forward(X, state);

	    auto l = loss(y_hat, y.to(torch::kLong)).mean();

	    updater.zero_grad();
	    l.backward();

	    //"""Clip the gradient."""
	    torch::nn::utils::clip_grad_norm_(net.parameters(), 0.5);

	    updater.step();

	    ppx += (l.data().item<float>() * y.numel());
	    tot_tk += y.numel();
	}
	unsigned int dul = timer.stop<unsigned int, std::chrono::microseconds>();
	auto t = (dul/1000000.0);

	//****************************************
	//change back the precision
	std::cout.precision(ss);
	// another way std::setprecision(ss)

	return { std::exp(ppx*1.0 / tot_tk), (tot_tk * 1.0 / t) };
}

template<typename T>
std::pair<std::vector<double>, std::vector<double>> train_ch9_gru(T& net, std::vector<std::pair<torch::Tensor, torch::Tensor>> train_iter,
		Vocab vocab, torch::Device device, float lr, int64_t num_epochs, bool use_random_iter) {

	std::string s = "time traveller", s2 = "traveller";
	std::vector<char> v(s.begin(), s.end()), v2(s2.begin(), s2.end());
	std::vector<std::string> prefix, prefix2;
	for(int i = 0; i < v.size(); i++ ) {
	    std::string tc(1, v[i]);
	    prefix.push_back(tc);
	}

	for(int i = 0; i < v2.size(); i++ ) {
		std::string tc(1, v[i]);
		prefix2.push_back(tc);
	}

    //"""Train a model (defined in Chapter 8)."""
	auto loss = torch::nn::CrossEntropyLoss();
	auto updater = torch::optim::SGD(net.parameters(), lr);

    std::vector<double> epochs, ppl;
    // Train and predict
    double eppl, speed;
    for(int64_t  epoch = 0; epoch < num_epochs; epoch++ ) {

        auto rlt = train_epoch_ch9_gru(net, train_iter, loss, updater, device, lr, use_random_iter);
        eppl = rlt.first;
        speed = rlt.second;
        if( (epoch + 1) % 10 == 0 ) {
        	std::cout << predict_ch9_gru(prefix, 50, net, vocab, device) << std::endl;

            ppl.push_back(eppl);
            epochs.push_back((epoch+1)*1.0);
        }
    }

    printf("perplexity: %.1f, speed: %.1f tokens/sec on %s\n", eppl, speed, device.str().c_str());
    std::cout << predict_ch9_gru(prefix, 50, net, vocab, device) << std::endl;
    std::cout << predict_ch9_gru(prefix2, 50, net, vocab, device) << std::endl;

    return {epochs, ppl};
}


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
	auto net = RNNModelScratchGru(vocab.length(), num_hiddens, device);

	torch::Tensor state = net.begin_state(X.size(0), device);
	std::tuple<torch::Tensor, torch::Tensor>  rlt = net.forward(X.to(device), state);
	auto Z = std::get<0>(rlt);
	torch::Tensor new_state = std::get<1>(rlt);

	std::cout << "Z: " << Z.sizes() << std::endl;
	//std::cout << Z << std::endl;
	std::cout << "new_state H: " << new_state.sizes() << std::endl;
	//std::cout << new_state << std::endl;

	//================================================
	// RNNModelScratch
	//================================================
	// Let us [first define the prediction function to generate new characters following the user-provided prefix]
	bool is_lstm = false;
	std::string s = "time traveller ";
	std::vector<char> v(s.begin(), s.end());
	std::vector<std::string> prefix;
	for(int i = 0; i < v.size(); i++ ) {
		std::string tc(1, v[i]);
		prefix.push_back(tc);
	}
	std::string prd = predict_ch9_gru(prefix, 10, net, vocab, device);
	std::cout << prd << std::endl;

	std::vector<int> tokens_ids;
	for( size_t i = 0; i < tokens.size(); i++ )
		tokens_ids.push_back(vocab[tokens[i]]);

	// Training and Predicting
	std::vector<std::pair<torch::Tensor, torch::Tensor>> train_iter = seq_data_iter_random(tokens_ids, batch_size, num_steps);

	int64_t num_epochs = 200;
	float lr = 1.0;
	bool use_random_iter = false;
	auto nett = RNNModelScratchGru(vocab.length(), num_hiddens, device);

	std::pair<std::vector<double>, std::vector<double>> trlt = train_ch9_gru(nett, train_iter, vocab, device, lr,
			num_epochs, use_random_iter);

	//================================================
	// RNNModel concise
	//================================================
	std::vector<std::pair<torch::Tensor, torch::Tensor>> ctrain_iter = seq_data_iter_random(tokens_ids, batch_size, num_steps);
	auto rnn_layer = torch::nn::RNN(vocab.length(), num_hiddens);
	auto cnet = RNNModelGru( rnn_layer, vocab.length() );
	cnet.to(device);
	num_epochs = 200;
	lr = 1.0;
	use_random_iter = false;

	std::pair<std::vector<double>, std::vector<double>> ctrlt = train_ch9_gru( cnet, ctrain_iter, vocab, device, lr,
			num_epochs, use_random_iter);

	plt::figure_size(1400, 500);
	plt::subplot(int(1),int(2),int(1));
	plt::named_plot("train", trlt.first, trlt.second, "b");
	plt::xlabel("epoch");
	plt::ylabel("perplexity");
	plt::title("RNNModelScratch GRU");
	plt::legend();

	plt::subplot(int(1),int(2),int(2));
	plt::named_plot("train", ctrlt.first, ctrlt.second, "b");
	plt::xlabel("epoch");
	plt::ylabel("perplexity");
	plt::title("RNNModel concise GRU");
	plt::legend();
	plt::show();

	std::cout << "Done!\n";
	return 0;
}



