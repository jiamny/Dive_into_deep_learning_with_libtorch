#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../utils/ch_8_9_util.h"
#include "../utils.h"

#include <matplot/matplot.h>
using namespace matplot;

torch::Tensor normal(int d1, int d2, torch::Device device) {
	return torch::randn({d1, d2}, device) * 0.01;
}

//===============================================
// Initializing the Model Parameters
//===============================================
std::vector<torch::Tensor> get_params(int vocab_size, int num_hiddens, torch::Device device) {
    int num_inputs = vocab_size;
    int num_outputs = vocab_size;

    std::vector<torch::Tensor> params;
    // Hidden layer parameters
    torch::Tensor W_xh = normal(num_inputs, num_hiddens, device).requires_grad_(true);
    params.push_back(W_xh);
    torch::Tensor W_hh = normal(num_hiddens, num_hiddens, device).requires_grad_(true);
    params.push_back(W_hh);
	torch::Tensor b_h = torch::zeros(num_hiddens, device).requires_grad_(true);
	params.push_back(b_h);
    // Output layer parameters
	torch::Tensor W_hq = normal(num_hiddens, num_outputs, device).requires_grad_(true);
	params.push_back(W_hq);
	torch::Tensor b_q = torch::zeros(num_outputs, device).requires_grad_(true);
	params.push_back(b_q);

    return params;
}

//==================================================
// RNN Model
//==================================================

/*
 * To define an RNN model, we first need [an init_rnn_state function to return the hidden state at initialization.]
 * It returns a tensor filled with 0 and with a shape of (batch size, number of hidden units).
 */
torch::Tensor init_rnn_state(int batch_size, int num_hiddens, torch::Device device) {
    return torch::zeros({batch_size, num_hiddens}, device);
}

// The following rnn function defines how to compute the hidden state and output at a time step
std::tuple<torch::Tensor, torch::Tensor> rnn(torch::Tensor inputs, torch::Tensor& state, std::vector<torch::Tensor>& params) {
    // Here `inputs` shape: (`num_steps`, `batch_size`, `vocab_size`)
	/*
	torch::Tensor W_xh = params[0];
	torch::Tensor W_hh = params[1];
	torch::Tensor b_h  = params[2];
	torch::Tensor W_hq = params[3];
	torch::Tensor b_q  = params[4];
	*/
    //auto H = state;
	std::vector<torch::Tensor> outputs;

	for(int i = 0; i < inputs.size(0); i++ ) {
	    state = torch::tanh(torch::mm(inputs[i], params[0]) + torch::mm(state, params[1]) + params[2]);
	    auto Y = torch::mm(state, params[3]) + params[4];
	    outputs.push_back(Y);
	}

    return std::make_tuple(torch::cat(outputs, 0), state);
}

struct RNNModelScratch {
	std::vector<torch::Tensor> params;
	int vocab_size, num_hiddens;

    //A RNN Model implemented from scratch.
	RNNModelScratch(int vocab_sz, int number_hiddens, torch::Device device ) {
        vocab_size = vocab_sz;
        num_hiddens = number_hiddens;
        params = get_params(vocab_size, num_hiddens, device);
	}

	std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor X, torch::Tensor state) {

		X = torch::nn::functional::one_hot(X.transpose(0, 1), vocab_size).to(torch::kFloat32);

        return rnn(X, state, params);
	}

	torch::Tensor begin_state(int batch_size, torch::Device device) {
        return init_rnn_state(batch_size, num_hiddens, device);
    }
};

//================================================
// Prediction
//================================================
template<typename T>
std::string predict_ch8(std::vector<std::string> prefix, int64_t num_preds, T net, Vocab vocab, torch::Device device) {
    //"""Generate new characters following the `prefix`."""
    auto state = net.begin_state(1, device);

    std::vector<int64_t> outputs;
    outputs.push_back(vocab[prefix[0]]);

    //outputs = [vocab[prefix[0]]]
    //get_input = lambda: d2l.reshape(d2l.tensor([outputs[-1]], device=device),
    //                                (1, 1))

    for( int i = 1; i < prefix.size(); i ++ ) { //# Warm-up period
    	std::string y = prefix[i];
    	torch::Tensor xx;
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
std::pair<double, double> train_epoch_ch8(T& net, std::vector<std::pair<torch::Tensor, torch::Tensor>> train_iter,
		torch::nn::CrossEntropyLoss loss, torch::optim::Optimizer& updater, torch::Device device, float lr, bool use_random_iter) {
	//***********************************************
	//two ways of setting the precision
	std::streamsize ss = std::cout.precision();
	std::cout.precision(15);
	// another way std::setprecision(N)

	double ppx = 0.0;
	int64_t tot_tk = 0;

	precise_timer timer;
	torch::Tensor state = torch::empty({0});;

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
	    torch::nn::utils::clip_grad_norm_(net.params, 0.5);

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
std::pair<std::vector<double>, std::vector<double>> train_ch8(T& net, std::vector<std::pair<torch::Tensor, torch::Tensor>> train_iter,
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

	torch::optim::SGD updater = torch::optim::SGD(net.params, lr);

    std::vector<double> epochs, ppl;
    // Train and predict
    double eppl, speed;
    for(int64_t  epoch = 0; epoch < num_epochs; epoch++ ) {

        auto rlt = train_epoch_ch8(net, train_iter, loss, updater, device, lr, use_random_iter);
        eppl = rlt.first;
        speed = rlt.second;
        if( (epoch + 1) % 10 == 0 ) {
        	std::cout << predict_ch8(prefix, 50, net, vocab, device) << std::endl;

            ppl.push_back(eppl);
            epochs.push_back((epoch+1)*1.0);
        }
    }

    printf("perplexity: %.1f, speed: %.1f tokens/sec on %s\n", eppl, speed, device.str().c_str());
    std::cout << predict_ch8(prefix, 50, net, vocab, device) << std::endl;
    std::cout << predict_ch8(prefix2, 50, net, vocab, device) << std::endl;

    return {epochs, ppl};
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	//=========================================================================
	// Implementation of Recurrent Neural Networks from Scratch
	//=========================================================================
	int64_t max_tokens = 10000;
	int64_t batch_size = 32, num_steps = 35;

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

	//=====================================================
	// One-Hot Encoding
	//=====================================================
	std::cout << torch::nn::functional::one_hot(torch::tensor({0, 2}), vocab.length()) << std::endl;

	// (The shape of the minibatch) that we sample each time (is (batch size, number of time steps).
	// The one_hot function transforms such a minibatch into a three-dimensional tensor with the last dimension
	// equals to the vocabulary size (len(vocab)).)
	auto X = torch::arange(10).reshape({2, 5});
	std::cout << torch::nn::functional::one_hot(X.transpose(0, 1), vocab.length()).sizes() << std::endl;
	std::cout << X.transpose(0, 1) << std::endl;
	auto Y = torch::nn::functional::one_hot(X.transpose(0, 1), vocab.length());
	std::vector<torch::Tensor> outputs;
	for(int i = 0; i < Y.size(0); i++ ) {
		//std::cout << Y[i] << std::endl;
		outputs.push_back(Y[i]);
	}
	std::cout << torch::cat(outputs, 0).sizes() << std::endl;

	torch::Tensor a = normal(10, 20, device);
	std::cout << a.sizes() << std::endl;

	//=============================================================
	// Let us [check whether the outputs have the correct shapes]
	int num_hiddens = 512;
	auto net = RNNModelScratch(vocab.length(), num_hiddens, device);

	torch::Tensor state = net.begin_state(X.size(0), device);
	std::tuple<torch::Tensor, torch::Tensor>  rlt = net.forward(X.to(device), state);
	auto Z = std::get<0>(rlt);
	auto new_state = std::get<1>(rlt);

	std::cout << "Z: " << Z.sizes() << std::endl;
	//std::cout << Z << std::endl;
	std::cout << "new_state: " << new_state.sizes() << std::endl;
	//std::cout << new_state << std::endl;

	//================================================
	// Prediction
	//================================================
	// Let us [first define the prediction function to generate new characters following the user-provided prefix]
	std::string s = "time traveller ";
	std::vector<char> v(s.begin(), s.end());
	std::vector<std::string> prefix;
	for(int i = 0; i < v.size(); i++ ) {
		std::string tc(1, v[i]);
		prefix.push_back(tc);
	}
	std::string prd = predict_ch8(prefix, 10, net, vocab, device);
	std::cout << prd << std::endl;

	std::vector<int> tokens_ids;
	for( size_t i = 0; i < tokens.size(); i++ )
		tokens_ids.push_back(vocab[tokens[i]]);

	// Training and Predicting
	std::vector<std::pair<torch::Tensor, torch::Tensor>> train_iter = seq_data_iter_random(tokens_ids, batch_size, num_steps);

	int64_t num_epochs = 300;
	float lr = 1.0;
	bool use_random_iter = false;
	auto nett = RNNModelScratch(vocab.length(), num_hiddens, device);

	std::pair<std::vector<double>, std::vector<double>> trlt = train_ch8(nett, train_iter, vocab, device, lr, num_epochs, use_random_iter);

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::plot(ax1, trlt.first, trlt.second, "b")->line_width(2);
    matplot::xlabel(ax1, "epoch");
    matplot::ylabel(ax1, "perplexity");
    matplot::show();

	std::cout << "Done!\n";
	return 0;
}





