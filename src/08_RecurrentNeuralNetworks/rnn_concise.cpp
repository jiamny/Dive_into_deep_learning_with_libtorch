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


struct RNNModel : public torch::nn::Module {
    //"""The RNN model."""
	torch::nn::RNN rnn{nullptr};
	int64_t vocab_size;
	int64_t num_hiddens;
	int64_t num_directions;
	torch::nn::Linear linear{nullptr};

	RNNModel( torch::nn::RNN rnn_layer, int64_t vocab_size) {

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



std::string predict_ch8(std::vector<std::string> prefix, int64_t num_preds, RNNModel net, Vocab vocab, torch::Device device) {
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

// Defined in file: ./chapter_recurrent-neural-networks/rnn-scratch.md
std::pair<double, double> train_epoch_ch8(RNNModel& net, std::vector<std::pair<torch::Tensor,
		torch::Tensor>> train_iter, torch::nn::CrossEntropyLoss loss, torch::optim::Optimizer& updater,
		torch::Device device, bool use_random_iter) {
	//***********************************************
	//two ways of setting the precision
	std::streamsize ss = std::cout.precision();
	std::cout.precision(15);
	// another way std::setprecision(N)

	double ppx = 0.0;
	int64_t tot_tk = 0;

    //"""Train a net within one epoch (defined in Chapter 8)."""
    //state, timer = None, d2l.Timer()
    //metric = d2l.Accumulator(2)  // Sum of training loss, no. of tokens
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
	    auto l = loss(y_hat, y.to(torch::kLong)).mean();				//loss(y_hat, y.long()).mean();

	    updater.zero_grad();
	    l.backward();
	    //grad_clipping(net, 1);

	    //"""Clip the gradient."""
	    torch::nn::utils::clip_grad_norm_(net.parameters(), 0.5); //  clip_grad_norm_(net.parameters(), 1.0);

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

std::pair<std::vector<double>, std::vector<double>> train_ch8( RNNModel& net, std::vector<std::pair<torch::Tensor, torch::Tensor>> train_iter,
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
    // Initialize
    //if isinstance(net, nn.Module):
	torch::optim::SGD updater = torch::optim::SGD(net.parameters(), lr);
    //else:
    //updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    //predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device);

    std::vector<double> epochs, ppl;
    // Train and predict
    double eppl, speed;
    for(int64_t  epoch = 0; epoch < num_epochs; epoch++ ) {

        auto rlt = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter);
        eppl = rlt.first;
        speed = rlt.second;
        if( (epoch + 1) % 10 == 0 ) {
        	std::cout << predict_ch8(prefix, 50, net, vocab, device) << std::endl;
            //animator.add(epoch + 1, [ppl])
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

	int64_t max_tokens = 10000;
	int64_t batch_size = 32, num_steps = 35;
	//		train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
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

	// Defining the Model
	int num_hiddens = 512;
	auto rnn_layer = torch::nn::RNN(vocab.length(), num_hiddens);

	// use a tensor to initialize the hidden state
	auto state_ = torch::zeros({1, batch_size, num_hiddens});
	std::cout << state_.sizes() << std::endl;

	std::cout << "size_ = " << state_.numel() << std::endl;
	/*
	 * With a hidden state and an input, we can compute the output with the updated hidden state.
	 * It should be emphasized that the "output" (Y) of rnn_layer does not involve computation of output layers:
	 * it refers to the hidden state at each time step, and they can be used as the input to the subsequent output layer.
	 */
	auto XX = torch::rand({num_steps, batch_size, vocab.length()});

	torch::Tensor output, state_new;
	std::tie(output, state_new) = rnn_layer->forward(XX, state_);
	//std::cout << std::get<0>(state_new).sizes()  << "\nHHH:\n" << std::get<1>(state_new).sizes() << std::endl;
	std::cout << output.sizes() << "\nHHH:\n" << state_new.sizes() << std::endl;

	auto nett = RNNModel( rnn_layer, vocab.length() );
	nett.to(device);

	std::string ss = "time traveller";
	std::vector<char> t(ss.begin(), ss.end());
	std::vector<std::string> prefx;
	for(int i = 0; i < t.size(); i++ ) {
		std::string tc(1, t[i]);
		prefx.push_back(tc);
	}
	std::cout << prefx[0] << "\n" << torch::tensor({{vocab[prefx[0]]}}, device).sizes() << std::endl;

	std::string ppred = predict_ch8(prefx, 10, nett, vocab, device);

	std::cout << "pred: " << ppred << std::endl;

	std::vector<int> tokens_ids;
	for( size_t i = 0; i < tokens.size(); i++ )
		tokens_ids.push_back(vocab[tokens[i]]);

	// Training and Predicting
	std::vector<std::pair<torch::Tensor, torch::Tensor>> train_iter = seq_data_iter_random(tokens_ids, batch_size, num_steps);

	auto net = RNNModel( rnn_layer, vocab.length() );
	net.to(device);

	int64_t num_epochs = 300;
	float lr = 1.0;
	bool use_random_iter = false;

	std::pair<std::vector<double>, std::vector<double>> trlt = train_ch8( net, train_iter, vocab, device, lr, num_epochs, use_random_iter);

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

