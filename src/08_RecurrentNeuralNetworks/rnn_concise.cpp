#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "util.h"


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	int64_t batch_size = 32, num_steps = 35;
	//		train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
	std::vector<std::string> lines = read_time_machine("./data/timemachine.txt");

	// split words
	std::vector<std::string> tokens = tokenize(lines, "char", true);
	std::vector<std::pair<std::string, int64_t>> counter = count_corpus( tokens );
	auto vocab = Vocab(counter, 0.0);

	std::cout << vocab[3] << "\n";

	// Defining the Model
	int num_hiddens = 256;
	auto rnn_layer = torch::nn::RNN(vocab.length(), num_hiddens);

	// use a tensor to initialize the hidden state
	auto state = torch::zeros({1, batch_size, num_hiddens});
	std::cout << state.sizes() << std::endl;

	/*
	 * With a hidden state and an input, we can compute the output with the updated hidden state.
	 * It should be emphasized that the "output" (Y) of rnn_layer does not involve computation of output layers:
	 * it refers to the hidden state at each time step, and they can be used as the input to the subsequent output layer.
	 */
	auto X = torch::rand({num_steps, batch_size, vocab.length()});
	auto state_new = rnn_layer->forward(X, state);
	std::cout << std::get<0>(state_new).sizes()  << "\nHHH:\n" << std::get<1>(state_new).sizes() << std::endl;


	std::cout << "Done!\n";
	return 0;
}

