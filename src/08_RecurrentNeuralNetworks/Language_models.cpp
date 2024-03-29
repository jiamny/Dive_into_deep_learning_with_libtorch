#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>

#include "../utils/ch_8_9_util.h"

#include <matplot/matplot.h>
using namespace matplot;

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	//=============================================
	// Natural Language Statistics
	//=============================================
	std::vector<std::string> lines = read_time_machine("./data/timemachine.txt");

	// split words
	std::vector<std::string> tokens = tokenize(lines, "word", false);
	printf("[");
	for (int i = 0; i < 30; i++) {
		std::cout << tokens[i] << " ";
	}
	printf("]\n");

	// Since each text line is not necessarily a sentence or a paragraph, we
	// concatenate all text lines
	std::vector<std::pair<std::string, int64_t>> counter = count_corpus( tokens );

	std::vector<std::string> rv(0);
	auto vocab = Vocab(counter, 0.0, rv);;
	auto tk_frq = vocab.token_freqs();
	auto it = tk_frq.begin();
	for(int i =0; i < 10; i++ ) {
		std::cout << it->first << ":" << it->second << "\n";
		it++;
	}

	std::vector<float> freqs;
	std::vector<float> x;
	int i = 0;
	for( const auto& it : tk_frq ) {
		freqs.push_back(it.second * 1.0);
		x.push_back(i * 1.0);
		i++;
	}
	//std::reverse(x.begin(), x.end());

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::loglog(ax1, x, freqs, "b")->line_width(2);
    matplot::xlabel(ax1, "token: x");
    matplot::ylabel(ax1, "frequency: n(x)");
    matplot::show();

	// bigrams
	std::vector<std::string> bigram_tokens;
	for( int i = 0; i < (tokens.size() - 1); i++) {
		bigram_tokens.push_back(tokens[i] + " " + tokens[i+1]);
	}

	std::vector<std::pair<std::string, int64_t>> bigram_counter = count_corpus( bigram_tokens );

	auto bigram_token_freqs = Vocab(bigram_counter, 0.0, rv).token_freqs();
	it = bigram_token_freqs.begin();
	for(int i =0; i < 10; i++ ) {
		std::cout << "(" << it->first << "):" << it->second << "\n";
		it++;
	}

	std::vector<float> bigram_freqs;
	std::vector<float> bigram_x;
	i = 1;
	for( const auto& it : bigram_token_freqs ) {
		bigram_freqs.push_back(it.second * 1.0);
		bigram_x.push_back(i * 1.0);
		i++;
	}

	// trigrams
	std::vector<std::string> trigram_tokens;
	for( int i = 0; i < (tokens.size() - 2); i++) {
		trigram_tokens.push_back(tokens[i] + " " + tokens[i+1] + " " + tokens[i+2]);
	}

	std::vector<std::pair<std::string, int64_t>> trigram_counter = count_corpus( trigram_tokens );

	auto trigram_token_freqs = Vocab(trigram_counter, 0.0, rv).token_freqs();
	it = trigram_token_freqs.begin();
	for(int i =0; i < 10; i++ ) {
		std::cout << "(" << it->first << "):" << it->second << "\n";
		it++;
	}

	std::vector<float> trigram_freqs;
	std::vector<float> trigram_x;
	i = 1;
	for( const auto& it : trigram_token_freqs ) {
		trigram_freqs.push_back(it.second * 1.0);
		trigram_x.push_back(i * 1.0);
		i++;
	}

	auto F2 = figure(true);
	F2->size(1200, 500);
	F2->add_axes(false);
	F2->reactive_mode(false);

    auto ax2 = subplot(1, 2, 0);
    ax2->xlabel("token: x");
    ax2->ylabel("frequency: n(x)");
    loglog(ax2, x, freqs, "b")->line_width(2).display_name("unigram");
    legend({});

   	auto ax3 = subplot(1, 2, 1);
   	loglog(ax3, bigram_x, bigram_freqs, "r-")->line_width(2).display_name("bigram");
   	hold(on);
   	loglog(ax3, trigram_x, trigram_freqs, "m.")->line_width(2).display_name("trigram");
   	hold(on);
    legend({});
   	ax3->xlabel("token: x");
   	ax3->ylabel("frequency: n(x)");
   	hold(off);
	F2->draw();
	show();

	//=============================================
	// Reading Long Sequence Data
	//=============================================

	// manually generate a sequence from 0 to 34
	// With a minibatch size of 2, we only get 3 minibatches.
	std::vector<int> my_seq;
	for(int i = 0; i < 35; i++ )
		my_seq.push_back(i);

	int64_t batch_size = 2, num_steps = 5;

	std::cout << "\nRandom Sampling" << std::endl;
	std::vector<std::pair<torch::Tensor, torch::Tensor>> outpairs = seq_data_iter_random(my_seq, batch_size, num_steps);

    for( int i = 0; i < outpairs.size(); i++ ) {
    	std::cout <<"X:\n" << outpairs[i].first << std::endl;
    	std::cout <<"Y:\n" << outpairs[i].second << std::endl;
    }

    std::cout << "\nSequential Partitioning" << std::endl;
    outpairs = seq_data_iter_sequential(my_seq, batch_size, num_steps);

    for( int i = 0; i < outpairs.size(); i++ ) {
    	std::cout <<"X:\n" << outpairs[i].first << std::endl;
    	std::cout <<"Y:\n" << outpairs[i].second << std::endl;
    }

    /*
	vector<int> v({1, 2, 3, 4});
	auto opts = torch::TensorOptions().dtype(torch::kInt32);
	torch::Tensor t = torch::from_blob(t.data(), {4}, opts).to(torch::kInt64);
    std::cout << "t:\n" << t.reshape({2, -1}) << std::endl;
    */

	std::cout << "\nDone!\n";
	return 0;
}






