
#ifndef SRC_08_RECURRENTNEURALNETWORKS_UTIL_H_
#define SRC_08_RECURRENTNEURALNETWORKS_UTIL_H_

#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>

#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <regex>
#include <sstream>

#include <fstream>
#include <algorithm>
#include <map>
#include <set>
#include <random>

using torch::indexing::Slice;
using torch::indexing::None;


// The comparison function for sorting the set by increasing order of its pair's
// second value. If the second value is equal, order by the pair's first value
struct comp {
    template<typename T>
    bool operator()(const T &l, const T &r) const
    {
        if (l.second != r.second) {
            return l.second < r.second;
        }

        return l.first < r.first;
    }
};


std::vector<std::string> read_time_machine( std::string filename );

std::vector<std::pair<std::string, int64_t>> count_corpus( std::vector<std::string> tokens );

std::vector<std::string> tokenize(const std::vector<std::string> lines, const std::string token, bool max_cut);


// Vocabulary
/*
 * build a dictionary, often called vocabulary as well, to map string tokens into numerical indices starting from 0
 * we first count the unique tokens in all the documents from the training set, namely a corpus, and then assign a numerical
 * index to each unique token according to its frequency. Rarely appeared tokens are often removed to reduce the complexity.
 * Any token that does not exist in the corpus or has been removed is mapped into a special unknown token “<unk>”.
 * We optionally add a list of reserved tokens, such as “<pad>” for padding, “<bos>” to present the beginning for a sequence,
 * and “<eos>” for the end of a sequence.
 */

class Vocab {

public:
	std::map<int64_t, std::string> idx_to_token;
	std::set<std::pair<std::string, int64_t>, comp> order_token_to_idx;

    //Vocabulary for text.
	Vocab(std::vector<std::pair<std::string, int64_t>> corpus, float min_freq);

	~Vocab(void) {}

	int64_t length(void);

    int64_t unk(void);

    std::vector<std::string> to_tokens( std::vector<int64_t> indices );

    std::vector<std::pair<std::string, int64_t>> token_freqs(void);

    // Overload the + operator
    int64_t operator [] (const std::string s);

private:
    std::map<std::string, int64_t> token_to_idx;
    std::vector<std::pair<std::string, int64_t>> _token_freqs;
};


// Reading Long Sequence Data
// Random Sampling
template<typename T>
std::vector<std::pair<torch::Tensor, torch::Tensor>>  seq_data_iter_random(std::vector<T> my_seq, int batch_size, int num_steps) {
    // Generate a minibatch of subsequences using random sampling
    // Start with a random offset (inclusive of `num_steps - 1`) to partition a sequence
	std::random_device rd;
	std::default_random_engine eng(rd());
	std::uniform_int_distribution<int> distr(0, num_steps - 1);
	//std::cout << distr(eng) << std::endl;

	std::vector<int> sub_seq;
	for( int i = distr(eng); i < my_seq.size(); i++ ) {
		sub_seq.push_back(my_seq[i]);
	}

	//Subtract 1 since we need to account for labels
    int num_subseqs = static_cast<int>((my_seq.size() - 1) * 1.0 / num_steps);

    //The starting indices for subsequences of length `num_steps`
    std::vector<int> initial_indices;
    //= list(range(0, num_subseqs * num_steps, num_steps))
//    std::cout << "initial_indices:\n";
    for(int i  = 0; i < num_subseqs; i ++ ) {
    	auto idx = i * num_steps;
    	initial_indices.push_back(idx);
//    	std::cout << idx << " ";
    }
//    printf("\n");

    //In random sampling, the subsequences from two adjacent random
    //minibatches during iteration are not necessarily adjacent on the  original sequence
    std::random_shuffle(initial_indices.begin(), initial_indices.end());

    //std::copy(initial_indices.begin(), initial_indices.end(), std::ostream_iterator<int>(std::cout, " "));
    //std::cout << "\n";

    int64_t num_batches = static_cast<int>(num_subseqs * 1.0 / batch_size);
    //std::cout << "num_batches: " << num_batches << std::endl;

    std::vector<int64_t> batch_initial_indices;

//    std::cout << "batch_initial_indices:\n";
    for(int i  = 0; i < num_batches; i ++ ) {
    	//Here, `initial_indices` contains randomized starting indices for subsequences
//    	std::cout << i * batch_size << std::endl;
    	batch_initial_indices.push_back( i * batch_size );
    }
//    printf("\n");

    auto opts = torch::TensorOptions().dtype(torch::kInt);

    std::vector<std::pair<torch::Tensor, torch::Tensor>> outpairs;

    for(int i  = 0; i < batch_initial_indices.size(); i ++ ) {
    	torch::Tensor X = torch::zeros({batch_size, num_steps}, opts);
    	torch::Tensor Y = torch::zeros({batch_size, num_steps}, opts);

    	for(int r = 0; r < batch_size; r++ )
    		for(int j = initial_indices[i + r], c= 0; j < (initial_indices[i + r] + num_steps); j++, c++ )
    			X.index({r, c}) = my_seq[j];
    	//std::cout << X.sizes() << std::endl;

    	for(int r = 0; r < batch_size; r++ )
    		for(int j = initial_indices[i + r] + 1, c=0; j < (initial_indices[i + r] + 1 + num_steps); j++, c++ )
    			Y.index({r, c}) = my_seq[j];

    	outpairs.push_back({X.clone().to(torch::kInt64), Y.clone().to(torch::kInt64)});
    }

	return outpairs;
}

// Sequential Partitioning
/*
 * we can also ensure that the subsequences from two adjacent minibatches during iteration are adjacent on the original sequence
 */
template<typename T>
std::vector<std::pair<torch::Tensor, torch::Tensor>>  seq_data_iter_sequential(std::vector<T> my_seq, int batch_size, int num_steps) {
    //Generate a minibatch of subsequences using sequential partitioning.
    // Start with a random offset to partition a sequence
	std::random_device rd;
	std::default_random_engine eng(rd());
	std::uniform_int_distribution<int> distr(0, num_steps);
    int offset = distr(eng);

    int num_tokens = static_cast<int>(((my_seq.size() - offset - 1)*1.0 / batch_size)) * batch_size;

    auto opts = torch::TensorOptions().dtype(torch::kInt);

    std::vector<std::pair<torch::Tensor, torch::Tensor>> outpairs;

    torch::Tensor Xs = torch::zeros({num_tokens}, opts);
    torch::Tensor Ys = torch::zeros({num_tokens}, opts);

    for( int i = 0; i < num_tokens; i++ ) {
    	Xs.index({i}) = my_seq[offset + i];
    	Ys.index({i}) = my_seq[offset + 1 + i];
    }
    Xs = Xs.reshape({batch_size,-1});
    Ys = Ys.reshape({batch_size,-1});

	int num_batches = static_cast<int>(Xs.size(1) * 1.0 / num_steps);   //.shape[1] // num_steps

    for(int i = 0; i < num_batches; i++ ) {
    	int idx = i * num_steps;
        auto X = Xs.index({Slice(None),  Slice(i, i + num_steps)});
        auto Y = Ys.index({Slice(None),  Slice(i, i + num_steps)});
        outpairs.push_back({X.clone().to(torch::kInt64), Y.clone().to(torch::kInt64)});
    }

	return outpairs;
}

#endif /* SRC_08_RECURRENTNEURALNETWORKS_UTIL_H_ */
