#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <vector>

#include <regex>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <map>
#include <set>
#include <random>

#include "../utils/ch_8_9_util.h"
//#include "../utils.h"
#include "../TempHelpFunctions.hpp"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;

void show_list_len_pair_hist(std::vector<std::vector<std::string>> source,
							 std::vector<std::vector<std::string>> target) {
	// show_list_len_pair_hist
	std::vector<float> xs(60);
	std::vector<float> xt(60);
	std::vector<float> ys(60);
	std::vector<float> yt(60);
	for( int i = 0; i < 60; i++ ) {
		xs[i] = i*1.0 - 0.5;
		xt[i] = i*1.0 + 0.5;
		ys[i] = 0;
		yt[i] = 0;
	}

	for( size_t i = 0; i < source.size(); i++ ) {
		ys[source[i].size()] += 1;
		yt[target[i].size()] += 1;
	}

	plt::figure_size(600, 450);
	plt::bar(xs, ys, "blue", "-", 0.25, {{"label", "source"}});
	plt::bar(xt, yt, "orange", "-", 0.25, {{"label", "target"}});
	plt::xlabel("# tokens per sequence");
	plt::ylabel("Count");
	plt::legend();
	plt::show();
	plt::close();
}

std::vector<std::pair<std::string, int64_t>> count_corpus2( std::vector<std::string> tokens ) {
	std::vector<std::pair<std::string, int64_t>> _token_freqs;
	std::map<std::string, int64_t> cntTks;


    for (auto tk : tokens) {

        if (cntTks.find(tk) == cntTks.end()) {  // if key is NOT present already
        	cntTks[tk] = 1; 					// initialize the key with value 1
        } else {
        	cntTks[tk]++; 						// key is already present, increment the value by 1
        }
    }

    // copy key-value pairs from the map to the vector
    std::copy(cntTks.begin(), cntTks.end(), std::back_inserter<std::vector<std::pair<std::string, int64_t>>>(_token_freqs));

    // sort the vector by increasing the order of its pair's second value
    // if the second value is equal, order by the pair's first value
    std::sort(_token_freqs.begin(), _token_freqs.end(),
                      [](const std::pair<std::string, int64_t> &l, const std::pair<std::string, int64_t> &r) {
                          if (l.second != r.second) {
                              return l.second > r.second;
                          }
                          return l.first > r.first;
                      });

    return _token_freqs;
}

void torch_cat_test() {
	torch::Tensor src_array = torch::tensor({{1,  1,  1,  0,  0,  0,  0,  0},
											 {1,  1,  1,  1,  1,  0,  0,  0},
											 {1,  1,  1,  0,  0,  0,  0 , 0}}).to(torch::kLong);

	torch::Tensor tgt_array = torch::tensor({{0,  0, 1,  1,  1,  1,  0,  0},
				{0,  0,  0,  0,  3,  1,  1,  1}, {1,  1,  1,  1,  1,  1,  0,  0}}).to(torch::kLong);

	torch::Tensor src_valid_len = torch::tensor({3, 3, 3, 3, 3, 3, 3, 3, 3, 3}).to(torch::kLong);
	torch::Tensor tgt_valid_len = torch::tensor({3, 3, 3, 3, 3, 4, 4, 4, 3, 4}).to(torch::kLong);

	torch::NoGradGuard no_grad;
	torch::Tensor features = torch::cat({src_array, tgt_array}, -1).to(torch::kLong);
	std::cout << "----------------------------\n";
	std::cout << src_array << std::endl;
	std::cout << tgt_array << std::endl;
	std::cout << "============================\n";
	std::cout << features << std::endl;
	std::cout << src_valid_len << std::endl;
	std::cout << tgt_valid_len << std::endl;
	torch::Tensor labels = (torch::cat({src_valid_len, tgt_valid_len}, -1).reshape({-1, src_valid_len.size(0)})).transpose(1, 0);
	std::cout << labels << std::endl;

	/*
	// merge tensors
	torch::Tensor features = torch::cat({src_array, tgt_array}, -1);

	torch::Tensor labels = (torch::cat({src_valid_len, tgt_valid_len}, -1).reshape({-1, src_valid_len.size(0)})).transpose(1, 0);

	std::pair<torch::Tensor, torch::Tensor> data_arrays = {features, labels};

	auto X = features.index({Slice(), Slice(None, num_steps)});
	auto X_valid_len = labels.index({Slice(), 0});
	auto Y = features.index({Slice(), Slice(num_steps, None)});
	auto Y_valid_len = labels.index({Slice(), 1});
	*/
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(7);
	int num_examples = 600;

	std::string filename = "./data/fra-eng/fra.txt";

	std::string raw_test = read_data_nmt(filename);

	std::cout << raw_test.size() << std::endl;
	std::cout << raw_test.substr(0, 75) << std::endl;

	std::string processed = preprocess_nmt( raw_test );

	std::cout << processed.size() << std::endl;
	std::cout << processed.substr(0, 75) << std::endl;

	std::vector<std::vector<std::string>> source, target;
	std::tie(source, target) = tokenize_nmt(processed, num_examples); // set 0 to use all samples

	std::cout << source.size() << std::endl;
	std::cout << target.size() << std::endl;

//	show_list_len_pair_hist(source, target);
	std::vector<std::string> tokens;
	for(std::vector<std::string>& tk : source ) {
		for(std::string& t : tk) {
			tokens.push_back(t);
		}
	}

	std::vector<std::pair<std::string, int64_t>> counter = count_corpus( tokens );

	std::vector<std::string> rv({"<pad>", "<bos>", "<eos>"});
	auto src_vocab = Vocab(counter, 2.0, rv);
	std::cout << src_vocab.length() << "\n";
	std::cout << src_vocab.to_tokens({47}) << "\n";

	for(std::string& t : source[0])
		std::cout << t << " " << src_vocab[t] << " ";
	std::cout << "\n";

	auto idx = src_vocab[source[0]];
	std::cout << idx.size() << "\n";
	for(auto& t : idx)
		std::cout << t << " ";
	std::cout << "\n";

	auto rlt = truncate_pad(src_vocab[source[0]], 10, src_vocab["<pad>"]);
	for(auto& t : rlt)
		std::cout << t << " ";
	std::cout << "\n";

	// transform text sequences into minibatches for training.
	int num_steps = 8;

	torch::Tensor src_array, src_valid_len;
	std::tie(src_array, src_valid_len) = build_array_nmt(source, src_vocab, num_steps);
	std::cout << src_valid_len.sizes() << "\n";
	std::cout << src_array[0] << std::endl; // index({Slice(None, 3), Slice()})

	std::vector<std::string> tgt_tokens;
	for(std::vector<std::string>& tk : target ) {
		for(std::string& t : tk) {
			tgt_tokens.push_back(t);
		}
	}
	std::vector<std::pair<std::string, int64_t>> tgt_counter = count_corpus( tgt_tokens );
	auto tgt_vocab = Vocab(tgt_counter, 2.0, rv);

	torch::Tensor tgt_array, tgt_valid_len;
	std::tie(tgt_array, tgt_valid_len) = build_array_nmt(target, tgt_vocab, num_steps);
	std::cout << tgt_valid_len.sizes() << "\n";
	std::cout << tgt_array[0] << std::endl;

	// merge tensors
	torch::Tensor features = torch::cat({src_array, tgt_array}, -1);
//	std::cout << features << std::endl;

	torch::Tensor labels = (torch::cat({src_valid_len, tgt_valid_len}, -1).reshape(
									   {-1, src_valid_len.size(0)})).transpose(1, 0);
//	std::cout << labels << "\n";

	std::pair<torch::Tensor, torch::Tensor> data_arrays = {features, labels};

	int batch_size = 2;
	auto dataset = LRdataset(data_arrays)
	    				.map(torch::data::transforms::Stack<>());
	auto data_iter = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
	    	        std::move(dataset), batch_size);

	// read the first minibatch from the English-French dataset
	for(auto& batch : *data_iter){
		auto features = batch.data;
		auto labels = batch.target;
		auto X = features.index({Slice(), Slice(None, num_steps)});
		auto X_valid_len = labels.index({Slice(), 0});
		auto Y = features.index({Slice(), Slice(num_steps, None)});
		auto Y_valid_len = labels.index({Slice(), 1});

		std::cout << "X:\n" << X << std::endl;
		std::cout << "valid lengths for X:\n" << X_valid_len << std::endl;
		std::cout << "Y:\n" << Y << std::endl;
		std::cout << "valid lengths for Y:\n" << Y_valid_len << std::endl;

        X.to(device);
        X_valid_len.to(device);
        Y.to(device);
        Y_valid_len.to(device);
        std::vector<int64_t> tmp;
        for(int i = 0; i < Y.size(0); i++)
        	tmp.push_back(tgt_vocab["<bos>"]);
        auto options = torch::TensorOptions().dtype(torch::kLong);
        torch::Tensor bos = torch::from_blob(tmp.data(), {1, Y.size(0)}, options);
        bos = bos.clone().to(device).reshape({-1, 1});
//        auto bos = torch::tensor({tgt_vocab["<bos>"]} * Y.size(0)).to(device).reshape(-1, 1);
        std::cout << bos << std::endl;
        torch::Tensor dec_input = torch::cat({bos, Y.index({Slice(), Slice(None, -1)})}, 1); // Y[:, :-1]
        std::cout << dec_input << std::endl;
	    break;
	}

	std::cout << "Done!\n";
	return 0;
}



