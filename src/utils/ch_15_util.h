
#ifndef SRC_UTILS_CH15_UTIL_H_
#define SRC_UTILS_CH15_UTIL_H_

#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <torch/nn.h>
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
#include <iostream>
#include <filesystem>
#include <fstream>

#include "../TempHelpFunctions.hpp"
#include "ch_8_9_util.h"

std::pair<std::vector<std::string>, std::vector<int64_t>> read_imdb(std::string data_dir, bool is_train = true, int num_files = 0);

std::pair<std::vector<std::string>, int> count_num_tokens(std::string text);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, Vocab>
load_data_imdb(std::string data_dir, size_t num_steps, int num_files = 0); // num_files = 0, load all data files

std::string extract_text(std::string s);

std::tuple<std::vector<std::string>, std::vector<std::string>, std::vector<int64_t>>
read_snli(const std::string data_dir, const bool is_train, const int num_sample = 0);

Vocab get_snil_vocab(std::tuple<std::vector<std::string>, std::vector<std::string>, std::vector<int64_t>> dt,
					 float min_freq, std::vector<std::string> reserved_tokens);


class SNLIDataset : public torch::data::datasets::Dataset<SNLIDataset> {
    //A customized dataset to load the SNLI dataset
private:
    int num_steps;
    Vocab vocab;
    std::vector<std::string> all_premise_tokens, all_hypothesis_tokens;
    torch::Tensor labels, features;
    int premise_w = 0, hypothese_w = 0;

public:

	explicit SNLIDataset(std::tuple<std::vector<std::string>, std::vector<std::string>, std::vector<int64_t>> dataset,
						 int num_steps, Vocab vocab) {
        this->num_steps = num_steps;
        this->all_premise_tokens = std::get<0>(dataset);
		this->all_hypothesis_tokens = std::get<1>(dataset);

        if( vocab.length() == 0 ) {
        	float min_freq = 5.0f;
        	std::vector<std::string> reserved_tokens;
        	reserved_tokens.push_back("<pad>");

            this->vocab = get_snil_vocab( dataset, min_freq, reserved_tokens);

        } else {
            this->vocab = vocab;
        }

        auto premises = _pad(all_premise_tokens);
        auto hypotheses = _pad(all_hypothesis_tokens);

        this->premise_w = premises.size(1);
        this->hypothese_w = hypotheses.size(1);

        std::vector<int64_t> labels_dt = std::get<2>(dataset);

        this->labels= torch::from_blob(labels_dt.data(),
        							 {static_cast<long>(labels_dt.size()), 1}, torch::TensorOptions(torch::kLong)).clone();

        this->labels.to(torch::kLong);

        this->features = torch::concat({premises, hypotheses}, 1).to(torch::kLong);

        std::cout << "read " << features.size(0) << " examples\n";

	}

    torch::Tensor _pad( std::vector<std::string> data) {
    	size_t num_steps = this->num_steps;

    	std::vector<torch::Tensor> tensors;

    	for( int i = 0; i < data.size(); i++ ) {
    		auto dt = truncate_pad(vocab[count_num_tokens(data[i]).first], num_steps, vocab["<pad>"]);

    		auto TT = torch::from_blob(dt.data(), {1, static_cast<long>(num_steps)}, torch::TensorOptions(torch::kLong)).clone();

    		tensors.push_back(TT);
    	}

    	torch::Tensor Tdata = torch::concat(tensors, 0).to(torch::kLong);
        //return torch.tensor([d2l.truncate_pad(
        //    self.vocab[line], self.num_steps, self.vocab['<pad>'])
        //                 for line in lines])
    	return Tdata;
    }

    torch::data::Example<> get(size_t idx) override {
        return {features[idx], labels[idx]};
    }

    torch::optional<size_t> size() const override {
        return features.size(0);
    }
};


class TokenEmbedding {
//Token Embedding

public:
	//std::vector<std::string> idx_to_token;
	std::map<int64_t, std::string> idx_to_token;
	std::map<std::string, int64_t> token_to_idx;
        torch::Tensor idx_to_vec;
	int64_t unknown_idx;

	TokenEmbedding(std::string embedding_name) {
        //Defined in :numref:`sec_synonyms`
        auto edt = _load_embedding(embedding_name);
        idx_to_token = edt.first;
        idx_to_vec   = edt.second;
        unknown_idx = 0;

        std::vector<int64_t> keys;
        std::transform(
        		idx_to_token.begin(),
				idx_to_token.end(),
				std::back_inserter(keys),
				[](const std::map<int64_t, std::string>::value_type &pair){return pair.first;});

        for(auto& k : keys) {
        	token_to_idx[idx_to_token[k]] = k;
        }
	}

    std::pair<std::map<int64_t, std::string>, torch::Tensor> _load_embedding(std::string embedding_name) {
        //idx_to_token = ['<unk>'],
        std::string text;
        std::ifstream fL(embedding_name.c_str());

        size_t idx = 0;
        std::map<int64_t, std::string> idx_to_token;
        idx_to_token[idx++] = "<unk>";

        torch::Tensor idx_to_vec;
        std::vector<torch::Tensor> tensors;
        tensors.push_back(torch::zeros({1, 100}, at::TensorOptions(torch::kDouble)));

        if( fL.is_open() ) {
        	while ( std::getline(fL, text) ) {
        		std::string space_delimiter = " ";
        		std::vector<double> words{};

        		size_t pos = 0;
        		size_t cnt = 0;
        		while ((pos = text.find(space_delimiter)) != std::string::npos) {
        			if( text.substr(0, pos).length() > 0 ) {
        				if(cnt > 0) {
        						words.push_back(std::stod(text.substr(0, pos).c_str()));
        				} else {
        						idx_to_token[idx++] = text.substr(0, pos);
        				}
        				cnt++;
        			}
        			text.erase(0, pos + space_delimiter.length());
        		}
        		if( text.substr(0, pos).length() > 0 )
        			words.push_back(std::stod(text.substr(0, pos).c_str()));

        		if( words.size() == 100 ) {
        			auto TT = torch::from_blob(words.data(), {1, 100}, torch::TensorOptions(torch::kDouble)).clone();
        			tensors.push_back(TT);
        		}
        	}
        }

        fL.close();
        idx_to_vec = torch::concat(tensors, 0);
        return std::make_pair(idx_to_token, idx_to_vec);
    }

    torch::Tensor getitem(std::vector<std::string> tokens) {
    	std::vector<int64_t> indices;
    	for(auto& tk : tokens) {
    		indices.push_back(token_to_idx[tk]);
    	}
        //indices = [token_to_idx.get(token, self.unknown_idx) for token in tokens]
    	// sort indices
    	std::sort(indices.begin(), indices.end());
    	auto index = torch::from_blob(indices.data(),
    			{static_cast<long>(indices.size())}, torch::TensorOptions(torch::kLong));
        torch::Tensor vecs = torch::index_select(idx_to_vec, 0, index).clone(); //[d2l.tensor(indices)]
        return vecs;
    }

    int64_t length() {
        return idx_to_token.size();
    }

    // Overload the [] operator
    torch::Tensor operator [] (const std::vector<std::string> ss) {
    	return getitem(ss);
    }

    torch::Tensor operator [] (const std::map<int64_t, std::string> ss ) {
    	std::vector<std::string> values;
    	std::transform( ss.begin(),
    					ss.end(),
    					std::back_inserter(values),
    					[](const std::map<int64_t, std::string>::value_type &pair){return pair.second;});

    	return getitem(values);
    }
};


#endif /* SRC_UTILS_CH15_UTIL_H_ */
