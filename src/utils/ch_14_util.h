
#ifndef SRC_UTILS_CH_14_UTIL_H_
#define SRC_UTILS_CH_14_UTIL_H_
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

#include "../utils/ch_8_9_util.h"
#include "../utils.h"

std::string strip( const std::string& s );

std::vector<std::vector<std::vector<std::string>>> _read_wiki(const std::string data_dir, size_t num_read=0);

std::tuple<std::vector<std::string>, std::vector<std::string>, bool> _get_next_sentence(std::vector<std::string> sentence,
							std::vector<std::string> next_sentence, std::vector<std::vector<std::vector<std::string>>> paragraphs);

std::pair<std::vector<std::string>, std::vector<int64_t>> get_tokens_and_segments(std::vector<std::string> tokens_a,
																				std::vector<std::string> tokens_b);

std::tuple<std::vector<std::string>, std::vector<std::string>, bool> _get_next_sentence(std::vector<std::string> sentence,
							std::vector<std::string> next_sentence, std::vector<std::vector<std::vector<std::string>>> paragraphs);

std::vector<std::tuple<std::vector<std::string>, std::vector<int64_t>, bool>> _get_nsp_data_from_paragraph(
		std::vector<std::vector<std::string>> paragraph, std::vector<std::vector<std::vector<std::string>>> paragraphs,
		Vocab vocab, size_t max_len);

std::pair<std::vector<std::string>, std::map<int64_t, std::string> > _replace_mlm_tokens(std::vector<std::string> tokens,
		std::vector<int64_t> candidate_pred_positions, int64_t num_mlm_preds, Vocab vocab);

std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>> _get_mlm_data_from_tokens(
		std::vector<std::string> tokens, Vocab vocab);

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
		   std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
_pad_bert_inputs(std::vector<std::tuple<std::vector<int64_t>, std::vector<int64_t>,
							 std::vector<int64_t>, std::vector<int64_t>, bool>> examples, size_t max_len, Vocab vocab);


class _WikiTextDataset {
public:
	Vocab vocab;

	_WikiTextDataset(std::vector<std::vector<std::vector<std::string>>> paragraphs, int64_t max_len) {
        // Input `paragraphs[i]` is a list of sentence strings representing a
        // paragraph; while output `paragraphs[i]` is a list of sentences
        // representing a paragraph, where each sentence is a list of tokens
		//std::vector<std::vector<std::string>> setences;
		std::vector<std::string> tokens;

		for(auto& paragraph : paragraphs) {
			for(auto& setence : paragraph) {

				for(auto& tk : setence)
					tokens.push_back(tk);

				//setences.push_back(setence);
			}
		}
//		std::cout << "tokens: " << tokens.size() << '\n';
//		std::cout << "tokens[0]: " << tokens[0] << '\n';

		std::vector<std::pair<std::string, int64_t>> counter = count_corpus( tokens );

		std::vector<std::string> reserved_tokens;
		reserved_tokens.push_back("<pad>");
		reserved_tokens.push_back("<mask>");
		reserved_tokens.push_back("<cls>");
		reserved_tokens.push_back("<sep>");
		vocab = Vocab(counter, 5.0, reserved_tokens);

//		std::cout << "the: " << vocab["the"] << "\n";
//		std::cout << "counter: " << counter[0].second << "\n";

	    std::vector<std::tuple<std::vector<int64_t>, std::vector<int64_t>,
									 std::vector<int64_t>, std::vector<int64_t>, bool>> examples;
	    for( auto& paragraph : paragraphs ) {
	    	// Get data for the next sentence prediction task
//	    	std::cout << "paragraph: " << paragraph.size() << "\n";
//	    	printVector(paragraph);

	    	auto nsp_data = _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len);

//	    	std::cout << "nsp_data: " << nsp_data.size() << "\n";

	    	for( auto& nsp : nsp_data ) {
	    		std::vector<std::string> tks;
	    		std::vector<int64_t>     segments;
	    		bool is_next;
	    		tks = std::get<0>(nsp);
				segments = std::get<1>(nsp);
				is_next = std::get<2>(nsp);
	    		// Get data for the masked language model task
	    		std::vector<int64_t> token_ids, pred_positions, mlm_pred_label_ids;
	    		std::tie(token_ids, pred_positions, mlm_pred_label_ids) = _get_mlm_data_from_tokens(tks, vocab);

	    		examples.push_back(std::make_tuple(token_ids, pred_positions, mlm_pred_label_ids, segments, is_next));
	    	}
	    }
//	    std::cout << "examples: " << examples.size() << "\n";

        // Pad inputs
		std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
				   std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
		rlt = _pad_bert_inputs(examples, max_len, vocab);
        all_token_ids = std::get<0>(rlt);
        all_segments = std::get<1>(rlt);
		valid_lens = std::get<2>(rlt);
        all_pred_positions = std::get<3>(rlt);
		all_mlm_weights = std::get<4>(rlt);
        all_mlm_labels = std::get<5>(rlt);
		nsp_labels = std::get<6>(rlt);
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get(int64_t idx) {
        return std::make_tuple(all_token_ids[idx], all_segments[idx],
                valid_lens[idx], all_pred_positions[idx],
                all_mlm_weights[idx], all_mlm_labels[idx],
                nsp_labels[idx]);
	}

	int64_t size() {
        return all_token_ids.size();
	}

private:
	std::vector<torch::Tensor> all_token_ids, all_segments, valid_lens,
							   all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels;
};


#endif /* SRC_UTILS_CH_14_UTIL_H_ */
