
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

#endif /* SRC_UTILS_CH_14_UTIL_H_ */
