
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


#endif /* SRC_UTILS_CH15_UTIL_H_ */
