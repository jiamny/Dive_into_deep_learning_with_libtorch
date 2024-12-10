

#ifndef SRC_UTILS_CH_16_UTIL_H_
#define SRC_UTILS_CH_16_UTIL_H_

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

using torch::indexing::Slice;
using torch::indexing::None;

std::string strip( const std::string& s );

std::vector<std::string> stringSplit(const std::string& str, char delim);

torch::Tensor RangeTensorIndex(int64_t num, bool suffle = false);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> train_test_split(torch::Tensor X,
		torch::Tensor y, double test_size=0.3, bool suffle=true);


#endif /* SRC_UTILS_CH_16_UTIL_H_ */
