
#include <torch/torch.h>
#include <torch/nn.h>
#include <torch/utils.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <tuple>
#include <map>
#include <vector>
#include <functional>
#include <utility> 		// make_pair etc.

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;

#ifndef SRC_UTILS_CH_11_UTIL_H_
#define SRC_UTILS_CH_11_UTIL_H_

std::list<std::pair<torch::Tensor, torch::Tensor>> get_data_ch11(torch::Tensor X, torch::Tensor Y, int64_t batch_size=8);

double f_2d(double x1, double x2);

#endif /* SRC_UTILS_CH_11_UTIL_H_ */
