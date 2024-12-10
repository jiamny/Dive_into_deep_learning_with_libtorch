
#ifndef SRC_UTILS_CH_20_UTIL_H_
#define SRC_UTILS_CH_20_UTIL_H_

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

#include "../utils.h"
#include "../TempHelpFunctions.hpp"

using torch::indexing::Slice;
using torch::indexing::None;

torch::Tensor distance_matrix(torch::Tensor x, torch::Tensor y);

torch::Tensor rbfkernel(torch::Tensor x1, torch::Tensor x2, float ls=4.);

std::vector<double> tensorTovec(torch::Tensor A);

class MultivariateNormalx{
    torch::Tensor mean, stddev, var, L;
    int d = 0;
    // Define epsilon.
    double epsilon = 0.0001;
public:
    MultivariateNormalx(const torch::Tensor &mean, const torch::Tensor &std) : mean(mean), stddev(std), var(std * std) {
      	d = mean.size(0);
    	// Add small pertturbation.
    	torch::Tensor K = stddev + epsilon*torch::eye(d).to(mean.dtype());
    	// Cholesky decomposition.
    	L = torch::linalg::cholesky(K);
    }

    torch::Tensor rsample(int n = 1) {
    	torch::Tensor u = torch::normal(0., 1., d*n).reshape({d, n}).to(mean.dtype());
    	torch::Tensor x = mean.reshape({d, 1}) + torch::mm(L, u);
    	return x;
    }
};

#endif /* SRC_UTILS_CH_20_UTIL_H_ */
