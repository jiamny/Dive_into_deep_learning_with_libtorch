#include <torch/torch.h>
#include <iostream>
#include <vector>

#ifndef SRC_UTILS_CH_18_UTIL_H_
#define SRC_UTILS_CH_18_UTIL_H_

const torch::Tensor torch_PI = torch::acos(torch::zeros(1)) * 2;  // Define pi in torch

template<typename T>
std::vector<T> vector_slice(std::vector<T> &v, size_t m, size_t n) {
    std::vector<T> vec(n - m + 1);
    std::copy(v.begin() + m, v.begin() + n + 1, vec.begin());
    return vec;
}

#endif /* SRC_UTILS_CH_18_UTIL_H_ */
