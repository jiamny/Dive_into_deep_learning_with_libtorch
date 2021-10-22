

#ifndef UTILS_H_
#define UTILS_H_

#pragma once

#include <torch/torch.h>
#include <torch/utils.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <atomic>
#include <algorithm>

#include "fashion.h"

class LRdataset : public torch::data::datasets::Dataset<LRdataset> {
 public:

    explicit LRdataset(std::pair<torch::Tensor, torch::Tensor> data_and_labels);
    explicit LRdataset(torch::Tensor data, torch::Tensor labels);

    // Returns the `Example` at the given `index`.
    torch::data::Example<> get(size_t index) override;

    // Returns the size of the dataset.
    torch::optional<size_t> size() const override;

    // Returns all features.
    const torch::Tensor& features() const;

    // Returns all targets
    const torch::Tensor& labels() const;

 private:
    torch::Tensor features_;
    torch::Tensor labels_;
};


std::pair<torch::Tensor, torch::Tensor> synthetic_data(torch::Tensor w, float b, int64_t num_examples);

torch::Tensor linreg(torch::Tensor X, torch::Tensor w, torch::Tensor b);

torch::Tensor squared_loss(torch::Tensor y_hat, torch::Tensor y);

void sgd(torch::Tensor& w, torch::Tensor& b, float lr, int64_t batch_size);

template <typename Clock = std::chrono::high_resolution_clock>
class Timer{
    public:
		std::vector<std::chrono::high_resolution_clock::duration::rep> times;

        Timer() : start_point(Clock::now()) {
        	times.clear();
        }

        ~Timer           ()         = default;

        template <typename Rep = typename Clock::duration::rep, typename Units = typename Clock::duration>
        Rep stop() {
            std::atomic_thread_fence(std::memory_order_relaxed);
            typename Clock::time_point now = Clock::now();
            auto counted_time = std::chrono::duration_cast<Units>( now - start_point).count();
            start_point = now;
            std::atomic_thread_fence(std::memory_order_relaxed);

            times.push_back(static_cast<Rep>(counted_time));

            return static_cast<Rep>(counted_time);
        }

        void restartTimer () {
        	start_point = Clock::now();
        	times.clear();
        }

        template <typename Rep = typename Clock::duration::rep, typename Units = typename Clock::duration>
        Rep avg() {
        	unsigned int sum_of_elems = 0;
        	for (auto& n : times)
        	    sum_of_elems += n;

        	return static_cast<Rep>((sum_of_elems / times.size()));
        }

        template <typename Rep = typename Clock::duration::rep, typename Units = typename Clock::duration>
        Rep sum() {
        	unsigned int sum_of_elems = 0;
        	for (auto& n : times)
        	    sum_of_elems += n;
            return static_cast<Rep>(sum_of_elems);
        }

        template <typename Rep = typename Clock::duration::rep, typename Units = typename Clock::duration>
        std::vector<Rep> cumsum() {
        	std::vector<std::chrono::high_resolution_clock::duration::rep> results;
        	results.push_back(times[0]);
        	for (int i = 1; i < times.size(); i++) {
        		results.push_back(results[i - 1] + times[i]);
        	}
        	return results;
        }
    private:
        typename Clock::time_point start_point;
};

using precise_timer = Timer<>;
using system_timer = Timer<std::chrono::system_clock>;
using monotonic_timer = Timer<std::chrono::steady_clock>;

/*
 * There is no way to change the precision via to_string() but the setprecision IO manipulator could be used instead:
 */
template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

std::unordered_map<int, std::string> get_fashion_mnist_labels(void);
/*
template<typename Sampler = samplers::RandomSampler, typename Dataset>
torch::disable_if_t<Dataset::is_stateful || !std::is_constructible<Sampler, size_t>::value, std::unique_ptr<StatelessDataLoader<Dataset, Sampler>>>
auto load_data_fashion_mnist(size_t batch_szie, std::string data_path, bool train_data);
*/
torch::Tensor softmax(torch::Tensor X);

int64_t accuracy(torch::Tensor y_hat, torch::Tensor y);

torch::Tensor d2l_relu(torch::Tensor x);

torch::Tensor l2_penalty(torch::Tensor x);

std::pair<torch::Tensor, torch::Tensor> init_params(int64_t num_inputs);

#endif /* UTILS_H_ */
