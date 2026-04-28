#pragma once

#include "nn/loss.hpp"
#include "math/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <numeric>
#include <vector>

namespace nn {
    template<std::floating_point T>
    class CrossEntropyLoss : public Loss<T> {
    public:
        T forward(const math::Matrix<T> &input, const math::Matrix<T> &target) override;
        math::Matrix<T> backward() override;

    private:
        math::Matrix<T> cached_softmax;
    };

    template<std::floating_point T>
    T CrossEntropyLoss<T>::forward(const math::Matrix<T> &input, const math::Matrix<T> &target) {
        this->cached_input = input;
        this->cached_target = target;

        T n = input.size();

        std::vector<T> max_pred(input.cols(), -std::numeric_limits<T>::infinity());
        for (std::size_t i = 0; i < input.rows(); i++) {
            for (std::size_t j = 0; j < input.cols(); j++) {
                max_pred[j] = std::max(max_pred[j], input[i][j]);
            }
        }

        std::vector<T> log_sum_exp(input.cols(), T{0});
        for (std::size_t i = 0; i < input.rows(); i++) {
            for (std::size_t j = 0; j < input.cols(); j++) {
                log_sum_exp[j] += std::exp(input[i][j] - max_pred[j]);
            }
        }
        for (std::size_t j = 0; j < input.cols(); j++) {
            log_sum_exp[j] = max_pred[j] + std::log(log_sum_exp[j]);
        }

        cached_softmax = math::Matrix<T>(
            input.rows(),
            input.cols()
        );
        for (std::size_t i = 0; i < input.rows(); i++) {
            for (std::size_t j = 0; j < input.cols(); j++) {
                cached_softmax(i, j) = std::exp(input(i, j) - log_sum_exp[j]);
            }
        }

        T result = target.elementwise_reduce(
            cached_softmax,
            T{0}, std::plus<>{},
            [](T y, T pred) {
                return y * (-std::log(pred));
            }
        ) / n;
        return result;
    }

    template<std::floating_point T>
    math::Matrix<T> CrossEntropyLoss<T>::backward() {
        T n = this->cached_input.size();

        return (cached_softmax - this->cached_target) / n;
    }
}
