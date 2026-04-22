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
            log_sum_exp[j] = std::log(log_sum_exp[j]);
        }

        T result = T{0};
        for (std::size_t i = 0; i < input.rows(); i++) {
            for (std::size_t j = 0; j < input.cols(); j++) {
                T pred = input[i][j];
                T y = target[i][j];
                result += y * (-pred + log_sum_exp[j]);
            }
        }
        return result;
    }

    template<std::floating_point T>
    math::Matrix<T> CrossEntropyLoss<T>::backward() {
        T n = this->cached_input.size();

        std::vector<T> max_pred(this->cached_input.cols(), -std::numeric_limits<T>::infinity());
        for (std::size_t i = 0; i < this->cached_input.rows(); i++) {
            for (std::size_t j = 0; j < this->cached_input.cols(); j++) {
                max_pred[j] = std::max(max_pred[j], this->cached_input[i][j]);
            }
        }
        std::vector<T> sum_exp_inv(this->cached_input.cols(), T{0});
        for (std::size_t i = 0; i < this->cached_input.rows(); i++) {
            for (std::size_t j = 0; j < this->cached_input.cols(); j++) {
                sum_exp_inv[j] += std::exp(this->cached_input[i][j] - max_pred[j]);
            }
        }
        for (std::size_t j = 0; j < this->cached_input.cols(); j++) {
            sum_exp_inv[j] = T{1} / sum_exp_inv[j];
        }

        math::Matrix<T> grad_output(
            this->cached_input.rows(),
            this->cached_input.cols()
        );
        for (std::size_t i = 0; i < this->cached_input.rows(); i++) {
            for (std::size_t j = 0; j < this->cached_input.cols(); j++) {
                T pred = this->cached_input[i][j];
                T y = this->cached_target[i][j];
                T p = std::exp(pred - max_pred[j]) * sum_exp_inv[j];
                grad_output[i][j] = p - y;
            }
        }
        return grad_output;
    }
}
