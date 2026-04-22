#pragma once

#include "nn/loss.hpp"
#include "math/matrix.hpp"

#include <algorithm>
#include <concepts>
#include <cstdlib>
#include <functional>
#include <numeric>

namespace nn {
    template<std::floating_point T>
    class BCEWithLogitsLoss : public Loss<T> {
    public:
        T forward(const math::Matrix<T> &input, const math::Matrix<T> &target) override;
        math::Matrix<T> backward() override;
    };

    template<std::floating_point T>
    T BCEWithLogitsLoss<T>::forward(const math::Matrix<T> &input, const math::Matrix<T> &target) {
        this->cached_input = input;
        this->cached_target = target;

        T n = input.size();
        
        return std::transform_reduce(
            input.begin(), input.end(),
            target.begin(),
            T{0},
            std::plus<>(),
            [](T pred, T y) {
                return std::max(pred, T{0}) - pred * y + std::log1p(std::exp(-std::abs(pred)));
            }
        ) / n;
    }

    template<std::floating_point T>
    math::Matrix<T> BCEWithLogitsLoss<T>::backward() {
        T n = this->cached_input.size();

        math::Matrix<T> grad_output(
            this->cached_input.rows(),
            this->cached_input.cols()
        );
        std::transform(
            this->cached_input.begin(), this->cached_input.end(),
            this->cached_target.begin(),
            grad_output.begin(),
            [](T pred, T y) {
                T p = T{1} / (T{1} + std::exp(-pred));
                return p - y;
            }
        );
        grad_output /= n;
        return grad_output;
    }
}
