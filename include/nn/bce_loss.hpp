#pragma once

#include "nn/loss.hpp"
#include "math/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <numeric>

namespace nn {
    template<std::floating_point T>
    class BCELoss : public Loss<T> {
    public:
        T forward(const math::Matrix<T> &input, const math::Matrix<T> &target) override;
        math::Matrix<T> backward() override;
    };

    template<std::floating_point T>
    T BCELoss<T>::forward(const math::Matrix<T> &input, const math::Matrix<T> &target) {
        this->cached_input = input;
        this->cached_target = target;

        T n = input.size();
        T eps = T{1e-7};
        return input.elementwise_reduce(
            target,
            T{0}, std::plus<>{},
            [eps](T pred, T y) {
                T p = std::clamp(pred, eps, T{1} - eps);
                return -(y * std::log(p) + (T{1} - y) * std::log(T{1} - p));
            }
        ) / n;
    }

    template<std::floating_point T>
    math::Matrix<T> BCELoss<T>::backward() {
        T n = this->cached_input.size();
        T eps = T{1e-7};
        return this->cached_input.elementwise(
            this->cached_target,
            [eps](T pred, T y) {
                T p = std::clamp(pred, eps, T{1} - eps);
                return (p - y) / (p * (T{1} - p));
            }
        ) / n;
    }
}
