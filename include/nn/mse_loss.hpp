#pragma once

#include "nn/loss.hpp"

#include <concepts>
#include <functional>
#include <numeric>

namespace nn {
    template<std::floating_point T>
    class MSELoss : public Loss<T> {
    public:
        T forward(const math::Matrix<T> &input, const math::Matrix<T> &target) override;
        math::Matrix<T> backward() override;
    };
    
    template<std::floating_point T>
    T MSELoss<T>::forward(const math::Matrix<T> &input, const math::Matrix<T> &target) {
        this->cached_input = input;
        this->cached_target = target;

        T n = input.size();
        return input.elementwise_reduce(
            target,
            T{0}, std::plus<>{},
            [](T pred, T y) {
                T diff = pred - y;
                return diff * diff;
            }
        ) / n;
    }

    template<std::floating_point T>
    math::Matrix<T> MSELoss<T>::backward() {
        T n = this->cached_input.size();
        return (this->cached_input - this->cached_target) * (T{2} / n);
    }
}
