#pragma once

#include "nn/module.hpp"
#include "math/matrix.hpp"

#include <algorithm>
#include <concepts>

namespace nn {
    template<std::floating_point T>
    class Sigmoid : public Module<T> {
    public:
        math::Matrix<T> forward(const math::Matrix<T> &input) override;
        math::Matrix<T> backward(const math::Matrix<T> &grad_output) override;

    private:
        math::Matrix<T> cached_input;
    };

    template<std::floating_point T>
    math::Matrix<T> Sigmoid<T>::forward(const math::Matrix<T> &input) {
        cached_input = input;

        return input.elementwise(
            [](T x) { return T{1} / (T{1} + std::exp(-x)); }
        );
    }

    template<std::floating_point T>
    math::Matrix<T> Sigmoid<T>::backward(const math::Matrix<T> &grad_output) {
        return grad_output.elementwise(
            cached_input,
            [](T dy, T x) {
                T y = T{1} / (T{1} + std::exp(-x));
                return dy * y * (T{1} - y);
            }
        );
    }
}
