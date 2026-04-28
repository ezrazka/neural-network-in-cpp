#pragma once

#include "nn/module.hpp"
#include "math/matrix.hpp"

#include <algorithm>
#include <concepts>

namespace nn {
    template<std::floating_point T>
    class ReLU : public Module<T> {
    public:
        math::Matrix<T> forward(const math::Matrix<T> &input) override;
        math::Matrix<T> backward(const math::Matrix<T> &grad_output) override;

    private:
        math::Matrix<T> cached_input;
    };

    template<std::floating_point T>
    math::Matrix<T> ReLU<T>::forward(const math::Matrix<T> &input) {
        cached_input = input;

        return input.elementwise(
            [](T x) { return std::max(T{0}, x); }
        );
    }

    template<std::floating_point T>
    math::Matrix<T> ReLU<T>::backward(const math::Matrix<T> &grad_output) {
        return grad_output.elementwise(
            cached_input,
            [](T dy, T x) { return dy * (x > T{0}); }
        );
    }
}
