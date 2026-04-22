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
        math::Matrix<T> output = input;
        std::transform(
            input.begin(), input.end(),
            output.begin(),
            [](T x) { return std::max(T{0}, x); }
        );
        return output;
    }

    template<std::floating_point T>
    math::Matrix<T> ReLU<T>::backward(const math::Matrix<T> &grad_output) {
        math::Matrix<T> grad_input = grad_output;
        std::transform(
            grad_output.begin(), grad_output.end(),
            cached_input.begin(),
            grad_input.begin(),
            [](T dy, T x) { return dy * (x > T{0} ? T{1} : T{0}); }
        );
        return grad_input;
    }
}
