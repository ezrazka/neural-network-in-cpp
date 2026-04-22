#pragma once

#include "math/matrix.hpp"
#include "nn/parameter.hpp"

#include <concepts>

namespace nn {
    template<std::floating_point T>
    class Module {
    public:
        virtual ~Module() = default;

        virtual std::vector<Parameter<T>> parameters();
        virtual math::Matrix<T> forward(const math::Matrix<T> &input) = 0;
        virtual math::Matrix<T> backward(const math::Matrix<T> &grad_output) = 0;
        virtual void zero_grad();
    };

    template<std::floating_point T>
    std::vector<Parameter<T>> Module<T>::parameters() {
        return {};
    }

    template<std::floating_point T>
    void Module<T>::zero_grad() {}
}
