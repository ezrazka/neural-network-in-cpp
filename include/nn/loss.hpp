#pragma once

#include "math/matrix.hpp"

#include <concepts>

namespace nn {
    template<std::floating_point T>
    class Loss {
    public:
        virtual T forward(const math::Matrix<T> &input, const math::Matrix<T> &target) = 0;
        virtual math::Matrix<T> backward() = 0;

    protected:
        math::Matrix<T> cached_input;
        math::Matrix<T> cached_target;
    };
}
