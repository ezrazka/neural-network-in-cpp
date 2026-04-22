#pragma once

#include "nn/parameter.hpp"

#include <concepts>

namespace nn {
    template<std::floating_point T>
    class Optimizer {
    public:
        Optimizer(const std::vector<Parameter<T>> &params);
        virtual ~Optimizer() = default;

        virtual void step() = 0;
        virtual void zero_grad() = 0;

    protected:
        std::vector<Parameter<T>> params;
    };

    template<std::floating_point T>
    Optimizer<T>::Optimizer(const std::vector<Parameter<T>> &params) : params(params) {}
}
