#pragma once

#include "nn/optimizer.hpp"

namespace nn {
    template<std::floating_point T>
    class SGD : public Optimizer<T> {
    public:
        SGD(const std::vector<Parameter<T>> &params, T lr = 0.001);
        void step() override;
        void zero_grad() override;

    private:
        T lr;
    };

    template<std::floating_point T>
    SGD<T>::SGD(const std::vector<Parameter<T>> &params, T lr) : Optimizer<T>(params), lr(lr) {}

    template<std::floating_point T>
    void SGD<T>::step() {
        for (auto &[value, grad] : this->params) {
            value.get() -= lr * grad.get();
        }
    }

    template<std::floating_point T>
    void SGD<T>::zero_grad() {
        for (auto &[value, grad] : this->params) {
            grad.get() = T{0};
        }
    }
}
