#pragma once

#include "math/matrix.hpp"
#include "nn/module.hpp"
#include "nn/parameter.hpp"

#include <concepts>
#include <memory>
#include <utility>

namespace nn {
    template<std::floating_point T>
    class Sequential : public Module<T> {
    public:
        template<typename ...Modules>
        Sequential(Modules &&...modules);

        std::vector<Parameter<T>> parameters() override;
        math::Matrix<T> forward(const math::Matrix<T> &input) override;
        math::Matrix<T> backward(const math::Matrix<T> &grad_output) override;
        void zero_grad() override;

    private:
        std::vector<std::unique_ptr<Module<T>>> layers;
    };

    template<std::floating_point T>
    template<typename ...Modules>
    Sequential<T>::Sequential(Modules &&...modules) {
        (
            layers.push_back(
                std::make_unique<std::decay_t<Modules>>(std::forward<Modules>(modules))
            ),
            ...
        );
    }

    template<std::floating_point T>
    std::vector<Parameter<T>> Sequential<T>::parameters() {
        std::vector<Parameter<T>> result;
        for (const std::unique_ptr<Module<T>> &layer : layers) {
            std::vector<Parameter<T>> params = layer->parameters();
            result.insert(result.end(), params.begin(), params.end());
        }
        return result;
    }

    template<std::floating_point T>
    math::Matrix<T> Sequential<T>::forward(const math::Matrix<T> &input) {
        math::Matrix<T> output = input;
        for (std::unique_ptr<Module<T>> &layer : layers) {
            output = layer->forward(output);
        }
        return output;
    }

    template<std::floating_point T>
    math::Matrix<T> Sequential<T>::backward(const math::Matrix<T> &grad_output) {
        math::Matrix<T> grad_input = grad_output;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            std::unique_ptr<Module<T>> &layer = *it;
            grad_input = layer->backward(grad_input);
        }
        return grad_input;
    }

    template<std::floating_point T>
    void Sequential<T>::zero_grad() {
        for (std::unique_ptr<Module<T>> &layer : layers) {
            layer->zero_grad();
        }
    }
}
