#pragma once

#include "nn/module.hpp"
#include "math/matrix.hpp"
#include "nn/parameter.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <numeric>

namespace nn {
    template<std::floating_point T>
    class Linear : public Module<T> {
    public:
        Linear(std::size_t input_size, std::size_t output_size);
        std::vector<Parameter<T>> parameters() override;
        math::Matrix<T> forward(const math::Matrix<T> &input) override;
        math::Matrix<T> backward(const math::Matrix<T> &grad_output) override;
        void zero_grad() override;

    private:
        math::Matrix<T> weights;
        math::Matrix<T> biases;
        math::Matrix<T> grad_weights;
        math::Matrix<T> grad_biases;
        math::Matrix<T> cached_input;
    };

    template<std::floating_point T>
    Linear<T>::Linear(std::size_t input_size, std::size_t output_size)
        : weights(math::Matrix<T>::random(output_size, input_size))
        , biases(math::Matrix<T>::random(output_size, 1))
        , grad_weights(output_size, input_size)
        , grad_biases(output_size, 1) {}

    template<std::floating_point T>
    std::vector<Parameter<T>> Linear<T>::parameters() {
        std::vector<Parameter<T>> result;
        result.reserve(weights.size());
        for (std::size_t i = 0; i < weights.size(); ++i) {
            result.push_back(Parameter<T>{std::ref(weights[i]), std::ref(grad_weights[i])});
        }
        for (std::size_t i = 0; i < biases.size(); ++i) {
            result.push_back(Parameter<T>{std::ref(biases[i]), std::ref(grad_biases[i])});
        }
        return result;
    }

    template<std::floating_point T>
    math::Matrix<T> Linear<T>::forward(const math::Matrix<T> &input) {
        cached_input = input;

        math::Matrix<T> output = weights * input;
        for (std::size_t i = 0; i < output.rows(); i++) {
            for (std::size_t j = 0; j < output.cols(); j++) {
                output(i, j) += biases(i, 0);
            }
        }

        return output;
    }

    template<std::floating_point T>
    math::Matrix<T> Linear<T>::backward(const math::Matrix<T> &grad_output) {
        math::Matrix<T> temp_grad_weights = grad_output * cached_input.transposed();
        std::copy(temp_grad_weights.begin(), temp_grad_weights.end(), grad_weights.begin());

        for (std::size_t i = 0; i < grad_biases.rows(); ++i) {
            grad_biases(i, 0) = std::accumulate(
                grad_output.begin() + i * grad_output.cols(),
                grad_output.begin() + (i + 1) * grad_output.cols(),
                T{0}
            );
        }

        return weights.transposed() * grad_output;
    }

    template<std::floating_point T>
    void Linear<T>::zero_grad() {
        fill(grad_weights.begin(), grad_weights.end(), T{0});
        fill(grad_biases.begin(), grad_biases.end(), T{0});
    }
}
