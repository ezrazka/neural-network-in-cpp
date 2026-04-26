#pragma once

#include <concepts>
#include <functional>

namespace nn {
    template<std::floating_point T>
    struct Parameter {
        std::reference_wrapper<T> value;
        std::reference_wrapper<T> grad;
    };
}
