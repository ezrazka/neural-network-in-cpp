#pragma once

#include <cstdint>
#include <random>

namespace math {
    namespace detail {
        inline std::mt19937 &get_rng() {
            static std::mt19937 rng{std::random_device{}()};
            return rng;
        }
    }

    inline void set_seed(std::uint32_t seed) {
        detail::get_rng().seed(seed);
    }
}
