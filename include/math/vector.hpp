#pragma once

#include <concepts>
#include <initializer_list>
#include <vector>

namespace math {
    template<std::floating_point T>
    class Vector {
    public:
        Vector(std::size_t length);
        Vector(std::initializer_list<T> init);
        template<std::input_iterator InputIt>
        Vector(InputIt first, InputIt last);

        std::size_t size() const noexcept;
        std::vector<T>::const_iterator begin() const noexcept;
        std::vector<T>::iterator begin() noexcept;
        std::vector<T>::const_iterator end() const noexcept;
        std::vector<T>::iterator end() noexcept;

        T magnitude_squared() const;
        T magnitude() const;
        Vector normalized() const;
        Vector &normalize();

        T dot(const Vector &other) const;
        Vector cross(const Vector &other) const;
        Vector hadamard(const Vector &other) const;
        Vector &hadamard_inplace(const Vector &other);
        T cosine_similarity(const Vector &other) const;

        T at(std::size_t i) const;
        T &at(std::size_t i);

        Vector operator-() const;
        
        Vector operator+(const Vector &other) const;
        Vector &operator+=(const Vector &other);
        Vector operator-(const Vector &other) const;
        Vector &operator-=(const Vector &other);

        Vector operator*(T k) const;
        friend Vector operator*(T k, const Vector<T> &v);
        Vector &operator*=(T k);
        Vector operator/(T k) const;
        Vector &operator/=(T k);
        
        bool operator==(const Vector &other) const;

        T operator[](std::size_t i) const noexcept;
        T &operator[](std::size_t i) noexcept;

    private:
        static constexpr T tolerance = []() {
            if constexpr (sizeof(T) <= 4) return static_cast<T>(1e-5);
            return static_cast<T>(1e-9);
        }();

        static void throw_zero_division();
        static void throw_null_vector();
        static void throw_not_3d(std::size_t i);
        static void throw_size_mismatch(std::size_t i, std::size_t j);

        std::vector<T> data;
    };
}
