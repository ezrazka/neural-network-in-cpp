#include "math/vector.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <format>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace math {
    template<std::floating_point T>
    Vector<T>::Vector(std::size_t length) : data(length) {}

    template<std::floating_point T>
    Vector<T>::Vector(std::initializer_list<T> init) : data(init) {}
    
    template<std::floating_point T>
    template<std::input_iterator InputIt>
    Vector<T>::Vector(InputIt first, InputIt last) : data(first, last) {}

    template<std::floating_point T>
    std::size_t Vector<T>::size() const noexcept {
        return data.size();
    }

    template<std::floating_point T>
    std::vector<T>::const_iterator Vector<T>::begin() const noexcept {
        return data.begin();
    }

    template<std::floating_point T>
    std::vector<T>::iterator Vector<T>::begin() noexcept {
        return data.begin();
    }

    template<std::floating_point T>
    std::vector<T>::const_iterator Vector<T>::end() const noexcept {
        return data.end();
    }

    template<std::floating_point T>
    std::vector<T>::iterator Vector<T>::end() noexcept {
        return data.end();
    }

    template<std::floating_point T>
    T Vector<T>::magnitude_squared() const {
        T sum_of_squares = std::inner_product(
            data.begin(), data.end(),
            data.begin(),
            static_cast<T>(0.0)
        );
        return sum_of_squares;
    }

    template<std::floating_point T>
    T Vector<T>::magnitude() const {
        return std::sqrt(magnitude_squared());
    }

    template<std::floating_point T>
    Vector<T> Vector<T>::normalized() const {
        T mag = magnitude();
        if (mag < tolerance) {
            throw_null_vector();
        }

        Vector<T> result_vec(size());
        std::transform(
            data.begin(), data.end(),
            result_vec.data.begin(),
            [mag](T x) { return x / mag; }
        );
        return result_vec;
    }

    template<std::floating_point T>
    Vector<T> &Vector<T>::normalize() {
        T mag = magnitude();
        if (mag < tolerance) {
            throw_null_vector();
        }

        std::transform(
            data.begin(), data.end(),
            data.begin(),
            [mag](T x) { return x / mag; }
        );
        return *this;
    }

    template<std::floating_point T>
    T Vector<T>::dot(const Vector<T> &other) const {
        if (size() != other.size()) {
            throw_size_mismatch(size(), other.size());
        }

        return std::inner_product(
            data.begin(), data.end(),
            other.data.begin(),
            static_cast<T>(0.0)
        );
    }

    template<std::floating_point T>
    Vector<T> Vector<T>::cross(const Vector<T> &other) const {
        if (size() != 3) {
            throw_not_3d(size());
        }

        if (other.size() != 3) {
            throw_not_3d(other.size());
        }

        return {
            data[1] * other.data[2] - data[2] * other.data[1],
            data[2] * other.data[0] - data[0] * other.data[2],
            data[0] * other.data[1] - data[1] * other.data[0]
        };
    }

    template<std::floating_point T>
    Vector<T> Vector<T>::hadamard(const Vector<T> &other) const {
        if (size() != other.size()) {
            throw_size_mismatch(size(), other.size());
        }

        Vector<T> result_vec(size());
        std::transform(
            data.begin(), data.end(),
            other.data.begin(),
            result_vec.data.begin(),
            [](T a, T b) { return a * b; }
        );
        return result_vec;
    }

    template<std::floating_point T>
    Vector<T> &Vector<T>::hadamard_inplace(const Vector<T> &other) {
        if (size() != other.size()) {
            throw_size_mismatch(size(), other.size());
        }

        std::transform(
            data.begin(), data.end(),
            other.data.begin(),
            data.begin(),
            [](T a, T b) { return a * b; }
        );
        return *this;
    }

    template<std::floating_point T>
    T Vector<T>::cosine_similarity(const Vector<T> &other) const {
        T mag_a = magnitude();
        T mag_b = other.magnitude();

        if (mag_a < tolerance) {
            throw_null_vector();
        }

        if (mag_b < tolerance) {
            throw_null_vector();
        }

        return dot(other) / (mag_a * mag_b);
    }

    template<std::floating_point T>
    T Vector<T>::at(std::size_t i) const {
        return data.at(i);
    }

    template<std::floating_point T>
    T &Vector<T>::at(std::size_t i) {
        return data.at(i);
    }

    template<std::floating_point T>
    Vector<T> Vector<T>::operator-() const {
        Vector<T> result_vec(size());
        std::transform(
            data.begin(), data.end(),
            result_vec.data.begin(),
            [](T x) { return -x; }
        );
        return result_vec;
    }
    
    template<std::floating_point T>
    Vector<T> Vector<T>::operator+(const Vector<T> &other) const {
        if (size() != other.size()) {
            throw_size_mismatch(size(), other.size());
        }

        Vector<T> result_vec(size());
        std::transform(
            data.begin(), data.end(),
            other.data.begin(),
            result_vec.data.begin(),
            [](T a, T b) { return a + b; }
        );
        return result_vec;
    }

    template<std::floating_point T>
    Vector<T> &Vector<T>::operator+=(const Vector<T> &other) {
        if (size() != other.size()) {
            throw_size_mismatch(size(), other.size());
        }

        std::transform(
            data.begin(), data.end(),
            other.data.begin(),
            data.begin(),
            [](T a, T b) { return a + b; }
        );
        return *this;
    }

    template<std::floating_point T>
    Vector<T> Vector<T>::operator-(const Vector<T> &other) const {
        if (size() != other.size()) {
            throw_size_mismatch(size(), other.size());
        }

        Vector<T> result_vec(size());
        std::transform(
            data.begin(), data.end(),
            other.data.begin(),
            result_vec.data.begin(),
            [](T a, T b) { return a - b; }
        );
        return result_vec;
    }

    template<std::floating_point T>
    Vector<T> &Vector<T>::operator-=(const Vector<T> &other) {
        if (size() != other.size()) {
            throw_size_mismatch(size(), other.size());
        }

        std::transform(
            data.begin(), data.end(),
            other.data.begin(),
            data.begin(),
            [](T a, T b) { return a - b; }
        );
        return *this;
    }

    template<std::floating_point T>
    Vector<T> Vector<T>::operator*(T k) const {
        Vector<T> result_vec(size());
        std::transform(
            data.begin(), data.end(),
            result_vec.data.begin(),
            [k](T x) { return x * k; }
        );
        return result_vec;
    }

    template<std::floating_point T>
    Vector<T> operator*(T k, const Vector<T> &v) {
        Vector<T> result_vec(v.size());
        std::transform(
            v.data.begin(), v.data.end(),
            result_vec.data.begin(),
            [k](T x) { return x * k; }
        );
        return result_vec;
    }

    template<std::floating_point T>
    Vector<T> &Vector<T>::operator*=(T k) {
        std::transform(
            data.begin(), data.end(),
            data.begin(),
            [k](T x) { return x * k; }
        );
        return *this;
    }

    template<std::floating_point T>
    Vector<T> Vector<T>::operator/(T k) const {
        if (std::abs(k) < tolerance) {
            throw_zero_division();
        }

        Vector<T> result_vec(size());
        std::transform(
            data.begin(), data.end(),
            result_vec.data.begin(),
            [k](T x) { return x / k; }
        );
        return result_vec;
    }

    template<std::floating_point T>
    Vector<T> &Vector<T>::operator/=(T k) {
        if (std::abs(k) < tolerance) {
            throw_zero_division();
        }

        std::transform(
            data.begin(), data.end(),
            data.begin(),
            [k](T x) { return x / k; }
        );
        return *this;
    }
    
    template<std::floating_point T>
    bool Vector<T>::operator==(const Vector<T> &other) const {
        if (size() != other.size()) {
            return false;
        }

        return std::equal(
            data.begin(), data.end(),
            other.data.begin(),
            [](T a, T b) { return std::abs(a - b) < tolerance; }
        );
    }

    template<std::floating_point T>
    T Vector<T>::operator[](std::size_t i) const noexcept {
        return data[i];
    }

    template<std::floating_point T>
    T &Vector<T>::operator[](std::size_t i) noexcept {
        return data[i];
    }
    
    template<std::floating_point T>
    void Vector<T>::throw_zero_division() {
        throw std::domain_error("Division by zero");
    }

    template<std::floating_point T>
    void Vector<T>::throw_null_vector() {
        throw std::domain_error("Vector has zero magnitude");
    }

    template<std::floating_point T>
    void Vector<T>::throw_not_3d(std::size_t i) {
        throw std::invalid_argument(
            std::format("Vector must be 3-dimensional: got {}", i)
        );
    }

    template<std::floating_point T>
    void Vector<T>::throw_size_mismatch(std::size_t i, std::size_t j) {
        throw std::invalid_argument(
            std::format("Size mismatch: {} and {}", i, j)
        );
    }
}
