#include "math/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <initializer_list>
#include <vector>

namespace math {
    template<std::floating_point T>
    Matrix<T> Matrix<T>::identity(std::size_t n) {
        Matrix<T> result_mat = Matrix<T>(n, n);
        for (std::size_t i = 0; i < n * n; i += n + 1) {
            result_mat.data[i] = 1;
        }
        return result_mat;
    }

    template<std::floating_point T>
    Matrix<T>::Matrix(std::size_t rows, std::size_t cols) : rows(rows), cols(cols), data(rows * cols) {}

    template<std::floating_point T>
    Matrix<T>::Matrix(std::size_t rows, std::size_t cols, std::initializer_list<T> init) : rows(rows), cols(cols), data(init) {}

    template<std::floating_point T>
    template<std::input_iterator InputIt>
    Matrix<T>::Matrix(std::size_t rows, std::size_t cols, InputIt first, InputIt last) : rows(rows), cols(cols), data(first, last) {}

    template<std::floating_point T>
    Matrix<T>::Matrix(const Vector<T> &v) : rows(v.size()), cols(1), data(v.begin(), v.end()) {}

    template<std::floating_point T>
    std::pair<std::size_t, std::size_t> Matrix<T>::shape() const noexcept {
        return {rows, cols};
    }

    template<std::floating_point T>
    std::vector<T>::const_iterator Matrix<T>::begin() const noexcept {
        return data.begin();
    }

    template<std::floating_point T>
    std::vector<T>::iterator Matrix<T>::begin() noexcept {
        return data.begin();
    }

    template<std::floating_point T>
    std::vector<T>::const_iterator Matrix<T>::end() const noexcept {
        return data.end();
    }

    template<std::floating_point T>
    std::vector<T>::iterator Matrix<T>::end() noexcept {
        return data.end();
    }

    template<std::floating_point T>
    T Matrix<T>::determinant() const {}

    template<std::floating_point T>
    Matrix<T> Matrix<T>::tranposed() const {}

    template<std::floating_point T>
    Matrix<T> &Matrix<T>::transpose() {}

    template<std::floating_point T>
    Matrix<T> Matrix<T>::inverse() const {}

    template<std::floating_point T>
    Matrix<T> &Matrix<T>::invert() {}

    template<std::floating_point T>
    Matrix<T> Matrix<T>::hadamard(const Matrix<T> &other) const {
        Matrix<T> result_mat(rows, cols);
        std::transform(
            data.begin(), data.end(),
            other.data.begin(),
            result_mat.data.begin(),
            [](T a, T b) { return a * b; }
        );
        return result_mat;
    }

    template<std::floating_point T>
    Matrix<T> &Matrix<T>::hadamard_inplace(const Matrix<T> &other) {
        std::transform(
            data.begin(), data.end(),
            other.data.begin(),
            data.begin(),
            [](T a, T b) { return a * b; }
        );
        return *this;
    }

    template<std::floating_point T>
    Vector<T> Matrix<T>::solve(const Vector<T> &b) const {}

    template<std::floating_point T>
    T Matrix<T>::at(std::size_t i, std::size_t j) const {
        return data.at(i * cols + j);
    }

    template<std::floating_point T>
    T &Matrix<T>::at(std::size_t i, std::size_t j) {
        return data.at(i * cols + j);
    }

    template<std::floating_point T>
    Matrix<T> Matrix<T>::operator-() const {
        Matrix<T> result_mat(rows, cols);
        std::transform(
            data.begin(), data.end(),
            result_mat.data.begin(),
            [](T x) { return -x; }
        );
        return result_mat;
    }

    template<std::floating_point T>
    Matrix<T> Matrix<T>::operator+(const Matrix<T> &other) const {
        Matrix<T> result_mat(rows, cols);
        std::transform(
            data.begin(), data.end(),
            other.data.begin(),
            result_mat.data.begin(),
            [](T a, T b) { return a + b; }
        );
        return result_mat;
    }

    template<std::floating_point T>
    Matrix<T> &Matrix<T>::operator+=(const Matrix<T> &other) {
        std::transform(
            data.begin(), data.end(),
            other.data.begin(),
            data.begin(),
            [](T a, T b) { return a + b; }
        );
        return *this;
    }

    template<std::floating_point T>
    Matrix<T> Matrix<T>::operator-(const Matrix<T> &other) const {
        Matrix<T> result_mat(rows, cols);
        std::transform(
            data.begin(), data.end(),
            other.data.begin(),
            result_mat.data.begin(),
            [](T a, T b) { return a - b; }
        );
        return result_mat;
    }

    template<std::floating_point T>
    Matrix<T> &Matrix<T>::operator-=(const Matrix<T> &other) {
        std::transform(
            data.begin(), data.end(),
            other.data.begin(),
            data.begin(),
            [](T a, T b) { return a - b; }
        );
        return *this;
    }

    template<std::floating_point T>
    Matrix<T> Matrix<T>::operator*(T k) const {
        Matrix<T> result_mat(rows, cols);
        std::transform(
            data.begin(), data.end(),
            result_mat.data.begin(),
            [k](T x) { return x * k; }
        );
        return result_mat;
    }

    template<std::floating_point T>
    Vector<T> Matrix<T>::operator*(const Vector<T> &v) const {}

    template<std::floating_point T>
    Matrix<T> Matrix<T>::operator*(const Matrix<T> &other) const {}

    template<std::floating_point T>
    Matrix<T> operator*(T k, const Matrix<T> &m) {
        Matrix<T> result_mat(m.rows, m.cols);
        std::transform(
            m.data.begin(), m.data.end(),
            result_mat.data.begin(),
            [k](T x) { return x * k; }
        );
        return result_mat;
    }

    template<std::floating_point T>
    Matrix<T> operator*(const Vector<T> &v, const Matrix<T> &m) {}

    template<std::floating_point T>
    Matrix<T> &Matrix<T>::operator*=(T k) {
        std::transform(
            data.begin(), data.end(),
            data.begin(),
            [k](T x) { return x * k; }
        );
        return *this;
    }

    template<std::floating_point T>
    Matrix<T> &Matrix<T>::operator*=(const Matrix<T> &other) {}

    template<std::floating_point T>
    Matrix<T> Matrix<T>::operator/(T k) const {
        Matrix<T> result_mat(rows, cols);
        std::transform(
            data.begin(), data.end(),
            result_mat.data.begin(),
            [k](T x) { return x / k; }
        );
        return result_mat;
    }

    template<std::floating_point T>
    Matrix<T> &Matrix<T>::operator/=(T k) {
        std::transform(
            data.begin(), data.end(),
            data.begin(),
            [k](T x) { return x / k; }
        );
        return *this;
    }

    template<std::floating_point T>
    bool Matrix<T>::operator==(const Matrix<T> &other) const {
        if (rows != other.rows || cols != other.cols) {
            return false;
        }

        return std::equal(
            data.begin(), data.end(),
            other.data.begin(),
            [](T a, T b) { return std::abs(a - b) < tolerance; }
        );
    }

    template<std::floating_point T>
    T Matrix<T>::operator()(std::size_t i, std::size_t j) const noexcept {
        return data[i * cols + j];
    }

    template<std::floating_point T>
    T &Matrix<T>::operator()(std::size_t i, std::size_t j) noexcept {
        return data[i * cols + j];
    }
}
