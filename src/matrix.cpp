#include "math/matrix.hpp"

#include "math/random.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <initializer_list>
#include <numeric>
#include <random>
#include <vector>

namespace math {
    template<std::floating_point T>
    Matrix<T> Matrix<T>::identity(std::size_t rows, std::size_t cols) {
        Matrix<T> result_mat(rows, cols);
        for (std::size_t i = 0; i < std::min(rows, cols); i++) {
            result_mat(i, i) = 1;
        }
        return result_mat;
    }

    template<std::floating_point T>
    Matrix<T> Matrix<T>::random(std::size_t rows, std::size_t cols) {
        return random(rows, cols, static_cast<T>(0.0), static_cast<T>(1.0));
    }

    template<std::floating_point T>
    Matrix<T> Matrix<T>::random(std::size_t rows, std::size_t cols, T min, T max) {
        std::mt19937 &rng = detail::get_rng();
        std::uniform_real_distribution<T> dist(min, max);

        Matrix result_mat(rows, cols);
        std::generate(
            result_mat.begin(), result_mat.end(),
            [&dist, &rng]() { return dist(rng); }
        );
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
    Matrix<T> Matrix<T>::transposed() const {
        Matrix<T> result_mat(cols, rows);
        for (std::size_t ii = 0; ii < rows; ii += block_size) {
            for (std::size_t jj = 0; jj < cols; jj += block_size) {
                for (std::size_t i = ii; i < std::min(ii + block_size, rows); i++) {
                    for (std::size_t j = jj; j < std::min(jj + block_size, cols); j++) {
                        result_mat(j, i) = (*this)(i, j);
                    }
                }
            }
        }
        return result_mat;
    }

    template<std::floating_point T>
    Matrix<T> &Matrix<T>::transpose() {
        if (rows == cols) {
            return this->transpose_square();
        }
        return *this = this->transposed();
    }

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
    Vector<T> Matrix<T>::operator*(const Vector<T> &v) const {
        Vector<T> result_vec(rows);
        for (std::size_t ii = 0; ii < rows; ii += block_size) {
            for (std::size_t jj = 0; jj < cols; jj += block_size) {
                for (std::size_t i = ii; i < std::min(ii + block_size, rows); i++) {
                    result_vec[i] += std::inner_product(
                        this->begin() + i * cols + jj,
                        this->begin() + i * cols + std::min(jj + block_size, cols),
                        v.begin() + jj,
                        static_cast<T>(0.0)
                    );
                }
            }
        }
        return result_vec;
    }

    template<std::floating_point T>
    Matrix<T> Matrix<T>::operator*(const Matrix<T> &other) const {
        Matrix result_mat(rows, other.cols);
        Matrix<T> other_T = other.transposed();
        for (std::size_t ii = 0; ii < rows; ii += block_size) {
            for (std::size_t jj = 0; jj < other.cols; jj += block_size) {
                for (std::size_t kk = 0; kk < cols; kk += block_size) {
                    for (std::size_t i = ii; i < std::min(ii + block_size, rows); i++) {
                        for (std::size_t j = jj; j < std::min(jj + block_size, other.cols); j++) {
                            result_mat(i, j) += std::inner_product(
                                this->begin() + i * cols + kk,
                                this->begin() + i * cols + std::min(kk + block_size, cols),
                                other_T.begin() + j * other_T.cols + kk,
                                static_cast<T>(0.0)
                            );
                        }
                    }
                }
            }
        }
        return result_mat;
    }

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
    Matrix<T> operator*(const Vector<T> &v, const Matrix<T> &m) {
        Matrix result_mat(v.size(), m.cols);
        for (std::size_t i = 0; i < v.size(); i++) {
            for (std::size_t j = 0; j < m.cols; j++) {
                result_mat(i, j) = v[i] * m(0, j);
            }
        }
        return result_mat;
    }

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
    Matrix<T> &Matrix<T>::operator*=(const Matrix<T> &other) {
        return *this = *this * other;
    }

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

    template<std::floating_point T>
    Matrix<T> &Matrix<T>::transpose_square() {
        for (std::size_t ii = 0; ii < rows; ii += block_size) {
            for (std::size_t jj = 0; jj < cols; jj += block_size) {
                if (ii != jj) {
                    for (std::size_t i = ii; i < std::min(ii + block_size, rows); i++) {
                        for (std::size_t j = jj; j < std::min(jj + block_size, cols); j++) {
                            std::swap((*this)(i, j), (*this)(j, i));
                        }
                    }
                } else {
                    for (std::size_t i = ii; i < std::min(ii + block_size, rows); i++) {
                        for (std::size_t j = i + 1; j < std::min(jj + block_size, cols); j++) {
                            std::swap((*this)(i, j), (*this)(j, i));
                        }
                    }
                }
            }
        }
        return *this;
    }
}
