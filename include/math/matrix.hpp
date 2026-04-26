#pragma once

#include "math/random.hpp"
#include "math/vector.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace math {
    template<std::floating_point T>
    class Matrix {
    public:
        static Matrix identity(std::size_t rows, std::size_t cols);
        static Matrix random(std::size_t rows, std::size_t cols);
        static Matrix random(std::size_t rows, std::size_t cols, T min, T max);

        Matrix() = default;
        Matrix(std::size_t rows, std::size_t cols);
        template<std::input_iterator InputIt>
        Matrix(std::size_t rows, std::size_t cols, InputIt first, InputIt last);
        Matrix(const Vector<T> &v);
        Matrix(std::initializer_list<std::initializer_list<T>> init);
        Matrix(std::vector<std::vector<T>> init);

        std::size_t size() const noexcept;
        std::size_t rows() const noexcept;
        std::size_t cols() const noexcept;
        std::vector<T>::const_iterator begin() const noexcept;
        std::vector<T>::iterator begin() noexcept;
        std::vector<T>::const_iterator end() const noexcept;
        std::vector<T>::iterator end() noexcept;

        Matrix transposed() const;
        Matrix &transpose();

        Matrix hadamard(const Matrix &other) const;
        Matrix &hadamard_inplace(const Matrix &other);

        const T &at(std::size_t i, std::size_t j) const;
        T &at(std::size_t i, std::size_t j);

        Matrix operator-() const;
        
        Matrix operator+(const Matrix &other) const;
        Matrix &operator+=(const Matrix &other);
        Matrix operator-(const Matrix &other) const;
        Matrix &operator-=(const Matrix &other);

        Matrix operator*(T k) const;
        Vector<T> operator*(const Vector<T> &v) const;
        Matrix operator*(const Matrix<T> &other) const;
        friend Matrix operator*(T k, const Matrix<T> &m);
        friend Matrix operator*(const Vector<T> &v, const Matrix<T> &m);
        Matrix &operator*=(T k);
        Matrix &operator*=(const Matrix<T> &other);
        Matrix operator/(T k) const;
        Matrix &operator/=(T k);
        
        bool operator==(const Matrix &other) const;

        const T &operator()(std::size_t i, std::size_t j) const noexcept;
        T &operator()(std::size_t i, std::size_t j) noexcept;
        
        const T &operator[](std::size_t i) const noexcept;
        T &operator[](std::size_t i) noexcept;

    private:
        static constexpr T tolerance = []() {
            if constexpr (sizeof(T) <= 4) return T{1e-5};
            return T{1e-9};
        }();
        static constexpr std::size_t block_size = []() {
            if constexpr (sizeof(T) <= 4) return std::size_t{96};
            if constexpr (sizeof(T) <= 8) return std::size_t{64};
            return std::numeric_limits<std::size_t>::max() / 2;
        }();

        static void throw_non_rectangular();
        static void throw_zero_division();
        static void throw_size_mismatch(std::size_t n, std::size_t m);
        static void throw_inner_dimension_mismatch(std::size_t cols, std::size_t rows);
        static void throw_shape_mismatch(std::size_t rows_1, std::size_t cols_1, std::size_t rows_2, std::size_t cols_2);

        std::size_t rows_;
        std::size_t cols_;
        std::vector<T> data;

        Matrix &transpose_square();
    };

    template<std::floating_point T>
    Matrix<T> Matrix<T>::identity(std::size_t rows, std::size_t cols) {
        Matrix<T> result_mat(rows, cols);
        for (std::size_t i = 0; i < std::min(rows, cols); i++) {
            result_mat(i, i) = T{1};
        }
        return result_mat;
    }

    template<std::floating_point T>
    Matrix<T> Matrix<T>::random(std::size_t rows, std::size_t cols) {
        return random(rows, cols, T{0}, T{1});
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
    Matrix<T>::Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows)
        , cols_(cols)
        , data(rows * cols) {}

    template<std::floating_point T>
    template<std::input_iterator InputIt>
    Matrix<T>::Matrix(std::size_t rows, std::size_t cols, InputIt first, InputIt last)
        : rows_(rows)
        , cols_(cols)
        , data(rows * cols)
    {
        std::size_t index = 0;
        for (; first != last && index < rows_ * cols_; first++) {
            data[index] = *first;
            index++;
        }

        if (index != rows_ * cols_) {
            throw_size_mismatch(index, rows_ * cols_);
        }
    }

    template<std::floating_point T>
    Matrix<T>::Matrix(const Vector<T> &v)
        : rows_(v.size())
        , cols_(1)
        , data(v.begin(), v.end()) {}

    template<std::floating_point T>
    Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> init)
        : rows_(init.size())
        , cols_(init.size() > 0 ? init.begin()->size() : 0)
        , data(rows_ * cols_)
    {
        std::size_t index = 0;
        for (const std::initializer_list<T> &row : init) {
            for (T value : row) {
                if (row.size() != cols_) {
                    throw_non_rectangular();
                }

                data[index] = value;
                index++;
            }
        }

        if (index != rows_ * cols_) {
            throw_size_mismatch(index, rows_ * cols_);
        }
    }

    template<std::floating_point T>
    Matrix<T>::Matrix(std::vector<std::vector<T>> init)
        : rows_(init.size())
        , cols_(init.size() > 0 ? init.begin()->size() : 0)
        , data(rows_ * cols_)
    {
        std::size_t index = 0;
        for (const std::vector<T> &row : init) {
            if (row.size() != cols_) {
                throw_non_rectangular();
            }

            for (T value : row) {
                data[index] = value;
                index++;
            }
        }

        if (index != rows_ * cols_) {
            throw_size_mismatch(index, rows_ * cols_);
        }
    }

    template<std::floating_point T>
    std::size_t Matrix<T>::size() const noexcept {
        return data.size();
    }

    template<std::floating_point T>
    std::size_t Matrix<T>::rows() const noexcept {
        return rows_;
    }

    template<std::floating_point T>
    std::size_t Matrix<T>::cols() const noexcept {
        return cols_;
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
        Matrix<T> result_mat(cols_, rows_);
        for (std::size_t ii = 0; ii < rows_; ii += block_size) {
            for (std::size_t jj = 0; jj < cols_; jj += block_size) {
                for (std::size_t i = ii; i < std::min(ii + block_size, rows_); i++) {
                    for (std::size_t j = jj; j < std::min(jj + block_size, cols_); j++) {
                        result_mat(j, i) = (*this)(i, j);
                    }
                }
            }
        }
        return result_mat;
    }

    template<std::floating_point T>
    Matrix<T> &Matrix<T>::transpose() {
        if (rows_ == cols_) {
            return this->transpose_square();
        }
        return *this = this->transposed();
    }

    template<std::floating_point T>
    Matrix<T> Matrix<T>::hadamard(const Matrix<T> &other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw_shape_mismatch(rows_, cols_, other.rows_, other.cols_);
        }

        Matrix<T> result_mat(rows_, cols_);
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
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw_shape_mismatch(rows_, cols_, other.rows_, other.cols_);
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
    const T &Matrix<T>::at(std::size_t i, std::size_t j) const {
        return data.at(i * cols_ + j);
    }

    template<std::floating_point T>
    T &Matrix<T>::at(std::size_t i, std::size_t j) {
        return data.at(i * cols_ + j);
    }

    template<std::floating_point T>
    Matrix<T> Matrix<T>::operator-() const {
        Matrix<T> result_mat(rows_, cols_);
        std::transform(
            data.begin(), data.end(),
            result_mat.data.begin(),
            [](T x) { return -x; }
        );
        return result_mat;
    }

    template<std::floating_point T>
    Matrix<T> Matrix<T>::operator+(const Matrix<T> &other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw_shape_mismatch(rows_, cols_, other.rows_, other.cols_);
        }

        Matrix<T> result_mat(rows_, cols_);
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
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw_shape_mismatch(rows_, cols_, other.rows_, other.cols_);
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
    Matrix<T> Matrix<T>::operator-(const Matrix<T> &other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw_shape_mismatch(rows_, cols_, other.rows_, other.cols_);
        }

        Matrix<T> result_mat(rows_, cols_);
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
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            throw_shape_mismatch(rows_, cols_, other.rows_, other.cols_);
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
    Matrix<T> Matrix<T>::operator*(T k) const {
        Matrix<T> result_mat(rows_, cols_);
        std::transform(
            data.begin(), data.end(),
            result_mat.data.begin(),
            [k](T x) { return x * k; }
        );
        return result_mat;
    }

    template<std::floating_point T>
    Vector<T> Matrix<T>::operator*(const Vector<T> &v) const {
        if (cols_ != v.size()) {
            throw_inner_dimension_mismatch(cols_, v.size());
        }

        Vector<T> result_vec(rows_);
        for (std::size_t ii = 0; ii < rows_; ii += block_size) {
            for (std::size_t jj = 0; jj < cols_; jj += block_size) {
                for (std::size_t i = ii; i < std::min(ii + block_size, rows_); i++) {
                    result_vec[i] += std::inner_product(
                        this->begin() + i * cols_ + jj,
                        this->begin() + i * cols_ + std::min(jj + block_size, cols_),
                        v.begin() + jj,
                        T{0}
                    );
                }
            }
        }
        return result_vec;
    }

    template<std::floating_point T>
    Matrix<T> Matrix<T>::operator*(const Matrix<T> &other) const {
        if (cols_ != other.rows_) {
            throw_inner_dimension_mismatch(cols_, other.rows_);
        }

        Matrix result_mat(rows_, other.cols_);
        Matrix<T> other_T = other.transposed();
        for (std::size_t ii = 0; ii < rows_; ii += block_size) {
            for (std::size_t jj = 0; jj < other.cols_; jj += block_size) {
                for (std::size_t kk = 0; kk < cols_; kk += block_size) {
                    for (std::size_t i = ii; i < std::min(ii + block_size, rows_); i++) {
                        for (std::size_t j = jj; j < std::min(jj + block_size, other.cols_); j++) {
                            result_mat(i, j) += std::inner_product(
                                this->begin() + i * cols_ + kk,
                                this->begin() + i * cols_ + std::min(kk + block_size, cols_),
                                other_T.begin() + j * other_T.cols_ + kk,
                                T{0}
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
        Matrix<T> result_mat(m.rows_, m.cols_);
        std::transform(
            m.data.begin(), m.data.end(),
            result_mat.data.begin(),
            [k](T x) { return x * k; }
        );
        return result_mat;
    }

    template<std::floating_point T>
    Matrix<T> operator*(const Vector<T> &v, const Matrix<T> &m) {
        if (m.rows_ != 1) {
            Matrix<T>::throw_inner_dimension_mismatch(1, m.rows_);
        }

        Matrix result_mat(v.size(), m.cols_);
        for (std::size_t i = 0; i < v.size(); i++) {
            for (std::size_t j = 0; j < m.cols_; j++) {
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
        if (k < tolerance) {
            throw_zero_division();
        }

        Matrix<T> result_mat(rows_, cols_);
        T k_inv = T{1} / k;
        std::transform(
            data.begin(), data.end(),
            result_mat.data.begin(),
            [k_inv](T x) { return x * k_inv; }
        );
        return result_mat;
    }

    template<std::floating_point T>
    Matrix<T> &Matrix<T>::operator/=(T k) {
        if (k < tolerance) {
            throw_zero_division();
        }

        T k_inv = T{1} / k;
        std::transform(
            data.begin(), data.end(),
            data.begin(),
            [k_inv](T x) { return x * k_inv; }
        );
        return *this;
    }

    template<std::floating_point T>
    bool Matrix<T>::operator==(const Matrix<T> &other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            return false;
        }
        return std::equal(
            data.begin(), data.end(),
            other.data.begin(),
            [](T a, T b) { return std::abs(a - b) < tolerance; }
        );
    }

    template<std::floating_point T>
    const T &Matrix<T>::operator()(std::size_t i, std::size_t j) const noexcept {
        return data[i * cols_ + j];
    }

    template<std::floating_point T>
    T &Matrix<T>::operator()(std::size_t i, std::size_t j) noexcept {
        return data[i * cols_ + j];
    }

    template<std::floating_point T>
    const T &Matrix<T>::operator[](std::size_t i) const noexcept {
        return data[i];
    }

    template<std::floating_point T>
    T &Matrix<T>::operator[](std::size_t i) noexcept {
        return data[i];
    }

    template<std::floating_point T>
    void Matrix<T>::throw_zero_division() {
        throw std::domain_error("Division by zero");
    }

    template<std::floating_point T>
    void Matrix<T>::throw_non_rectangular() {
        throw std::invalid_argument("Input contains inconsistent row sizes");
    }

    template<std::floating_point T>
    void Matrix<T>::throw_size_mismatch(std::size_t n, std::size_t m) {
        throw std::invalid_argument(
            std::format("Size mismatch: {} and {}", n, m)
        );
    }

    template<std::floating_point T>
    void Matrix<T>::throw_inner_dimension_mismatch(std::size_t cols, std::size_t rows) {
        throw std::invalid_argument(
            std::format("Inner dimension mismatch: {} and {}", cols, rows)
        );
    }

    template<std::floating_point T>
    void Matrix<T>::throw_shape_mismatch(std::size_t rows_1, std::size_t cols_1, std::size_t rows_2, std::size_t cols_2) {
        throw std::invalid_argument(
            std::format("Shape mismatch: ({}, {}) and ({}, {})", rows_1, cols_1, rows_2, cols_2)
        );
    }

    template<std::floating_point T>
    Matrix<T> &Matrix<T>::transpose_square() {
        for (std::size_t ii = 0; ii < rows_; ii += block_size) {
            for (std::size_t jj = 0; jj < cols_; jj += block_size) {
                if (ii != jj) {
                    for (std::size_t i = ii; i < std::min(ii + block_size, rows_); i++) {
                        for (std::size_t j = jj; j < std::min(jj + block_size, cols_); j++) {
                            std::swap((*this)(i, j), (*this)(j, i));
                        }
                    }
                } else {
                    for (std::size_t i = ii; i < std::min(ii + block_size, rows_); i++) {
                        for (std::size_t j = i + 1; j < std::min(jj + block_size, cols_); j++) {
                            std::swap((*this)(i, j), (*this)(j, i));
                        }
                    }
                }
            }
        }
        return *this;
    }
}
