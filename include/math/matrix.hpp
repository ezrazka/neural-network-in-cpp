#pragma once

#include "math/vector.hpp"

#include <concepts>
#include <initializer_list>
#include <vector>

namespace math {
    template<std::floating_point T>
    class Matrix {
    public:
        static Matrix identity(std::size_t n);

        Matrix(std::size_t rows, std::size_t cols);
        Matrix(std::size_t rows, std::size_t cols, std::initializer_list<T> init);
        template<std::input_iterator InputIt>
        Matrix(std::size_t rows, std::size_t cols, InputIt first, InputIt last);
        Matrix(const Vector<T> &v);

        std::pair<std::size_t, std::size_t> shape() const noexcept;
        std::vector<T>::const_iterator begin() const noexcept;
        std::vector<T>::iterator begin() noexcept;
        std::vector<T>::const_iterator end() const noexcept;
        std::vector<T>::iterator end() noexcept;

        T determinant() const;
        Matrix tranposed() const;
        Matrix &transpose();
        Matrix inverse() const;
        Matrix &invert();

        Matrix hadamard(const Matrix &other) const;
        Matrix &hadamard_inplace(const Matrix &other);
        Vector<T> solve(const Vector<T> &b) const;

        T at(std::size_t i, std::size_t j) const;
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

        T operator()(std::size_t i, std::size_t j) const noexcept;
        T &operator()(std::size_t i, std::size_t j) noexcept;
        
        T operator[](std::size_t i) const noexcept;
        T &operator[](std::size_t i) noexcept;

    private:
        static constexpr T tolerance = []() {
            if constexpr (sizeof(T) <= 4) return static_cast<T>(1e-5);
            return static_cast<T>(1e-9);
        }();

        std::size_t rows;
        std::size_t cols;
        std::vector<T> data;
    };
}
