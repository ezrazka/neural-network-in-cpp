#include "nn/sequential.hpp"
#include "nn/linear.hpp"
#include "nn/relu.hpp"
#include "nn/sgd.hpp"
#include "nn/cross_entropy_loss.hpp"
#include "math/matrix.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

template <typename T>
uint32_t read_u32(std::ifstream &f) {
    uint8_t b1, b2, b3, b4;
    f.read((char*) &b1, 1);
    f.read((char*) &b2, 1);
    f.read((char*) &b3, 1);
    f.read((char*) &b4, 1);
    return (uint32_t(b1) << 24) | (uint32_t(b2) << 16) | (uint32_t(b3) << 8) | uint32_t(b4);
}

template <typename T>
math::Matrix<T> load_images(const std::string &path) {
    std::ifstream f(path, std::ios::binary);

    uint32_t magic = read_u32<T>(f);
    uint32_t n = read_u32<T>(f);
    uint32_t rows = read_u32<T>(f);
    uint32_t cols = read_u32<T>(f);

    std::vector<std::vector<T>> x(n, std::vector<T>(rows * cols));

    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < rows * cols; j++) {
            unsigned char p;
            f.read((char*) &p, 1);
            x[i][j] = T(p) / 255.0;
        }
    }

    return math::Matrix<T>(x);
}

template <typename T>
math::Matrix<T> load_labels(const std::string &path) {
    std::ifstream f(path, std::ios::binary);

    uint32_t magic = read_u32<T>(f);
    uint32_t n = read_u32<T>(f);

    std::vector<std::vector<T>> y(n, std::vector<T>(10, 0));

    for (uint32_t i = 0; i < n; i++) {
        unsigned char label;
        f.read((char*) &label, 1);
        y[i][label] = 1;
    }

    return math::Matrix<T>(y);
}

int main() {
    try {
        using T = double;

        nn::Sequential<T> model(
            nn::Linear<T>(784, 10)
            // nn::Linear<T>(784, 128),
            // nn::ReLU<T>(),
            // nn::Linear<T>(128, 64),
            // nn::ReLU<T>(),
            // nn::Linear<T>(64, 10)
        );

        nn::CrossEntropyLoss<T> loss_fn;
        nn::SGD<T> optimizer(model.parameters(), 0.01);

        math::Matrix<T> X_train = load_images<T>("dataset/train-images.idx3-ubyte").transposed();
        math::Matrix<T> Y_train = load_labels<T>("dataset/train-labels.idx1-ubyte").transposed();

        math::Matrix<T> X_test = load_images<T>("dataset/t10k-images.idx3-ubyte").transposed();
        math::Matrix<T> Y_test = load_labels<T>("dataset/t10k-labels.idx1-ubyte").transposed();

        for (int epoch = 0; epoch < 100; epoch++) {
            const std::size_t batch_size = 64;
            const std::size_t n_samples = X_train.cols();
            T epoch_loss = 0;

            for (std::size_t batch_start = 0; batch_start < n_samples; batch_start += batch_size) {
                std::size_t batch_end = std::min(batch_start + batch_size, n_samples);
                std::size_t cur_batch_size = batch_end - batch_start;

                std::vector<std::vector<T>> x_batch_data(X_train.rows(), std::vector<T>(cur_batch_size));
                std::vector<std::vector<T>> y_batch_data(Y_train.rows(), std::vector<T>(cur_batch_size));

                for (std::size_t i = 0; i < X_train.rows(); i++)
                    for (std::size_t j = 0; j < cur_batch_size; j++)
                        x_batch_data[i][j] = X_train(i, batch_start + j);

                for (std::size_t i = 0; i < Y_train.rows(); i++)
                    for (std::size_t j = 0; j < cur_batch_size; j++)
                        y_batch_data[i][j] = Y_train(i, batch_start + j);

                math::Matrix<T> X_batch(x_batch_data);
                math::Matrix<T> Y_batch(y_batch_data);

                auto logits = model.forward(X_batch);
                T loss = loss_fn.forward(logits, Y_batch);
                auto grad = loss_fn.backward();
                model.zero_grad();
                model.backward(grad);
                optimizer.step();
                epoch_loss += loss;
            }

            std::cout << "Epoch " << epoch << " Loss: " << epoch_loss / (n_samples / batch_size) << "\n";
        }

        auto test_logits = model.forward(X_test);
        T test_loss = loss_fn.forward(test_logits, Y_test);

        std::cout << "\nTest Loss: " << test_loss << "\n";
        for (std::size_t i = 0; i < 10; i++) {
            std::cout << test_logits(i, 0) << "\n";
        }
        std::cout << X_train.rows() << " " << X_train.cols() << "\n";
        std::cout << X_test.rows() << " " << X_test.cols() << "\n";
        std::cout << Y_test.rows() << " " << Y_test.cols() << "\n";
        std::cout << Y_train.rows() << " " << Y_train.cols() << "\n";
        std::cout << test_logits.rows() << " " << test_logits.cols() << "\n";
    } catch (const std::exception &e) {
        std::cerr << "Exception caught: " << e.what() << "\n";
    }
}
