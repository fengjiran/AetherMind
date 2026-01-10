//
// Created by richard on 11/28/25.
//

#ifndef AETHERMIND_TEST_UTILS_H
#define AETHERMIND_TEST_UTILS_H

#include "container/string.h"
#include <chrono>
#include <iostream>
#include <random>
#include <utility>

namespace aethermind {

class Timer {
public:
    explicit Timer(String name) : name_(std::move(name)),
                                  start_(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        std::cout << name_ << " took " << duration.count() << " microseconds." << std::endl;
    }

private:
    String name_;
    std::chrono::high_resolution_clock::time_point start_;
};

inline std::vector<std::pair<int, int>> GenerateRandomData(size_t size) {
    std::vector<std::pair<int, int>> data;
    data.reserve(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, size * 10);
    for (size_t i = 0; i < size; ++i) {
        data.emplace_back(dist(gen), dist(gen));
    }
    return data;
}


}// namespace aethermind

#endif//AETHERMIND_TEST_UTILS_H
