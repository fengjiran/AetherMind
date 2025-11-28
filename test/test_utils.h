//
// Created by richard on 11/28/25.
//

#ifndef AETHERMIND_TEST_UTILS_H
#define AETHERMIND_TEST_UTILS_H

#include "container/string.h"
#include <chrono>
#include <iostream>

namespace aethermind {

class Timer {
public:
    explicit Timer(const String& name)
        : name_(name),
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

}// namespace aethermind

#endif//AETHERMIND_TEST_UTILS_H
