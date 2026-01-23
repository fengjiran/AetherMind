//
// Created by richard on 1/22/26.
//
module;
#include <cmath>

module test_module;

namespace aethermind {
int add(int a, int b) {
    return a + b;
}

double square(double x) {
    return x * x;
}

double Point::distance(const Point& other) const {
    return sqrt(square(x - other.x) + square(y - other.y));
}
}// namespace aethermind
