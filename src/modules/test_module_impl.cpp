//
// Created by richard on 1/22/26.
//

module test_module;

#include <cmath>

int add(int a, int b) {
    return a + b;
}

double square(double x) {
    return x * x;
}

double Point::distance(const Point& other) const {
    return sqrt(square(x - other.x) + square(y - other.y));
}