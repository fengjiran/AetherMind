//
// Created by richard on 1/22/26.
//

export module test_module;

export int add(int a, int b);

export struct Point {
    int x;
    int y;

    [[nodiscard]] double distance(const Point& other) const;
};
