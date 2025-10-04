//
// Created by 赵丹 on 25-6-19.
//
#include "tensor.h"

#include <fmt/compile.h>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <string>
#include <tuple>

namespace aethermind {

enum class FormatType {
    Default,   // 'g' format (defaultfloat equivalent)
    Scientific,// 'e' format with precision 4
    Fixed      // 'f' format with precision 4
};

struct PrintFormat {
    double scale;
    int width;
    FormatType type;

    PrintFormat(double s, int w, FormatType t = FormatType::Default)
        : scale(s), width(w), type(t) {}
};

static PrintFormat get_print_format(const Tensor& t) {
    // using data_type = DataType2CPPType<DLDataTypeCode::kFloat, 32, 1>::type;
    auto size = t.numel();
    if (size == 0) {
        return {1.0, 0};
    }

    bool int_mod = true;
    auto data = t.const_data_ptr<double>();
    for (int64_t i = 0; i < size; ++i) {
        auto z = data[i];
        if (std::isfinite(z)) {
            if (z != std::ceil(z)) {
                int_mod = false;
                break;
            }
        }
    }

    int64_t offset = 0;
    while (offset < size && !std::isfinite(data[offset])) {
        ++offset;
    }

    double exp_min = 1;
    double exp_max = 1;
    if (offset != size) {
        exp_min = std::fabs(data[offset]);
        exp_max = std::fabs(data[offset]);
        for (int64_t i = offset; i < size; ++i) {
            double z = std::fabs(data[i]);
            if (std::isfinite(z)) {
                exp_min = std::min(exp_min, z);
                exp_max = std::max(exp_max, z);
            }
        }

        if (exp_min != 0) {
            exp_min = std::floor(std::log10(exp_min)) + 1;
        } else {
            exp_min = 1;
        }

        if (exp_max != 0) {
            exp_max = std::floor(std::log10(exp_max)) + 1;
        } else {
            exp_max = 1;
        }
    }

    double scale = 1;
    int width = 11;
    if (int_mod) {
        if (exp_max > 9) {
            width = 11;
            return {scale, width, FormatType::Scientific};
        }
        width = static_cast<int>(exp_max) + 1;
        return {scale, width, FormatType::Default};
    }

    if (exp_max - exp_min > 4) {
        width = 11;
        if (std::fabs(exp_max) > 99 || std::fabs(exp_min) > 99) {
            ++width;
        }
        return {scale, width, FormatType::Scientific};
    }

    if (exp_max > 5 || exp_max < 0) {
        width = 7;
        scale = std::pow(10, exp_max - 1);
        return {scale, width, FormatType::Fixed};
    }

    width = exp_max == 0 ? 7 : (static_cast<int>(exp_max) + 6);
    return {scale, width, FormatType::Fixed};
}

// Precompiled format specs
static constexpr auto FMT_G = FMT_COMPILE("{:>{}g}");
static constexpr auto FMT_E4 = FMT_COMPILE("{:>{}.4e}");
static constexpr auto FMT_F4 = FMT_COMPILE("{:>{}.4f}");

// Print a single value directly into the stream buffer with no temporaries
static void print_value(std::ostream& stream, double v, const PrintFormat& pf) {
    auto it = std::ostreambuf_iterator<char>(stream);
    double val = v / pf.scale;
    switch (pf.type) {
        case FormatType::Default:
            fmt::format_to(it, FMT_G, val, pf.width);
            break;
        case FormatType::Scientific:
            fmt::format_to(it, FMT_E4, val, pf.width);
            break;
        case FormatType::Fixed:
            fmt::format_to(it, FMT_F4, val, pf.width);
            break;
    }
}

}// namespace atp