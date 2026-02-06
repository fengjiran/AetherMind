//
// Created by richard on 2/6/26.
//

#include "ammalloc/common.h"

#include <cctype>
#include <cstdlib>
#include <limits>
#include <string_view>

namespace aethermind {
namespace details {

size_t ParseSize(const char* str) {
    if (!str || *str == '\0') {
        return 0;
    }
    char* end_ptr = nullptr;
    // 1. 解析数字部分
    // strtoull 会自动跳过前导空格，并处理数字
    auto value = strtoul(str, &end_ptr, 10);

    // 如果 end_ptr 等于 str，说明没有读到任何数字
    if (str == end_ptr) {
        return 0;
    }

    // 2. 跳过数字和单位之间的空格 (例如 "64 KB")
    while (*end_ptr && std::isspace(static_cast<unsigned char>(*end_ptr))) {
        ++end_ptr;
    }

    // 3. 检查单位
    if (*end_ptr == '\0') {
        return value;// 无单位，默认为 Bytes
    }

    size_t multiplier = 1;
    switch (std::tolower(static_cast<unsigned char>(*end_ptr))) {
        case 'b':
            multiplier = 1;
            break;
        case 'k':
            multiplier = 1ULL << 10;
            break;// KB
        case 'm':
            multiplier = 1ULL << 20;
            break;// MB
        case 'g':
            multiplier = 1ULL << 30;
            break;// GB
        case 't':
            multiplier = 1ULL << 40;
            break;// TB
        default:
            return value;// 未知单位，按 Bytes 处理或报错
    }

    // 4. 溢出检查 (Overflow Check)
    // 检查 value * multiplier 是否会超过 size_t 的最大值
    if (value > std::numeric_limits<size_t>::max() / multiplier) {
        return std::numeric_limits<size_t>::max();
    }

    return multiplier * value;
}

bool ParseBool(const char* str) {
    if (!str) {
        return false;
    }

    std::string_view sv(str);
    // 1. 去除前导空格 (Trim Leading)
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front()))) {
        sv.remove_prefix(1);
    }

    // 2. 去除尾部空格 (Trim Trailing)
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) {
        sv.remove_suffix(1);
    }

    if (sv.empty()) {
        return false;
    }

    // 3. 快速检查 "1"
    if (sv == "1") {
        return true;
    }

    // 4. 定义大小写不敏感比较 Lambda
    auto is_equal = [](std::string_view a, std::string_view b) {
        if (a.size() != b.size()) {
            return false;
        }

        return std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
            return std::tolower(static_cast<unsigned char>(x)) == std::tolower(static_cast<unsigned char>(y));
        });
    };

    if (is_equal(sv, "true") ||
        is_equal(sv, "on") ||
        is_equal(sv, "yes")) {
        return true;
    }

    return false;
}

}// namespace details
}// namespace aethermind
