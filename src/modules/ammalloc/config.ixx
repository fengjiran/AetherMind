//
// Created by richard on 2/5/26.
//
module;

#include "macros.h"

#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <limits>

export module ammalloc_config;

namespace aethermind {

// ===========================================================================
// 1. 系统与硬件架构配置 (System & Architecture)
// ===========================================================================
export struct SystemConfig {
    // page size (default: 4KB)
    constexpr static size_t PAGE_SIZE = 4096;
    // page shift
    constexpr static size_t PAGE_SHIFT = 12;
    // huge page size(2MB)
    constexpr static size_t HUGE_PAGE_SIZE = 2 * 1024 * 1024;
    // cache line size
    constexpr static size_t CACHE_LINE_SIZE = 64;
    // bitmap bits
    constexpr static size_t BITMAP_BITS = 64;
    // alignment
    constexpr static size_t ALIGNMENT = 16;
};

// ===========================================================================
// 2. 尺寸分级与前端配置 (Size Class & Frontend)
// ===========================================================================
export struct SizeConfig {
    // max thread cache size(32KB)
    constexpr static size_t MAX_TC_SIZE = 32 * 1024;
    // For size class index
    constexpr static int kStepsPerGroup = 4;
    constexpr static int kStepShift = 2;
};

// ===========================================================================
// 3. 页缓存与后端配置 (Page Cache & Backend)
// ===========================================================================
export struct PageConfig {
    // Maximum number of consecutive pages managed by Page Cache
    // (to avoid excessively large Spans)
    constexpr static size_t MAX_PAGE_NUM = 128;
    constexpr static size_t RADIX_BITS = 9;
    constexpr static size_t RADIX_NODE_SIZE = 1 << RADIX_BITS;
    constexpr static size_t RADIX_MASK = RADIX_NODE_SIZE - 1;
};

export class RuntimeConfig {
public:
    static RuntimeConfig& GetInstance() {
        static RuntimeConfig instance;
        return instance;
    }

    AM_NODISCARD size_t MaxTCSize() const {
        return max_tc_size_;
    }

    /**
     * @brief 解析带单位的内存大小字符串
     *
     * 支持格式示例：
     * - "1024"      -> 1024
     * - "64KB"      -> 64 * 1024
     * - "16 M"      -> 16 * 1024 * 1024 (允许空格)
     * - "1gb"       -> 1 * 1024 * 1024 * 1024 (忽略大小写)
     *
     * @param str 环境变量字符串
     * @return size_t 解析后的字节数。解析失败返回 0。
     */
    static size_t ParseSize(const char* str) {
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

private:
    RuntimeConfig() {
        InitFromEnv();
    }

    void InitFromEnv() {
        if (const char* env = std::getenv("AM_TC_SIZE")) {
            if (const auto val = ParseSize(env); val > 0) {
                max_tc_size_ = val < SizeConfig::MAX_TC_SIZE ? val : SizeConfig::MAX_TC_SIZE;
            }
        }
    }

    size_t max_tc_size_ = SizeConfig::MAX_TC_SIZE;
};

}// namespace aethermind