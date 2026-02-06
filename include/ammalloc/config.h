//
// Created by richard on 2/6/26.
//

#ifndef AETHERMIND_AMMALLOC_CONFIG_H
#define AETHERMIND_AMMALLOC_CONFIG_H

#include "ammalloc/common.h"
#include "macros.h"

#include <cctype>
#include <cstddef>
#include <cstdlib>

namespace aethermind {

// ===========================================================================
// 1. 系统与硬件架构配置 (System & Architecture)
// ===========================================================================
struct SystemConfig {
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
struct SizeConfig {
    // max thread cache size(32KB)
    constexpr static size_t MAX_TC_SIZE = 32 * 1024;
    // For size class index
    constexpr static int kStepsPerGroup = 4;
    constexpr static int kStepShift = 2;
    constexpr static size_t kSmallSizeThreshold = 1024;
};

// ===========================================================================
// 3. 页缓存与后端配置 (Page Cache & Backend)
// ===========================================================================
struct PageConfig {
    // Maximum number of consecutive pages managed by Page Cache
    // (to avoid excessively large Spans)
    constexpr static size_t MAX_PAGE_NUM = 128;
    constexpr static size_t RADIX_BITS = 9;
    constexpr static size_t RADIX_NODE_SIZE = 1 << RADIX_BITS;
    constexpr static size_t RADIX_MASK = RADIX_NODE_SIZE - 1;
};


class RuntimeConfig {
public:
    static RuntimeConfig& GetInstance() {
        static RuntimeConfig instance;
        return instance;
    }

    AM_NODISCARD size_t MaxTCSize() const {
        return max_tc_size_;
    }

    AM_NODISCARD bool UseMapPopulate() const {
        return use_map_populate;
    }

private:
    RuntimeConfig() {
        InitFromEnv();
    }

    void InitFromEnv() {
        if (const char* env = std::getenv("AM_TC_SIZE")) {
            if (const auto val = details::ParseSize(env); val > 0) {
                max_tc_size_ = val < SizeConfig::MAX_TC_SIZE ? val : SizeConfig::MAX_TC_SIZE;
            }
        }

        if (const char* env = std::getenv("AM_USE_MAP_POPULATE")) {
            use_map_populate = details::ParseBool(env);
        }
    }

    size_t max_tc_size_ = SizeConfig::MAX_TC_SIZE;
    bool use_map_populate = false;// default: Lazy Allocation
};

}// namespace aethermind

#endif//AETHERMIND_AMMALLOC_CONFIG_H
