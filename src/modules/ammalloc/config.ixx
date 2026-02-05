//
// Created by richard on 2/5/26.
//
module;

#include <cstddef>

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

class RuntimeConfig {
public:
    static RuntimeConfig& GetInstance() {
        static RuntimeConfig instance;
        return instance;
    }

private:
    RuntimeConfig() = default;


};

}// namespace aethermind