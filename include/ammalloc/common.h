//
// Created by richard on 2/6/26.
//

#ifndef AETHERMIND_MALLOC_COMMON_H
#define AETHERMIND_MALLOC_COMMON_H

#include "ammalloc/config.h"
#include "macros.h"
#include "utils/logging.h"

#include <immintrin.h>

namespace aethermind {
namespace details {

/**
 * @brief Aligns 'size' up to the specified 'align'.
 * @return The smallest multiple of 'align' that is greater than or equal to 'size'.
 */
AM_NODISCARD constexpr size_t AlignUp(size_t size,
                                      size_t align = SystemConfig::ALIGNMENT) noexcept {
    if (size == 0) AM_UNLIKELY {
            return align;
        }
    // Optimization for power-of-two alignments (the most common case)
    if (std::has_single_bit(align)) AM_LIKELY {
            return (size + align - 1) & ~(align - 1);
        }
    // Fallback for non-power-of-two alignments
    return (size + align - 1) / align * align;
}

/**
 * @brief Maps a raw memory pointer to its global page index.
 *
 * This function calculates the page number by dividing the memory address
 * by the system page size. It is a critical path component for PageMap
 * and Span lookups.
 *
 * @param ptr The raw pointer to be converted.
 * @return The corresponding page number (address >> shift).
 *
 * @note Performance:
 * - Constant-time O(1) complexity.
 * - Uses `if constexpr` to eliminate division overhead at compile-time.
 * - If PAGE_SIZE is a power of two, this lowers to a single bitwise SHR instruction.
 */
AM_NODISCARD inline size_t PtrToPageIdx(void* ptr) noexcept {
    // 1. Cast pointer to integer. Note: This limits 'true' constexpr usage
    // but enables massive runtime inlining optimizations.
    const auto addr = reinterpret_cast<uintptr_t>(ptr);
    // 2. Static dispatch for page size alignment.
    if constexpr (std::has_single_bit(SystemConfig::PAGE_SIZE)) {
        // Optimization: Address / 2^n -> Address >> n
        constexpr size_t shift = std::countr_zero(SystemConfig::PAGE_SIZE);
        return addr >> shift;
    } else {
        // Fallback for non-standard page sizes.
        return addr / SystemConfig::PAGE_SIZE;
    }
}

AM_NODISCARD inline void* PageIDToPtr(size_t page_idx) noexcept {
    if constexpr (std::has_single_bit(SystemConfig::PAGE_SIZE)) {
        constexpr size_t shift = std::countr_zero(SystemConfig::PAGE_SIZE);
        return reinterpret_cast<void*>(page_idx << shift);
    } else {
        return reinterpret_cast<void*>(page_idx * SystemConfig::PAGE_SIZE);
    }
}

inline void CPUPause() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    // x86 环境：_mm_pause 是最稳妥的，由编译器映射为 PAUSE 指令
    _mm_pause();
#elif defined(__aarch64__) || defined(_M_ARM64)
    // ARM 环境：使用 ISB 或 YIELD
    __asm__ volatile("yield" ::: "memory");
#else
    // 其他架构：简单的空操作，防止编译器把循环优化掉
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
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
size_t ParseSize(const char* str);

/**
 * @brief 解析环境变量中的布尔值
 *
 * 支持的 Truthy 值 (不区分大小写，忽略首尾空格):
 * - "1"
 * - "true"
 * - "on"
 * - "yes"
 *
 * 其他所有值均返回 false。
 */
bool ParseBool(const char* str);

}// namespace details

}// namespace aethermind

#endif//AETHERMIND_MALLOC_COMMON_H
