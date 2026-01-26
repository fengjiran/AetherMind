//
// Created by richard on 1/23/26.
//

/* Global Module Fragment Begin*/
module;
#include "macros.h"
#include "utils/logging.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
/* Global Module Fragment end*/

export module AMString;

import concept_module;

namespace aethermind {

namespace details {

/**
 * @brief High-performance memory copy for POD (Plain Old Data) types.
 *
 * This function leverages C++20 Concepts to ensure that bitwise copying is only
 * performed on types that are safe to copy via std::memcpy. It bypasses the
 * overhead of copy constructors and destructors.
 *
 * @tparam T A type satisfying the is_pod constraint (trivially copyable).
 *
 * @param begin Pointer to the start of the source range.
 * @param end   Pointer to the end of the source range (exclusive).
 * @param dst   Pointer to the destination memory address.
 *
 * @section optimization Performance Optimizations:
 * - Uses AM_DCHECK for validation: Ensures pointer sanity and detects memory overlaps
 *   during development, while incurring zero cost in Release builds.
 * - [[unlikely]] attribute: Informs the compiler that zero-length copies are rare,
 *   keeping the hot path optimized for actual data transfers.
 * - Standard Compliance: Maps directly to std::memcpy, enabling compiler
 *   auto-vectorization (e.g., AVX-512, AMX) on modern 2026 processors.
 *
 * @warning
 * - The destination buffer must be large enough to hold (end - begin) elements.
 * - Memory regions [begin, end) and [dst, dst + n) must not overlap. Use
 *   std::memmove if overlap is possible.
 */
template<is_pod T>
void PodCopy(const T* begin, const T* end, T* dst) noexcept {
    AM_DCHECK(begin != nullptr);
    AM_DCHECK(end != nullptr);
    AM_DCHECK(dst != nullptr);
    AM_DCHECK(end >= begin);
    AM_DCHECK(dst >= end || dst + (end - begin) <= begin);

    const auto n = static_cast<size_t>(end - begin);
    if (n == 0) AM_UNLIKELY {
            return;
        }

    std::memcpy(dst, begin, n * sizeof(T));
}

/**
 * @brief Performs a high-performance bitwise copy between two memory spans.
 *
 * This version uses std::span to provide a safe, modern interface for memory
 * transfers. It is restricted to trivially copyable (POD) types to ensure
 * that bitwise copying does not violate object lifecycle rules.
 *
 * @tparam T A type satisfying the is_pod constraint (trivially copyable).
 *
 * @param src The source memory view (read-only).
 * @param dst The destination memory view.
 *
 * @note
 * - The function uses std::memcpy internally, which is optimized for modern
 *   SIMD architectures (AVX-512/AMX) in 2026.
 * - Performance: O(N) bitwise transfer. Branch prediction is optimized for
 *   non-empty spans using [[unlikely]].
 *
 * @warning
 * - Destination span size must be greater than or equal to source span size.
 * - Memory regions must not overlap. Use PodMove for overlapping ranges.
 */
template<is_pod T>
void PodCopy(std::span<const T> src, std::span<T> dst) noexcept {
    // Ensure destination has enough space for the source data
    AM_DCHECK(dst.size() >= src.size());

    // Strict non-overlap check required by memcpy
    AM_DCHECK(dst.data() >= src.data() + src.size() ||
              dst.data() + src.size() <= src.data());

    if (src.empty()) AM_UNLIKELY {
            return;
        }

    std::memcpy(dst.data(), src.data(), src.size_bytes());
}

template<is_pod T>
void PodMove(const T* begin, const T* end, T* dst) noexcept {
    AM_DCHECK(end >= begin);
    const auto n = static_cast<size_t>(end - begin);
    if (n == 0) AM_UNLIKELY {
            return;
        }
    std::memmove(dst, begin, n * sizeof(T));
}

/**
 * @brief Safely moves a range of POD elements, supporting overlapping regions.
 *
 * This function uses std::memmove via std::span interfaces to provide a safe
 * way to relocate data. It is specifically designed for cases where the
 * destination may overlap with the source (e.g., shifting elements within
 * a buffer).
 *
 * @tparam T A trivially copyable (POD) type.
 *
 * @param src The source memory span (read-only).
 * @param dst The destination memory span.
 *
 * @note
 * - Performance: O(N) bitwise move. Optimized for modern SIMD architectures.
 * - The [[unlikely]] attribute minimizes branch overhead for empty spans.
 *
 * @warning
 * - The destination span (dst.size()) must be at least as large as the
 *   source span (src.size()).
 */
template<is_pod T>
void PodMove(std::span<const T> src, std::span<T> dst) noexcept {
    AM_DCHECK(dst.size() >= src.size());
    if (src.empty()) AM_UNLIKELY {
            return;
        }

    std::memmove(dst.data(), src.data(), src.size_bytes());
}

/**
 * @brief High-performance fill operation for trivially copyable types.
 *
 * This function uses C++20 std::span to provide a safe, non-owning view of a
 * contiguous memory range. It leverages std::ranges::fill, which is highly
 * optimized by 2026-era compilers to use hardware-level instructions.
 *
 * @tparam Pod The type of elements in the span. Must be trivially copyable
 *             to ensure bitwise safety.
 * @tparam T   The type of the value to fill. Must be convertible to Pod.
 *
 * @param storage A std::span representing the memory range to be filled.
 * @param c       The value to be assigned to each element in the range.
 *
 * @section perf Performance Characteristics:
 * - Zero-overhead: Acts as a thin wrapper over raw pointers.
 * - Instruction Mapping: Compilers (GCC 15+, Clang 19+) typically lower this to:
 *   - `std::memset` for 1-byte types (e.g., char, uint8_t).
 *   - SIMD vectorized loops (AVX-512/AMX) for multi-byte types.
 * - [[unlikely]] hint: Optimizes the branch predictor for non-empty buffers.
 */
template<typename Pod, typename T>
    requires is_pod<Pod> && std::is_convertible_v<T, Pod>
void PodFill(std::span<Pod> storage, const T& c) noexcept {
    if (storage.empty()) AM_UNLIKELY {
            return;
        }

    std::ranges::fill(storage, static_cast<Pod>(c));
}

}// namespace details

struct Constants {
    // constexpr bool kIsLittleEndian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;
    constexpr static auto kIsBigEndian = __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__;
    constexpr static auto kIsLittleEndian = std::endian::native == std::endian::little;

#ifdef AM_SANITIZE
    constexpr static auto kIsSanitize = true;
#else
    constexpr static auto kIsSanitize = false;
#endif
};

template<typename Char>
class AMStringCore {
public:
    AMStringCore() {
        reset();
    }

    const Char* c_str() const {
        const auto* ptr = ml_.data_;
        ptr = category() == Category::isSmall ? small_ : ptr;// CMOV optimization
        return ptr;
    }

    AM_NODISCARD size_t size() const {
        auto res = ml_.size_;
        if constexpr (Constants::kIsLittleEndian) {
            using UChar = std::make_unsigned_t<Char>;
            auto small_size = maxSmallSize - static_cast<size_t>(static_cast<UChar>(small_[maxSmallSize]));
            res = static_cast<std::ptrdiff_t>(small_size) >= 0 ? small_size : res;// gcc will generate a CMOV instead of a branch
        } else {
            res = category() == Category::isSmall ? SmallSize() : res;
        }
        return res;
    }

private:
    Char* c_str() {
        auto* ptr = ml_.data_;
        ptr = category() == Category::isSmall ? small_ : ptr;// CMOV optimization
        return ptr;
    }

    void reset() {
        SetSmallSize(0);
    }

    enum class Category : uint8_t {
        isSmall = 0,
        isMedium = Constants::kIsLittleEndian ? 0x80 : 0x02,
        isLarge = Constants::kIsLittleEndian ? 0x40 : 0x01,
    };

    Category category() const {
        // compatible with little-endian and big-endian
        return static_cast<Category>(std::to_integer<uint8_t>(bytes_[lastChar] & categoryExtractMask));
    }

    struct MediumLarge {
        Char* data_;
        size_t size_;
        size_t cap_;

        AM_NODISCARD size_t capacity() const {
            return Constants::kIsLittleEndian ? cap_ & capacityExtractMask : cap_ >> 2;
        }

        void SetCapacity(size_t cap, Category cat) {
            cap_ = Constants::kIsLittleEndian ? cap | (static_cast<size_t>(cat) << kCategoryShift)
                                              : (cap << 2) | static_cast<size_t>(cat);
        }
    };

    union {
        std::byte bytes_[sizeof(MediumLarge)];
        Char small_[sizeof(MediumLarge) / sizeof(Char)];
        MediumLarge ml_;
    };

    static_assert(sizeof(MediumLarge) % sizeof(Char) == 0, "Corrupt memory layout.");

    AM_NODISCARD size_t SmallSize() const {
        AM_CHECK(category() == Category::isSmall);
        constexpr auto shift = Constants::kIsLittleEndian ? 0 : 2;
        auto diff = static_cast<size_t>(small_[maxSmallSize]) >> shift;
        AM_CHECK(maxSmallSize >= diff);
        return maxSmallSize - diff;
    }

    void SetSmallSize(size_t sz) {
        AM_CHECK(sz <= maxSmallSize);
        constexpr auto shift = Constants::kIsLittleEndian ? 0 : 2;
        small_[maxSmallSize] = static_cast<char>((maxSmallSize - sz) << shift);
        small_[sz] = '\0';
        AM_DCHECK(category() == Category::isSmall && size() == sz);
    }

    void InitSmall(std::span<const Char> src);
    void InitMedium(std::span<const Char> src);

    constexpr static size_t lastChar = sizeof(MediumLarge) - 1;
    constexpr static size_t maxSmallSize = lastChar / sizeof(Char);
    constexpr static size_t maxMediumSize = 254 / sizeof(Char);
    constexpr static uint8_t categoryExtractMask = Constants::kIsLittleEndian ? 0xC0 : 0x03;
    constexpr static size_t kCategoryShift = (sizeof(size_t) - 1) * 8;
    constexpr static size_t capacityExtractMask = Constants::kIsLittleEndian ? ~(static_cast<size_t>(categoryExtractMask) << kCategoryShift)
                                                                             : 0x0 /* unused */;
};

/**
 * @brief Fast initialization for Small String Optimization (SSO).
 *
 * Uses a technique called "Safe Over-reading": if the source data is within a
 * single 4KB memory page, we copy the entire 24-byte block regardless of
 * actual size to eliminate branching.
 *
 * @param src A span representing the source characters to initialize from.
 */
template<typename Char>
void AMStringCore<Char>::InitSmall(std::span<const Char> src) {
    // Layout is: Char* data_, size_t size_, size_t capacity_
    static_assert(sizeof(*this) == sizeof(Char*) + 2 * sizeof(size_t),
                  "amstring has unexpected size");
    static_assert(sizeof(Char*) == sizeof(size_t),
                  "amstring size assumption violation");
    // sizeof(size_t) must be a power of 2
    static_assert((sizeof(size_t) & (sizeof(size_t) - 1)) == 0,
                  "amstring size assumption violation");

    const size_t sz = src.size();
    constexpr size_t page_size = 4096;
    const auto addr = reinterpret_cast<uintptr_t>(src.data());

    // Optimization: Block copy 24 bytes if safe (no page crossing and no ASAN)
    // We add parenthesis around XOR operands for correct precedence
    if (!Constants::kIsSanitize &&// sanitizer would trap on over-reads
        sz > 0 &&
        (addr ^ (addr + sizeof(small_) - 1)) < page_size) AM_LIKELY {
            // the input data is all within one page so over-reads will not segfault.
            // Direct word-aligned copy, likely lowered to 3-4 LDP/STP or SIMD instructions.
            std::memcpy(small_, src.data(), sizeof(small_));
        }
    else {
        // Precise copy path
        if (sz > 0) {
            details::PodCopy(src.data(), src.data() + sz, small_);
        }
    }

    SetSmallSize(sz);
}

template<typename Char>
void AMStringCore<Char>::InitMedium(std::span<const Char> src) {
    //
}

}// namespace aethermind
