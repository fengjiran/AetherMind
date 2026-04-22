#ifndef AETHERMIND_CONTAINER_AMSTRING_H
#define AETHERMIND_CONTAINER_AMSTRING_H

#include "any_utils.h"
#include "macros.h"
#include "utils/logging.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace aethermind {

namespace details {

/// Bitwise copy for trivially copyable (POD) types.
///
/// Uses std::memcpy for maximum performance. The destination buffer must have
/// sufficient space, and source/destination ranges must not overlap.
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

/// Bitwise copy using std::span interface.
///
/// Destination span must be at least as large as source. Ranges must not overlap.
template<is_pod T>
void PodCopy(std::span<const T> src, std::span<T> dst) noexcept {
    AM_DCHECK(dst.size() >= src.size());
    AM_DCHECK(dst.data() >= src.data() + src.size() ||
              dst.data() + src.size() <= src.data());

    if (src.empty()) AM_UNLIKELY {
            return;
        }

    std::memcpy(dst.data(), src.data(), src.size_bytes());
}

template<is_pod T>
void PodMove(const T* begin, const T* end, T* dst) noexcept {
    const auto n = static_cast<size_t>(end - begin);
    if (n == 0) AM_UNLIKELY {
            return;
        }
    std::memmove(dst, begin, n * sizeof(T));
}

/// Bitwise move supporting overlapping regions.
///
/// Uses std::memmove for safe relocation when destination may overlap with source.
template<is_pod T>
void PodMove(std::span<const T> src, std::span<T> dst) noexcept {
    AM_DCHECK(dst.size() >= src.size());
    if (src.empty()) AM_UNLIKELY {
            return;
        }

    std::memmove(dst.data(), src.data(), src.size_bytes());
}

/// Bitwise fill for trivially copyable types.
template<typename Pod, typename T>
    requires is_pod<Pod> && std::is_convertible_v<T, Pod>
void PodFill(std::span<Pod> storage, const T& c) noexcept {
    if (storage.empty()) AM_UNLIKELY {
            return;
        }

    std::ranges::fill(storage, static_cast<Pod>(c));
}

}// namespace details


/// Platform-specific constants for AMStringCore layout optimization.
struct MagicConstants {
    constexpr static auto kIsBigEndian = __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__;
    constexpr static auto kIsLittleEndian = std::endian::native == std::endian::little;

#ifdef AM_SANITIZE
    constexpr static auto kIsSanitize = true;
#else
    constexpr static auto kIsSanitize = false;
#endif
};

/// Core implementation for Small String Optimization (SSO).
///
/// Stores small strings inline (up to maxSmallSize characters) to avoid heap allocation.
/// Medium/large strings use external storage with capacity/size/category packed into
/// a single word for compact representation.
///
/// Layout differs between little-endian and big-endian platforms for optimal bit packing.
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
        if constexpr (MagicConstants::kIsLittleEndian) {
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
        isMedium = MagicConstants::kIsLittleEndian ? 0x80 : 0x02,
        isLarge = MagicConstants::kIsLittleEndian ? 0x40 : 0x01,
    };

    Category category() const {
        return static_cast<Category>(std::to_integer<uint8_t>(bytes_[lastChar] & categoryExtractMask));
    }

    struct MediumLarge {
        Char* data_;
        size_t size_;
        size_t cap_;

        AM_NODISCARD size_t capacity() const {
            return MagicConstants::kIsLittleEndian ? cap_ & capacityExtractMask : cap_ >> 2;
        }

        void SetCapacity(size_t cap, Category cat) {
            cap_ = MagicConstants::kIsLittleEndian ? cap | (static_cast<size_t>(cat) << kCategoryShift)
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
        constexpr auto shift = MagicConstants::kIsLittleEndian ? 0 : 2;
        auto diff = static_cast<size_t>(small_[maxSmallSize]) >> shift;
        AM_CHECK(maxSmallSize >= diff);
        return maxSmallSize - diff;
    }

    void SetSmallSize(size_t sz) {
        AM_CHECK(sz <= maxSmallSize);
        constexpr auto shift = MagicConstants::kIsLittleEndian ? 0 : 2;
        small_[maxSmallSize] = static_cast<char>((maxSmallSize - sz) << shift);
        small_[sz] = '\0';
        AM_DCHECK(category() == Category::isSmall && size() == sz);
    }

    void InitSmall(std::span<const Char> src);
    void InitMedium(std::span<const Char> src);

    constexpr static size_t lastChar = sizeof(MediumLarge) - 1;
    constexpr static size_t maxSmallSize = lastChar / sizeof(Char);
    constexpr static size_t maxMediumSize = 254 / sizeof(Char);
    constexpr static uint8_t categoryExtractMask = MagicConstants::kIsLittleEndian ? 0xC0 : 0x03;
    constexpr static size_t kCategoryShift = (sizeof(size_t) - 1) * 8;
    constexpr static size_t capacityExtractMask = MagicConstants::kIsLittleEndian ? ~(static_cast<size_t>(categoryExtractMask) << kCategoryShift)
                                                                                  : 0x0 /* unused */;
};

/// Fast small-string initialization using "safe over-reading".
///
/// If source data lies within a single 4KB page, copies the full 24-byte block
/// to eliminate branching. Disabled under sanitizers which trap on over-reads.
template<typename Char>
void AMStringCore<Char>::InitSmall(std::span<const Char> src) {
    static_assert(sizeof(*this) == sizeof(Char*) + 2 * sizeof(size_t));
    static_assert(sizeof(Char*) == sizeof(size_t));
    static_assert((sizeof(size_t) & (sizeof(size_t) - 1)) == 0);

    const size_t sz = src.size();
    const auto addr = reinterpret_cast<uintptr_t>(src.data());

    // Fast path: block copy if data stays within one page (no page crossing, no ASAN)
    if (constexpr size_t page_size = 4096; !MagicConstants::kIsSanitize &&
                                           sz > 0 &&
                                           (addr ^ (addr + sizeof(small_) - 1)) < page_size) AM_LIKELY {
            std::memcpy(small_, src.data(), sizeof(small_));
        }
    else {
        if (sz > 0) {
            details::PodCopy(src.data(), src.data() + sz, small_);
        }
    }

    SetSmallSize(sz);
}

template<typename Char>
void AMStringCore<Char>::InitMedium(std::span<const Char> src) {
    // TODO: Implement medium string allocation
}

}// namespace aethermind

#endif// AETHERMIND_CONTAINER_AMSTRING_H
