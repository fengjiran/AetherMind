//
// Created by richard on 1/23/26.
//
module;
#include "macros.h"
#include "utils/logging.h"

#include <bit>
#include <cstddef>
#include <cstdint>

export module AMString;

namespace aethermind {

// constexpr auto kIsLittleEndian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;
// constexpr auto kIsBigEndian = __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__;

constexpr auto kIsLittleEndian = std::endian::native == std::endian::little;

template<typename Char>
class AMStringCore {
public:
    AMStringCore() {
        reset();
    }

private:
    void reset() {
        SetSmallSize(0);
    }

    enum class Category : uint8_t {
        isSmall = 0,
        isMedium = kIsLittleEndian ? 0x80 : 0x02,
        isLarge = kIsLittleEndian ? 0x40 : 0x01,
    };

    Category category() const {
        // compatible with little-endian and big-endian
        return static_cast<Category>(std::to_integer<uint8_t>(bytes_[lastChar] & categoryExtractMask));
    }

    struct MediumLarge {
        Char* data_;
        size_t size_;
        size_t cap_;

        NODISCARD size_t capacity() const {
            return kIsLittleEndian ? cap_ & capacityExtractMask : cap_ >> 2;
        }

        void SetCapacity(size_t cap, Category cat) {
            cap_ = kIsLittleEndian ? cap | (static_cast<size_t>(cat) << kCategoryShift)
                                   : (cap << 2) | static_cast<size_t>(cat);
        }
    };

    union {
        std::byte bytes_[sizeof(MediumLarge)]{};
        Char small_[sizeof(MediumLarge) / sizeof(Char)];
        MediumLarge ml_;
    };

    NODISCARD size_t SmallSize() const {
        AM_CHECK(category() == Category::isSmall);
        constexpr auto shift = kIsLittleEndian ? 0 : 2;
        auto diff = static_cast<size_t>(small_[maxSmallSize]) >> shift;
        AM_CHECK(maxSmallSize >= diff);
        return maxSmallSize - diff;
    }

    void SetSmallSize(size_t sz) {
        AM_CHECK(sz <= maxSmallSize);
        constexpr auto shift = kIsLittleEndian ? 0 : 2;
        small_[maxSmallSize] = static_cast<char>((maxSmallSize - sz) << shift);
        small_[sz] = '\0';
        // AM_DCHECK(category() == Category::isSmall && size() == sz);
    }


    constexpr static size_t lastChar = sizeof(MediumLarge) - 1;
    constexpr static size_t maxSmallSize = lastChar / sizeof(Char);
    constexpr static size_t maxMediumSize = 254 / sizeof(Char);
    constexpr static uint8_t categoryExtractMask = kIsLittleEndian ? 0xC0 : 0x03;
    constexpr static size_t kCategoryShift = (sizeof(size_t) - 1) * 8;
    constexpr static size_t capacityExtractMask = kIsLittleEndian ? ~(static_cast<size_t>(categoryExtractMask) << kCategoryShift)
                                                                  : 0x0 /* unused */;
};


}// namespace aethermind
