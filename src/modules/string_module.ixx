//
// Created by richard on 1/23/26.
//
module;
#include "macros.h"

#include <bit>
#include <cstddef>
#include <cstdint>

export module String;

namespace aethermind {

// constexpr auto kIsLittleEndian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;
// constexpr auto kIsBigEndian = __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__;

constexpr auto kIsLittleEndian = std::endian::native == std::endian::little;

template<typename Char>
class AMStringCore {
public:
private:
    constexpr static uint8_t categoryExtractMask = kIsLittleEndian ? 0xC0 : 0x3;
    constexpr static size_t kCategoryShift = (sizeof(size_t) - 1) * 8;
    constexpr static size_t capacityExtractMask = kIsLittleEndian ? ~(static_cast<size_t>(categoryExtractMask) << kCategoryShift)
                                                                  : 0x0 /* unused */;
    // using category_type = uint8_t;
    enum class Category : uint8_t {
        isSmall = 0,
        isMedium = kIsLittleEndian ? 0x80 : 0x02,
        isLarge = kIsLittleEndian ? 0x40 : 0x01,
    };

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
        std::byte bytes_[sizeof(MediumLarge)];
        Char small_[sizeof(MediumLarge) / sizeof(Char)];
        MediumLarge ml_;
    };
};


}// namespace aethermind
