#ifndef AETHERMIND_MODEL_RAW_WEIGHT_H
#define AETHERMIND_MODEL_RAW_WEIGHT_H

#include "aethermind/dtypes/data_type.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace aethermind {

struct RawStorage {
    virtual ~RawStorage() = default;
};

struct RawWeightView {
    const std::byte* data = nullptr;
    size_t bytes = 0;
    DataType dtype{};
    std::vector<int64_t> shape{};
    std::shared_ptr<const RawStorage> storage{};
    bool is_contiguous = true;

    AM_NODISCARD bool IsValid() const noexcept {
        return storage != nullptr && dtype.bits() > 0 && (data != nullptr || bytes == 0);
    }

    AM_NODISCARD bool IsAligned(size_t alignment) const noexcept {
        return alignment != 0 && data != nullptr &&
               reinterpret_cast<std::uintptr_t>(data) % alignment == 0;
    }
};

using RawWeightTable = std::unordered_map<std::string, RawWeightView>;

}// namespace aethermind

#endif
