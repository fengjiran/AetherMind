#ifndef AETHERMIND_MODEL_RESOLVED_TENSOR_INDEX_H
#define AETHERMIND_MODEL_RESOLVED_TENSOR_INDEX_H

#include "data_type.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace aethermind {

struct RawTensorBacking {
    virtual ~RawTensorBacking() = default;
};

struct RawTensorView {
    const std::byte* data = nullptr;
    size_t byte_size = 0;
    DataType dtype{};
    std::vector<int64_t> shape{};
    std::shared_ptr<const RawTensorBacking> backing{};

    AM_NODISCARD bool IsValid() const noexcept {
        return backing != nullptr && dtype.bits() > 0 && (data != nullptr || byte_size == 0);
    }
};

using RawTensorMap = std::unordered_map<std::string, RawTensorView>;

}// namespace aethermind

#endif
