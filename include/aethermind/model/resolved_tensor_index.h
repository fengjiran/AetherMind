#ifndef AETHERMIND_MODEL_RESOLVED_TENSOR_INDEX_H
#define AETHERMIND_MODEL_RESOLVED_TENSOR_INDEX_H

#include "data_type.h"

#include <cstddef>
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
    size_t bytes = 0;
    DataType dtype{};
    std::vector<int64_t> shape{};
    std::shared_ptr<const RawTensorBacking> backing{};

    AM_NODISCARD bool IsValid() const noexcept {
        return backing != nullptr && dtype.bits() > 0 && (data != nullptr || bytes == 0);
    }
};

using RawTensorMap = std::unordered_map<std::string, RawTensorView>;

}// namespace aethermind

#endif
