#ifndef AETHERMIND_TEST_TENSOR_BUFFER_HELPERS_H
#define AETHERMIND_TEST_TENSOR_BUFFER_HELPERS_H

#include "aethermind/base/tensor_view.h"
#include "aethermind/dtypes/data_type.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace aethermind::test {

/// @brief Zeroed tensor bytes that own the metadata a TensorView borrows.
///
/// A TensorView borrows its shape and stride arrays, so a test that keeps a view
/// past the expression that built it must keep this buffer alive too. Compact
/// row-major strides are derived from the dimensions.
class TestBuffer {
public:
    TestBuffer(DataType dtype, std::initializer_list<int64_t> dimensions)
        : dtype_(dtype), shape_(dimensions) {
        int64_t count = 1;
        for (const int64_t dim: shape_) {
            count *= dim;
        }
        bytes_.assign(static_cast<size_t>(count) * static_cast<size_t>(dtype_.nbytes()),
                      std::byte{0});
        strides_.resize(shape_.size());
        int64_t stride = 1;
        for (size_t i = shape_.size(); i-- > 0;) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }

    AM_NODISCARD TensorView view() const {
        return TensorView(bytes_.data(), dtype_, IntArrayView{shape_}, IntArrayView{strides_});
    }

    AM_NODISCARD const void* data() const noexcept {
        return bytes_.data();
    }

    AM_NODISCARD void* mutable_data() noexcept {
        return bytes_.data();
    }

private:
    DataType dtype_{};
    std::vector<std::byte> bytes_{};
    std::vector<int64_t> shape_{};
    std::vector<int64_t> strides_{};
};

} // namespace aethermind::test

#endif
