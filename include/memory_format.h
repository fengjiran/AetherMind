//
// Created by richard on 10/12/25.
//

#ifndef AETHERMIND_MEMORY_FORMAT_H
#define AETHERMIND_MEMORY_FORMAT_H

#include <cstdint>
#include <ostream>
#include <vector>

namespace aethermind {

// Memory format is not the property of a Tensor. It is the way to tell an
// operator how the result should be organized in memory and nothing more. That
// means memory format should never be used as return value for any tensor state
// interrogation functions (internally and externally).
//
// Possible options are:
//  Preserve:
//    If any of the input tensors is in channels_last format, operator output
//    should be in channels_last format
//
//  Contiguous:
//    Regardless of input tensors format, the output should be contiguous
//    Tensor.
//
//  ChannelsLast:
//    Regardless of input tensors format, the output should be in channels_last
//    format.
// https://juejin.cn/post/7234454112955580475
enum class MemoryFormat : uint8_t {
    // NCHW, dense, no overlap,
    // strides[0] > strides[1] > strides[2] > strides[3] == 1
    Contiguous,
    Preserve,
    // NHWC, dense, no overlap
    // strides[0] > strides[2] > strides[3] > strides[1] == 1
    ChannelsLast,
    ChannelsLast3d,// NDHWC
    NumOptions
};

inline MemoryFormat get_contiguous_memory_format() {
    return MemoryFormat::Contiguous;
}

template<typename T>
std::vector<T> get_channels_last_strides_2d(ArrayView<T> shape) {
    std::vector<T> strides(shape.size());
    switch (shape.size()) {
        case 4: {
            strides[1] = 1;
            strides[3] = shape[1];
            strides[2] = strides[3] * shape[3];
            strides[0] = strides[2] * shape[2];
        }
        case 3: {
            strides[0] = 1;
            strides[2] = shape[0];
            strides[1] = strides[2] * shape[2];
        }
        default: {
            CHECK(false) << "ChannelsLast2d doesn't support size " << shape.size();
        }
    }
    return strides;
}

inline std::ostream& operator<<(std::ostream& os, MemoryFormat format) {
    switch (format) {
        case MemoryFormat::Preserve:
            os << "Preserve";
        case MemoryFormat::Contiguous:
            os << "Contiguous";
        case MemoryFormat::ChannelsLast:
            os << "ChannelsLast";
        case MemoryFormat::ChannelsLast3d:
            os << "ChannelsLast3d";
        default:
            CHECK(false) << "Unknown memory format.";
    }
    return os;
}

}// namespace aethermind

#endif//AETHERMIND_MEMORY_FORMAT_H
