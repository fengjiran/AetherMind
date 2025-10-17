//
// Created by richard on 10/12/25.
//

#ifndef AETHERMIND_MEMORY_FORMAT_H
#define AETHERMIND_MEMORY_FORMAT_H

#include "container/array_view.h"

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
// https://zhuanlan.zhihu.com/p/494620090
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

// shape and stride is NCHW
// underlying data in memory is NHWC
template<typename T>
std::vector<T> get_channels_last_strides_2d(ArrayView<T> shape) {
    auto n = shape.size();
    std::vector<T> strides(n);// (stride_n, stride_c, stride_h, stride_w)
    switch (n) {
        case 4: {
            strides[1] = 1;
            strides[3] = shape[1];
            strides[2] = strides[3] * shape[3];
            strides[0] = strides[2] * shape[2];
            break;
        }

        case 3: {
            strides[0] = 1;
            strides[2] = shape[0];
            strides[1] = strides[2] * shape[2];
            break;
        }

        default: {
            CHECK(false) << "ChannelsLast2d doesn't support size " << shape.size();
        }
    }
    return strides;
}

inline std::vector<int64_t> get_channels_last_strides_2d(IntArrayView shape) {
    return get_channels_last_strides_2d<int64_t>(shape);
}

// shape and stride is NCDHW
// underlying data in memory is NDHWC
template<typename T>
std::vector<T> get_channels_last_strides_3d(ArrayView<T> shape) {
    auto n = shape.size();
    std::vector<T> strides(n);// NCDHW
    switch (n) {
        case 5: {
            strides[1] = 1;
            strides[4] = strides[1] * shape[1];
            strides[3] = strides[4] * shape[4];
            strides[2] = strides[3] * shape[3];
            strides[0] = strides[2] * shape[2];
            break;
        }

        case 4: {
            strides[0] = 1;
            strides[3] = strides[0] * shape[0];
            strides[2] = strides[3] * shape[3];
            strides[1] = strides[2] * shape[2];
            break;
        }

        default: {
            CHECK(false) << "ChannelsLast3d doesn't support size " << shape.size();
        }
    }

    return strides;
}

inline std::vector<int64_t> get_channels_last_strides_3d(IntArrayView shape) {
    return get_channels_last_strides_3d<int64_t>(shape);
}

template<typename T>
bool is_channels_last_strides_2d_s4(ArrayView<T> shape, ArrayView<T> strides) {
    T pre = 0;
    // special case for trivial C dimension. default to NCHW
    if (strides[1] == 0) {
        return false;
    }

    for (int d: {1, 3, 2, 0}) {
        if (shape[d] == 0 || strides[d] < pre) {
            return false;
        }

        if (d == 0 && pre == strides[1]) {
            return false;
        }

        pre = strides[d];
        if (shape[d] > 1) {
            pre *= shape[d];
        }
    }
    return true;
}

template<typename T>
bool is_channels_last_strides_3d_s5(ArrayView<T> shape, ArrayView<T> strides) {
    T min = 0;
    // special case for trivial C dimension. default to NCHW
    if (strides[1] == 0) {
        return false;
    }

    for (int d: {1, 4, 3, 2, 0}) {
        if (shape[d] == 0) {
            return false;
        }

        if (strides[d] < min) {
            return false;
        }

        if (d == 0 && min == strides[1]) {
            return false;
        }

        min = strides[d];
        if (shape[d] > 1) {
            min *= shape[d];
        }
    }
    return true;
}

template<typename T>
bool is_channels_last_strides_2d(ArrayView<T> shape, ArrayView<T> strides) {
    switch (shape.size()) {
        case 4: {
            return is_channels_last_strides_2d_s4(shape, strides);
        }

        case 3: {
            return false;
        }

        default:
            return false;
    }
}

template<typename T>
bool is_channels_last_strides_3d(ArrayView<T> shape, ArrayView<T> strides) {
    switch (shape.size()) {
        case 5: {
            return is_channels_last_strides_3d_s5(shape, strides);
        }

        case 4: {
            return false;
        }

        default:
            return false;
    }
}

inline bool is_channels_last_strides_2d(IntArrayView shape, IntArrayView strides) {
    return is_channels_last_strides_2d<int64_t>(shape, strides);
}

inline bool is_channels_last_strides_3d(IntArrayView shape, IntArrayView strides) {
    return is_channels_last_strides_3d<int64_t>(shape, strides);
}

inline std::ostream& operator<<(std::ostream& os, MemoryFormat format) {
    switch (format) {
        case MemoryFormat::Preserve:
            os << "Preserve";
            break;
        case MemoryFormat::Contiguous:
            os << "Contiguous";
            break;
        case MemoryFormat::ChannelsLast:
            os << "ChannelsLast";
            break;
        case MemoryFormat::ChannelsLast3d:
            os << "ChannelsLast3d";
            break;
        default:
            CHECK(false) << "Unknown memory format.";
    }
    return os;
}

}// namespace aethermind

#endif//AETHERMIND_MEMORY_FORMAT_H
