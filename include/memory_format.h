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
    kContiguous,
    kPreserve,
    // NHWC, dense, no overlap
    // strides[0] > strides[2] > strides[3] > strides[1] == 1
    kChannelsLast,
    kChannelsLast3d,// NDHWC
    NumOptions
};

inline MemoryFormat GetContiguousMemoryFormat() {
    return MemoryFormat::kContiguous;
}

// shape and stride is NCHW
// underlying data in memory is NHWC
template<typename T>
std::vector<T> GetChannelsLastStrides2d(ArrayView<T> shape) {
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
            AM_CHECK(false, "ChannelsLast2d doesn't support size {}", shape.size());
        }
    }
    return strides;
}

inline std::vector<int64_t> GetChannelsLastStrides2d(IntArrayView shape) {
    return GetChannelsLastStrides2d<int64_t>(shape);
}

// shape and stride is NCDHW
// underlying data in memory is NDHWC
template<typename T>
std::vector<T> GetChannelsLastStrides3d(ArrayView<T> shape) {
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
            AM_CHECK(false, "ChannelsLast3d doesn't support size {}", shape.size());
        }
    }

    return strides;
}

inline std::vector<int64_t> GetChannelsLastStrides3d(IntArrayView shape) {
    return GetChannelsLastStrides3d<int64_t>(shape);
}

template<typename T>
bool IsChannelsLastStrides2ds4(ArrayView<T> shape, ArrayView<T> strides) {
    T pre_stride = 0;
    // special case for trivial C dimension. default to NCHW
    // for broadcast case.
    if (strides[1] == 0) {
        return false;
    }

    for (int d: {1, 3, 2, 0}) {
        if (shape[d] == 0 || strides[d] < pre_stride) {
            return false;
        }

        // Fallback to NCHW as default layout for ambiguous cases,
        // This is the flaw of implicit memory_format from strides.
        // N111 tensor with identical strides for size 1 dimension;
        // Two cases could lead us here:
        // a. N111 contiguous Tensor ([N,1,1,1]@[1,1,1,1])
        // b. N11W contiguous Tensor sliced on the W-dimension.
        // ([N,1,1,1]@[W,W,W,W])
        if (d == 0 && pre_stride == strides[1]) {
            return false;
        }

        pre_stride = strides[d];
        if (shape[d] > 1) {
            pre_stride *= shape[d];
        }
    }
    return true;
}

template<typename T>
bool IsChannelsLastStrides3ds5(ArrayView<T> shape, ArrayView<T> strides) {
    T pre_stride = 0;
    // special case for trivial C dimension. default to NCHW
    if (strides[1] == 0) {
        return false;
    }

    for (int d: {1, 4, 3, 2, 0}) {
        if (shape[d] == 0 || strides[d] < pre_stride) {
            return false;
        }

        if (d == 0 && pre_stride == strides[1]) {
            return false;
        }

        pre_stride = strides[d];
        if (shape[d] > 1) {
            pre_stride *= shape[d];
        }
    }
    return true;
}

template<typename T>
bool IsChannelsLastStrides2d(ArrayView<T> shape, ArrayView<T> strides) {
    switch (shape.size()) {
        case 4: {
            return IsChannelsLastStrides2ds4(shape, strides);
        }

        case 3: {
            return false;
        }

        default:
            return false;
    }
}

template<typename T>
bool IsChannelsLastStrides3d(ArrayView<T> shape, ArrayView<T> strides) {
    switch (shape.size()) {
        case 5: {
            return IsChannelsLastStrides3ds5(shape, strides);
        }

        case 4: {
            return false;
        }

        default:
            return false;
    }
}

inline bool IsChannelsLastStrides2d(IntArrayView shape, IntArrayView strides) {
    return IsChannelsLastStrides2d<int64_t>(shape, strides);
}

inline bool IsChannelsLastStrides3d(IntArrayView shape, IntArrayView strides) {
    return IsChannelsLastStrides3d<int64_t>(shape, strides);
}

inline std::ostream& operator<<(std::ostream& os, MemoryFormat format) {
    switch (format) {
        case MemoryFormat::kPreserve:
            os << "Preserve";
            break;
        case MemoryFormat::kContiguous:
            os << "Contiguous";
            break;
        case MemoryFormat::kChannelsLast:
            os << "ChannelsLast";
            break;
        case MemoryFormat::kChannelsLast3d:
            os << "ChannelsLast3d";
            break;
        default:
            AM_CHECK(false, "Unknown memory format.");
    }
    return os;
}

}// namespace aethermind

#endif//AETHERMIND_MEMORY_FORMAT_H
