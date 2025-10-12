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
enum class MemoryFormat : uint8_t {
    Contiguous,
    Preserve,
    ChannelsLast,
    ChannelsLast3d,
    NumOptions
};

inline MemoryFormat get_contiguous_memory_format() {
}

}// namespace aethermind

#endif//AETHERMIND_MEMORY_FORMAT_H
