//
// Created by richard on 6/25/25.
//

#ifndef AETHERMIND_ALIGNMENT_H
#define AETHERMIND_ALIGNMENT_H

namespace aethermind {

constexpr size_t gAlignment = 64;

constexpr size_t gPagesize = 4096;
// since the default thp pagesize is 2MB, enable thp only
// for buffers of size 2MB or larger to avoid memory bloating
constexpr size_t gAlloc_threshold_thp = static_cast<size_t>(2) * 1024 * 1024;
}// namespace aethermind

#endif//AETHERMIND_ALIGNMENT_H
