//
// Created by richard on 6/25/25.
//

#ifndef AETHERMIND_ALIGNMENT_H
#define AETHERMIND_ALIGNMENT_H

#include <cstddef>

namespace aethermind {

#ifdef KALLOC_ALIGNMENT
constexpr size_t gAlignment = KALLOC_ALIGNMENT;
#else
constexpr size_t gAlignment = 64;
#endif

constexpr size_t gPagesize = 4096;
constexpr size_t gAlloc_threshold_thp = static_cast<size_t>(2) * 1024 * 1024;

}// namespace aethermind

#endif// AETHERMIND_ALIGNMENT_H
