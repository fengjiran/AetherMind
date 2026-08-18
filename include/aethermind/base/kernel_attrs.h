// Copyright 2026 The AetherMind Authors
// SPDX-License-Identifier: Apache-2.0
//
// Backend-independent execution attributes shared by graph lowering, execution
// planning, and kernel selection.
//
// These are pure data enums: they describe how a step may be executed (ISA
// level, execution phase, weight layout) without referencing any backend or
// kernel implementation. They live in the base layer so the graph module can
// carry them through lowering without depending on backend or execution
// headers.

#ifndef AETHERMIND_BASE_KERNEL_ATTRS_H
#define AETHERMIND_BASE_KERNEL_ATTRS_H

#include "aethermind/base/macros.h"

#include <cstdint>

namespace aethermind {

/// CPU instruction-set level a kernel may target.
enum class IsaLevel : uint8_t {
    kScalar = 0,
    kAVX2,
    kAVX512,
    kAMX,
};

/// Execution phase in which a step runs.
enum class ExecPhase : uint8_t {
    kPrefill = 0,
    kDecode,
    kBoth,
};

/// Weight layout a step consumes.
enum class WeightFormat : uint8_t {
    kPlain = 0,
    kPacked,
    kQuantizedInt8,
    kQuantizedInt4,
};

AM_NODISCARD const char* ToString(IsaLevel isa) noexcept;
AM_NODISCARD const char* ToString(ExecPhase phase) noexcept;
AM_NODISCARD const char* ToString(WeightFormat format) noexcept;

AM_NODISCARD inline bool PhaseMatches(ExecPhase candidate,
                                      ExecPhase request) noexcept {
    return candidate == request || candidate == ExecPhase::kBoth;
}

}// namespace aethermind

#endif// AETHERMIND_BASE_KERNEL_ATTRS_H
