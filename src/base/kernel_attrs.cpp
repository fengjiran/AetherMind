// Copyright 2026 The AetherMind Authors
// SPDX-License-Identifier: Apache-2.0

#include "aethermind/base/kernel_attrs.h"

namespace aethermind {

const char* ToString(ExecPhase phase) noexcept {
    switch (phase) {
        case ExecPhase::kPrefill:
            return "Prefill";
        case ExecPhase::kDecode:
            return "Decode";
        case ExecPhase::kBoth:
            return "Both";
        default:
            return "Unknown";
    }
}

const char* ToString(WeightFormat format) noexcept {
    switch (format) {
        case WeightFormat::kPlain:
            return "Plain";
        case WeightFormat::kPacked:
            return "Packed";
        case WeightFormat::kQuantizedInt8:
            return "QuantizedInt8";
        case WeightFormat::kQuantizedInt4:
            return "QuantizedInt4";
        default:
            return "Unknown";
    }
}

}// namespace aethermind
