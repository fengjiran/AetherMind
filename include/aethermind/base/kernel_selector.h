#ifndef AETHERMIND_BASE_KERNEL_SELECTOR_H
#define AETHERMIND_BASE_KERNEL_SELECTOR_H

/// @file kernel_selector.h
/// @brief Cross-module execution request descriptor.
///
/// A KernelSelector names the structural execution request (device, dtypes,
/// weight layout, phase). It is a pure data contract shared by
/// graph lowering (which records it on ExecutionPlanNodeSpec), execution
/// planning (which resolves kernels against it), backend kernel registries
/// (which describe kernel capabilities with it), and model weight prepacking.
///
/// It lives in the base layer so that the graph and execution public headers
/// can carry it without depending on backend headers.

#include "aethermind/base/device.h"
#include "aethermind/base/kernel_attrs.h"
#include "aethermind/dtypes/data_type.h"
#include "utils/hash.h"

#include <cstdint>

namespace aethermind {

/// @brief Pure-data description of the execution capabilities a step
/// requires: device, activation and weight dtypes, weight layout, and
/// execution phase.
struct KernelSelector {
    DeviceType device_type = DeviceType::kUndefined;
    DataType act_dtype{};
    DataType weight_dtype{};
    WeightFormat weight_format = WeightFormat::kPlain;
    ExecPhase phase = ExecPhase::kBoth;

    friend bool operator==(const KernelSelector& lhs, const KernelSelector& rhs) noexcept {
        return lhs.device_type == rhs.device_type &&
               lhs.act_dtype == rhs.act_dtype &&
               lhs.weight_dtype == rhs.weight_dtype &&
               lhs.weight_format == rhs.weight_format &&
               lhs.phase == rhs.phase;
    }

    friend bool operator!=(const KernelSelector& lhs, const KernelSelector& rhs) {
        return !(lhs == rhs);
    }
};

/// @brief Reports whether a kernel described by `candidate` can serve
/// `request`.
///
/// CPU instruction requirements are deliberately not part of this contract;
/// CpuBackend filters them against its immutable CpuCapabilities snapshot
/// after structural matching.
///
/// @param candidate Capabilities advertised by a registered kernel.
/// @param request Capabilities the step requires.
/// @return True if a kernel described by `candidate` can execute `request`.
AM_NODISCARD inline bool SelectorMatches(const KernelSelector& candidate,
                                         const KernelSelector& request) noexcept {
    return candidate.device_type == request.device_type &&
           candidate.act_dtype == request.act_dtype &&
           candidate.weight_dtype == request.weight_dtype &&
           candidate.weight_format == request.weight_format &&
           PhaseMatches(candidate.phase, request.phase);
}

/// @brief Returns a human-readable description of all selector fields.
///
/// @param selector Selector to render.
/// @return Compact single-line description, mainly for diagnostics and error
///         messages.
std::string ToString(const KernelSelector& selector);

}// namespace aethermind

// Must hash exactly the fields compared by operator== so equal selectors
// produce equal hashes.
template<>
struct std::hash<aethermind::KernelSelector> {
    std::size_t operator()(const aethermind::KernelSelector& s) const noexcept {
        std::size_t seed = 0;
        seed = aethermind::hash_combine(seed, static_cast<std::size_t>(s.device_type));
        seed = aethermind::hash_combine(seed, std::hash<aethermind::DataType>{}(s.act_dtype));
        seed = aethermind::hash_combine(seed, std::hash<aethermind::DataType>{}(s.weight_dtype));
        seed = aethermind::hash_combine(seed, static_cast<std::size_t>(s.weight_format));
        seed = aethermind::hash_combine(seed, static_cast<std::size_t>(s.phase));
        return seed;
    }
};

#endif// AETHERMIND_BASE_KERNEL_SELECTOR_H
