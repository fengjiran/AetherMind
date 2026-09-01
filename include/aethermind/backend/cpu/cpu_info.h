#ifndef AETHERMIND_CPU_INFO_H
#define AETHERMIND_CPU_INFO_H

/// @file cpu_info.h
/// @brief CPU capability detection and policy application.
///
/// Probes hardware and OS state to produce an immutable `CpuCapabilities`
/// snapshot, then derives the effective feature set by applying a
/// `CpuFeaturePolicy`. The file owns only detection and policy logic;
/// kernel selection lives in `cpu_capabilities.h`.

#include "aethermind/backend/cpu/cpu_capabilities.h"
#include "aethermind/base/status.h"

namespace aethermind::cpu {

/// @brief Detects an immutable CPU capability snapshot for the current process.
///
/// The returned effective features are always derived from usable features by
/// applying `policy`; a policy cannot enable unsupported instructions.
///
/// @param policy Feature policy that may only restrict usable features.
///               An empty policy returns the full usable set as effective.
/// @return Snapshot with hardware, usable, and effective feature sets on
///         success, or `InvalidArgument`/`FailedPrecondition` when `policy`
///         violates invariants or requires unavailable features.
/// @note On x86-64 Linux, detection may issue a process-level OS request via
///       `arch_prctl(ARCH_REQ_XCOMP_PERM)` to acquire AMX `XTILEDATA`
///       permission; no other CPU state is modified.
StatusOr<CpuCapabilities> DetectCpuCapabilities(
        const CpuFeaturePolicy& policy = {}) noexcept;

/// @brief Applies a feature policy to an existing capability snapshot.
///
/// Validates both `capabilities` and `policy` invariants, then derives
/// `effective_features` from `usable_features`. Exposed separately so tests
/// can exercise deterministic synthetic snapshots without probing hardware.
///
/// @param capabilities Snapshot to apply the policy to. Must satisfy
///        `hardware_features` superset of `usable_features` and describe a
///        CPU device.
/// @param policy Policy to apply. `disabled_features` are removed from the
///        usable set; `required_features` must remain in the resulting
///        effective set.
/// @return Updated snapshot with `effective_features` derived, or
///         `InvalidArgument` when the snapshot is malformed, or
///         `FailedPrecondition` when the policy requires unsupported
///         effective features.
StatusOr<CpuCapabilities> ApplyCpuFeaturePolicy(
        CpuCapabilities capabilities,
        const CpuFeaturePolicy& policy) noexcept;

} // namespace aethermind::cpu

#endif // AETHERMIND_CPU_INFO_H
