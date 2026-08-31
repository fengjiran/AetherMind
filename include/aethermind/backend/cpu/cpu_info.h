//
// Created by richard on 6/6/26.
//

#ifndef AETHERMIND_CPU_INFO_H
#define AETHERMIND_CPU_INFO_H

#include "aethermind/backend/cpu/cpu_capabilities.h"
#include "aethermind/base/status.h"

namespace aethermind {
namespace cpu {

/// Detects an immutable CPU capability snapshot for the current process.
/// The returned effective features are always derived from usable features by
/// applying `policy`; a policy cannot enable unsupported instructions.
///
/// 注意：检测过程可能发起进程级 OS 请求——在 x86-64 Linux 上会调用
/// arch_prctl(ARCH_REQ_XCOMP_PERM) 申请 AMX XTILEDATA 权限；除此类权限申请外，
/// 不会改变任何 CPU 状态。
AM_NODISCARD StatusOr<CpuCapabilities> DetectCpuCapabilities(
        const CpuFeaturePolicy& policy = {}) noexcept;

/// Applies `policy` to a snapshot, validating snapshot and policy invariants.
/// Exposed separately so tests can exercise deterministic synthetic snapshots.
AM_NODISCARD StatusOr<CpuCapabilities> ApplyCpuFeaturePolicy(
        CpuCapabilities capabilities,
        const CpuFeaturePolicy& policy) noexcept;

}// namespace cpu
}// namespace aethermind

#endif// AETHERMIND_CPU_INFO_H
