//
// Created by richard on 4/15/26.
//
#ifndef AETHERMIND_BACKEND_DISPATCHER_BRIDGE_H
#define AETHERMIND_BACKEND_DISPATCHER_BRIDGE_H

#include "aethermind/backend/kernel_key.h"
#include "aethermind/backend/kernel_selector.h"
#include "aethermind/base/status.h"
#include "aethermind/operators/op_type.h"

#include "data_type.h"

namespace aethermind {

// Transitional helper for legacy call sites that still construct KernelKey.
// Batch 1 keeps this bridge alive only to avoid mixing migration work with the
// selector-based resolve changes scheduled for later batches.
KernelKey MakeKernelKey(DeviceType device_type, const OperatorName& op_name);

// Transitional helper for code paths that still start from OperatorName but
// need to enter the new OpType-centered dispatch mainline.
AM_NODISCARD StatusOr<OpType> ToOpType(const OperatorName& op_name) noexcept;

// Transitional helper for assembling a selector from legacy call sites without
// reintroducing the old global dispatcher path.
AM_NODISCARD KernelSelector MakeKernelSelector(DeviceType device_type,
                                               const DataType& activation_dtype,
                                               const DataType& weight_dtype,
                                               WeightFormat weight_format = WeightFormat::kPlain,
                                               IsaLevel isa = IsaLevel::kScalar,
                                               ExecPhase phase = ExecPhase::kBoth) noexcept;

}// namespace aethermind
#endif
