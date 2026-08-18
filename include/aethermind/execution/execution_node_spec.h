#ifndef AETHERMIND_EXECUTION_EXECUTION_NODE_SPEC_H
#define AETHERMIND_EXECUTION_EXECUTION_NODE_SPEC_H

#include "aethermind/base/device.h"
#include "aethermind/base/kernel_attrs.h"
#include "aethermind/base/workspace_types.h"
#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/shape_inference/shape_constraint.h"
#include "aethermind/shape_inference/tensor_spec.h"

#include <cstddef>
#include <vector>

namespace aethermind {

struct ExecutionPlanNodeSpec {
    OpType op_type = OpType::kUnknown;
    DeviceType device_type = DeviceType::kCPU;
    DataType act_dtype{};
    DataType weight_dtype{};
    WeightFormat weight_format = WeightFormat::kPlain;
    IsaLevel isa = IsaLevel::kScalar;
    ExecPhase phase = ExecPhase::kBoth;
    WorkspaceRequirement workspace_requirement{};
    /// Complete schema-port-ordered input specs, including state ports that
    /// do not contribute to runtime tensor bindings. Use MakeCompactInputSpecs
    /// to derive the compact contributing-only view consumed by operator
    /// semantic inference and ExecutionStep::input_specs.
    std::vector<TensorSpec> input_specs{};
    std::vector<TensorSpec> output_specs{};
    /// Deferred runtime shape constraints derived during graph construction
    /// and carried through lowering without re-inference.
    std::vector<ShapeConstraint> runtime_checks{};
    OpParams op_params{};
};

}// namespace aethermind

#endif
