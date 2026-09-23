#include "aethermind/inference/executable_model.h"

#include "aethermind/compiler/packing_request_builder.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/graph/graph_types.h"
#include "aethermind/model/raw_weight.h"
#include "aethermind/model/resolved_model_weights.h"
#include "aethermind/model/weight/weight_binding_resolver.h"
#include "aethermind/model/weight/weight_packing.h"
#include "aethermind/runtime/runtime.h"
#include "utils/overflow_check.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace aethermind {
namespace {

/// @brief Names a Transformer weight role for error messages.
///
/// Labels mirror ModelGraph dumps so a failing preparation can be read against
/// a graph dump. Kept file-local because the dump header is outside inference's
/// allowed dependency surface.
const char* WeightRoleLabel(TransformerWeightRole role) noexcept {
    switch (role) {
        case TransformerWeightRole::kTokenEmbedding:
            return "TokenEmbedding";
        case TransformerWeightRole::kInputNorm:
            return "InputNorm";
        case TransformerWeightRole::kAttentionQ:
            return "AttentionQ";
        case TransformerWeightRole::kAttentionK:
            return "AttentionK";
        case TransformerWeightRole::kAttentionV:
            return "AttentionV";
        case TransformerWeightRole::kAttentionO:
            return "AttentionO";
        case TransformerWeightRole::kMlpGate:
            return "MlpGate";
        case TransformerWeightRole::kMlpUp:
            return "MlpUp";
        case TransformerWeightRole::kMlpDown:
            return "MlpDown";
        case TransformerWeightRole::kPostAttentionNorm:
            return "PostAttentionNorm";
        case TransformerWeightRole::kFinalNorm:
            return "FinalNorm";
        case TransformerWeightRole::kLmHead:
            return "LmHead";
        case TransformerWeightRole::kMoERouter:
            return "MoERouter";
    }
    return "UnknownTransformerWeightRole";
}

/// @brief Describes a weight binding's identity for error messages.
///
/// A nullptr resolution is otherwise indistinguishable from a structurally
/// absent weight, so the message carries the semantic role and layer index.
std::string DescribeWeightBinding(const WeightBinding& binding) {
    const std::optional<TransformerWeightRole> role = TryGetTransformerWeightRole(binding);
    std::string description = "role=";
    description += role.has_value() ? WeightRoleLabel(*role) : "<none>";
    if (binding.decoder_layer_index.has_value()) {
        description += ", layer=" + std::to_string(*binding.decoder_layer_index);
    }
    return description;
}

/// @brief Derives the artifact's single execution phase from its steps.
///
/// Lowering records one selector per step. A mixture would make "one shared plan
/// serves this phase" undecidable, so it is rejected at preparation time rather
/// than deferred to a phase query.
StatusOr<ExecPhase> ResolveArtifactPhase(const ExecutionPlan& plan) {
    ExecPhase phase = ExecPhase::kBoth;
    bool first = true;
    for (const ExecutionStep& step: plan.steps()) {
        if (first) {
            phase = step.selector.phase;
            first = false;
        } else if (step.selector.phase != phase) {
            return Status::FailedPrecondition(
                    "PrepareExecutableModel: artifact mixes execution phases (" +
                    std::string(ToString(phase)) + " and " +
                    std::string(ToString(step.selector.phase)) + ")");
        }
    }
    return phase;
}

/// @brief Materializes one constant's inline bytes as a tensor view.
///
/// Execution never reads ConstantBinding::inline_data directly: it requires an
/// external binding for every constant, so preparation has to turn the payload
/// bytes into a view. The bytes are owned by the artifact payload and stay alive
/// because the artifact is owned by the model being prepared.
StatusOr<TensorView> MaterializeConstant(WeightBindingStorage& storage,
                                         const ConstantValue& constant,
                                         const TensorSpec& spec,
                                         uint32_t value_index) {
    if (constant.binding.inline_data == nullptr) {
        return Status::FailedPrecondition(
                "PrepareExecutableModel: constant value " + std::to_string(value_index) +
                " carries no inline data");
    }
    if (!spec.shape.IsStatic()) {
        return Status::FailedPrecondition(
                "PrepareExecutableModel: constant value " + std::to_string(value_index) +
                " has a non-static shape");
    }

    std::vector<int64_t> dims;
    dims.reserve(spec.shape.rank().value_or(0));
    uint64_t elements = 1;
    for (const ShapeSymbol& dim: spec.shape) {
        const int64_t extent = dim.GetStaticValue();
        if (extent < 0) {
            return Status::FailedPrecondition(
                    "PrepareExecutableModel: constant value " + std::to_string(value_index) +
                    " has a negative dimension");
        }
        dims.push_back(extent);
        if (CheckOverflowMul(elements, static_cast<uint64_t>(extent), &elements)) {
            return Status::FailedPrecondition(
                    "PrepareExecutableModel: constant value " + std::to_string(value_index) +
                    " shape product overflows");
        }
    }

    uint64_t expected_bytes = 0;
    if (CheckOverflowMul(elements, static_cast<uint64_t>(spec.dtype.nbytes()),
                         &expected_bytes) ||
        constant.binding.inline_data->size() != expected_bytes) {
        return Status::FailedPrecondition(
                "PrepareExecutableModel: constant value " + std::to_string(value_index) +
                " inline data size does not match its logical shape");
    }

    return storage.Add(constant.binding.inline_data->data(), spec.dtype, std::move(dims));
}

/// @brief Materializes one required weight or constant value as a tensor view.
StatusOr<TensorView> MaterializeBinding(WeightBindingStorage& storage,
                                        const LoweredValueDesc& value,
                                        const ResolvedModelWeights& resolved,
                                        ExecutionValueKind kind,
                                        uint32_t value_index) {
    if (kind == ExecutionValueKind::kConstant) {
        const auto* constant = std::get_if<ConstantValue>(&value.payload);
        if (constant == nullptr) {
            return Status::Internal(
                    "PrepareExecutableModel: constant value " + std::to_string(value_index) +
                    " has no ConstantValue payload");
        }
        return MaterializeConstant(storage, *constant, value.spec, value_index);
    }
    if (kind != ExecutionValueKind::kWeight) {
        return Status::Internal(
                "PrepareExecutableModel: value " + std::to_string(value_index) +
                " requires an external binding but is neither a weight nor a constant");
    }

    const auto* weight = std::get_if<WeightValue>(&value.payload);
    if (weight == nullptr) {
        return Status::Internal(
                "PrepareExecutableModel: weight value " + std::to_string(value_index) +
                " has no WeightValue payload");
    }
    const RawWeightView* raw = ResolveWeightBinding(weight->binding, resolved);
    if (raw == nullptr) {
        return Status::FailedPrecondition(
                "PrepareExecutableModel: weight value " + std::to_string(value_index) +
                " (" + DescribeWeightBinding(weight->binding) +
                ") has no resolvable raw weight");
    }
    AM_RETURN_IF_ERROR(ValidateRawWeightView(*raw));
    if (!raw->is_contiguous) {
        return Status::FailedPrecondition(
                "PrepareExecutableModel: weight value " + std::to_string(value_index) +
                " is not contiguous");
    }
    return storage.Add(raw->data, raw->dtype, raw->shape);
}

} // namespace

ExecutableModel::ExecutableModel(LoweredModelArtifact artifact,
                                 PackedWeightStore packed_weights,
                                 WeightBindingStorage binding_storage,
                                 ExternalTensorBindings bindings,
                                 ExecutionPlan plan,
                                 ExecPhase phase) noexcept
    : artifact_(std::move(artifact)),
      packed_weights_(std::move(packed_weights)),
      binding_storage_(std::move(binding_storage)),
      bindings_(std::move(bindings)),
      plan_(std::move(plan)),
      phase_(phase) {}

Status ExecutableModel::CheckPhase(ExecPhase phase) const noexcept {
    if (!PhaseMatches(phase_, phase)) {
        return Status::FailedPrecondition(
                std::string("ExecutableModel was compiled for phase ") + ToString(phase_) +
                ", but phase " + ToString(phase) + " was requested");
    }
    return Status::Ok();
}

StatusOr<const ExecutionPlan*> ExecutableModel::plan(ExecPhase phase) const noexcept {
    AM_RETURN_IF_ERROR(CheckPhase(phase));
    return &plan_;
}

StatusOr<const ExternalTensorBindings*> ExecutableModel::immutable_weight_bindings(
        ExecPhase phase) const noexcept {
    AM_RETURN_IF_ERROR(CheckPhase(phase));
    return &bindings_;
}

uint64_t ExecutableModel::artifact_id() const noexcept {
    return artifact_.graph.artifact_id();
}

ExecPhase ExecutableModel::phase() const noexcept {
    return phase_;
}

StatusOr<ExecutableModel> PrepareExecutableModel(Runtime& runtime,
                                                 LoweredModelArtifact artifact) {
    if (artifact.loaded_model == nullptr) {
        return Status::InvalidArgument(
                "PrepareExecutableModel: artifact carries no loaded model");
    }
    const ResolvedModelWeights& resolved = artifact.loaded_model->GetResolvedWeights();

    const auto requests = BuildWeightPackingRequests(artifact.graph, resolved);
    if (!requests.ok()) {
        return requests.status();
    }

    PackedWeightStore packed_weights;
    AM_RETURN_IF_ERROR(packed_weights.SetSourceId(artifact.graph.artifact_id()));
    AM_RETURN_IF_ERROR(PrepackWeightRequests(packed_weights, *requests));

    const auto plan = ExecutionPlanBuilder::Build(runtime, packed_weights, artifact.graph);
    if (!plan.ok()) {
        return plan.status();
    }
    const auto phase = ResolveArtifactPhase(*plan);
    if (!phase.ok()) {
        return phase.status();
    }
    const auto required = ComputeExternalReadRequirements(*plan);
    if (!required.ok()) {
        return required.status();
    }

    WeightBindingStorage binding_storage;
    ExternalTensorBindings bindings;
    std::vector<bool> bound(plan->values().size(), false);
    for (uint32_t index = 0; index < required->size(); ++index) {
        if (!(*required)[index]) {
            continue;
        }
        const ExecutionValueKind kind = plan->values()[index].kind;
        // Model inputs are session-supplied per phase; preparation holds no data
        // for them.
        if (kind == ExecutionValueKind::kModelInput) {
            continue;
        }
        if (index >= artifact.graph.values().size()) {
            return Status::Internal(
                    "PrepareExecutableModel: plan value " + std::to_string(index) +
                    " is beyond the artifact's values");
        }
        const auto view = MaterializeBinding(binding_storage, artifact.graph.values()[index],
                                             resolved, kind, index);
        if (!view.ok()) {
            return view.status();
        }
        bindings.readable.push_back({.value = {.index = index}, .tensor = *view});
        bound[index] = true;
    }

    // Reconcile in both directions. A binding the plan does not require would
    // shadow a packed artifact or an activation, and a required value left unbound
    // would only fail at the first PrepareExecutionBindings call.
    for (uint32_t index = 0; index < bound.size(); ++index) {
        const bool expected = (*required)[index] &&
                              plan->values()[index].kind != ExecutionValueKind::kModelInput;
        if (bound[index] != expected) {
            return Status::Internal(
                    "PrepareExecutableModel: binding reconciliation failed for value " +
                    std::to_string(index));
        }
    }

    return ExecutableModel(std::move(artifact), std::move(packed_weights),
                           std::move(binding_storage), std::move(bindings),
                           std::move(*plan), *phase);
}

} // namespace aethermind
