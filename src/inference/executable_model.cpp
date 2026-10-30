#include "aethermind/inference/executable_model.h"
#include "aethermind/model/weight/weight_binding_resolver.h"
#include "inference/inference_internal.h"

#include "aethermind/backend/backend.h"
#include "aethermind/compiler/weight_packing_request_builder.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/graph/graph_types.h"
#include "aethermind/inference/model_execution_policy.h"
#include "aethermind/model/model_loader.h"
#include "aethermind/model/raw_weight.h"
#include "aethermind/model/resolved_model_weights.h"
#include "aethermind/model/weight/packed_weight_collection.h"
#include "aethermind/model/weight/weight_packing_request.h"
#include "aethermind/runtime/runtime.h"
#include "utils/overflow_check.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
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
    for (const auto& step: plan.steps()) {
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
StatusOr<TensorView> MaterializeConstant(WeightBindingMetadata& metadata,
                                         const ConstantValue& constant,
                                         const TensorSpec& spec,
                                         uint32_t value_index) {
    if (constant.binding.inline_data == nullptr) {
        return Status::FailedPrecondition(
                "PrepareExecutableModel: constant value " +
                std::to_string(value_index) + " carries no inline data");
    }

    if (!spec.shape.IsStatic()) {
        return Status::FailedPrecondition(
                "PrepareExecutableModel: constant value " +
                std::to_string(value_index) + " has a non-static shape");
    }

    std::vector<int64_t> dims;
    dims.reserve(spec.shape.rank().value_or(0));
    uint64_t elements = 1;
    for (const auto& dim: spec.shape) {
        const int64_t extent = dim.GetStaticValue();
        if (extent < 0) {
            return Status::FailedPrecondition(
                    "PrepareExecutableModel: constant value " +
                    std::to_string(value_index) + " has a negative dimension");
        }

        dims.push_back(extent);
        if (CheckOverflowMul(elements, static_cast<uint64_t>(extent), &elements)) {
            return Status::FailedPrecondition(
                    "PrepareExecutableModel: constant value " +
                    std::to_string(value_index) + " shape product overflows");
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

    return metadata.Add(constant.binding.inline_data->data(), spec.dtype, std::move(dims));
}

/// @brief Materializes one required weight or constant value as a tensor view.
StatusOr<TensorView> MaterializeBinding(WeightBindingMetadata& metadata,
                                        const LoweredValueDesc& value,
                                        const ResolvedModelWeights& resolved,
                                        ExecutionValueKind kind,
                                        uint32_t value_index) {
    if (kind == ExecutionValueKind::kConstant) {
        const auto* constant = std::get_if<ConstantValue>(&value.payload);
        if (constant == nullptr) {
            return Status::Internal(
                    "PrepareExecutableModel: constant value " +
                    std::to_string(value_index) + " has no ConstantValue payload");
        }
        return MaterializeConstant(metadata, *constant, value.spec, value_index);
    }

    if (kind != ExecutionValueKind::kWeight) {
        return Status::Internal(
                "PrepareExecutableModel: value " + std::to_string(value_index) +
                " requires an external binding but is neither a weight nor a constant");
    }

    const auto* weight = std::get_if<WeightValue>(&value.payload);
    if (weight == nullptr) {
        return Status::Internal(
                "PrepareExecutableModel: weight value " +
                std::to_string(value_index) + " has no WeightValue payload");
    }

    const RawWeightView* raw = ResolveWeightBinding(weight->binding, resolved);
    if (raw == nullptr) {
        return Status::FailedPrecondition(
                "PrepareExecutableModel: weight value " + std::to_string(value_index) +
                " (" + DescribeWeightBinding(weight->binding) + ") has no resolvable raw weight");
    }

    AM_RETURN_IF_ERROR(ValidateRawWeightView(*raw));
    if (!raw->is_contiguous) {
        return Status::FailedPrecondition(
                "PrepareExecutableModel: weight value " +
                std::to_string(value_index) + " is not contiguous");
    }
    return metadata.Add(raw->data, raw->dtype, raw->shape);
}

} // namespace

ExecutableModel::ExecutableModel(Runtime& runtime,
                                 LoweredModelArtifact artifact,
                                 PackedWeightCollection packed_weights,
                                 WeightBindingMetadata binding_metadata,
                                 ExternalTensorBindings bindings,
                                 ExecutionPlan plan,
                                 ExecPhase phase,
                                 size_t context_limit,
                                 size_t vocab_size,
                                 ModelResourceUsage resource_usage) noexcept
    : runtime_(&runtime),
      artifact_(std::move(artifact)),
      packed_weights_(std::move(packed_weights)),
      binding_metadata_(std::move(binding_metadata)),
      bindings_(std::move(bindings)),
      plan_(std::move(plan)),
      phase_(phase),
      context_limit_(context_limit),
      vocab_size_(vocab_size),
      resource_usage_(resource_usage) {}

const ModelResourceUsage& ExecutableModel::resource_usage() const noexcept {
    return resource_usage_;
}

Status ExecutableModel::CheckPhase(ExecPhase phase) const noexcept {
    if (!PhaseMatches(phase_, phase)) {
        return Status::FailedPrecondition(
                std::string("ExecutableModel was compiled for phase ") +
                ToString(phase_) + ", but phase " + ToString(phase) + " was requested");
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

size_t ExecutableModel::context_limit() const noexcept {
    return context_limit_;
}

size_t ExecutableModel::vocab_size() const noexcept {
    return vocab_size_;
}

bool ExecutableModel::IsPreparedFor(const Runtime& runtime) const noexcept {
    return runtime_ == &runtime;
}

StatusOr<std::vector<WeightPackingRequest>>
inference::internal::ResolveWeightPackingRequests(const Backend& backend,
                                                  std::vector<WeightPackingRequest> requests,
                                                  DeviceType expected_device) {
    std::vector<WeightPackingRequest> resolved_requests;
    std::unordered_map<uint32_t, std::unordered_map<KernelSelector, size_t>> request_index;
    for (auto& request: requests) {
        if (request.selector.device_type != expected_device) {
            return Status::InvalidArgument(
                    "PrepareExecutableModel: packed weights span multiple devices");
        }

        AM_ASSIGN_OR_RETURN(request.recipe,
                            backend.GetPackingRecipe(request.op_type, request.selector));
        auto& selectors = request_index[request.value_index];
        if (const auto existing = selectors.find(request.selector); existing != selectors.end()) {
            if (const auto& first = resolved_requests[existing->second];
                first.op_type != request.op_type || first.binding != request.binding ||
                first.recipe != request.recipe) {
                return Status::InvalidArgument(
                        "PrepareExecutableModel: shared packed weight consumers "
                        "require incompatible op types or packing recipes");
            }
            continue;
        }
        selectors.emplace(request.selector, resolved_requests.size());
        resolved_requests.push_back(std::move(request));
    }
    return resolved_requests;
}

StatusOr<ExecutableModel> PrepareExecutableModel(Runtime& runtime,
                                                 LoweredModelArtifact artifact) {
    return PrepareExecutableModel(runtime, std::move(artifact), {});
}

StatusOr<ExecutableModel> PrepareExecutableModel(Runtime& runtime,
                                                 LoweredModelArtifact artifact,
                                                 const ExecutableModelPrepareOptions& options) try {
    if (artifact.loaded_model == nullptr) {
        return Status::InvalidArgument("PrepareExecutableModel: artifact "
                                       "carries no loaded model");
    }

    const HfModelConfig& model_config = artifact.loaded_model->GetConfig();
    if (model_config.vocab_size <= 0 || model_config.max_position_embeddings <= 0) {
        return Status::InvalidArgument("PrepareExecutableModel: loaded model has "
                                       "invalid vocabulary or context limit");
    }

    if (!std::in_range<size_t>(model_config.vocab_size) ||
        !std::in_range<size_t>(model_config.max_position_embeddings)) {
        return Status::Overflow("PrepareExecutableModel: vocabulary or "
                                "context limit does not fit size_t");
    }

    const auto vocab_size = static_cast<size_t>(model_config.vocab_size);
    const auto context_limit = static_cast<size_t>(model_config.max_position_embeddings);

    const ResolvedModelWeights& resolved = artifact.loaded_model->GetResolvedWeights();
    ModelResourceUsage usage;
    usage.weight_normalization = artifact.loaded_model->GetWeightNormalizationUsage();
    auto weight_usage = SummarizeModelWeightStorage(resolved);
    if (!weight_usage.ok()) {
        if (options.max_model_payload_bytes != 0 ||
            weight_usage.status().code() == StatusCode::kResourceExhausted) {
            return weight_usage.status();
        }
        // Reporting must not strengthen the generic unlimited preparation
        // contract for raw roles the graph never consumes.
        usage.weights.backing_complete = false;
    } else {
        usage.weights = *weight_usage;
    }
    // Distinct vector objects may alias one shared control block; charge each
    // visible payload once, rather than treating that entire owner as one vector.
    std::set<const std::vector<std::byte>*, std::less<const std::vector<std::byte>*>> constants;
    for (const auto& value: artifact.graph.values()) {
        const auto* constant = std::get_if<ConstantValue>(&value.payload);
        if (constant != nullptr && constant->binding.inline_data != nullptr &&
            constants.insert(constant->binding.inline_data.get()).second) {
            if (CheckOverflowAdd(usage.inline_constant_bytes,
                                 constant->binding.inline_data->size(), &usage.inline_constant_bytes)) {
                return Status::Overflow("Inline constant payload overflowed size_t");
            }
        }
    }
    if (CheckOverflowAdd(usage.weights.owned_bytes, usage.inline_constant_bytes,
                         &usage.managed_payload_bytes)) {
        return Status::Overflow("Model payload overflowed size_t");
    }
    std::optional<size_t> packed_cap;
    if (options.max_packed_weight_bytes != 0) {
        packed_cap = options.max_packed_weight_bytes;
    }
    if (options.max_model_payload_bytes != 0) {
        if (!usage.weights.backing_complete) {
            return Status::Unimplemented("Model payload limit requires complete raw storage footprints");
        }
        if (usage.managed_payload_bytes > options.max_model_payload_bytes) {
            return Status::ResourceExhausted("Model payload exceeds limit: required=" +
                                             std::to_string(usage.managed_payload_bytes) + ", limit=" +
                                             std::to_string(options.max_model_payload_bytes));
        }
        const size_t remaining = options.max_model_payload_bytes - usage.managed_payload_bytes;
        packed_cap = packed_cap.has_value() ? std::min(*packed_cap, remaining) : remaining;
    }
    auto requests = BuildWeightPackingRequests(
            artifact.graph, resolved);
    if (!requests.ok()) {
        return requests.status();
    }

    // Resolve the backend once from the first request's device (empty requests
    // skip packing entirely and default to CPU, mirroring plan construction).
    const DeviceType device = requests->empty()
                                      ? DeviceType::kCPU
                                      : requests->front().selector.device_type;
    const auto backend = runtime.GetBackend(device);
    AM_RETURN_IF_ERROR(backend.status());

    AM_ASSIGN_OR_RETURN(std::vector<WeightPackingRequest> resolved_requests,
                        inference::internal::ResolveWeightPackingRequests(
                                **backend, std::move(*requests), device));
    // Packed artifacts are managed storage, so they are allocated through the
    // Runtime's allocator and follow its storage policy.
    AM_ASSIGN_OR_RETURN(Allocator * pack_allocator,
                        runtime.TryGetAllocator(Device(device)));
    AM_ASSIGN_OR_RETURN(PackedWeightCollection packed_weights,
                        PrepackWeightRequests(**backend, resolved_requests,
                                              *pack_allocator, packed_cap));

    AM_ASSIGN_OR_RETURN(usage.packed_weight_bytes, packed_weights.total_storage_bytes());
    usage.packing_plan_complete = packed_weights.storage_plan_complete();
    if (CheckOverflowAdd(usage.managed_payload_bytes, usage.packed_weight_bytes,
                         &usage.managed_payload_bytes)) {
        return Status::Overflow("Model payload overflowed size_t");
    }
    auto plan = ExecutionPlanBuilder::Build(runtime,
                                            packed_weights, artifact.graph);
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

    WeightBindingMetadata binding_metadata;
    ExternalTensorBindings bindings;
    std::vector<bool> bound(plan->values().size(), false);
    for (uint32_t i = 0; i < required->size(); ++i) {
        if (!(*required)[i]) {
            continue;
        }

        const auto kind = plan->values()[i].kind;
        // Model inputs are session-supplied per phase; preparation holds no data
        // for them.
        if (kind == ExecutionValueKind::kModelInput) {
            continue;
        }

        if (i >= artifact.graph.values().size()) {
            return Status::Internal(
                    "PrepareExecutableModel: plan value " + std::to_string(i) +
                    " is beyond the artifact's values");
        }

        auto view = MaterializeBinding(binding_metadata,
                                       artifact.graph.values()[i],
                                       resolved, kind, i);
        if (!view.ok()) {
            return view.status();
        }

        bindings.read_only.push_back({.value = {.index = i}, .tensor = *view});
        bound[i] = true;
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

    return ExecutableModel(runtime, std::move(artifact), std::move(packed_weights),
                           std::move(binding_metadata), std::move(bindings),
                           std::move(*plan), *phase,
                           context_limit,
                           vocab_size, usage);
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

StatusOr<ExecutableModel> LoadAndPrepareExecutableModel(
        Runtime& runtime, const std::filesystem::path& model_dir,
        const ModelCompileOptions& compile_options) {
    return LoadAndPrepareExecutableModel(runtime, model_dir, compile_options, {});
}

StatusOr<ExecutableModel> LoadAndPrepareExecutableModel(
        Runtime& runtime,
        const std::filesystem::path& model_dir,
        const ModelCompileOptions& compile_options,
        const ExecutableModelPrepareOptions& prepare_options) {
    auto artifact = ModelCompiler::LoadAndCompile(model_dir, compile_options);
    if (!artifact.ok()) {
        const Status& status = artifact.status();
        return status.WithMessage(
                "LoadAndPrepareExecutableModel loading/compilation failed: " +
                status.message());
    }

    auto executable = PrepareExecutableModel(runtime, std::move(*artifact), prepare_options);
    if (!executable.ok()) {
        const Status& status = executable.status();
        return status.WithMessage(
                "LoadAndPrepareExecutableModel preparation failed: " +
                status.message());
    }
    return std::move(*executable);
}

StatusOr<ExecutableModel> LoadAndPrepareCpuExecutableModel(
        Runtime& runtime, const std::filesystem::path& model_dir,
        const CpuModelPrepareOptions& options) try {
    if (options.precision != InferencePrecisionPreset::kStrictFp32 &&
        options.precision != InferencePrecisionPreset::kFp32Compatible) {
        return Status::InvalidArgument("Unknown inference precision preset");
    }
    AM_ASSIGN_OR_RETURN(auto compile_options, MakeCpuCompileOptions(options.cpu_preset));
    const bool compatible = options.precision == InferencePrecisionPreset::kFp32Compatible;
    if (compatible && options.cpu_preset != CpuInferencePreset::kO1Plain) {
        return Status::Unimplemented("FP32 compatibility requires CPU O1/plain");
    }
    compile_options.graph_build.activation_dtype = DataType::Float32();
    compile_options.graph_build.kv_dtype = DataType::Float32();

    AM_ASSIGN_OR_RETURN(auto loaded, ModelLoader::Load(model_dir));
    if (compatible) {
        AM_RETURN_IF_ERROR(ValidateFp32CompatibleSourceInputs(*loaded));
        AM_ASSIGN_OR_RETURN(Allocator * allocator, runtime.TryGetAllocator(Device::CPU()));
        auto normalized = NormalizeLoadedModelWeightsToFp32(
                *loaded, *allocator,
                {.max_owned_weight_bytes = options.resources.max_model_payload_bytes});
        if (!normalized.ok()) {
            return normalized.status().WithMessage(
                    "FP32 compatibility normalization failed: " + normalized.status().message());
        }
        // Drop source-only mappings before compiling or packing. Shared F32
        // source owners remain alive through the normalized model's views.
        loaded = std::move(*normalized);
        AM_RETURN_IF_ERROR(ValidateFp32ExecutionWeights(*loaded));
    } else {
        AM_RETURN_IF_ERROR(ValidateFp32ExecutionInputs(*loaded));
    }

    auto artifact = ModelCompiler::Compile(std::move(loaded), compile_options);
    if (!artifact.ok()) {
        return artifact.status().WithMessage(
                "CPU model compilation failed: " + artifact.status().message());
    }
    auto executable = PrepareExecutableModel(runtime, std::move(*artifact), options.resources);
    if (!executable.ok()) {
        return executable.status().WithMessage(
                "CPU model preparation failed: " + executable.status().message());
    }
    return std::move(*executable);
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

} // namespace aethermind
