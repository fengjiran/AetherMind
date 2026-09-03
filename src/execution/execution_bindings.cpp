#include "aethermind/execution/execution_bindings.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/memory/allocator.h"
#include "aethermind/runtime/workspace.h"
#include "aethermind/shape_inference/shape_constraint_evaluator.h"
#include "utils/overflow_check.h"

#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

namespace aethermind {
namespace {

constexpr size_t kActivationAlignment = 64;
constexpr size_t kUnassignedOffset = std::numeric_limits<size_t>::max();
constexpr size_t kNoKernelParams = std::numeric_limits<size_t>::max();
constexpr size_t kKernelParamsAlignment = alignof(std::max_align_t);

StatusOr<size_t> AlignKernelParamsOffset(size_t offset) noexcept {
    const size_t remainder = offset % kKernelParamsAlignment;
    if (remainder == 0) {
        return offset;
    }

    size_t aligned = 0;
    if (CheckOverflowAdd(offset, kKernelParamsAlignment - remainder, &aligned)) {
        return Status::Overflow(
                "Kernel params arena alignment overflowed size_t");
    }
    return aligned;
}

struct ConcreteTensorMetadata {
    std::vector<int64_t> shape{};
    std::vector<int64_t> strides{};
};

TensorView MakeReadOnlyView(const MutableTensorView& view) noexcept {
    return {view.data(), view.dtype(), view.shape(), view.strides(), view.alignment()};
}

void SnapshotMetadata(ConcreteTensorMetadata& destination,
                      IntArrayView shape,
                      IntArrayView strides) {
    destination.shape.assign(shape.begin(), shape.end());
    destination.strides.assign(strides.begin(), strides.end());
}


Status ValidateExternalView(const TensorView& view,
                            const TensorSpec& spec,
                            std::unordered_map<int64_t, int64_t>& symbol_values) {
    if (!view.is_valid()) {
        return Status::InvalidArgument(
                "External read-only tensor view is invalid");
    }
    return ValidateConcreteShapeAgainstSpec(
            spec, view.dtype(), view.shape(), "external input", 0, symbol_values);
}

Status ValidateExternalView(const MutableTensorView& view,
                            const TensorSpec& spec,
                            std::unordered_map<int64_t, int64_t>& symbol_values) {
    if (!view.is_valid()) {
        return Status::InvalidArgument(
                "External writable tensor view is invalid");
    }
    return ValidateConcreteShapeAgainstSpec(
            spec, view.dtype(), view.shape(), "external output", 0, symbol_values);
}

StatusOr<ConcreteTensorMetadata> ResolveConcreteMetadata(
        const TensorSpec& spec,
        const std::unordered_map<int64_t, int64_t>& symbol_values) {
    if (!spec.shape.IsRanked()) {
        return Status::FailedPrecondition(
                "Cannot allocate an activation with unranked shape");
    }

    ConcreteTensorMetadata metadata;
    metadata.shape.reserve(*spec.shape.rank());
    for (const ShapeSymbol symbol: spec.shape) {
        if (symbol.IsStatic()) {
            metadata.shape.push_back(symbol.GetStaticValue());
        } else if (symbol.IsSymbolic()) {
            const auto it = symbol_values.find(symbol.value());
            if (it == symbol_values.end()) {
                return Status::FailedPrecondition(
                        "Cannot resolve an activation ShapeSymbol from external bindings");
            }
            metadata.shape.push_back(it->second);
        } else {
            return Status::FailedPrecondition(
                    "Cannot allocate an activation with unconstrained dimensions");
        }
    }

    metadata.strides.assign(metadata.shape.size(), 1);
    for (size_t i = metadata.shape.size(); i > 1; --i) {
        size_t product = 0;
        if (CheckOverflowMul(static_cast<size_t>(metadata.strides[i - 1]),
                             static_cast<size_t>(metadata.shape[i - 1]), &product)) {
            return Status::Overflow(
                    "Activation stride computation overflowed size_t");
        }
        if (product > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
            return Status::Overflow(
                    "Activation stride exceeds int64_t range");
        }
        metadata.strides[i - 2] = static_cast<int64_t>(product);
    }
    return metadata;
}

StatusOr<size_t> ComputeByteSize(const ConcreteTensorMetadata& metadata,
                                 DataType dtype) {
    if (dtype.IsUndefined()) {
        return Status::InvalidArgument(
                "Activation value has undefined dtype");
    }

    size_t elements = 1;
    for (const int64_t dimension: metadata.shape) {
        if (dimension < 0) {
            return Status::InvalidArgument(
                    "Activation dimension cannot be negative");
        }

        if (CheckOverflowMul(elements, static_cast<size_t>(dimension), &elements)) {
            return Status::Overflow(
                    "Activation element count overflowed size_t");
        }
    }

    size_t bytes = 0;
    if (CheckOverflowMul(elements, static_cast<size_t>(dtype.nbytes()), &bytes)) {
        return Status::Overflow(
                "Activation byte size overflowed size_t");
    }
    return bytes;
}

} // namespace

class PreparedExecutionBindingsStorage {
public:
    ExecutionPlanBindingKey plan_key{};
    std::vector<BoundValue> values{};
    std::vector<StepTensorBinding> steps{};
    std::vector<ConcreteTensorMetadata> metadata{};
    Buffer act_storage{};
    /// Type-erased per-step prepared kernel params plus their offsets into
    /// `kernel_params_storage_`. The arena is finalized (resized) before any
    /// params builder runs, so placement-constructed params keep stable
    /// addresses for the prepared bindings' lifetime. Steps without a builder keep the
    /// `kNoKernelParams` sentinel.
    std::vector<std::max_align_t> kernel_params_storage{};
    std::vector<size_t> kernel_params_offsets{};
};

namespace {

void* MutableKernelParamsPointer(PreparedExecutionBindingsStorage& storage,
                                 size_t step_index) noexcept {
    AM_DCHECK(step_index < storage.kernel_params_offsets.size());
    const size_t offset = storage.kernel_params_offsets[step_index];
    AM_DCHECK(offset != kNoKernelParams);
    auto* base = reinterpret_cast<std::byte*>(storage.kernel_params_storage.data());
    return base + offset;
}

} // namespace

PreparedExecutionBindings::PreparedExecutionBindings(
        std::unique_ptr<PreparedExecutionBindingsStorage> storage) noexcept
    : storage_(std::move(storage)) {}

PreparedExecutionBindings::PreparedExecutionBindings() noexcept = default;
PreparedExecutionBindings::PreparedExecutionBindings(PreparedExecutionBindings&&) noexcept = default;
PreparedExecutionBindings& PreparedExecutionBindings::operator=(PreparedExecutionBindings&&) noexcept = default;
PreparedExecutionBindings::~PreparedExecutionBindings() = default;

std::span<const BoundValue> PreparedExecutionBindings::values() const noexcept {
    return storage_ == nullptr ? std::span<const BoundValue>{}
                               : std::span<const BoundValue>{storage_->values};
}

const StepTensorBinding& PreparedExecutionBindings::step(size_t step_index) const noexcept {
    AM_DCHECK(storage_ != nullptr && step_index < storage_->steps.size());
    return storage_->steps[step_index];
}

const void* PreparedExecutionBindings::kernel_params(size_t step_index) const noexcept {
    AM_DCHECK(storage_ != nullptr);
    AM_DCHECK(step_index < storage_->kernel_params_offsets.size());

    const size_t offset = storage_->kernel_params_offsets[step_index];
    if (offset == kNoKernelParams) {
        return nullptr;
    }

    const auto* base = reinterpret_cast<const std::byte*>(
            storage_->kernel_params_storage.data());
    return base + offset;
}

size_t PreparedExecutionBindings::step_count() const noexcept {
    return storage_ == nullptr ? 0 : storage_->steps.size();
}

bool PreparedExecutionBindings::empty() const noexcept {
    return storage_ == nullptr;
}

bool PreparedExecutionBindings::IsCompatible(const ExecutionPlan& plan) const noexcept {
    return storage_ != nullptr && storage_->plan_key.value != 0 &&
           storage_->plan_key == plan.binding_key() && storage_->steps.size() == plan.size();
}

StatusOr<PreparedExecutionBindings> PrepareExecutionBindings(const ExecutionPlan& plan,
                                                             const ExternalTensorBindings& external,
                                                             Allocator& act_allocator) {
    auto storage = std::make_unique<PreparedExecutionBindingsStorage>();
    storage->plan_key = plan.binding_key();
    storage->values.resize(plan.values().size());
    storage->metadata.resize(plan.values().size());
    std::vector<const TensorView*> readable(plan.values().size(), nullptr);
    std::vector<const MutableTensorView*> writable(plan.values().size(), nullptr);
    std::unordered_map<int64_t, int64_t> symbol_values;

    for (const auto& [value, tensor]: external.readable) {
        if (value.index >= plan.values().size()) {
            return Status::InvalidArgument(
                    "External readable binding references an invalid ExecutionValueId");
        }

        if (readable[value.index] != nullptr) {
            return Status::InvalidArgument(
                    "External readable bindings contain a duplicate ExecutionValueId");
        }

        if (auto kind = plan.values()[value.index].kind;
            kind == ExecutionValueKind::kState || kind == ExecutionValueKind::kActivation) {
            return Status::InvalidArgument(
                    "State and activation values cannot use read-only external bindings");
        }

        AM_RETURN_IF_ERROR(ValidateExternalView(tensor, plan.values()[value.index].spec,
                                                symbol_values));
        readable[value.index] = &tensor;
    }

    for (const auto& [value, tensor]: external.writable) {
        if (value.index >= plan.values().size()) {
            return Status::InvalidArgument(
                    "External writable binding references an invalid ExecutionValueId");
        }

        if (writable[value.index] != nullptr) {
            return Status::InvalidArgument(
                    "External writable bindings contain a duplicate ExecutionValueId");
        }

        if (plan.values()[value.index].kind != ExecutionValueKind::kActivation) {
            return Status::InvalidArgument(
                    "Only activation values may use writable external bindings");
        }

        AM_RETURN_IF_ERROR(ValidateExternalView(
                tensor, plan.values()[value.index].spec, symbol_values));
        writable[value.index] = &tensor;
    }

    std::vector<size_t> act_offsets(plan.values().size(), kUnassignedOffset);
    size_t act_bytes = 0;
    for (size_t i = 0; i < plan.values().size(); ++i) {
        const auto& value = plan.values()[i];
        BoundValue& bound = storage->values[i];
        switch (value.kind) {
            case ExecutionValueKind::kModelInput:
            case ExecutionValueKind::kWeight:
            case ExecutionValueKind::kConstant:
                if (readable[i] == nullptr) {
                    return Status::FailedPrecondition("ExecutionPlan value requires "
                                                      "an external read-only binding");
                }

                SnapshotMetadata(storage->metadata[i], readable[i]->shape(),
                                 readable[i]->strides());
                bound.readable = TensorView(
                        readable[i]->data(), readable[i]->dtype(),
                        storage->metadata[i].shape, storage->metadata[i].strides,
                        readable[i]->alignment());
                break;
            case ExecutionValueKind::kState:
                if (readable[i] != nullptr || writable[i] != nullptr) {
                    return Status::InvalidArgument("State values are bound through "
                                                   "ExecutionContext, not TensorView");
                }
                break;
            case ExecutionValueKind::kActivation:
                if (writable[i] != nullptr) {
                    SnapshotMetadata(storage->metadata[i], writable[i]->shape(),
                                     writable[i]->strides());
                    bound.writable = MutableTensorView(
                            writable[i]->data(), writable[i]->dtype(),
                            storage->metadata[i].shape, storage->metadata[i].strides,
                            writable[i]->alignment());
                    bound.readable = MakeReadOnlyView(bound.writable);
                    bound.has_writable = true;
                    break;
                }

                {
                    auto metadata =
                            ResolveConcreteMetadata(value.spec, symbol_values);
                    if (!metadata.ok()) {
                        return metadata.status();
                    }

                    auto bytes = ComputeByteSize(*metadata, value.spec.dtype);
                    if (!bytes.ok()) {
                        return bytes.status();
                    }

                    const auto aligned_offset =
                            AlignWorkspaceOffset(act_bytes, kActivationAlignment);
                    if (!aligned_offset.ok()) {
                        return aligned_offset.status();
                    }

                    size_t next = 0;
                    if (CheckOverflowAdd(*aligned_offset, *bytes, &next)) {
                        return Status::Overflow(
                                "Activation arena size overflowed size_t");
                    }

                    act_offsets[i] = *aligned_offset;
                    act_bytes = next;
                    storage->metadata[i] = std::move(*metadata);
                }
                break;
        }
    }

    storage->act_storage = act_allocator.Allocate(act_bytes);
    if (!storage->act_storage.is_initialized()) {
        return Status::ResourceExhausted(
                "Activation allocator returned an uninitialized Buffer");
    }

    for (size_t i = 0; i < plan.values().size(); ++i) {
        if (act_offsets[i] == kUnassignedOffset) {
            continue;
        }

        const auto& value = plan.values()[i];
        const auto& [shape, strides] = storage->metadata[i];
        auto* data = static_cast<std::byte*>(storage->act_storage.mutable_data()) + act_offsets[i];
        storage->values[i].readable = TensorView(data, value.spec.dtype, shape, strides,
                                                 storage->act_storage.alignment());
        storage->values[i].writable = MutableTensorView(data, value.spec.dtype, shape, strides,
                                                        storage->act_storage.alignment());
        storage->values[i].has_writable = true;
    }

    // Plan the kernel params arena before building any step: align every
    // builder-owned slot to max_align_t and size the arena exactly once. The
    // storage must not be resized after builders placement-construct into it,
    // otherwise prepared params would be invalidated.
    storage->kernel_params_offsets.resize(plan.size(), kNoKernelParams);

    size_t params_bytes = 0;
    for (size_t i = 0; i < plan.size(); ++i) {
        const auto& kernel = plan.steps()[i].kernel;
        if (kernel.params_builder == nullptr) {
            continue;
        }

        const auto aligned = AlignKernelParamsOffset(params_bytes);
        if (!aligned.ok()) {
            return aligned.status();
        }
        params_bytes = *aligned;
        storage->kernel_params_offsets[i] = params_bytes;

        if (CheckOverflowAdd(params_bytes, kernel.params_size, &params_bytes)) {
            return Status::Overflow("Kernel params arena size overflowed size_t");
        }
    }

    constexpr size_t word_size = sizeof(std::max_align_t);
    storage->kernel_params_storage.resize((params_bytes + word_size - 1) / word_size);

    storage->steps.reserve(plan.size());
    for (size_t step_index = 0; step_index < plan.size(); ++step_index) {
        const auto& step = plan.steps()[step_index];
        StepTensorBinding binding;
        binding.inputs.reserve(step.kernel_input_ports.size());
        binding.outputs.reserve(step.kernel_output_ports.size());
        for (const uint32_t port: step.kernel_input_ports) {
            const auto& value = storage->values[step.inputs[port].index];
            if (!value.readable.is_valid()) {
                return Status::FailedPrecondition(
                        "ExecutionPlan kernel input has no canonical TensorView binding");
            }
            binding.inputs.push_back(value.readable);
        }

        for (const uint32_t port: step.kernel_output_ports) {
            const auto& value = storage->values[step.outputs[port].index];
            if (!value.has_writable || !value.writable.is_valid()) {
                return Status::FailedPrecondition(
                        "ExecutionPlan kernel output has no "
                        "canonical writable TensorView binding");
            }
            binding.outputs.push_back(value.writable);
        }

        std::vector<TensorSpec> input_specs;
        input_specs.reserve(step.kernel_input_ports.size());
        for (uint32_t port: step.kernel_input_ports) {
            input_specs.push_back(plan.values()[step.inputs[port].index].spec);
        }

        std::vector<TensorSpec> output_specs;
        output_specs.reserve(step.kernel_output_ports.size());
        for (uint32_t port: step.kernel_output_ports) {
            output_specs.push_back(plan.values()[step.outputs[port].index].spec);
        }

        AM_RETURN_IF_ERROR(ValidateTensorBindingPremises(
                input_specs, output_specs, binding.inputs, binding.outputs));
        AM_RETURN_IF_ERROR(ValidateShapeConstraints(
                step.runtime_checks, binding.inputs, binding.outputs));
        if (step.kernel.params_builder != nullptr) {
            void* params_buffer = MutableKernelParamsPointer(*storage, step_index);
            AM_RETURN_IF_ERROR(step.kernel.params_builder(
                    KernelParamsBuildContext{
                            .inputs = binding.inputs,
                            .outputs = binding.outputs,
                            .attrs = step.kernel.attrs,
                    },
                    params_buffer));
        }
        storage->steps.push_back(std::move(binding));
    }
    return PreparedExecutionBindings(std::move(storage));
}

} // namespace aethermind
