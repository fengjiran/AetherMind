#include "aethermind/model/graph/optimization/constant_folding_pass.h"
#include "aethermind/model/graph/operator_schema.h"
#include "aethermind/model/graph/optimization/const_evaluator.h"

#include <cstddef>
#include <memory>
#include <span>
#include <variant>
#include <vector>

namespace aethermind {
namespace {

// Aggregates read-only TensorViews with their backing shape/strides.
// Backing storage is declared first so it is destroyed after the borrowing views.
struct InputViews {
    std::vector<std::vector<int64_t>> shapes{};
    std::vector<std::vector<int64_t>> strides{};
    std::vector<TensorView> views{};
};

// Aggregates mutable output views with the owning buffers.
// `buffers` owns the byte storage; views borrow into both the buffers and
// metadata. OutputStorage must outlive any use of its views.
struct OutputStorage {
    std::vector<std::shared_ptr<std::vector<std::byte>>> buffers{};
    std::vector<std::vector<int64_t>> shapes{};
    std::vector<std::vector<int64_t>> strides{};
    std::vector<MutableTensorView> views{};
};

StatusOr<std::vector<GraphValueDesc>> CollectValueDescs(const GraphRewriteSession& session,
                                                        std::span<const GraphValueId> values) {
    std::vector<GraphValueDesc> descs;
    descs.reserve(values.size());
    for (const auto value: values) {
        auto desc = session.GetValueOutputMetadata(value);
        AM_RETURN_IF_ERROR(desc.status());
        descs.push_back(std::move(*desc));
    }
    return descs;
}

// Returns true when the output desc carries a ConstantValue with inline_data
// whose byte count matches the tensor spec. Dynamic shapes (Unimplemented
// from CountBytes) are treated as "no" — they cannot be folded.
StatusOr<bool> HasInlineConstantBytes(const GraphValueDesc& desc) {
    const auto* constant = std::get_if<ConstantValue>(&desc.payload);
    if (constant == nullptr || !constant->binding.inline_data) {
        return false;
    }

    auto expected_bytes = CountBytes(desc.spec);
    if (!expected_bytes.ok()) {
        if (expected_bytes.status().code() == StatusCode::kUnimplemented) {
            return false;
        }
        return expected_bytes.status();
    }
    return constant->binding.inline_data->size() == *expected_bytes;
}

// Iterates all inputs and returns false if any lacks inline constant data
// with matching byte count. This is the gate that ensures only fully-materialised
// constant subgraphs enter the evaluator.
StatusOr<bool> AllInputsAreInlineConstantValues(std::span<const GraphValueDesc> inputs) {
    for (const GraphValueDesc& input: inputs) {
        auto has_inline_bytes = HasInlineConstantBytes(input);
        AM_RETURN_IF_ERROR(has_inline_bytes.status());
        if (!*has_inline_bytes) {
            return false;
        }
    }
    return true;
}

// Converts each input's ConstantValue::inline_data into a TensorView so the
// evaluator can read the bytes. The views borrow into result.shapes,
// result.strides, and the ConstantBinding inline_data heap; all backing
// storage must outlive the evaluator call.
StatusOr<InputViews> BuildInputViews(std::span<const GraphValueDesc> inputs) {
    InputViews result;
    // reserve() on shapes/strides is critical: TensorView stores IntArrayView
    // (span) pointing into these vectors. Without reserve, push_back reallocation
    // would dangle spans already stored in result.views.
    result.views.reserve(inputs.size());
    result.shapes.reserve(inputs.size());
    result.strides.reserve(inputs.size());

    for (const auto& input: inputs) {
        const auto* constant = std::get_if<ConstantValue>(&input.payload);
        if (constant == nullptr || !constant->binding.inline_data) {
            return Status::InvalidArgument(
                    "constant folding input lacks inline constant data");
        }

        auto shape = ExtractStaticShape(input.spec);
        AM_RETURN_IF_ERROR(shape.status());
        auto strides = MakeContiguousStrides(*shape);
        AM_RETURN_IF_ERROR(strides.status());
        result.shapes.push_back(std::move(*shape));
        result.strides.push_back(std::move(*strides));
        result.views.emplace_back(constant->binding.inline_data->data(),
                                  input.spec.dtype,
                                  result.shapes.back(),
                                  result.strides.back());
    }
    return result;
}

// Allocates output buffers per the evaluator's plan, validates the byte size
// and stride rank, and builds the MutableTensorViews for Evaluate() to write
// into. The buffers are shared_ptr so they can be transferred to
// ConstantBinding::inline_data without copying.
StatusOr<OutputStorage> AllocateOutputViews(const ConstEvalPlan& plan) {
    OutputStorage result;
    // reserve() on shapes/strides is critical: MutableTensorView stores
    // IntArrayView (span) pointing into these vectors. Without reserve,
    // push_back reallocation would dangle spans already stored in result.views.
    result.views.reserve(plan.outputs.size());
    result.buffers.reserve(plan.outputs.size());
    result.shapes.reserve(plan.outputs.size());
    result.strides.reserve(plan.outputs.size());

    for (const auto& output: plan.outputs) {
        auto shape = ExtractStaticShape(output.spec);
        AM_RETURN_IF_ERROR(shape.status());
        auto expected_bytes = CountBytes(output.spec);
        AM_RETURN_IF_ERROR(expected_bytes.status());
        if (output.nbytes != *expected_bytes) {
            return Status::Internal(
                    "constant folding plan output byte size does not match spec");
        }

        std::vector<int64_t> strides;
        if (output.strides.empty()) {
            auto contiguous_strides = MakeContiguousStrides(*shape);
            AM_RETURN_IF_ERROR(contiguous_strides.status());
            strides = std::move(*contiguous_strides);
        } else {
            strides = output.strides;
        }
        if (strides.size() != shape->size()) {
            return Status::Internal(
                    "constant folding plan output strides rank mismatch");
        }

        result.buffers.push_back(std::make_shared<std::vector<std::byte>>(output.nbytes));
        result.shapes.push_back(std::move(*shape));
        result.strides.push_back(std::move(strides));
        result.views.emplace_back(result.buffers.back()->data(),
                                  output.spec.dtype,
                                  result.shapes.back(),
                                  result.strides.back());
    }
    return result;
}

}// namespace

std::string_view ConstantFoldingPass::Name() const noexcept {
    return "ConstantFoldingPass";
}

Status ConstantFoldingPass::Run(GraphRewriteSession& session, const PassContext& ctx) {
    if (!ctx.enable_constant_folding) {
        return Status::Ok();
    }

    auto order = session.GetTopologicalOrder();
    AM_RETURN_IF_ERROR(order.status());
    for (const auto node_id: *order) {
        auto node = session.GetNodeView(node_id);
        if (!node.ok()) {
            // Node was removed during a prior rewrite — skip stale id.
            continue;
        }

        // kNotFound means the op type is not registered — skip silently.
        // Any other error (e.g. Internal) is a real bug and must propagate.
        auto schema = GetOperatorSchema(node->op_type);
        if (!schema.ok()) {
            if (schema.status() == StatusCode::kNotFound) {
                continue;
            }
            return schema.status();
        }

        if (!IsCompileTimeEvaluable(*schema)) {
            continue;
        }

        const auto* evaluator = FindConstEvaluator(node->op_type);
        if (evaluator == nullptr) {
            continue;
        }

        auto input_descs = CollectValueDescs(session, node->inputs);
        AM_RETURN_IF_ERROR(input_descs.status());
        auto all_inputs_inline = AllInputsAreInlineConstantValues(*input_descs);
        AM_RETURN_IF_ERROR(all_inputs_inline.status());
        if (!*all_inputs_inline) {
            continue;
        }

        auto output_descs = CollectValueDescs(session, node->outputs);
        AM_RETURN_IF_ERROR(output_descs.status());

        // Plan before allocation: validate feasibility and compute layout
        // before spending memory on output buffers. Unimplemented or Overflow
        // means the evaluator cannot fold this node under the current constraints.
        auto plan = evaluator->Plan(*input_descs,
                                    *output_descs,
                                    node->op_params,
                                    ctx.const_eval_policy);
        if (plan.status() == StatusCode::kUnimplemented ||
            plan.status() == StatusCode::kOverflow) {
            continue;
        }
        AM_RETURN_IF_ERROR(plan.status());

        if (plan->outputs.size() != node->outputs.size()) {
            return Status::Internal("constant folding plan output count mismatch");
        }

        // Invariant: Plan()'s output specs must match the graph's declared output
        // specs. The folded constant replaces the original value via ReplaceValue,
        // so downstream consumers must see the same dtype/shape. An evaluator
        // returning a mismatched spec is an implementation bug, not an unsupported
        // case, so this returns Internal rather than Unimplemented.
        for (size_t i = 0; i < node->outputs.size(); ++i) {
            if (plan->outputs[i].spec.dtype != (*output_descs)[i].spec.dtype ||
                plan->outputs[i].spec.shape != (*output_descs)[i].spec.shape) {
                return Status::Internal("constant folding plan output spec mismatch");
            }
        }

        auto input_views = BuildInputViews(*input_descs);
        AM_RETURN_IF_ERROR(input_views.status());
        auto output_storage = AllocateOutputViews(*plan);
        AM_RETURN_IF_ERROR(output_storage.status());

        // Evaluate writes into the pre-allocated buffers. Unimplemented at
        // this stage is unexpected but recoverable — the node is simply skipped.
        Status eval_status = evaluator->Evaluate(input_views->views,
                                                 output_storage->views,
                                                 node->op_params);
        if (eval_status == StatusCode::kUnimplemented ||
            eval_status == StatusCode::kOverflow) {
            continue;
        }
        AM_RETURN_IF_ERROR(eval_status);

        // Move the shared_ptr — zero-copy: buffers are not copied, only
        // the reference count is bumped when the graph commits.
        for (size_t i = 0; i < node->outputs.size(); ++i) {
            const auto& plan_output = plan->outputs[i];
            ConstantBinding binding{
                    .inline_data = std::move(output_storage->buffers[i]),
                    .name = plan_output.debug_name,
            };
            const auto folded = session.AddConstant(plan_output.spec,
                                                    std::move(binding),
                                                    plan_output.quantization,
                                                    plan_output.debug_name);
            AM_RETURN_IF_ERROR(session.ReplaceValue(node->outputs[i], folded));
        }
    }
    return Status::Ok();
}

}// namespace aethermind
