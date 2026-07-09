#include "aethermind/model/graph/constant_folding_pass.h"
#include "aethermind/model/graph/const_evaluator.h"
#include "aethermind/model/graph/operator_schema.h"

#include <cstddef>
#include <memory>
#include <span>
#include <variant>
#include <vector>

namespace aethermind {
namespace {

struct BorrowedViewStorage {
    std::vector<std::vector<int64_t>> shapes{};
    std::vector<std::vector<int64_t>> strides{};
};

struct InputViews {
    std::vector<TensorView> views{};
    BorrowedViewStorage metadata{};
};

struct OutputStorage {
    std::vector<MutableTensorView> views{};
    std::vector<std::shared_ptr<std::vector<std::byte>>> buffers{};
    BorrowedViewStorage metadata{};
};

StatusOr<std::vector<NodeOutputDesc>> CollectValueDescs(
        const GraphRewriteSession& session,
        std::span<const GraphValueId> values) {
    std::vector<NodeOutputDesc> descs;
    descs.reserve(values.size());
    for (const auto value: values) {
        auto desc = session.GetValueOutputDesc(value);
        AM_RETURN_IF_ERROR(desc.status());
        descs.push_back(std::move(*desc));
    }
    return descs;
}

StatusOr<bool> HasInlineConstantBytes(const NodeOutputDesc& desc) {
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

StatusOr<bool> AllInputsAreInlineConstantValues(std::span<const NodeOutputDesc> inputs) {
    for (const NodeOutputDesc& input: inputs) {
        auto has_inline_bytes = HasInlineConstantBytes(input);
        AM_RETURN_IF_ERROR(has_inline_bytes.status());
        if (!*has_inline_bytes) {
            return false;
        }
    }
    return true;
}

StatusOr<InputViews> BuildInputViews(std::span<const NodeOutputDesc> inputs) {
    InputViews result;
    result.views.reserve(inputs.size());
    result.metadata.shapes.reserve(inputs.size());
    result.metadata.strides.reserve(inputs.size());

    for (const auto& input: inputs) {
        const auto* constant = std::get_if<ConstantValue>(&input.payload);
        if (constant == nullptr || !constant->binding.inline_data) {
            return Status::InvalidArgument(
                    "constant folding input lacks inline constant data");
        }

        auto shape = ExtractStaticShape(input.spec);
        AM_RETURN_IF_ERROR(shape.status());
        auto strides = MakeContiguousStrides(*shape);
        result.metadata.shapes.push_back(std::move(*shape));
        result.metadata.strides.push_back(std::move(strides));
        result.views.emplace_back(constant->binding.inline_data->data(),
                                  input.spec.dtype,
                                  result.metadata.shapes.back(),
                                  result.metadata.strides.back());
    }
    return result;
}

StatusOr<OutputStorage> AllocateOutputViews(const ConstEvalPlan& plan) {
    OutputStorage result;
    result.views.reserve(plan.outputs.size());
    result.buffers.reserve(plan.outputs.size());
    result.metadata.shapes.reserve(plan.outputs.size());
    result.metadata.strides.reserve(plan.outputs.size());

    for (const auto& output: plan.outputs) {
        auto shape = ExtractStaticShape(output.spec);
        AM_RETURN_IF_ERROR(shape.status());
        auto expected_bytes = CountBytes(output.spec);
        AM_RETURN_IF_ERROR(expected_bytes.status());
        if (output.nbytes != *expected_bytes) {
            return Status::Internal(
                    "constant folding plan output byte size does not match spec");
        }

        std::vector<int64_t> strides = output.strides.empty()
                                               ? MakeContiguousStrides(*shape)
                                               : output.strides;
        if (strides.size() != shape->size()) {
            return Status::Internal(
                    "constant folding plan output strides rank mismatch");
        }

        result.buffers.push_back(std::make_shared<std::vector<std::byte>>(output.nbytes));
        result.metadata.shapes.push_back(std::move(*shape));
        result.metadata.strides.push_back(std::move(strides));
        result.views.emplace_back(result.buffers.back()->data(),
                                  output.spec.dtype,
                                  result.metadata.shapes.back(),
                                  result.metadata.strides.back());
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
            continue;
        }

        if (auto schema = GetOperatorSchema(node->op_type);
            !schema.ok() || !IsCompileTimeEvaluable(*schema)) {
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

        auto plan = evaluator->Plan(
                *input_descs, *output_descs, node->op_params, ctx.const_eval_policy);
        if (plan.status().code() == StatusCode::kUnimplemented) {
            continue;
        }
        AM_RETURN_IF_ERROR(plan.status());

        if (plan->outputs.size() != node->outputs.size()) {
            return Status::Internal("constant folding plan output count mismatch");
        }

        auto input_views = BuildInputViews(*input_descs);
        AM_RETURN_IF_ERROR(input_views.status());
        auto output_storage = AllocateOutputViews(*plan);
        AM_RETURN_IF_ERROR(output_storage.status());

        Status eval_status = evaluator->Evaluate(input_views->views,
                                                 output_storage->views,
                                                 node->op_params);
        if (eval_status.code() == StatusCode::kUnimplemented) {
            continue;
        }
        AM_RETURN_IF_ERROR(eval_status);

        for (size_t i = 0; i < node->outputs.size(); ++i) {
            const auto& output = plan->outputs[i];
            ConstantBinding binding{
                    .inline_data = std::move(output_storage->buffers[i]),
                    .name = output.debug_name,
            };
            const auto folded = session.AddConstant(output.spec,
                                                    std::move(binding),
                                                    output.quantization,
                                                    output.debug_name);
            AM_RETURN_IF_ERROR(session.ReplaceValue(node->outputs[i], folded));
        }
    }
    return Status::Ok();
}

}// namespace aethermind
