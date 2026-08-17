#include "aethermind/graph/graph_op_builder.h"
#include "aethermind/graph/lowering/graph_lowering.h"
#include "aethermind/graph/optimization/dead_code_elimination_pass.h"
#include "aethermind/graph/optimization/gate_up_linear_fusion_pass.h"
#include "aethermind/graph/optimization/optimize_model_graph.h"
#include "test_optimization_helpers.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <optional>
#include <string>

namespace {

using namespace aethermind;
using namespace test_utils;

struct GateUpProjections {
    GraphValueId input{};
    GraphValueId gate{};
    GraphValueId up{};
};

StatusOr<GraphValueId> AddProjection(ModelGraph& graph,
                                     GraphValueId input,
                                     int64_t rows,
                                     uint32_t layer,
                                     TransformerWeightRole role,
                                     DataType weight_dtype,
                                     std::string name) {
    const TensorSpec& input_spec = graph.GetValue(input).spec;
    if (!input_spec.shape.IsRanked() || input_spec.shape.rank() == 0U) {
        return Status::InvalidArgument("projection requires ranked input");
    }

    const GraphValueId weight = graph.AddWeight(
            {.dtype = weight_dtype,
             .shape = {ShapeSymbol::CreateFromValue(rows),
                       input_spec.shape[*input_spec.shape.rank() - 1U]}},
            MakeTransformerWeightBinding(layer, role),
            name + ".weight");
    const StatusOr<AddedNode> node = graph.AddNode(
            OpType::kLinear,
            layer,
            {input, weight},
            {NodeOutputDesc{.payload = ActivationValue{}, .name = std::move(name)}},
            LinearParams{});
    if (!node.ok()) {
        return node.status();
    }
    return node->outputs[0];
}

GateUpProjections AddGateUpProjections(ModelGraph& graph,
                                       uint32_t layer,
                                       int64_t gate_rows = 16,
                                       int64_t up_rows = 16,
                                       std::optional<GraphValueId> input = std::nullopt,
                                       bool up_before_gate = false) {
    if (!input.has_value()) {
        input = AddActivation(graph, "hidden");
    }

    GateUpProjections result{.input = *input};
    const auto add_gate = [&] {
        return AddProjection(graph,
                             *input,
                             gate_rows,
                             layer,
                             TransformerWeightRole::kMlpGate,
                             DataType::Float32(),
                             "layers." + std::to_string(layer) + ".mlp.gate_proj");
    };
    const auto add_up = [&] {
        return AddProjection(graph,
                             *input,
                             up_rows,
                             layer,
                             TransformerWeightRole::kMlpUp,
                             DataType::Float32(),
                             "layers." + std::to_string(layer) + ".mlp.up_proj");
    };
    StatusOr<GraphValueId> gate = Status::Unknown("uninitialized gate projection");
    StatusOr<GraphValueId> up = Status::Unknown("uninitialized up projection");
    if (up_before_gate) {
        up = add_up();
        gate = add_gate();
    } else {
        gate = add_gate();
        up = add_up();
    }
    AM_CHECK(gate.ok(), "failed to add gate projection: {}", gate.status().ToString());
    AM_CHECK(up.ok(), "failed to add up projection: {}", up.status().ToString());
    result.gate = *gate;
    result.up = *up;
    return result;
}

GraphNodeId Producer(const ModelGraph& graph, GraphValueId value) {
    const auto producer = graph.GetValue(value).producer;
    AM_CHECK(producer.has_value(), "expected a producer");
    return *producer;
}

GraphValueId WeightForOutput(const ModelGraph& graph, GraphValueId output) {
    const GraphNode& node = graph.GetNode(Producer(graph, output));
    AM_CHECK(node.inputs.size() == 2U, "expected a linear node");
    return node.inputs[1];
}

StatusOr<ModelGraph> RunGateUpFusion(const ModelGraph& graph,
                                     bool run_dce = false,
                                     PassContext context = {.enable_gate_up_fusion = true}) {
    GraphPassManager pipeline(context);
    pipeline.Add(std::make_unique<GateUpLinearFusionPass>());
    if (run_dce) {
        pipeline.Add(std::make_unique<DeadCodeEliminationPass>());
    }
    return pipeline.Run(graph);
}

const GraphNode& OnlyGateUpNode(const ModelGraph& graph) {
    const auto nodes = graph.FindNodesByOpType(OpType::kGateUpLinear);
    AM_CHECK(nodes.size() == 1U, "expected exactly one GateUpLinear node");
    return graph.GetNode(nodes[0]);
}

TEST(GateUpLinearFusionPass, FusesFixedGateUpOrderRegardlessOfInsertionAndPreservesSiluMulConsumers) {
    ModelGraph graph;
    const GateUpProjections projections = AddGateUpProjections(
            graph, 2U, 16, 16, std::nullopt, true);
    const QuantizationSpec gate_quantization{.kind = QuantizationKind::kInt8,
                                             .group_size = 64U,
                                             .scale_dtype = DataType::Float32(),
                                             .has_zero_point = false};
    graph.SetQuantization(projections.gate, gate_quantization);

    const auto swiglu = AddSiluMul(graph, 2U, projections.gate, projections.up, "mlp.swiglu");
    const auto additional_gate_consumer = AddSilu(graph, 2U, projections.gate, "mlp.gate.debug");
    ASSERT_TRUE(swiglu.ok()) << swiglu.status().ToString();
    ASSERT_TRUE(additional_gate_consumer.ok()) << additional_gate_consumer.status().ToString();
    graph.MarkOutput(*swiglu);
    graph.MarkOutput(*additional_gate_consumer);

    const StatusOr<ModelGraph> result = RunGateUpFusion(graph);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());
    EXPECT_EQ(result->FindNodesByOpType(OpType::kLinear).size(), 0U);
    const GraphNode& fused = OnlyGateUpNode(*result);
    ASSERT_EQ(fused.outputs.size(), 2U);
    EXPECT_EQ(fused.decoder_layer_index, 2U);
    EXPECT_EQ(result->GetValue(fused.outputs[0]).name, "layers.2.mlp.gate_proj");
    EXPECT_EQ(result->GetValue(fused.outputs[0]).quantization, gate_quantization);
    EXPECT_EQ(result->GetValue(fused.outputs[1]).name, "layers.2.mlp.up_proj");
    EXPECT_EQ(result->GetValue(fused.inputs[1]).spec.shape[0].GetStaticValue(), 32);
    const auto* binding = std::get_if<WeightValue>(&result->GetValue(fused.inputs[1]).payload);
    ASSERT_NE(binding, nullptr);
    EXPECT_TRUE(std::holds_alternative<GateUpWeightBinding>(binding->binding.spec));

    const auto silu_mul_nodes = result->FindNodesByOpType(OpType::kSiluMul);
    ASSERT_EQ(silu_mul_nodes.size(), 1U);
    const GraphNode& swiglu_node = result->GetNode(silu_mul_nodes[0]);
    EXPECT_EQ(swiglu_node.inputs[0], fused.outputs[0]);
    EXPECT_EQ(swiglu_node.inputs[1], fused.outputs[1]);
}

TEST(GateUpLinearFusionPass, SkipsIncompleteDuplicateAndIncompatibleGroups) {
    ModelGraph missing_graph;
    const GraphValueId input = AddActivation(missing_graph, "hidden");
    auto gate = AddProjection(missing_graph,
                              input,
                              16,
                              0U,
                              TransformerWeightRole::kMlpGate,
                              DataType::Float32(),
                              "gate");
    ASSERT_TRUE(gate.ok());
    auto gate_consumer = AddSilu(missing_graph, 0U, *gate, "gate.consumer");
    ASSERT_TRUE(gate_consumer.ok());
    missing_graph.MarkOutput(*gate_consumer);
    const StatusOr<ModelGraph> missing_result = RunGateUpFusion(missing_graph);
    ASSERT_TRUE(missing_result.ok()) << missing_result.status().ToString();
    EXPECT_TRUE(missing_result->FindNodesByOpType(OpType::kGateUpLinear).empty());

    ModelGraph duplicate_graph;
    const GateUpProjections projections = AddGateUpProjections(duplicate_graph, 0U);
    auto duplicate_gate = AddProjection(duplicate_graph,
                                        projections.input,
                                        16,
                                        0U,
                                        TransformerWeightRole::kMlpGate,
                                        DataType::Float32(),
                                        "duplicate.gate");
    ASSERT_TRUE(duplicate_gate.ok());
    auto main_consumer = AddSiluMul(duplicate_graph, 0U, projections.gate, projections.up, "swiglu");
    auto duplicate_consumer = AddSilu(duplicate_graph, 0U, *duplicate_gate, "duplicate.consumer");
    ASSERT_TRUE(main_consumer.ok());
    ASSERT_TRUE(duplicate_consumer.ok());
    duplicate_graph.MarkOutput(*main_consumer);
    duplicate_graph.MarkOutput(*duplicate_consumer);
    const StatusOr<ModelGraph> duplicate_result = RunGateUpFusion(duplicate_graph);
    ASSERT_TRUE(duplicate_result.ok()) << duplicate_result.status().ToString();
    EXPECT_TRUE(duplicate_result->FindNodesByOpType(OpType::kGateUpLinear).empty());

    ModelGraph quantized_graph;
    const GateUpProjections quantized = AddGateUpProjections(quantized_graph, 0U);
    quantized_graph.SetQuantization(
            WeightForOutput(quantized_graph, quantized.up),
            QuantizationSpec{.kind = QuantizationKind::kInt8,
                             .group_size = 64U,
                             .scale_dtype = DataType::Float32(),
                             .has_zero_point = false});
    const auto quantized_consumer = AddSiluMul(quantized_graph, 0U, quantized.gate, quantized.up, "swiglu");
    ASSERT_TRUE(quantized_consumer.ok());
    quantized_graph.MarkOutput(*quantized_consumer);
    const StatusOr<ModelGraph> quantized_result = RunGateUpFusion(quantized_graph);
    ASSERT_TRUE(quantized_result.ok()) << quantized_result.status().ToString();
    EXPECT_TRUE(quantized_result->FindNodesByOpType(OpType::kGateUpLinear).empty());

    ModelGraph dtype_graph;
    const GraphValueId dtype_input = AddActivation(dtype_graph, "hidden");
    const auto dtype_gate = AddProjection(dtype_graph,
                                          dtype_input,
                                          16,
                                          0U,
                                          TransformerWeightRole::kMlpGate,
                                          DataType::Float32(),
                                          "gate");
    const auto dtype_up = AddProjection(dtype_graph,
                                        dtype_input,
                                        16,
                                        0U,
                                        TransformerWeightRole::kMlpUp,
                                        DataType::Float(16),
                                        "up");
    ASSERT_TRUE(dtype_gate.ok());
    ASSERT_TRUE(dtype_up.ok());
    const auto dtype_consumer = AddSiluMul(dtype_graph, 0U, *dtype_gate, *dtype_up, "swiglu");
    ASSERT_TRUE(dtype_consumer.ok());
    dtype_graph.MarkOutput(*dtype_consumer);
    const StatusOr<ModelGraph> dtype_result = RunGateUpFusion(dtype_graph);
    ASSERT_TRUE(dtype_result.ok()) << dtype_result.status().ToString();
    EXPECT_TRUE(dtype_result->FindNodesByOpType(OpType::kGateUpLinear).empty());

    ModelGraph separate_input_graph;
    const GraphValueId gate_input = AddActivation(separate_input_graph, "gate_input");
    const GraphValueId up_input = AddActivation(separate_input_graph, "up_input");
    const auto separate_gate = AddProjection(separate_input_graph,
                                             gate_input,
                                             16,
                                             0U,
                                             TransformerWeightRole::kMlpGate,
                                             DataType::Float32(),
                                             "gate");
    const auto separate_up = AddProjection(separate_input_graph,
                                           up_input,
                                           16,
                                           0U,
                                           TransformerWeightRole::kMlpUp,
                                           DataType::Float32(),
                                           "up");
    ASSERT_TRUE(separate_gate.ok());
    ASSERT_TRUE(separate_up.ok());
    const auto separate_consumer = AddSiluMul(
            separate_input_graph, 0U, *separate_gate, *separate_up, "swiglu");
    ASSERT_TRUE(separate_consumer.ok());
    separate_input_graph.MarkOutput(*separate_consumer);
    const StatusOr<ModelGraph> separate_result = RunGateUpFusion(separate_input_graph);
    ASSERT_TRUE(separate_result.ok()) << separate_result.status().ToString();
    EXPECT_TRUE(separate_result->FindNodesByOpType(OpType::kGateUpLinear).empty());

    ModelGraph separate_layer_graph;
    const GraphValueId layer_input = AddActivation(separate_layer_graph, "hidden");
    const auto layer_gate = AddProjection(separate_layer_graph,
                                          layer_input,
                                          16,
                                          0U,
                                          TransformerWeightRole::kMlpGate,
                                          DataType::Float32(),
                                          "gate");
    const auto layer_up = AddProjection(separate_layer_graph,
                                        layer_input,
                                        16,
                                        1U,
                                        TransformerWeightRole::kMlpUp,
                                        DataType::Float32(),
                                        "up");
    ASSERT_TRUE(layer_gate.ok());
    ASSERT_TRUE(layer_up.ok());
    const auto layer_consumer = AddSiluMul(
            separate_layer_graph, std::nullopt, *layer_gate, *layer_up, "swiglu");
    ASSERT_TRUE(layer_consumer.ok());
    separate_layer_graph.MarkOutput(*layer_consumer);
    const StatusOr<ModelGraph> layer_result = RunGateUpFusion(separate_layer_graph);
    ASSERT_TRUE(layer_result.ok()) << layer_result.status().ToString();
    EXPECT_TRUE(layer_result->FindNodesByOpType(OpType::kGateUpLinear).empty());
}

TEST(GateUpLinearFusionPass, SkipsIncompatibleKReplacedAndDeadOutputs) {
    ModelGraph k_mismatch_graph;
    const GateUpProjections k_mismatch = AddGateUpProjections(k_mismatch_graph, 0U);
    k_mismatch_graph.MarkOutput(k_mismatch.gate);
    k_mismatch_graph.MarkOutput(k_mismatch.up);
    GraphRewriteSession k_mismatch_session(k_mismatch_graph);
    const GraphValueId incompatible_weight = k_mismatch_session.AddSessionWeight(
            Spec(DataType::Float32(), {16, 5}),
            MakeTransformerWeightBinding(0U, TransformerWeightRole::kMlpGate),
            {},
            "incompatible.gate.weight");
    ASSERT_TRUE(k_mismatch_session.RedirectInput(
                                          Producer(k_mismatch_graph, k_mismatch.gate), 1U, incompatible_weight)
                        .ok());
    GateUpLinearFusionPass pass;
    ASSERT_TRUE(pass.Run(k_mismatch_session, PassContext{.enable_gate_up_fusion = true}).ok());
    EXPECT_FALSE(k_mismatch_session.Commit().ok());

    ModelGraph replaced_graph;
    const GateUpProjections replaced = AddGateUpProjections(replaced_graph, 0U);
    const auto alternative = AddProjection(replaced_graph,
                                           replaced.input,
                                           16,
                                           0U,
                                           TransformerWeightRole::kMlpDown,
                                           DataType::Float32(),
                                           "alternative");
    ASSERT_TRUE(alternative.ok());
    replaced_graph.MarkOutput(replaced.gate);
    replaced_graph.MarkOutput(replaced.up);
    GraphRewriteSession replaced_session(replaced_graph);
    ASSERT_TRUE(replaced_session.ReplaceValue(replaced.gate, *alternative).ok());
    ASSERT_TRUE(pass.Run(replaced_session, PassContext{.enable_gate_up_fusion = true}).ok());
    const StatusOr<ModelGraph> replaced_result = replaced_session.Commit();
    ASSERT_TRUE(replaced_result.ok()) << replaced_result.status().ToString();
    EXPECT_TRUE(replaced_result->FindNodesByOpType(OpType::kGateUpLinear).empty());

    ModelGraph dead_graph;
    (void) AddGateUpProjections(dead_graph, 0U);
    const StatusOr<ModelGraph> dead_result = RunGateUpFusion(dead_graph);
    ASSERT_TRUE(dead_result.ok()) << dead_result.status().ToString();
    EXPECT_TRUE(dead_result->FindNodesByOpType(OpType::kGateUpLinear).empty());
}

TEST(GateUpLinearFusionPass, HonorsFlagRunsAtO2AndRemainsIdempotentAfterDce) {
    ModelGraph graph;
    const GateUpProjections projections = AddGateUpProjections(graph, 1U);
    const auto swiglu = AddSiluMul(graph, 1U, projections.gate, projections.up, "swiglu");
    ASSERT_TRUE(swiglu.ok());
    graph.MarkOutput(*swiglu);

    // The flag gates the pass at runtime: disabled context leaves the
    // gate/up Linear pair unfused even at O2.
    PassContext disabled_context{.enable_gate_up_fusion = false};
    const StatusOr<ModelGraph> disabled_result =
            OptimizeModelGraph(graph, disabled_context);
    ASSERT_TRUE(disabled_result.ok()) << disabled_result.status().ToString();
    EXPECT_TRUE(disabled_result->FindNodesByOpType(OpType::kGateUpLinear).empty());
    EXPECT_EQ(disabled_result->FindNodesByOpType(OpType::kLinear).size(), 2U);

    const StatusOr<ModelGraph> fused = RunGateUpFusion(graph, true);
    ASSERT_TRUE(fused.ok()) << fused.status().ToString();
    ASSERT_TRUE(fused->Validate().ok());
    EXPECT_EQ(fused->FindNodesByOpType(OpType::kGateUpLinear).size(), 1U);
    EXPECT_TRUE(fused->FindNodesByOpType(OpType::kLinear).empty());

    const StatusOr<LoweredGraph> lowered = LowerModelGraph(*fused);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    ASSERT_EQ(lowered->steps.size(), lowered->step_bindings.size());
    std::optional<size_t> gate_up_step_index;
    for (size_t i = 0; i < lowered->steps.size(); ++i) {
        if (lowered->steps[i].op_type == OpType::kGateUpLinear) {
            gate_up_step_index = i;
            break;
        }
    }
    ASSERT_TRUE(gate_up_step_index.has_value());
    const size_t step_index = *gate_up_step_index;
    EXPECT_EQ(lowered->step_bindings[step_index].input_values.size(), 2U);
    EXPECT_EQ(lowered->step_bindings[step_index].output_values.size(), 2U);
    EXPECT_EQ(lowered->steps[step_index].output_specs.size(), 2U);

    const StatusOr<ModelGraph> rerun = RunGateUpFusion(*fused, true);
    ASSERT_TRUE(rerun.ok()) << rerun.status().ToString();
    EXPECT_EQ(rerun->FindNodesByOpType(OpType::kGateUpLinear).size(), 1U);

    // The default O2 pipeline fuses gate/up (flag defaults to true).
    const StatusOr<ModelGraph> o2_result = OptimizeModelGraph(graph);
    ASSERT_TRUE(o2_result.ok()) << o2_result.status().ToString();
    EXPECT_EQ(o2_result->FindNodesByOpType(OpType::kGateUpLinear).size(), 1U);
}

}// namespace
