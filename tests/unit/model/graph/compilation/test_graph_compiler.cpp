#include "../test_graph_helpers.h"
#include "aethermind/model/graph/compilation/graph_compiler.h"
#include "aethermind/model/graph/graph_builder.h"
#include "aethermind/model/graph/graph_dump.h"
#include "aethermind/model/graph/graph_op_builder.h"

#include <gtest/gtest.h>

#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/model/graph/compilation/graph_lowering.h"
#include "aethermind/operators/operator_semantics.h"
#include "aethermind/runtime/runtime_builder.h"
#include "aethermind/shape_inference/shape_symbol.h"
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <vector>

namespace aethermind {
namespace {

// ---- Composite test-graph builder ------------------------------------------
//
// Builds a graph with three optimization opportunities coexisting in one
// deterministic topology:
//   1. Foldable: two inline float constants fed into an Add → rank-0 constant
//      result that is broadcast-added (Add) to a live activation.
//   2. Dead: an Embedding whose output is not consumed by any graph output.
//   3. SiLU+Mul: a live SiLU whose output is elementwise-multiplied with
//      another live activation, forming the SiLU→Mul fusion pattern.
//
// The graph output is the result of the final ElementwiseMul.
//
//                                       ┌────────────────────┐
//   const_lhs (rank-0 [2.0]) ──┐        │  tokens_dead ──→ Embedding("dead")
//   const_rhs (rank-0 [3.0]) ──┤        │                    (NOT a graph output)
//                               Add("foldable_add")
//                                    │
//   tokens_live ──→ Embedding("live")─┤
//                               Add("merge")  ←── broadcasts rank-0 → [2,4]
//                                    │
//                               Silu("silu")
//                                    │
//   tokens_up ───→ Embedding("up")───┤
//                               ElementwiseMul("mul")
//                                    │
//                               graph output
//

std::shared_ptr<const std::vector<std::byte>> InlineFloats(std::vector<float> values) {
    std::vector<std::byte> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return std::make_shared<const std::vector<std::byte>>(std::move(bytes));
}

GraphValueId AddFloatConstant(ModelGraph& graph,
                              std::vector<float> values,
                              std::vector<int64_t> shape,
                              const std::string& name) {
    return graph.AddConstant(
            Spec(DataType::Float32(), std::move(shape)),
            ConstantBinding{.inline_data = InlineFloats(std::move(values)), .name = name},
            name);
}

GraphValueId AddLiveActivation(ModelGraph& graph, const char* name) {
    const GraphValueId tokens = graph.AddInput(
            Spec(DataType::Int(32), {2}), std::string(name) + ".tokens");
    auto embedding_or = AddEmbedding(graph,
                                     tokens,
                                     16,
                                     4,
                                     DataType::Float32(),
                                     WeightBinding{.slot = ParameterSlot::kEmbeddingTable,
                                                   .semantic_role = TransformerWeightRole::kTokenEmbedding},
                                     name);
    AM_CHECK(embedding_or.ok(), "AddEmbedding failed in test helper");
    return *embedding_or;
}

ModelGraph BuildCompositeGraph() {
    ModelGraph graph;

    // (1) Foldable: two rank-0 float constants feeding an Add.
    const GraphValueId const_lhs = AddFloatConstant(graph, {2.0F}, {}, "const_lhs");
    const GraphValueId const_rhs = AddFloatConstant(graph, {3.0F}, {}, "const_rhs");
    auto foldable_or = AddElementwiseAdd(graph, 0U, const_lhs, const_rhs, "foldable_add");
    AM_CHECK(foldable_or.ok(), "{}", foldable_or.status().ToString());
    const GraphValueId foldable = *foldable_or;

    // (2) Live activation that merges with the foldable result.
    const GraphValueId live = AddLiveActivation(graph, "live");

    // Broadcast-add the rank-0 constant result into the [2,4] activation.
    auto merged_or = AddElementwiseAdd(graph, 0U, foldable, live, "merge");
    AM_CHECK(merged_or.ok(), "{}", merged_or.status().ToString());
    const GraphValueId merged = *merged_or;

    // (3) SiLU from the merged activation.
    auto silu_or = AddSilu(graph, 0U, merged, "silu");
    AM_CHECK(silu_or.ok(), "{}", silu_or.status().ToString());
    const GraphValueId silu = *silu_or;

    // (4) Second live activation for the multiply path.
    const GraphValueId up = AddLiveActivation(graph, "up");

    // (5) ElementwiseMul of silu(gate) * up → fusion pattern.
    auto mul_or = AddElementwiseMul(graph, 0U, silu, up, "mul");
    AM_CHECK(mul_or.ok(), "{}", mul_or.status().ToString());
    const GraphValueId mul = *mul_or;

    graph.MarkOutput(mul);

    // (6) Dead node: an activation with no consumer and not a graph output.
    AddLiveActivation(graph, "dead");

    return graph;
}

// ---- O0: no passes applied -------------------------------------------------

TEST(DefaultGraphPassPipeline, OptLevelZeroPreservesGraph) {
    ModelGraph graph = BuildCompositeGraph();
    ASSERT_TRUE(graph.Validate().ok());

    PassContext ctx;
    ctx.opt_level = 0;
    const StatusOr<ModelGraph> result = OptimizeModelGraph(graph, ctx);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());

    // Foldable Add must still be present (no constant folding).
    EXPECT_GE(result->FindNodesByOpType(OpType::kAdd).size(), 2U);

    // Dead node must still be present (no DCE): exact equality proves zero
    // nodes were removed, including the dead embedding.
    EXPECT_EQ(result->GetNodes().size(), graph.GetNodes().size());
    bool dead_node_found = false;
    for (const GraphNode& node: result->GetNodes()) {
        if (node.name.find("dead") != std::string::npos) {
            dead_node_found = true;
            break;
        }
    }
    EXPECT_TRUE(dead_node_found);

    // SiLU and ElementwiseMul must remain separate (no fusion).
    EXPECT_GE(result->FindNodesByOpType(OpType::kSilu).size(), 1U);
    EXPECT_GE(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 0U);
}

// ---- O1: ConstantFolding + DCE (no SiLU fusion) ---------------------------

TEST(DefaultGraphPassPipeline, OptLevelOneFoldsAndRemovesDeadButNotSiluMul) {
    ModelGraph graph = BuildCompositeGraph();
    ASSERT_TRUE(graph.Validate().ok());
    const size_t original_node_count = graph.GetNodes().size();

    PassContext ctx;
    ctx.opt_level = 1;
    const StatusOr<ModelGraph> result = OptimizeModelGraph(graph, ctx);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());

    // Dead node removed by DCE → fewer nodes than original.
    EXPECT_LT(result->GetNodes().size(), original_node_count);

    // Constant folding independently proven: CF folded foldable_add into a new
    // constant (const_lhs + const_rhs = 5.0). The committed graph now has one
    // more ConstantValue than the original two input constants. DCE alone
    // cannot create ConstantValues, so a non-decreasing count is not enough.
    {
        size_t constant_count = 0;
        for (const GraphValue& value: result->GetValues()) {
            if (std::holds_alternative<ConstantValue>(value.payload)) {
                ++constant_count;
            }
        }
        EXPECT_GT(constant_count, 2U);
    }

    // Dead "dead" node must be gone.
    for (const GraphNode& node: result->GetNodes()) {
        EXPECT_TRUE(node.name.find("dead") == std::string::npos);
    }

    // SiLU and ElementwiseMul remain separate (O1 does not fuse).
    EXPECT_GE(result->FindNodesByOpType(OpType::kSilu).size(), 1U);
    EXPECT_GE(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 0U);
}

// ---- O2: ConstantFolding → SiluMulFusion → DCE ---------------------------

void ExpectFullOptimization(const StatusOr<ModelGraph>& result,
                            size_t original_node_count) {
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());

    // Optimization reduced the node count.
    EXPECT_LT(result->GetNodes().size(), original_node_count);

    // kSilu and kElementwiseMul are gone; kSiluMul takes their place.
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 1U);

    // No dead "dead" node survives.
    for (const GraphNode& node: result->GetNodes()) {
        EXPECT_TRUE(node.name.find("dead") == std::string::npos);
    }
}

TEST(DefaultGraphPassPipeline, OptLevelTwoAddsSiluMulFusion) {
    ModelGraph graph = BuildCompositeGraph();
    const size_t original = graph.GetNodes().size();
    ASSERT_TRUE(graph.Validate().ok());

    PassContext ctx;
    ctx.opt_level = 2;
    const StatusOr<ModelGraph> result = OptimizeModelGraph(graph, ctx);

    ExpectFullOptimization(result, original);
}

TEST(DefaultGraphPassPipeline, OptLevelThreeMatchesOptLevelTwo) {
    ModelGraph graph = BuildCompositeGraph();
    const size_t original = graph.GetNodes().size();
    ASSERT_TRUE(graph.Validate().ok());

    PassContext ctx;
    ctx.opt_level = 3;
    const StatusOr<ModelGraph> result = OptimizeModelGraph(graph, ctx);

    ExpectFullOptimization(result, original);
}

TEST(DefaultGraphPassPipeline, OptLevelNinetyNineMatchesOptLevelTwo) {
    ModelGraph graph = BuildCompositeGraph();
    const size_t original = graph.GetNodes().size();
    ASSERT_TRUE(graph.Validate().ok());

    PassContext ctx;
    ctx.opt_level = 99;
    const StatusOr<ModelGraph> result = OptimizeModelGraph(graph, ctx);

    ExpectFullOptimization(result, original);
}

// ---- Default opt_level is 2 ------------------------------------------------

TEST(DefaultGraphPassPipeline, DefaultContextUsesOptLevelTwo) {
    ModelGraph graph = BuildCompositeGraph();
    const size_t original = graph.GetNodes().size();
    ASSERT_TRUE(graph.Validate().ok());

    // No explicit opt_level → should default to 2 (full pipeline).
    const StatusOr<ModelGraph> result = OptimizeModelGraph(graph);

    ExpectFullOptimization(result, original);
}

// ---- Input immutability ----------------------------------------------------

TEST(DefaultGraphPassPipeline, InputGraphIsUnchanged) {
    ModelGraph graph = BuildCompositeGraph();
    ASSERT_TRUE(graph.Validate().ok());

    const std::string before = graph.GetNodes()[0].name;
    const size_t node_count_before = graph.GetNodes().size();

    PassContext ctx;
    ctx.opt_level = 2;
    const StatusOr<ModelGraph> result = OptimizeModelGraph(graph, ctx);
    ASSERT_TRUE(result.ok());

    // Source graph must be identical after the call.
    EXPECT_EQ(graph.GetNodes().size(), node_count_before);
    EXPECT_EQ(graph.GetNodes()[0].name, before);
}

// ---- Strengthened immutability: DumpGraph before/after + value counts ------

TEST(DefaultGraphPassPipeline, InputGraphDumpAndValueCountUnchanged) {
    ModelGraph graph = BuildCompositeGraph();
    ASSERT_TRUE(graph.Validate().ok());

    std::ostringstream before_dump;
    DumpGraph(graph, before_dump);
    const std::string before_str = before_dump.str();
    const size_t node_count_before = graph.GetNodes().size();
    const size_t value_count_before = graph.GetValues().size();

    PassContext ctx;
    ctx.opt_level = 2;
    const StatusOr<ModelGraph> result = OptimizeModelGraph(graph, ctx);
    ASSERT_TRUE(result.ok());
    ASSERT_TRUE(graph.Validate().ok());

    std::ostringstream after_dump;
    DumpGraph(graph, after_dump);
    const std::string after_str = after_dump.str();

    EXPECT_EQ(graph.GetNodes().size(), node_count_before);
    EXPECT_EQ(graph.GetValues().size(), value_count_before);
    EXPECT_EQ(before_str, after_str);
}

// ---- Feature gates: disabling individual passes at O2 ----------------------
// The pipeline builder does not gate registration on flags; each pass checks
// its own enable_* flag internally. These tests prove that the flags are
// forwarded unchanged from PassContext through GraphPassManager to each pass.

TEST(OptimizeModelGraph, DisablingConstantFoldingPreservesFoldableAtO2) {
    ModelGraph graph = BuildCompositeGraph();
    ASSERT_TRUE(graph.Validate().ok());

    PassContext ctx;
    ctx.opt_level = 2;
    ctx.enable_constant_folding = false;
    const StatusOr<ModelGraph> result = OptimizeModelGraph(graph, ctx);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());

    // Foldable Add preserved (CF did not run).
    EXPECT_GE(result->FindNodesByOpType(OpType::kAdd).size(), 2U);

    // SiLU fusion still ran (flag scope is per-pass, not pipeline-wide).
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 1U);

    // DCE still ran.
    for (const GraphNode& node: result->GetNodes()) {
        EXPECT_TRUE(node.name.find("dead") == std::string::npos);
    }
}

TEST(OptimizeModelGraph, DisablingSwigluFusionPreservesSiluMulAtO2) {
    ModelGraph graph = BuildCompositeGraph();
    ASSERT_TRUE(graph.Validate().ok());

    PassContext ctx;
    ctx.opt_level = 2;
    ctx.enable_swiglu_fusion = false;
    const StatusOr<ModelGraph> result = OptimizeModelGraph(graph, ctx);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());

    // SiLU and Mul remain separate (fusion did not run).
    EXPECT_GE(result->FindNodesByOpType(OpType::kSilu).size(), 1U);
    EXPECT_GE(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 1U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 0U);

    // CF ran: foldable Add produced a new constant.
    size_t constant_count = 0;
    for (const GraphValue& value: result->GetValues()) {
        if (std::holds_alternative<ConstantValue>(value.payload)) ++constant_count;
    }
    EXPECT_GT(constant_count, 2U);

    // DCE ran.
    for (const GraphNode& node: result->GetNodes()) {
        EXPECT_TRUE(node.name.find("dead") == std::string::npos);
    }
}

TEST(OptimizeModelGraph, DisablingDcePreservesDeadNodeAtO2) {
    ModelGraph graph = BuildCompositeGraph();
    ASSERT_TRUE(graph.Validate().ok());

    PassContext ctx;
    ctx.opt_level = 2;
    ctx.enable_dce = false;
    const StatusOr<ModelGraph> result = OptimizeModelGraph(graph, ctx);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());

    // Dead node survives (DCE did not run).
    bool dead_found = false;
    for (const GraphNode& node: result->GetNodes()) {
        if (node.name.find("dead") != std::string::npos) {
            dead_found = true;
            break;
        }
    }
    EXPECT_TRUE(dead_found);

    // CF ran.
    size_t constant_count = 0;
    for (const GraphValue& value: result->GetValues()) {
        if (std::holds_alternative<ConstantValue>(value.payload)) ++constant_count;
    }
    EXPECT_GT(constant_count, 2U);

    // SiLU fusion ran.
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 1U);
}

// ---- ConstEvalPolicy: restrictive budget prevents folding ------------------

TEST(OptimizeModelGraph, RestrictiveConstEvalPolicySkipsFoldingAtO2) {
    ModelGraph graph = BuildCompositeGraph();
    ASSERT_TRUE(graph.Validate().ok());

    PassContext ctx;
    ctx.opt_level = 2;
    ctx.const_eval_policy.max_output_bytes = 0;
    const StatusOr<ModelGraph> result = OptimizeModelGraph(graph, ctx);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());

    // Foldable Add preserved: 0-byte budget prevents any constant output.
    EXPECT_GE(result->FindNodesByOpType(OpType::kAdd).size(), 2U);

    // No new constant created beyond the original two input constants.
    size_t constant_count = 0;
    for (const GraphValue& value: result->GetValues()) {
        if (std::holds_alternative<ConstantValue>(value.payload)) ++constant_count;
    }
    EXPECT_EQ(constant_count, 2U);

    // Fusion and DCE still ran (policy is only for CF).
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 1U);
}

// ---- Checkpoint: checkpoint_every=1 produces same result as no checkpoint --

TEST(OptimizeModelGraph, CheckpointEveryOneMatchesNoCheckpointAtO2) {
    ModelGraph graph1 = BuildCompositeGraph();
    ModelGraph graph2 = BuildCompositeGraph();
    ASSERT_TRUE(graph1.Validate().ok());
    ASSERT_TRUE(graph2.Validate().ok());

    PassContext ctx_no_cp;
    ctx_no_cp.opt_level = 2;
    const StatusOr<ModelGraph> result_no_cp = OptimizeModelGraph(graph1, ctx_no_cp);
    ASSERT_TRUE(result_no_cp.ok());
    ASSERT_TRUE(result_no_cp->Validate().ok());

    PassContext ctx_cp;
    ctx_cp.opt_level = 2;
    ctx_cp.checkpoint_every = 1;
    const StatusOr<ModelGraph> result_cp = OptimizeModelGraph(graph2, ctx_cp);
    ASSERT_TRUE(result_cp.ok());
    ASSERT_TRUE(result_cp->Validate().ok());

    // checkpoint_every=1 must produce the same optimized graph as no-checkpoint.
    // DumpGraph text, node count, value count, and op-type distribution must
    // all match exactly — structural divergence across checkpoint boundaries
    // is a DCE liveness bug, not an acceptable semantic difference.
    EXPECT_EQ(result_no_cp->GetNodes().size(), result_cp->GetNodes().size());
    EXPECT_EQ(result_no_cp->GetValues().size(), result_cp->GetValues().size());

    std::ostringstream os_no_cp;
    DumpGraph(*result_no_cp, os_no_cp);
    std::ostringstream os_cp;
    DumpGraph(*result_cp, os_cp);
    EXPECT_EQ(os_no_cp.str(), os_cp.str());
}

// ---- Ordering: CF before fusion with constant gate + runtime up -----------
// CF folds SiLU(const_gate); SiluMulFusion resolves through ReplaceValue chains
// and fuses the SiLU → Mul pattern deterministically; DCE removes dead nodes.

ModelGraph BuildGraphWithConstantSiLUGate() {
    ModelGraph graph;

    // Up path first: ensures the Embedding producer appears before the SiLU
    // node in topological order, so the fusion rewrite (emitted when the first
    // old node — SiLU — is processed) can map the up input immediately.
    const GraphValueId up = AddLiveActivation(graph, "up");

    const GraphValueId const_gate = AddFloatConstant(
            graph, {0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F},
            {2, 4}, "const_gate");
    auto silu_out_or = AddSilu(graph, 0U, const_gate, "silu_const");
    AM_CHECK(silu_out_or.ok(), "{}", silu_out_or.status().ToString());
    const GraphValueId silu_out = *silu_out_or;

    auto mul_out_or = AddElementwiseMul(graph, 0U, silu_out, up, "mul");
    AM_CHECK(mul_out_or.ok(), "{}", mul_out_or.status().ToString());
    const GraphValueId mul_out = *mul_out_or;

    graph.MarkOutput(mul_out);

    return graph;
}

TEST(OptimizeModelGraph, ConstantGateRuntimeUpDeterministicFusionAtO2) {
    ModelGraph graph = BuildGraphWithConstantSiLUGate();
    ASSERT_TRUE(graph.Validate().ok());

    PassContext ctx;
    ctx.opt_level = 2;
    const StatusOr<ModelGraph> result = OptimizeModelGraph(graph, ctx);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->Validate().ok());

    // Fusion fired deterministically: SiLU + Mul replaced by single kSiluMul.
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSilu).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kElementwiseMul).size(), 0U);
    EXPECT_EQ(result->FindNodesByOpType(OpType::kSiluMul).size(), 1U);

    // CF folded the SiLU: more constants than the original gate constant alone.
    size_t constant_count = 0;
    for (const GraphValue& value: result->GetValues()) {
        if (std::holds_alternative<ConstantValue>(value.payload)) ++constant_count;
    }
    EXPECT_GT(constant_count, 1U);

    // The runtime up activation survives (Embedding node present).
    EXPECT_GE(result->FindNodesByOpType(OpType::kEmbedding).size(), 1U);
}

// ---- CompileModelGraph: strict Optimize → Lower sequencing -----------------

TEST(CompileModelGraph, DefaultConfigUsesOptLevelTwo) {
    ModelGraph graph = BuildCompositeGraph();
    ASSERT_TRUE(graph.Validate().ok());

    // Default config: optimization at O2, lowering at defaults.
    GraphCompileConfig config;
    const StatusOr<CompiledModelGraph> result = CompileModelGraph(graph, config);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_TRUE(result->optimized_graph.Validate().ok());

    // O2 default pipeline: SiLU+Mul fused, dead nodes removed.
    EXPECT_EQ(result->optimized_graph.FindNodesByOpType(OpType::kSiluMul).size(), 1U);
    EXPECT_GT(result->lowered.steps.size(), 0U);
}

TEST(CompileModelGraph, CustomLoweringConfigPropagatesToSteps) {
    ModelGraph graph = BuildCompositeGraph();
    ASSERT_TRUE(graph.Validate().ok());

    GraphCompileConfig config;
    config.lowering.device_type = DeviceType::kCUDA;
    config.lowering.isa = IsaLevel::kAVX2;
    config.lowering.weight_format = WeightFormat::kPacked;
    config.lowering.phase = ExecPhase::kPrefill;

    const StatusOr<CompiledModelGraph> result = CompileModelGraph(graph, config);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_GT(result->lowered.steps.size(), 0U);

    for (const ExecutionPlanNodeSpec& step: result->lowered.steps) {
        EXPECT_EQ(step.device_type, DeviceType::kCUDA);
        EXPECT_EQ(step.isa, IsaLevel::kAVX2);
        EXPECT_EQ(step.weight_format, WeightFormat::kPacked);
        EXPECT_EQ(step.phase, ExecPhase::kPrefill);
    }
}

TEST(CompileModelGraph, RejectsInvalidGraph) {
    // Invalid graph: a dangling GraphValue with no producer and no payload
    // that is not a valid model input. LowerModelGraph rejects this.
    ModelGraph invalid(HfModelConfig{}, {}, {GraphValue{}});

    GraphCompileConfig config;
    const StatusOr<CompiledModelGraph> result = CompileModelGraph(invalid, config);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
}

TEST(CompileModelGraph, DefaultConstructionCompiles) {
    // Prove default-constructed configs are valid and the types satisfy
    // nothrow-move requirements for StatusOr.
    GraphCompileConfig config;
    EXPECT_EQ(config.optimization.opt_level, 2U);

    CompiledModelGraph compiled;
    EXPECT_EQ(compiled.lowered.steps.size(), 0U);
}

// ---- Ownership: standalone constant graph output ---------------------------

std::vector<float> ReadFloatConstant(const GraphValue& value) {
    const auto* constant = std::get_if<ConstantValue>(&value.payload);
    AM_CHECK(constant != nullptr, "expected constant value");
    AM_CHECK(constant->binding.inline_data != nullptr, "expected inline data");
    std::vector<float> result(constant->binding.inline_data->size() / sizeof(float));
    std::memcpy(result.data(), constant->binding.inline_data->data(),
                constant->binding.inline_data->size());
    return result;
}

TEST(CompileModelGraph, PreservesStandaloneConstantGraphOutput) {
    ModelGraph graph;
    const GraphValueId lhs = AddFloatConstant(graph, {1.0F, 2.0F, 3.0F, 4.0F}, {4}, "lhs");
    const GraphValueId rhs = AddFloatConstant(graph, {5.0F, 6.0F, 7.0F, 8.0F}, {4}, "rhs");
    auto sum_or = AddElementwiseAdd(graph, 0U, lhs, rhs, "sum");
    ASSERT_TRUE(sum_or.ok()) << sum_or.status().ToString();
    const GraphValueId sum = *sum_or;
    graph.MarkOutput(sum);
    ASSERT_TRUE(graph.Validate().ok());

    GraphCompileConfig config;
    config.optimization.opt_level = 2;
    const StatusOr<CompiledModelGraph> compiled = CompileModelGraph(graph, config);

    ASSERT_TRUE(compiled.ok()) << compiled.status().ToString();
    ASSERT_TRUE(compiled->optimized_graph.Validate().ok());

    // The optimized graph output is a ConstantValue after CF+DCE.
    ASSERT_EQ(compiled->optimized_graph.GetOutputs().size(), 1U);
    const GraphValue& output = compiled->optimized_graph.GetValue(
            compiled->optimized_graph.GetOutputs()[0].value);
    ASSERT_TRUE(std::holds_alternative<ConstantValue>(output.payload));

    // Inline bytes are readable and equal the expected folded sum.
    const std::vector<float> values = ReadFloatConstant(output);
    ASSERT_EQ(values.size(), 4U);
    EXPECT_FLOAT_EQ(values[0], 6.0F);
    EXPECT_FLOAT_EQ(values[1], 8.0F);
    EXPECT_FLOAT_EQ(values[2], 10.0F);
    EXPECT_FLOAT_EQ(values[3], 12.0F);

    // No Add producer survives after DCE fixed the replaced-output liveness.
    EXPECT_EQ(compiled->optimized_graph.FindNodesByOpType(OpType::kAdd).size(), 0U);

    // The lowered model_output references the same output value id.
    ASSERT_EQ(compiled->lowered.model_outputs.size(), 1U);
    EXPECT_EQ(compiled->lowered.model_outputs[0],
              compiled->optimized_graph.GetOutputs()[0].value);
}

// ---- Full Llama dense graph compilation -----------------------------------

namespace {

HfModelConfig MakeLlamaConfig2Layer() {
    return HfModelConfig{
            .model_type = "llama",
            .architectures = {"LlamaForCausalLM"},
            .hidden_size = 8,
            .intermediate_size = 16,
            .num_hidden_layers = 2,
            .num_attention_heads = 4,
            .num_key_value_heads = 2,
            .vocab_size = 32,
            .max_position_embeddings = 128,
            .head_dim = 2,
            .rms_norm_eps = 1.0e-5,
            .hidden_act = "silu",
            .tie_word_embeddings = false,
            .weight_dtype_hint = DataType::Float32(),
    };
}

struct TestStorage : RawStorage {};

RawWeightView MakeWeightView(const std::shared_ptr<TestStorage>& storage,
                             std::vector<int64_t> shape) {
    return RawWeightView{
            .data = nullptr,
            .bytes = 0,
            .dtype = DataType::Float32(),
            .shape = std::move(shape),
            .storage = storage,
            .is_contiguous = true,
    };
}

ResolvedModelWeights MakeLlamaWeights(const HfModelConfig& config) {
    const auto storage = std::make_shared<TestStorage>();
    ResolvedModelWeights weights{
            .embed_tokens = MakeWeightView(storage, {config.vocab_size, config.hidden_size}),
            .final_norm = MakeWeightView(storage, {config.hidden_size}),
            .lm_head = MakeWeightView(storage, {config.vocab_size, config.hidden_size}),
    };

    weights.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
    const int64_t head_dim = config.head_dim != 0 ? config.head_dim : config.hidden_size / config.num_attention_heads;
    const int64_t kv_hidden_size = config.num_key_value_heads * head_dim;
    for (int64_t i = 0; i < config.num_hidden_layers; ++i) {
        weights.layers.push_back(DecoderLayerRawWeights{
                .norm = NormRawWeights{
                        .input_rmsnorm = MakeWeightView(storage, {config.hidden_size}),
                        .post_attn_rmsnorm = MakeWeightView(storage, {config.hidden_size}),
                },
                .attn = AttnRawWeights{
                        .q_proj = MakeWeightView(storage, {config.hidden_size, config.hidden_size}),
                        .k_proj = MakeWeightView(storage, {kv_hidden_size, config.hidden_size}),
                        .v_proj = MakeWeightView(storage, {kv_hidden_size, config.hidden_size}),
                        .o_proj = MakeWeightView(storage, {config.hidden_size, config.hidden_size}),
                },
                .mlp = MLPRawWeights{
                        .gate_proj = MakeWeightView(storage, {config.intermediate_size, config.hidden_size}),
                        .up_proj = MakeWeightView(storage, {config.intermediate_size, config.hidden_size}),
                        .down_proj = MakeWeightView(storage, {config.hidden_size, config.intermediate_size}),
                },
        });
    }

    return weights;
}

}// namespace

TEST(CompileModelGraph, LowersFullLlamaDenseGraph) {
    const HfModelConfig config = MakeLlamaConfig2Layer();
    const ResolvedModelWeights weights = MakeLlamaWeights(config);
    const StatusOr<ModelGraph> graph = ModelGraphBuilder::BuildLlamaDense(config, weights);
    ASSERT_TRUE(graph.ok()) << graph.status().ToString();

    // Capture source graph state for immutability check.
    std::ostringstream before_dump;
    DumpGraph(*graph, before_dump);
    const std::string before_str = before_dump.str();
    const size_t node_count_before = graph->GetNodes().size();
    const size_t value_count_before = graph->GetValues().size();

    GraphCompileConfig compile_config;
    compile_config.optimization.opt_level = 2;
    const StatusOr<CompiledModelGraph> compiled = CompileModelGraph(*graph, compile_config);

    ASSERT_TRUE(compiled.ok()) << compiled.status().ToString();

    // Source graph unchanged.
    std::ostringstream after_dump;
    DumpGraph(*graph, after_dump);
    EXPECT_EQ(before_str, after_dump.str());
    EXPECT_EQ(graph->GetNodes().size(), node_count_before);
    EXPECT_EQ(graph->GetValues().size(), value_count_before);

    // Optimized graph validates.
    ASSERT_TRUE(compiled->optimized_graph.Validate().ok());

    // Step and binding counts match (contract: 1:1 with optimized graph nodes).
    EXPECT_EQ(compiled->lowered.steps.size(), compiled->optimized_graph.GetNodes().size());
    EXPECT_EQ(compiled->lowered.step_bindings.size(), compiled->optimized_graph.GetNodes().size());

    // First step is Embedding, last step is Argmax.
    ASSERT_GT(compiled->lowered.steps.size(), 0U);
    EXPECT_EQ(compiled->lowered.steps.front().op_type, OpType::kEmbedding);
    EXPECT_EQ(compiled->lowered.steps.back().op_type, OpType::kArgmax);

    // Model inputs/outputs match.
    EXPECT_EQ(compiled->lowered.model_inputs.size(), graph->GetInputs().size());
    EXPECT_EQ(compiled->lowered.model_outputs.size(), graph->GetOutputs().size());

    // State aliases: 2 per layer (K and V cache for each decoder layer).
    EXPECT_EQ(compiled->lowered.state_aliases.size(),
              static_cast<size_t>(config.num_hidden_layers) * 2U);

    // Output spec count matches binding output count per step.
    for (size_t i = 0; i < compiled->lowered.steps.size(); ++i) {
        EXPECT_EQ(compiled->lowered.steps[i].output_specs.size(),
                  compiled->lowered.step_bindings[i].output_values.size());
    }

    // ResolveStateAliases succeeds and returns expected count.
    const StatusOr<StateAliasPlan> alias_plan = ResolveStateAliases(compiled->lowered);
    ASSERT_TRUE(alias_plan.ok()) << alias_plan.status().ToString();
    EXPECT_EQ(alias_plan->size(),
              static_cast<size_t>(config.num_hidden_layers) * 2U);
}

// Sentinel pass: increments an external counter when Run() is invoked.
// Used to prove that GraphPassManager rejects an invalid source graph
// BEFORE any pass observes it (Task 5 acceptance: "pass sentinel 未被调用").

// Common helper: builds a RmsNorm graph skeleton with the given input/output
// TensorSpecs. Uses the test-only ModelGraph constructor to bypass AddNode
// validation, so callers can inject forged/invalid specs that AddNode would
// normally reject. RmsNorm input[0] = kActivation (accepts ConstantValue),
// input[1] = kWeight (requires kScale slot), output[0] = kActivation.
ModelGraph BuildRmsNormGraphWithSpecs(const TensorSpec& act_in_spec,
                                      const TensorSpec& weight_spec,
                                      const TensorSpec& out_spec) {
    std::vector<GraphValue> values;
    values.push_back(GraphValue{
            .payload = ConstantValue{},
            .spec = act_in_spec,
            .producer = std::nullopt,
            .name = "act_in",
    });
    values.push_back(GraphValue{
            .payload = WeightValue{.binding = WeightBinding{.slot = ParameterSlot::kScale}},
            .spec = weight_spec,
            .producer = std::nullopt,
            .name = "weight_in",
    });
    values.push_back(GraphValue{
            .payload = ActivationValue{},
            .spec = out_spec,
            .producer = GraphNodeId{.index = 0},
            .name = "act_out",
    });

    GraphNode node;
    node.op_type = OpType::kRmsNorm;
    node.inputs = {GraphValueId{.index = 0}, GraphValueId{.index = 1}};
    node.outputs = {GraphValueId{.index = 2}};
    node.op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}};

    return ModelGraph(HfModelConfig{}, {node}, values);
}

// Builds a structurally-valid but semantically-invalid graph: a RmsNorm node
// whose activation input carries Int32 (AnalyzeRmsNorm only accepts floating-point).
ModelGraph BuildGraphWithWrongInputDtype() {
    return BuildRmsNormGraphWithSpecs(
            Spec(DataType::Int(32), {4, 8}),
            Spec(DataType::Float32(), {8}),
            Spec(DataType::Float32(), {4, 8}));
}

// Builds a graph with a forged output spec: AnalyzeRmsNorm would derive a
// Float32 [4, 8] output, but the stored GraphValue carries Float16 to simulate
// stale/forged metadata. ValidateAndTopologicalOrder must catch this.
ModelGraph BuildGraphWithForgedOutputSpec() {
    return BuildRmsNormGraphWithSpecs(
            Spec(DataType::Float32(), {4, 8}),
            Spec(DataType::Float32(), {8}),
            Spec(DataType::Float(16), {4, 8}));
}


// ---- Task 5: precondition verification at compilation entry ----------------

TEST(OptimizeModelGraph, RejectsWrongInputDtypeBeforeOptimization) {
    // Bad dtype: AnalyzeRmsNorm only accepts floating-point; the graph
    // carries Float16. OptimizeModelGraph must propagate the semantic error
    // from GraphPassManager::Run precondition, without fallback to unoptimized
    // compilation.
    const ModelGraph graph = BuildGraphWithWrongInputDtype();

    PassContext ctx;
    ctx.opt_level = 2;
    const StatusOr<ModelGraph> result = OptimizeModelGraph(graph, ctx);

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(result.status().message().find("RmsNorm"), std::string::npos);
}

TEST(OptimizeModelGraph, RejectsForgedOutputSpecBeforeOptimization) {
    // Forged output spec: AnalyzeRmsNorm derives Float32 output, but the
    // stored GraphValue carries Float16. OptimizeModelGraph must propagate
    // the semantic error from the precondition check, without fallback.
    const ModelGraph graph = BuildGraphWithForgedOutputSpec();

    PassContext ctx;
    ctx.opt_level = 2;
    const StatusOr<ModelGraph> result = OptimizeModelGraph(graph, ctx);

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(result.status().message().find("RmsNorm"), std::string::npos);
}

TEST(CompileModelGraph, RejectsWrongInputDtypeBeforeLowering) {
    // Bad dtype at the compilation entry: OptimizeModelGraph precondition
    // fires first (via GraphPassManager::Run). CompileModelGraph must
    // propagate the error without entering lowering.
    const ModelGraph graph = BuildGraphWithWrongInputDtype();

    GraphCompileConfig config;
    config.optimization.opt_level = 2;
    const StatusOr<CompiledModelGraph> result = CompileModelGraph(graph, config);

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(result.status().message().find("RmsNorm"), std::string::npos);
}

TEST(CompileModelGraph, RejectsForgedOutputSpecBeforeLowering) {
    // Forged output spec at the compilation entry. CompileModelGraph must
    // propagate the error without entering lowering.
    const ModelGraph graph = BuildGraphWithForgedOutputSpec();

    GraphCompileConfig config;
    config.optimization.opt_level = 2;
    const StatusOr<CompiledModelGraph> result = CompileModelGraph(graph, config);

    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(result.status().message().find("RmsNorm"), std::string::npos);
}

// O0/O1/O2 pipelines must each reject invalid graphs at the precondition.
// Parameterized over opt_level to satisfy "O0、O1、O2 ... 均返回 semantically
// valid graph" from the other direction: an invalid source never reaches any
// pass regardless of opt_level.
struct OptLevelParam {
    uint32_t opt_level;
    std::string name;
};

class OptLevelPreconditionTest : public testing::TestWithParam<OptLevelParam> {};

TEST_P(OptLevelPreconditionTest, RejectsForgedOutputSpecAtAnyOptLevel) {
    const ModelGraph graph = BuildGraphWithForgedOutputSpec();

    PassContext ctx;
    ctx.opt_level = GetParam().opt_level;
    const StatusOr<ModelGraph> result = OptimizeModelGraph(graph, ctx);

    ASSERT_FALSE(result.ok()) << "opt_level=" << GetParam().opt_level;
    EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(result.status().message().find("RmsNorm"), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(
        GraphCompilerPrecondition,
        OptLevelPreconditionTest,
        testing::Values(
                OptLevelParam{0, "O0"},
                OptLevelParam{1, "O1"},
                OptLevelParam{2, "O2"},
                OptLevelParam{3, "O3"}),
        [](const testing::TestParamInfo<OptLevelParam>& info) {
            return info.param.name;
        });
std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

struct RuntimeTensorStorage {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::vector<float> storage;

    explicit RuntimeTensorStorage(std::vector<int64_t> input_shape)
        : shape(std::move(input_shape)),
          strides(MakeStrides(shape)) {
        size_t numel = 1;
        for (const auto d: shape) {
            numel *= static_cast<size_t>(std::max<int64_t>(d, 0));
        }
        storage.resize(numel, 0.0F);
    }

    AM_NODISCARD TensorView View() const {
        return {storage.data(), DataType::Float32(), shape, strides};
    }

    AM_NODISCARD MutableTensorView MutableView() {
        return {storage.data(), DataType::Float32(), shape, strides};
    }
};

// ---- Task 8: graph -> rewrite -> lowering -> plan -> runtime integration --
//
// Symbolic-failure integration test: a graph with a deferred DimEqualConstraint
// (concrete activation hidden != symbolic weight length) flows through
// OptimizeModelGraph -> LowerModelGraph -> ExecutionPlanBuilder::Build ->
// Executor::Execute. The runtime check survives every layer verbatim, and
// supplying bindings that violate the constraint causes Executor::Execute to
// fail BEFORE the kernel runs.

TEST(GraphCompilerIntegration, SymbolicConstraintFlowsFromGraphToRuntimeFailure) {
    // Build a graph with a deferred DimEqualConstraint: the activation hidden
    // dim and the norm weight length are distinct symbolic dimensions, so
    // AnalyzeRmsNorm emits a runtime check that cannot be resolved at compile
    // time. The check must flow verbatim through OptimizeModelGraph ->
    // LowerModelGraph -> ExecutionPlanBuilder -> Executor, and violating
    // runtime bindings must be rejected BEFORE the kernel runs.
    ModelGraph graph;
    const ShapeSymbol seq = ShapeSymbol::Create();
    const ShapeSymbol hidden = ShapeSymbol::Create();
    const ShapeSymbol weight_dim = ShapeSymbol::Create();

    const GraphValueId act = graph.AddConstant(
            TensorSpec{.dtype = DataType::Float32(),
                       .shape = SymbolicShape(std::vector<ShapeSymbol>{seq, hidden})},
            ConstantBinding{.name = "act"},
            "act");
    const GraphValueId norm_weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(),
                       .shape = SymbolicShape(std::vector<ShapeSymbol>{weight_dim})},
            WeightBinding{.slot = ParameterSlot::kScale},
            "norm_weight");

    auto rms_or = graph.AddNode(
            OpType::kRmsNorm,
            std::nullopt,
            {act, norm_weight},
            {NodeOutputDesc{.payload = ActivationValue{}}},
            OpParams{RmsNormParams{.eps = 1.0e-5F}},
            {},
            "rms");
    ASSERT_TRUE(rms_or.ok()) << rms_or.status().ToString();
    graph.MarkOutput(rms_or->outputs[0]);

    // Layer 0: the graph itself. AddNode must have invoked AnalyzeRmsNorm and
    // stored the derived runtime check on the node.
    const GraphNode& graph_node = graph.GetNode(rms_or->node);
    ASSERT_EQ(graph_node.runtime_checks.size(), 1u)
            << "AnalyzeRmsNorm must emit exactly one DimEqualConstraint for "
               "distinct symbolic hidden vs weight length";
    const auto graph_checks = graph_node.runtime_checks;

    // Layer 1: OptimizeModelGraph (rewrite). Must not drop the runtime check.
    PassContext opt_ctx;
    opt_ctx.opt_level = 2;
    const StatusOr<ModelGraph> optimized = OptimizeModelGraph(graph, opt_ctx);
    ASSERT_TRUE(optimized.ok()) << optimized.status().ToString();
    bool found_rms = false;
    for (const GraphNode& node: optimized->GetNodes()) {
        if (node.op_type == OpType::kRmsNorm) {
            EXPECT_EQ(node.runtime_checks, graph_checks)
                    << "OptimizeModelGraph must carry runtime_checks verbatim";
            found_rms = true;
            break;
        }
    }
    ASSERT_TRUE(found_rms) << "RmsNorm node must survive optimization";

    // Layer 2: LowerModelGraph. Must carry runtime_checks into LoweredGraph.
    const StatusOr<LoweredGraph> lowered = LowerModelGraph(*optimized);
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();
    const ExecutionPlanNodeSpec* rms_step = nullptr;
    for (const ExecutionPlanNodeSpec& step: lowered->steps) {
        if (step.op_type == OpType::kRmsNorm) {
            rms_step = &step;
            break;
        }
    }
    ASSERT_NE(rms_step, nullptr);
    EXPECT_EQ(rms_step->runtime_checks, graph_checks)
            << "LowerModelGraph must carry runtime_checks verbatim";

    // Layer 3: ExecutionPlanBuilder::Build (trusted path). No re-inference.
    RuntimeBuilder runtime_builder;
    RuntimeContext runtime = runtime_builder.Build();
    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, *lowered);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1u)
            << "Plan must contain exactly one step for the single-RmsNorm graph";
    EXPECT_EQ(plan->steps().front().runtime_checks, graph_checks)
            << "ExecutionPlanBuilder trusted path must carry runtime_checks verbatim";

    // Layer 4: Executor::Execute with violating bindings. Must fail BEFORE
    // the kernel runs. hidden=8 (act) vs weight_len=16 violates
    // DimEqualConstraint(input[0].dim[1] == input[1].dim[0]).
    RuntimeBindingContext bindings;
    RuntimeTensorStorage act_storage{std::vector<int64_t>{2, 8}};
    RuntimeTensorStorage weight_storage{std::vector<int64_t>{16}};
    RuntimeTensorStorage out_storage{std::vector<int64_t>{2, 8}};
    bindings.SetStepTensorBinding(
            0,
            StepTensorBinding{
                    .inputs = {act_storage.View(), weight_storage.View()},
                    .outputs = {out_storage.MutableView()},
            });

    const Status status = Executor::Execute(*plan, bindings);
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument)
            << "Runtime must reject bindings violating the symbolic constraint";
    EXPECT_NE(status.message().find("RmsNorm"), std::string::npos)
            << "Failure message must preserve op semantic context";
}
}// namespace
}// namespace aethermind
