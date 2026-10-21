#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/base/workspace_arena.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_context.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/kv_cache/kv_cache_manager.h"
#include "aethermind/memory/cpu_allocator.h"
#include "inference/inference_internal.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace {

using namespace aethermind;

constexpr size_t kPromptLength = 2;
constexpr size_t kVocabSize = 32;
constexpr ExecutionValueId kTokenOutput{.index = 2};

class RecordingWorkspaceArena final : public WorkspaceArena {
public:
    WorkspaceBinding Bind(const WorkspaceRequirement&) noexcept override { return {}; }
    void Reset() noexcept override { ++reset_count; }

    size_t reset_count = 0;
};

class Prefill : public ::testing::Test {
protected:
    void SetUp() override {
        token_kernel_calls_ = 0;
        ASSERT_TRUE(cache_.Init(allocator_, 1, 1, 8, 1, DataType::Float32()).ok());
        auto view = cache_.ReserveForSession(kPromptLength, 1);
        ASSERT_TRUE(view.ok()) << view.status().ToString();
        view_ = *view;
    }

    void TearDown() override {
        context_.reset();
        if (view_.valid()) {
            EXPECT_TRUE(cache_.ReleaseSession(view_).ok());
        }
        token_kernel_calls_ = 0;
    }

    // Token computation is outside this helper's contract; the fixture supplies
    // its output, while a real CPU update writes the KV range being committed.
    static Status RecordTokenKernel(const KernelContext&) noexcept {
        ++token_kernel_calls_;
        return Status::Ok();
    }

    static Status FailingTokenKernel(const KernelContext&) noexcept {
        ++token_kernel_calls_;
        return Status::Aborted("synthetic token kernel failure");
    }

    Status PrepareContext(bool append = true, bool with_kv = true,
                          KernelFunc token_kernel = &RecordTokenKernel) {
        const TensorSpec activation{.dtype = DataType::Float32(),
                                    .shape = SymbolicShape(IntArrayView(kv_shape_))};
        const TensorSpec token{.dtype = DataType::Int(64),
                               .shape = SymbolicShape(IntArrayView(token_shape_))};
        std::vector<ExecutionValueDesc> values{
                {.spec = activation, .kind = ExecutionValueKind::kModelInput},
                {.spec = activation, .kind = ExecutionValueKind::kModelInput},
                {.spec = token, .kind = ExecutionValueKind::kActivation}};
        if (with_kv) {
            const int64_t cache_shape[3] = {1, 8, 1};
            const TensorSpec state{.dtype = DataType::Float32(),
                                   .shape = SymbolicShape(IntArrayView(cache_shape))};
            for (const auto slot: {ExecutionKVCacheSlot::kKey, ExecutionKVCacheSlot::kValue,
                                   ExecutionKVCacheSlot::kKey, ExecutionKVCacheSlot::kValue}) {
                values.push_back({.spec = state,
                                  .kind = ExecutionValueKind::kState,
                                  .state_binding = ExecutionKVCacheStateIdentity{
                                          .decoder_layer_index = 0,
                                          .slot = slot}});
            }
        }

        const KernelSelector selector{.device_type = DeviceType::kCPU,
                                      .act_dtype = DataType::Float32(),
                                      .weight_dtype = DataType::Float32()};
        std::vector<ExecutionStep> steps;
        StateAliasPlan aliases;
        if (append) {
            CpuBackend backend;
            AM_ASSIGN_OR_RETURN(auto update,
                                backend.PrepareKernel(OpType::kKVCacheUpdate, selector,
                                                      OpParams{KVCacheUpdateParams{}}));
            steps.push_back({.selector = selector,
                             .kernel = std::move(update),
                             .inputs = {{.index = 0}, {.index = 1}, {.index = 3}, {.index = 4}},
                             .outputs = {{.index = 5}, {.index = 6}},
                             .kernel_input_ports = {0, 1}});
            aliases.aliases = {{.step_index = 0, .input_port = 2, .output_port = 0},
                               {.step_index = 0, .input_port = 3, .output_port = 1}};
        }
        steps.push_back({.selector = selector,
                         .kernel = {.op_type = OpType::kArgmax, .fn = token_kernel},
                         .inputs = {{.index = 0}},
                         .outputs = {kTokenOutput},
                         .kernel_input_ports = {0},
                         .kernel_output_ports = {0}});
        AM_ASSIGN_OR_RETURN(auto plan,
                            ExecutionPlan::Create(std::move(values), {{.index = 0}, {.index = 1}},
                                                  {kTokenOutput}, std::move(steps), std::move(aliases)));
        plan_.emplace(std::move(plan));
        AM_ASSIGN_OR_RETURN(auto prepared,
                            PrepareExecutionBindings(
                                    *plan_,
                                    {.read_only = {{.value = {.index = 0},
                                                    .tensor = TensorView(key_.data(), DataType::Float32(), kv_shape_, kv_strides_)},
                                                   {.value = {.index = 1},
                                                    .tensor = TensorView(value_.data(), DataType::Float32(), kv_shape_, kv_strides_)}},
                                     .writable = {{.value = kTokenOutput,
                                                   .tensor = MutableTensorView(tokens_.data(), DataType::Int(64), token_shape_, token_strides_)}}},
                                    allocator_));
        AM_ASSIGN_OR_RETURN(auto context,
                            ExecutionContext::Create(*plan_, std::move(prepared), &workspace_,
                                                     with_kv ? view_ : KVCacheSessionView{}));
        context_.emplace(std::move(context));
        return Status::Ok();
    }

    StatusOr<uint32_t> Run(size_t prompt_length = kPromptLength,
                           size_t vocab_size = kVocabSize,
                           ExecutionValueId output = kTokenOutput) {
        return inference::internal::RunPrefill(*plan_, *context_, output, prompt_length, vocab_size);
    }

    CPUAllocator allocator_{Device::CPU()};
    KVCacheManager cache_{};
    KVCacheSessionView view_{};
    RecordingWorkspaceArena workspace_{};
    std::array<float, 2> key_{1.0F, 2.0F};
    std::array<float, 2> value_{3.0F, 4.0F};
    std::array<int64_t, 2> tokens_{3, 7};
    const std::array<int64_t, 2> kv_shape_{2, 1};
    const std::array<int64_t, 2> kv_strides_{1, 1};
    const std::array<int64_t, 1> token_shape_{2};
    const std::array<int64_t, 1> token_strides_{1};
    std::optional<ExecutionPlan> plan_{};
    std::optional<ExecutionContext> context_{};
    static inline size_t token_kernel_calls_ = 0;
};

TEST_F(Prefill, ReturnsLastPositionTokenAfterCommittingPrompt) {
    ASSERT_TRUE(PrepareContext().ok());

    const auto token = Run();

    ASSERT_TRUE(token.ok()) << token.status().ToString();
    EXPECT_EQ(*token, 7U);
    EXPECT_EQ(view_.current_pos(), kPromptLength);
    EXPECT_FALSE(view_.awaiting_prefill());
    EXPECT_EQ(workspace_.reset_count, 1U);
    EXPECT_EQ(token_kernel_calls_, 1U);
    EXPECT_NE(context_->prepared_bindings(), nullptr);
}

TEST_F(Prefill, RejectsInvalidPromptLengthBeforeExecution) {
    ASSERT_TRUE(PrepareContext().ok());

    EXPECT_EQ(Run(0).status().code(), StatusCode::kInvalidArgument);
    if constexpr (std::numeric_limits<size_t>::max() >
                  static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        EXPECT_EQ(Run(std::numeric_limits<size_t>::max()).status().code(),
                  StatusCode::kInvalidArgument);
    }
    EXPECT_EQ(workspace_.reset_count, 0U);
    EXPECT_EQ(token_kernel_calls_, 0U);
    EXPECT_EQ(view_.current_pos(), 0U);
}

TEST_F(Prefill, RejectsZeroVocabularyBeforeExecution) {
    ASSERT_TRUE(PrepareContext().ok());

    EXPECT_EQ(Run(kPromptLength, 0).status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(workspace_.reset_count, 0U);
    EXPECT_EQ(token_kernel_calls_, 0U);
    EXPECT_EQ(view_.current_pos(), 0U);
}

TEST_F(Prefill, RejectsMissingKVSessionBeforeExecution) {
    ASSERT_TRUE(PrepareContext(/*append=*/false, /*with_kv=*/false).ok());

    EXPECT_EQ(Run().status().code(), StatusCode::kFailedPrecondition);
    EXPECT_EQ(workspace_.reset_count, 0U);
    EXPECT_EQ(token_kernel_calls_, 0U);
}

TEST_F(Prefill, RejectsReservationLengthMismatchBeforeExecution) {
    ASSERT_TRUE(PrepareContext().ok());

    EXPECT_EQ(Run(1).status().code(), StatusCode::kFailedPrecondition);
    EXPECT_EQ(workspace_.reset_count, 0U);
    EXPECT_EQ(token_kernel_calls_, 0U);
    EXPECT_EQ(view_.current_pos(), 0U);
}

TEST_F(Prefill, RejectsRepeatedPrefillBeforeExecution) {
    ASSERT_TRUE(PrepareContext().ok());
    ASSERT_TRUE(Run().ok());

    EXPECT_EQ(Run().status().code(), StatusCode::kFailedPrecondition);
    EXPECT_EQ(workspace_.reset_count, 1U);
    EXPECT_EQ(token_kernel_calls_, 1U);
    EXPECT_EQ(view_.current_pos(), kPromptLength);
}

TEST_F(Prefill, PropagatesKernelFailureWithoutCommittingKV) {
    ASSERT_TRUE(PrepareContext(/*append=*/true, /*with_kv=*/true, &FailingTokenKernel).ok());

    const auto token = Run();

    EXPECT_EQ(token.status().code(), StatusCode::kAborted);
    EXPECT_EQ(token.status().message(), "synthetic token kernel failure");
    EXPECT_EQ(view_.current_pos(), 0U);
    EXPECT_TRUE(view_.awaiting_prefill());
    EXPECT_EQ(workspace_.reset_count, 1U);
    EXPECT_EQ(token_kernel_calls_, 1U);
}

TEST_F(Prefill, RejectsPlanThatDoesNotCommitThePrompt) {
    ASSERT_TRUE(PrepareContext(/*append=*/false).ok());

    EXPECT_EQ(Run().status().code(), StatusCode::kInternal);
    EXPECT_EQ(view_.current_pos(), 0U);
    EXPECT_TRUE(view_.awaiting_prefill());
    EXPECT_EQ(token_kernel_calls_, 1U);
}

TEST_F(Prefill, InvalidOutputShapeDoesNotRollBackCommittedKV) {
    ASSERT_TRUE(PrepareContext().ok());

    EXPECT_EQ(Run(kPromptLength, kVocabSize, {.index = 0}).status().code(), StatusCode::kInternal);
    EXPECT_EQ(view_.current_pos(), kPromptLength);
    EXPECT_FALSE(view_.awaiting_prefill());
    EXPECT_EQ(token_kernel_calls_, 1U);
}

class PrefillInvalidToken : public Prefill, public ::testing::WithParamInterface<int64_t> {};

TEST_P(PrefillInvalidToken, RejectsTokenWithoutRollingBackCommittedKV) {
    tokens_.back() = GetParam();
    ASSERT_TRUE(PrepareContext().ok());
    const size_t vocab_size = std::numeric_limits<size_t>::max();

    EXPECT_EQ(Run(kPromptLength, vocab_size).status().code(), StatusCode::kOutOfRange);
    EXPECT_EQ(view_.current_pos(), kPromptLength);
    EXPECT_FALSE(view_.awaiting_prefill());
}

INSTANTIATE_TEST_SUITE_P(InvalidToken, PrefillInvalidToken,
                         ::testing::Values(int64_t{-1},
                                           static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1));

TEST_F(Prefill, OutOfVocabularyTokenDoesNotRollBackCommittedKV) {
    tokens_.back() = static_cast<int64_t>(kVocabSize);
    ASSERT_TRUE(PrepareContext().ok());

    EXPECT_EQ(Run().status().code(), StatusCode::kOutOfRange);
    EXPECT_EQ(view_.current_pos(), kPromptLength);
    EXPECT_FALSE(view_.awaiting_prefill());
}

} // namespace
