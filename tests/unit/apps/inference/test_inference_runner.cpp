#define AETHERMIND_INFERENCE_RUNNER_TESTING 1

#include "aethermind/backend/backend.h"
#include "aethermind/backend/backend_factory.h"
#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/memory/allocator.h"
#include "aethermind/model/loaded_model.h"
#include "aethermind/model/model_loader.h"
#include "aethermind/runtime/runtime_builder.h"
#include "inference_runner_test.h"
#include "model/test_utils.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <utility>

namespace {

using namespace aethermind;
using namespace aethermind::apps::inference;
namespace fs = std::filesystem;
namespace runner_test = aethermind::apps::inference::testing;

constexpr std::string_view kAllocationFailure = "runner injected allocation failure";
constexpr std::string_view kBackendFailure = "runner injected backend factory failure";
constexpr std::string_view kExecutionFailure = "runner injected execution failure";

fs::path BaseModelDirectory() {
    return fs::path(AETHERMIND_TEST_MODELS_DIR) / "tiny-random-LlamaForCausalLM";
}

RunOptions RunnerOptions() {
    RunOptions options;
    options.model_dir = BaseModelDirectory();
    options.prompt_tokens = {7};
    options.context_tokens = 8;
    options.generation.max_new_tokens = 3;
    return options;
}

struct RunnerResources {
    size_t allocation_attempts = 0;
    size_t allocations = 0;
    size_t live_allocations = 0;
    size_t releases = 0;
    size_t allocator_creations = 0;
    size_t allocator_destructions = 0;
    size_t provider_destructions = 0;
    size_t backend_creations = 0;
    size_t backend_destructions = 0;
    size_t factory_destructions = 0;
    size_t kernel_preparations = 0;
    size_t fail_allocation_attempt = 0;
};

struct AllocationOwner {
    std::shared_ptr<RunnerResources> resources;
};

void ReleaseAllocation(void* context, void* data) noexcept {
    std::unique_ptr<AllocationOwner> owner(static_cast<AllocationOwner*>(context));
    --owner->resources->live_allocations;
    ++owner->resources->releases;
    std::free(data);
}

class RecordingAllocator final : public Allocator {
public:
    explicit RecordingAllocator(std::shared_ptr<RunnerResources> resources)
        : resources_(std::move(resources)) {}

    ~RecordingAllocator() override {
        ++resources_->allocator_destructions;
    }

    Device device() const noexcept override {
        return Device::CPU();
    }

protected:
    StatusOr<MemoryHandle> AllocateStorage(const AllocationRequest& request) override {
        ++resources_->allocation_attempts;
        if (resources_->allocation_attempts == resources_->fail_allocation_attempt) {
            return Status::Aborted(std::string(kAllocationFailure));
        }

        const size_t alignment = std::max(request.alignment, size_t{64});
        const size_t requested = std::max(request.nbytes, size_t{1});
        if (requested > std::numeric_limits<size_t>::max() - (alignment - 1)) {
            return Status::Overflow("test allocation size overflow");
        }
        const size_t bytes = (requested + alignment - 1) & ~(alignment - 1);
        auto owner = std::make_unique<AllocationOwner>(AllocationOwner{resources_});
        void* const data = std::aligned_alloc(alignment, bytes);
        if (data == nullptr) {
            return Status::ResourceExhausted({});
        }
        ++resources_->allocations;
        ++resources_->live_allocations;
        // The handle owns the tracking context independently of the allocator.
        return MemoryHandle(data, owner.release(), &ReleaseAllocation, Device::CPU(), alignment);
    }

private:
    std::shared_ptr<RunnerResources> resources_;
};

class RecordingAllocatorProvider final : public AllocatorProvider {
public:
    explicit RecordingAllocatorProvider(std::shared_ptr<RunnerResources> resources)
        : resources_(std::move(resources)) {}

    ~RecordingAllocatorProvider() override {
        ++resources_->provider_destructions;
    }

    std::unique_ptr<Allocator> CreateAllocator(Device device) override {
        EXPECT_EQ(device, Device::CPU());
        ++resources_->allocator_creations;
        return std::make_unique<RecordingAllocator>(resources_);
    }

private:
    std::shared_ptr<RunnerResources> resources_;
};

void RegisterRecordingAllocator(RuntimeBuilder& builder,
                                const std::shared_ptr<RunnerResources>& resources) {
    builder.RegisterCustomAllocatorProvider(
            DeviceType::kCPU, std::make_unique<RecordingAllocatorProvider>(resources));
}

enum class BackendFailure {
    kFactoryStatus,
    kPrepareBadAlloc,
    kExecutionStatus,
};

Status FailExecution(const KernelContext&) noexcept {
    return Status::Aborted(std::string(kExecutionFailure));
}

class FailingBackend final : public Backend {
public:
    FailingBackend(std::unique_ptr<Backend> delegate,
                   std::shared_ptr<RunnerResources> resources, BackendFailure failure)
        : delegate_(std::move(delegate)), resources_(std::move(resources)), failure_(failure) {}

    ~FailingBackend() override {
        ++resources_->backend_destructions;
    }

    DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    StatusOr<ResolvedKernel> PrepareKernel(OpType op_type, const KernelSelector& selector,
                                           const OpParams& params) const override {
        ++resources_->kernel_preparations;
        if (failure_ == BackendFailure::kPrepareBadAlloc) {
            // Throw outside BackendRegistry::GetBackend's noexcept factory path.
            throw std::bad_alloc();
        }
        auto kernel = delegate_->PrepareKernel(op_type, selector, params);
        if (kernel.ok() && failure_ == BackendFailure::kExecutionStatus && op_type == OpType::kArgmax) {
            kernel->fn = &FailExecution;
        }
        return kernel;
    }

    StatusOr<PackingRecipe> GetPackingRecipe(OpType op_type,
                                             const KernelSelector& selector) const override {
        return delegate_->GetPackingRecipe(op_type, selector);
    }

    StatusOr<std::unique_ptr<PackedWeight>> PackWeights(
            OpType op_type, std::span<const TensorView> components,
            const KernelSelector& selector, const PackingRecipe& recipe,
            Allocator& allocator) const override {
        return delegate_->PackWeights(op_type, components, selector, recipe, allocator);
    }

    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return delegate_->TryGetKernelRegistryForDebug();
    }

private:
    std::unique_ptr<Backend> delegate_;
    std::shared_ptr<RunnerResources> resources_;
    BackendFailure failure_;
};

class FailingBackendFactory final : public BackendFactory {
public:
    FailingBackendFactory(std::shared_ptr<RunnerResources> resources, BackendFailure failure)
        : resources_(std::move(resources)), failure_(failure) {}

    ~FailingBackendFactory() override {
        ++resources_->factory_destructions;
    }

    DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    StatusOr<std::unique_ptr<Backend>> Create() const override {
        ++resources_->backend_creations;
        if (failure_ == BackendFailure::kFactoryStatus) {
            return Status::Unavailable(std::string(kBackendFailure));
        }
        auto backend = CpuBackendFactory{}.Create();
        if (!backend.ok()) {
            return backend.status();
        }
        return std::unique_ptr<Backend>(
                std::make_unique<FailingBackend>(std::move(*backend), resources_, failure_));
    }

private:
    std::shared_ptr<RunnerResources> resources_;
    BackendFailure failure_;
};

void ExpectReleasedResources(const RunnerResources& resources) {
    EXPECT_EQ(resources.live_allocations, 0U);
    EXPECT_EQ(resources.releases, resources.allocations);
    EXPECT_EQ(resources.allocator_destructions, resources.allocator_creations);
    EXPECT_EQ(resources.provider_destructions, 1U);
}

Status CopyModelWithConfigReplacement(const fs::path& destination,
                                      std::string_view original, std::string_view replacement) {
    const fs::path source = BaseModelDirectory();
    std::ifstream stream(source / "config.json", std::ios::binary);
    if (!stream.is_open()) {
        return Status::NotFound("M0 base config is missing");
    }
    std::string config((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    const size_t position = config.find(original);
    if (position == std::string::npos) {
        return Status::InvalidArgument("M0 base config does not contain the expected field");
    }
    config.replace(position, original.size(), replacement);
    WriteTextFile(destination / "config.json", config);
    fs::copy_file(source / "model.safetensors", destination / "model.safetensors");
    return Status::Ok();
}

TEST(InferenceRunnerFailure, ReportsLoadingFailureBeforeCreatingRuntimeResources) {
    RunOptions options = RunnerOptions();
    options.model_dir /= "missing-model-directory";
    auto resources = std::make_shared<RunnerResources>();
    {
        RuntimeBuilder builder;
        RegisterRecordingAllocator(builder, resources);
        RunStage stage = RunStage::kGenerate;
        const auto result = runner_test::RunInferenceWithBuilder(options, builder, &stage);
        EXPECT_FALSE(result.ok());
        EXPECT_EQ(result.status().code(), StatusCode::kNotFound);
        EXPECT_EQ(stage, RunStage::kLoad);
        EXPECT_EQ(resources->allocator_creations, 0U);
        EXPECT_EQ(resources->allocation_attempts, 0U);
    }
    ExpectReleasedResources(*resources);
}

TEST(InferenceRunnerFailure, ReportsInvalidRequestBeforeCreatingRuntimeResources) {
    RunOptions options = RunnerOptions();
    options.prompt_tokens.clear();
    auto resources = std::make_shared<RunnerResources>();
    {
        RuntimeBuilder builder;
        RegisterRecordingAllocator(builder, resources);
        RunStage stage = RunStage::kNone;
        const auto result = runner_test::RunInferenceWithBuilder(options, builder, &stage);
        EXPECT_FALSE(result.ok());
        EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
        EXPECT_EQ(stage, RunStage::kParameters);
        EXPECT_EQ(resources->allocator_creations, 0U);
        EXPECT_EQ(resources->allocation_attempts, 0U);
    }
    ExpectReleasedResources(*resources);
}

TEST(InferenceRunnerFailure, ReportsCapacityErrorBeforeCreatingRuntimeResources) {
    RunOptions options = RunnerOptions();
    options.context_tokens = 2;
    auto resources = std::make_shared<RunnerResources>();
    {
        RuntimeBuilder builder;
        RegisterRecordingAllocator(builder, resources);
        RunStage stage = RunStage::kNone;
        const auto result = runner_test::RunInferenceWithBuilder(options, builder, &stage);
        EXPECT_FALSE(result.ok());
        EXPECT_EQ(result.status().code(), StatusCode::kOutOfRange);
        EXPECT_EQ(stage, RunStage::kParameters);
        EXPECT_EQ(resources->allocation_attempts, 0U);
    }
    ExpectReleasedResources(*resources);
}

class RunnerKvAllocationFailure : public ::testing::TestWithParam<size_t> {};

TEST_P(RunnerKvAllocationFailure, PreservesStatusAndReleasesPartialKvStorage) {
    auto resources = std::make_shared<RunnerResources>();
    resources->fail_allocation_attempt = GetParam();
    RuntimeBuilder builder;
    RegisterRecordingAllocator(builder, resources);
    RunStage stage = RunStage::kNone;

    const auto result = runner_test::RunInferenceWithBuilder(RunnerOptions(), builder, &stage);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kAborted);
    EXPECT_EQ(result.status().message(), kAllocationFailure);
    EXPECT_EQ(stage, RunStage::kRuntimeBuild);
    EXPECT_EQ(resources->allocation_attempts, GetParam());
    EXPECT_EQ(resources->allocations, GetParam() - 1);
    EXPECT_EQ(resources->allocator_creations, 1U);
    ExpectReleasedResources(*resources);
}

INSTANTIATE_TEST_SUITE_P(
        KvPlanes, RunnerKvAllocationFailure, ::testing::Values(size_t{1}, size_t{2}),
        [](const ::testing::TestParamInfo<size_t>& info) {
            return info.param == 1 ? "FirstPlane" : "SecondPlane";
        });

TEST(InferenceRunnerFailure, ReportsLazyBackendFactoryFailureAsPreparationFailure) {
    auto resources = std::make_shared<RunnerResources>();
    RuntimeBuilder builder;
    RegisterRecordingAllocator(builder, resources);
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<FailingBackendFactory>(resources, BackendFailure::kFactoryStatus));
    RunStage stage = RunStage::kNone;

    const auto result = runner_test::RunInferenceWithBuilder(RunnerOptions(), builder, &stage);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kUnavailable);
    EXPECT_EQ(result.status().message(), kBackendFailure);
    EXPECT_EQ(stage, RunStage::kPrepare);
    EXPECT_EQ(resources->allocations, 2U);
    EXPECT_EQ(resources->backend_creations, 1U);
    EXPECT_EQ(resources->factory_destructions, 1U);
    ExpectReleasedResources(*resources);
}

TEST(InferenceRunnerFailure, ConvertsEscapingBadAllocToEmptyMessageAndReleasesStorage) {
    auto resources = std::make_shared<RunnerResources>();
    RuntimeBuilder builder;
    RegisterRecordingAllocator(builder, resources);
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<FailingBackendFactory>(resources, BackendFailure::kPrepareBadAlloc));
    RunStage stage = RunStage::kNone;

    const auto result = runner_test::RunInferenceWithBuilder(RunnerOptions(), builder, &stage);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(result.status().message().empty());
    EXPECT_EQ(stage, RunStage::kPrepare);
    EXPECT_EQ(resources->allocations, 2U);
    EXPECT_GT(resources->kernel_preparations, 0U);
    EXPECT_EQ(resources->backend_destructions, 1U);
    EXPECT_EQ(resources->factory_destructions, 1U);
    ExpectReleasedResources(*resources);
}

TEST(InferenceRunnerFailure, PreservesExecutionFailureAndReleasesRequestAndKvStorage) {
    auto resources = std::make_shared<RunnerResources>();
    RuntimeBuilder builder;
    RegisterRecordingAllocator(builder, resources);
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<FailingBackendFactory>(resources, BackendFailure::kExecutionStatus));
    RunStage stage = RunStage::kNone;

    const auto result = runner_test::RunInferenceWithBuilder(RunnerOptions(), builder, &stage);

    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kAborted);
    EXPECT_EQ(result.status().message(), kExecutionFailure);
    EXPECT_EQ(stage, RunStage::kGenerate);
    EXPECT_GT(resources->allocations, 2U);
    EXPECT_EQ(resources->backend_destructions, 1U);
    EXPECT_EQ(resources->factory_destructions, 1U);
    ExpectReleasedResources(*resources);
}

struct RejectedPrecisionCase {
    const char* directory;
    const char* parameter_name;
};

constexpr std::array<RejectedPrecisionCase, 16> kRejectedPrecisionCases{{
        {"f16-missing", "F16Missing"},
        {"f16-auto", "F16Auto"},
        {"bf16-missing", "BF16Missing"},
        {"bf16-auto", "BF16Auto"},
        {"f16-f32-hint", "F16F32Hint"},
        {"bf16-f32-hint", "BF16F32Hint"},
        {"f32-f16-hint", "F32F16Hint"},
        {"f32-bf16-hint", "F32BF16Hint"},
        {"mixed-norm-f16-missing", "MixedNormF16Missing"},
        {"mixed-norm-f16-auto", "MixedNormF16Auto"},
        {"mixed-norm-bf16-missing", "MixedNormBF16Missing"},
        {"mixed-norm-bf16-auto", "MixedNormBF16Auto"},
        {"single-norm-f16-missing", "SingleNormF16Missing"},
        {"single-norm-f16-auto", "SingleNormF16Auto"},
        {"single-norm-bf16-missing", "SingleNormBF16Missing"},
        {"single-norm-bf16-auto", "SingleNormBF16Auto"},
}};

class RunnerFp32Admission : public ::testing::TestWithParam<RejectedPrecisionCase> {};

TEST_P(RunnerFp32Admission, RejectsLoaderAcceptedPrecisionBeforeAllocatingKv) {
    RunOptions options = RunnerOptions();
    options.model_dir = fs::path(AETHERMIND_TEST_MODELS_DIR) / "inference-admission" / GetParam().directory;
    const auto loaded = ModelLoader::Load(options.model_dir);
    ASSERT_TRUE(loaded.ok()) << "Generate and check M0 fixtures before this test: " << loaded.status().ToString();
    auto resources = std::make_shared<RunnerResources>();
    {
        RuntimeBuilder builder;
        RegisterRecordingAllocator(builder, resources);
        RunStage stage = RunStage::kNone;
        const auto result = runner_test::RunInferenceWithBuilder(options, builder, &stage);
        EXPECT_FALSE(result.ok());
        EXPECT_EQ(result.status().code(), StatusCode::kUnimplemented);
        EXPECT_NE(result.status().message().find("required Float32"), std::string::npos);
        EXPECT_EQ(stage, RunStage::kModelAdmission);
        EXPECT_EQ(resources->allocator_creations, 0U);
        EXPECT_EQ(resources->allocation_attempts, 0U);
    }
    ExpectReleasedResources(*resources);
}

INSTANTIATE_TEST_SUITE_P(
        AdmissionFixtures, RunnerFp32Admission, ::testing::ValuesIn(kRejectedPrecisionCases),
        [](const ::testing::TestParamInfo<RejectedPrecisionCase>& info) { return info.param.parameter_name; });

TEST(InferenceRunnerFailure, RejectsUnsupportedFamilyAtCompileBeforeAllocatingKv) {
    TempDirectory directory;
    ASSERT_TRUE(CopyModelWithConfigReplacement(directory.path(), R"("model_type": "llama")",
                                               R"("model_type": "qwen2")")
                        .ok());
    const auto loaded = ModelLoader::Load(directory.path());
    ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
    EXPECT_EQ((*loaded)->GetConfig().model_type, "qwen2");
    RunOptions options = RunnerOptions();
    options.model_dir = directory.path();
    auto resources = std::make_shared<RunnerResources>();
    {
        RuntimeBuilder builder;
        RegisterRecordingAllocator(builder, resources);
        RunStage stage = RunStage::kNone;
        const auto result = runner_test::RunInferenceWithBuilder(options, builder, &stage);
        EXPECT_FALSE(result.ok());
        EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
        EXPECT_NE(result.status().message().find("unsupported model type"), std::string::npos);
        EXPECT_EQ(stage, RunStage::kCompile);
        EXPECT_EQ(resources->allocator_creations, 0U);
        EXPECT_EQ(resources->allocation_attempts, 0U);
    }
    ExpectReleasedResources(*resources);
}

TEST(InferenceRunnerFailure, RejectsUnknownRopeAtCompileBeforeAllocatingKv) {
    TempDirectory directory;
    ASSERT_TRUE(CopyModelWithConfigReplacement(directory.path(), R"("rope_type": "default")",
                                               R"("rope_type": "unsupported_admission_fixture")")
                        .ok());
    const auto loaded = ModelLoader::Load(directory.path());
    ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
    EXPECT_EQ((*loaded)->GetConfig().rope.algorithm, HfRoPEAlgorithm::kUnknown);
    RunOptions options = RunnerOptions();
    options.model_dir = directory.path();
    auto resources = std::make_shared<RunnerResources>();
    {
        RuntimeBuilder builder;
        RegisterRecordingAllocator(builder, resources);
        RunStage stage = RunStage::kNone;
        const auto result = runner_test::RunInferenceWithBuilder(options, builder, &stage);
        EXPECT_FALSE(result.ok());
        EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
        EXPECT_NE(result.status().message().find("RoPE"), std::string::npos);
        EXPECT_EQ(stage, RunStage::kCompile);
        EXPECT_EQ(resources->allocator_creations, 0U);
        EXPECT_EQ(resources->allocation_attempts, 0U);
    }
    ExpectReleasedResources(*resources);
}

} // namespace
