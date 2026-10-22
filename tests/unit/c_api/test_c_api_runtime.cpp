#include "c_api.h"
#include "c_api_runtime_internal.h"
#include "test_c_api_runtime_helpers.h"

#include "aethermind/compiler/model_compiler.h"
#include "aethermind/memory/buffer.h"
#include "aethermind/model/model_loader.h"
#include "aethermind/runtime/runtime_builder.h"
#include "inference/test_malloc_interposer.h"
#include "model/test_utils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

using namespace aethermind;
namespace ct = aethermind::test::c_api;
namespace injection = aethermind::test;

constexpr std::array<uint32_t, 3> kSentinel{0x12345678U, 0x23456789U, 0x3456789aU};

void ExpectError(const ct::Error& error, am_status_code code) {
    EXPECT_NE(error.value, nullptr);
    EXPECT_EQ(am_error_code(error.value), code) << error.message();
}

TEST(CAbiRuntime, ReportsFrozenVersion) {
    EXPECT_EQ(am_get_abi_version(), AM_ABI_VERSION);
}

TEST(CAbiRuntime, NullDestroyFunctionsAreNoOps) {
    am_runtime_destroy(nullptr);
    am_model_unload(nullptr);
    am_session_destroy(nullptr);
}

TEST(CAbiRuntime, NullConfigIsInvalidArgument) {
    ct::Error error;
    ct::RuntimeOwner runtime(am_runtime_create(nullptr, &error.value));
    EXPECT_EQ(runtime, nullptr);
    ExpectError(error, AM_STATUS_INVALID_ARGUMENT);
}

TEST(CAbiRuntime, NullErrorOutputIsAllowed) {
    EXPECT_EQ(am_runtime_create(nullptr, nullptr), nullptr);
    auto config = ct::RuntimeConfig();
    ct::RuntimeOwner runtime(am_runtime_create(&config, nullptr));
    EXPECT_NE(runtime, nullptr);
}

TEST(CAbiRuntime, SuccessClearsButDoesNotReleasePreviousErrorSlot) {
    ct::Error retained;
    retained.value = am_error_create(AM_STATUS_ABORTED, "retained by caller");
    ASSERT_NE(retained.value, nullptr);
    am_error_handle slot = retained.value;
    auto config = ct::RuntimeConfig();
    ct::RuntimeOwner runtime(am_runtime_create(&config, &slot));
    ASSERT_NE(runtime, nullptr);
    EXPECT_EQ(slot, nullptr);
    EXPECT_EQ(am_error_code(retained.value), AM_STATUS_ABORTED);
    EXPECT_STREQ(retained.message(), "retained by caller");
}

TEST(CAbiRuntime, RejectsUnsupportedMajorAndHigherMinor) {
    for (const uint32_t version: {0U, 0x00020000U, AM_ABI_VERSION + 1U}) {
        SCOPED_TRACE(version);
        auto config = ct::RuntimeConfig();
        config.abi_version = version;
        ct::Error error;
        ct::RuntimeOwner runtime(am_runtime_create(&config, &error.value));
        EXPECT_EQ(runtime, nullptr);
        ExpectError(error, AM_STATUS_UNIMPLEMENTED);
    }
}

TEST(CAbiRuntime, RejectsWrongStructSizes) {
    for (const uint32_t size: {0U, 8U, uint32_t{sizeof(am_config_t) - 1}, uint32_t{sizeof(am_config_t) + 8}}) {
        SCOPED_TRACE(size);
        auto config = ct::RuntimeConfig();
        config.struct_size = size;
        ct::Error error;
        ct::RuntimeOwner runtime(am_runtime_create(&config, &error.value));
        EXPECT_EQ(runtime, nullptr);
        ExpectError(error, AM_STATUS_INVALID_ARGUMENT);
    }
}

#if defined(__linux__)
class GuardedPrefix {
public:
    GuardedPrefix() {
        const long page = sysconf(_SC_PAGESIZE);
        if (page <= 0) {
            return;
        }
        page_size_ = static_cast<size_t>(page);
        memory_ = mmap(nullptr, page_size_ * 2, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (memory_ == MAP_FAILED) {
            memory_ = nullptr;
            return;
        }
        if (mprotect(static_cast<char*>(memory_) + page_size_, page_size_, PROT_NONE) != 0) {
            munmap(memory_, page_size_ * 2);
            memory_ = nullptr;
            return;
        }
        const std::array<uint32_t, 2> prefix{8, AM_ABI_VERSION};
        std::memcpy(data(), prefix.data(), sizeof(prefix));
    }

    GuardedPrefix(const GuardedPrefix&) = delete;
    GuardedPrefix& operator=(const GuardedPrefix&) = delete;
    ~GuardedPrefix() {
        if (memory_ != nullptr) {
            munmap(memory_, page_size_ * 2);
        }
    }

    void* data() const noexcept {
        return memory_ == nullptr ? nullptr : static_cast<char*>(memory_) + page_size_ - 8;
    }

private:
    void* memory_ = nullptr;
    size_t page_size_ = 0;
};

TEST(CAbiRuntime, WrongSizeReadsOnlyAccessibleEightBytePrefix) {
    GuardedPrefix prefix;
    ASSERT_NE(prefix.data(), nullptr);
    ct::Error error;
    ct::RuntimeOwner runtime(am_runtime_create(static_cast<const am_config_t*>(prefix.data()), &error.value));
    EXPECT_EQ(runtime, nullptr);
    ExpectError(error, AM_STATUS_INVALID_ARGUMENT);
}
#endif

TEST(CAbiRuntime, RejectsUnknownPresetAndDtype) {
    for (const bool preset: {false, true}) {
        auto config = ct::RuntimeConfig();
        (preset ? config.cpu_preset : config.kv_dtype) = std::numeric_limits<uint32_t>::max();
        ct::Error error;
        ct::RuntimeOwner runtime(am_runtime_create(&config, &error.value));
        EXPECT_EQ(runtime, nullptr);
        ExpectError(error, AM_STATUS_INVALID_ARGUMENT);
    }
}

TEST(CAbiRuntime, RejectsEachZeroGeometryField) {
    constexpr std::array fields{&am_config_t::num_layers, &am_config_t::num_kv_heads,
                                &am_config_t::head_dim, &am_config_t::max_tokens};
    for (const auto field: fields) {
        auto config = ct::RuntimeConfig();
        config.*field = 0;
        ct::Error error;
        ct::RuntimeOwner runtime(am_runtime_create(&config, &error.value));
        EXPECT_EQ(runtime, nullptr);
        ExpectError(error, AM_STATUS_INVALID_ARGUMENT);
    }
}

TEST(CAbiRuntime, RejectsLayoutOverflowBeforeAllocatingPlanes) {
    auto config = ct::RuntimeConfig();
    config.head_dim = std::numeric_limits<uint64_t>::max();
    ct::Error error;
    ct::RuntimeOwner runtime(am_runtime_create(&config, &error.value));
    EXPECT_EQ(runtime, nullptr);
    ExpectError(error, sizeof(size_t) < sizeof(uint64_t) ? AM_STATUS_OUT_OF_RANGE : AM_STATUS_OVERFLOW);
}

TEST(CAbiRuntime, RejectsRowAlignmentOverflowBeforeAllocatingPlanes) {
    auto config = ct::RuntimeConfig();
    config.head_dim = std::numeric_limits<size_t>::max() / sizeof(uint32_t);
    ct::Error error;
    ct::RuntimeOwner runtime(am_runtime_create(&config, &error.value));
    EXPECT_EQ(runtime, nullptr);
    ExpectError(error, AM_STATUS_OVERFLOW);
}

struct RuntimeAllocationCase {
    const char* name;
    injection::MallocFailureTarget target;
    size_t skip;
};

class CAbiRuntimeAllocationFailure : public ::testing::TestWithParam<RuntimeAllocationCase> {};

TEST_P(CAbiRuntimeAllocationFailure, FailureDoesNotPublishHandleAndCanRetry) {
    ASSERT_TRUE(injection::MallocInterposerAvailable()) << "Requires Linux/glibc allocation-injection evidence";
    const auto& param = GetParam();
    const bool metadata = param.target == injection::MallocFailureTarget::kMalloc;
    auto config = ct::RuntimeConfig();
    ct::Error error;
    injection::ScopedMallocFailure failure(metadata ? sizeof(BufferImpl) : 0, param.skip, false, param.target);
    ct::RuntimeOwner runtime(am_runtime_create(&config, &error.value));
    const auto result = failure.Stop();
    EXPECT_TRUE(result.allocation_failed);
    EXPECT_EQ(runtime, nullptr);
    ExpectError(error, AM_STATUS_RESOURCE_EXHAUSTED);
    const size_t allocated_planes = param.skip + (metadata ? 1 : 0);
    EXPECT_EQ(result.aligned_allocation_calls, allocated_planes);
    if (allocated_planes != 0) {
        EXPECT_TRUE(result.last_aligned_allocation_released);
    }
    error.Clear();
    runtime.reset(am_runtime_create(&config, &error.value));
    EXPECT_NE(runtime, nullptr) << error.message();
    EXPECT_EQ(error.value, nullptr);
}

INSTANTIATE_TEST_SUITE_P(
        KvPlanes, CAbiRuntimeAllocationFailure,
        ::testing::Values(
                RuntimeAllocationCase{"FirstPlaneStorage", injection::MallocFailureTarget::kPosixMemalign, 0},
                RuntimeAllocationCase{"SecondPlaneStorage", injection::MallocFailureTarget::kPosixMemalign, 1},
                RuntimeAllocationCase{"FirstPlaneMetadata", injection::MallocFailureTarget::kMalloc, 0},
                RuntimeAllocationCase{"SecondPlaneMetadata", injection::MallocFailureTarget::kMalloc, 1}),
        [](const ::testing::TestParamInfo<RuntimeAllocationCase>& info) { return info.param.name; });

TEST(CAbiRuntime, RegistryAllocationFailureReturnsResourceExhausted) {
    ASSERT_TRUE(injection::MallocInterposerAvailable());
    auto config = ct::RuntimeConfig();
    ct::Error error;
    injection::ScopedMallocFailure failure(0);
    ct::RuntimeOwner runtime(am_runtime_create(&config, &error.value));
    const auto result = failure.Stop();
    EXPECT_TRUE(result.allocation_failed);
    EXPECT_EQ(runtime, nullptr);
    ExpectError(error, AM_STATUS_RESOURCE_EXHAUSTED);
}

TEST(CAbiRuntime, ErrorObjectAllocationFailureLeavesDiagnosticUnavailable) {
    ASSERT_TRUE(injection::MallocInterposerAvailable());
    auto config = ct::RuntimeConfig();
    config.kv_dtype = std::numeric_limits<uint32_t>::max();
    ct::Error error;
    injection::BeginMallocCallCounting();
    // Status construction owns two copies of its text before the optional
    // error object is allocated. Reject that object, after status creation.
    injection::ScopedMallocFailure failure(0, 2);
    const am_runtime_t runtime = am_runtime_create(&config, &error.value);
    const auto result = failure.Stop();
    const auto counts = injection::EndMallocCallCounting();
    EXPECT_TRUE(result.allocation_failed);
    EXPECT_EQ(runtime, nullptr);
    EXPECT_EQ(error.value, nullptr);
    EXPECT_EQ(counts.malloc_calls, counts.free_calls + 1);
    am_runtime_destroy(runtime);
}

TEST(CAbiRuntime, ErrorMessageAllocationFailureReleasesDiagnosticObject) {
    ASSERT_TRUE(injection::MallocInterposerAvailable());
    constexpr std::string_view message = "Unknown KV dtype; required Float32";
    auto config = ct::RuntimeConfig();
    config.kv_dtype = std::numeric_limits<uint32_t>::max();
    ct::Error error;
    injection::BeginMallocCallCounting();
    // Original Status construction uses two copies of this text; reject its
    // third copy into the independently owned error object.
    injection::ScopedMallocFailure failure(message.size() + 1, 2);
    const am_runtime_t runtime = am_runtime_create(&config, &error.value);
    const auto result = failure.Stop();
    const auto counts = injection::EndMallocCallCounting();
    EXPECT_TRUE(result.allocation_failed);
    EXPECT_EQ(runtime, nullptr);
    EXPECT_EQ(error.value, nullptr);
    EXPECT_EQ(counts.malloc_calls, counts.free_calls + 1);
    am_runtime_destroy(runtime);
}

class CAbiModel : public ::testing::Test {
protected:
    void SetUp() override {
        const auto directory = ct::Fixture();
        ASSERT_TRUE(std::filesystem::is_regular_file(directory / "config.json")) << directory;
        auto loaded = ModelLoader::Load(directory);
        ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
        config_ = ct::RuntimeConfig((*loaded)->GetConfig());
        vocab_size_ = static_cast<uint32_t>((*loaded)->GetConfig().vocab_size);
        model_context_ = static_cast<size_t>((*loaded)->GetConfig().max_position_embeddings);
        path_ = directory.string();
        runtime_.reset(am_runtime_create(&config_, &error_.value));
        ASSERT_NE(runtime_, nullptr) << error_.message();
        ASSERT_EQ(error_.value, nullptr);
    }

    void LoadModel() {
        model_.reset(am_model_load(runtime_.get(), path_.c_str(), &error_.value));
        ASSERT_NE(model_, nullptr) << error_.message();
        ASSERT_EQ(error_.value, nullptr);
    }

    ct::RuntimeOwner runtime_;
    ct::ModelOwner model_;
    ct::Error error_;
    am_config_t config_{};
    uint32_t vocab_size_ = 0;
    size_t model_context_ = 0;
    std::string path_;
};

TEST_F(CAbiModel, NullRuntimeOrPathAndEmptyPathAreInvalid) {
    ct::ModelOwner invalid(am_model_load(nullptr, path_.c_str(), &error_.value));
    EXPECT_EQ(invalid, nullptr);
    ExpectError(error_, AM_STATUS_INVALID_ARGUMENT);
    for (const char* path: {static_cast<const char*>(nullptr), ""}) {
        error_.Clear();
        invalid.reset(am_model_load(runtime_.get(), path, &error_.value));
        EXPECT_EQ(invalid, nullptr);
        ExpectError(error_, AM_STATUS_INVALID_ARGUMENT);
    }
}

TEST_F(CAbiModel, MissingModelDoesNotInvalidateRuntime) {
    const std::string missing = (ct::Fixture() / "missing-model").string();
    ct::ModelOwner invalid(am_model_load(runtime_.get(), missing.c_str(), &error_.value));
    EXPECT_EQ(invalid, nullptr);
    EXPECT_NE(error_.value, nullptr);
    error_.Clear();
    LoadModel();
}

TEST_F(CAbiModel, CompileRejectionPreservesRuntimeAndOriginalStatus) {
    constexpr std::array replacements{
            std::pair{std::string_view{"\"model_type\": \"llama\""},
                      std::string_view{"\"model_type\": \"qwen2\""}},
            std::pair{std::string_view{"\"rope_type\": \"default\""},
                      std::string_view{"\"rope_type\": \"unsupported_c_abi_fixture\""}},
    };
    for (const auto& [original, replacement]: replacements) {
        SCOPED_TRACE(original);
        TempDirectory directory;
        std::ifstream input(ct::Fixture() / "config.json", std::ios::binary);
        ASSERT_TRUE(input.is_open());
        std::string config{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        const auto position = config.find(original);
        ASSERT_NE(position, std::string::npos);
        config.replace(position, original.size(), replacement);
        WriteTextFile(directory.path() / "config.json", config);
        std::filesystem::copy_file(ct::Fixture() / "model.safetensors", directory.path() / "model.safetensors");
        const auto loaded = ModelLoader::Load(directory.path());
        ASSERT_TRUE(loaded.ok()) << loaded.status().ToString();
        const std::string path = directory.path().string();
        ct::ModelOwner rejected(am_model_load(runtime_.get(), path.c_str(), &error_.value));
        EXPECT_EQ(rejected, nullptr);
        ExpectError(error_, AM_STATUS_INVALID_ARGUMENT);
        EXPECT_FALSE(std::string_view(error_.message()).empty());
        error_.Clear();
        LoadModel();
        model_.reset();
    }
}

TEST_F(CAbiModel, MismatchedGeometryIsRejectedBeforeModelPublication) {
    constexpr std::array fields{&am_config_t::num_layers, &am_config_t::num_kv_heads, &am_config_t::head_dim};
    for (const auto field: fields) {
        auto mismatch = config_;
        ++(mismatch.*field);
        ct::Error local_error;
        ct::RuntimeOwner runtime(am_runtime_create(&mismatch, &local_error.value));
        ASSERT_NE(runtime, nullptr) << local_error.message();
        ct::ModelOwner rejected(am_model_load(runtime.get(), path_.c_str(), &local_error.value));
        EXPECT_EQ(rejected, nullptr);
        ExpectError(local_error, AM_STATUS_INVALID_ARGUMENT);
    }
    LoadModel();
}

TEST_F(CAbiModel, CapacityAboveModelContextIsRejectedAtModelLoad) {
    auto config = config_;
    config.max_tokens = model_context_ + 1;
    ct::Error local_error;
    ct::RuntimeOwner runtime(am_runtime_create(&config, &local_error.value));
    ASSERT_NE(runtime, nullptr) << local_error.message();
    ct::ModelOwner rejected(am_model_load(runtime.get(), path_.c_str(), &local_error.value));
    EXPECT_EQ(rejected, nullptr);
    ExpectError(local_error, AM_STATUS_OUT_OF_RANGE);
}

TEST_F(CAbiModel, CapacityBelowAndAtContextAreBothAdmitted) {
    for (const size_t capacity: {size_t{1}, model_context_}) {
        auto config = config_;
        config.max_tokens = capacity;
        ct::Error local_error;
        ct::RuntimeOwner runtime(am_runtime_create(&config, &local_error.value));
        ASSERT_NE(runtime, nullptr) << local_error.message();
        ct::ModelOwner model(am_model_load(runtime.get(), path_.c_str(), &local_error.value));
        EXPECT_NE(model, nullptr) << local_error.message();
        EXPECT_EQ(local_error.value, nullptr);
    }
}

TEST_F(CAbiModel, EveryUnsupportedPrecisionVariantIsRejectedAndRuntimeCanRetry) {
    constexpr std::array variants{
            "f16-missing", "f16-auto", "bf16-missing", "bf16-auto",
            "f16-f32-hint", "bf16-f32-hint", "f32-f16-hint", "f32-bf16-hint",
            "mixed-norm-f16-missing", "mixed-norm-f16-auto", "mixed-norm-bf16-missing", "mixed-norm-bf16-auto",
            "single-norm-f16-missing", "single-norm-f16-auto", "single-norm-bf16-missing", "single-norm-bf16-auto"};
    for (const char* variant: variants) {
        SCOPED_TRACE(variant);
        const std::string path = ct::Fixture(variant).string();
        ASSERT_TRUE(std::filesystem::is_regular_file(ct::Fixture(variant) / "model.safetensors"));
        ct::ModelOwner rejected(am_model_load(runtime_.get(), path.c_str(), &error_.value));
        EXPECT_EQ(rejected, nullptr);
        ExpectError(error_, AM_STATUS_UNIMPLEMENTED);
        error_.Clear();
        LoadModel();
        model_.reset();
    }
}

TEST_F(CAbiModel, NullErrorOutputAllowsSuccessfulAndFailedLoads) {
    EXPECT_EQ(am_model_load(nullptr, path_.c_str(), nullptr), nullptr);
    ct::ModelOwner model(am_model_load(runtime_.get(), path_.c_str(), nullptr));
    EXPECT_NE(model, nullptr);
}

TEST_F(CAbiModel, AllocationFailureDoesNotPublishModelAndRuntimeCanRetry) {
    ASSERT_TRUE(injection::MallocInterposerAvailable());
    injection::ScopedMallocFailure failure(0);
    ct::ModelOwner model(am_model_load(runtime_.get(), path_.c_str(), &error_.value));
    const auto result = failure.Stop();
    EXPECT_TRUE(result.allocation_failed);
    EXPECT_EQ(model, nullptr);
    ExpectError(error_, AM_STATUS_RESOURCE_EXHAUSTED);
    error_.Clear();
    LoadModel();
}

TEST_F(CAbiModel, PackedPreparationAllocationFailureReleasesStorageAndCanRetry) {
    ASSERT_TRUE(injection::MallocInterposerAvailable());
    auto config = config_;
    config.cpu_preset = AM_CPU_PRESET_O2_PACKED;
    ct::Error error;
    ct::RuntimeOwner runtime(am_runtime_create(&config, &error.value));
    ASSERT_NE(runtime, nullptr) << error.message();
    // Loading maps raw bytes. The first new aligned allocation belongs to
    // packed-weight preparation; reject its owning Buffer metadata afterwards.
    injection::ScopedMallocFailure failure(sizeof(BufferImpl), 0, true);
    ct::ModelOwner model(am_model_load(runtime.get(), path_.c_str(), &error.value));
    const auto result = failure.Stop();
    EXPECT_TRUE(result.allocation_failed);
    EXPECT_GE(result.aligned_allocation_calls, 1U);
    EXPECT_TRUE(result.last_aligned_allocation_released);
    EXPECT_EQ(model, nullptr);
    ExpectError(error, AM_STATUS_RESOURCE_EXHAUSTED);
    error.Clear();
    model.reset(am_model_load(runtime.get(), path_.c_str(), &error.value));
    EXPECT_NE(model, nullptr) << error.message();
    EXPECT_EQ(error.value, nullptr);
}

class CAbiSession : public CAbiModel {
protected:
    void SetUp() override {
        CAbiModel::SetUp();
        if (::testing::Test::HasFatalFailure()) {
            return;
        }
        LoadModel();
    }
};

TEST_F(CAbiSession, NullModelOrConfigIsInvalidArgument) {
    auto config = ct::SessionConfig();
    ct::SessionOwner session(am_session_create(nullptr, &config, &error_.value));
    EXPECT_EQ(session, nullptr);
    ExpectError(error_, AM_STATUS_INVALID_ARGUMENT);
    error_.Clear();
    session.reset(am_session_create(model_.get(), nullptr, &error_.value));
    EXPECT_EQ(session, nullptr);
    ExpectError(error_, AM_STATUS_INVALID_ARGUMENT);
}

TEST_F(CAbiSession, RejectsWrongSizeAndVersion) {
    auto config = ct::SessionConfig();
    config.struct_size = 8;
    ct::SessionOwner session(am_session_create(model_.get(), &config, &error_.value));
    EXPECT_EQ(session, nullptr);
    ExpectError(error_, AM_STATUS_INVALID_ARGUMENT);
    for (const uint32_t version: {0U, 0x00020000U, AM_ABI_VERSION + 1U}) {
        error_.Clear();
        config = ct::SessionConfig();
        config.abi_version = version;
        session.reset(am_session_create(model_.get(), &config, &error_.value));
        EXPECT_EQ(session, nullptr);
        ExpectError(error_, AM_STATUS_UNIMPLEMENTED);
    }
}

#if defined(__linux__)
TEST_F(CAbiSession, WrongSizeReadsOnlyAccessibleEightBytePrefix) {
    GuardedPrefix prefix;
    ASSERT_NE(prefix.data(), nullptr);
    ct::SessionOwner session(am_session_create(model_.get(), static_cast<const am_session_config_t*>(prefix.data()), &error_.value));
    EXPECT_EQ(session, nullptr);
    ExpectError(error_, AM_STATUS_INVALID_ARGUMENT);
}
#endif

TEST_F(CAbiSession, RejectsUnknownEosFlagAndGenerationByteOverflow) {
    auto config = ct::SessionConfig();
    config.has_eos_token_id = 2;
    ct::SessionOwner session(am_session_create(model_.get(), &config, &error_.value));
    EXPECT_EQ(session, nullptr);
    ExpectError(error_, AM_STATUS_INVALID_ARGUMENT);
    error_.Clear();
    config = ct::SessionConfig(std::numeric_limits<uint64_t>::max());
    session.reset(am_session_create(model_.get(), &config, &error_.value));
    EXPECT_EQ(session, nullptr);
    ExpectError(error_, sizeof(size_t) < sizeof(uint64_t) ? AM_STATUS_OUT_OF_RANGE : AM_STATUS_OVERFLOW);
}

TEST_F(CAbiSession, AllocationFailureDoesNotPublishSessionAndCanRetry) {
    ASSERT_TRUE(injection::MallocInterposerAvailable());
    auto config = ct::SessionConfig();
    injection::ScopedMallocFailure failure(0);
    ct::SessionOwner session(am_session_create(model_.get(), &config, &error_.value));
    const auto result = failure.Stop();
    EXPECT_TRUE(result.allocation_failed);
    EXPECT_EQ(session, nullptr);
    ExpectError(error_, AM_STATUS_RESOURCE_EXHAUSTED);
    error_.Clear();
    session.reset(am_session_create(model_.get(), &config, &error_.value));
    EXPECT_NE(session, nullptr) << error_.message();
    EXPECT_EQ(error_.value, nullptr);
}

class CAbiGenerate : public CAbiSession {
protected:
    void SetUp() override {
        CAbiSession::SetUp();
        if (::testing::Test::HasFatalFailure()) {
            return;
        }
        CreateSession(ct::SessionConfig());
    }

    void CreateSession(const am_session_config_t& config) {
        session_.reset(am_session_create(model_.get(), &config, &error_.value));
        ASSERT_NE(session_, nullptr) << error_.message();
        ASSERT_EQ(error_.value, nullptr);
    }

    am_status_code Generate(const uint32_t* prompt, size_t prompt_len, uint32_t* output, size_t capacity, size_t* length) {
        return am_session_generate(session_.get(), prompt, prompt_len, output, capacity, length, &error_.value);
    }

    ct::SessionOwner session_;
};

TEST_F(CAbiGenerate, RejectsNullSessionAndRequiredPointers) {
    constexpr uint32_t prompt = 7;
    auto output = kSentinel;
    size_t length = 99;
    EXPECT_EQ(am_session_generate(nullptr, &prompt, 1, output.data(), output.size(), &length, &error_.value), AM_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(length, 0U);
    EXPECT_EQ(output, kSentinel);
    error_.Clear();
    EXPECT_EQ(Generate(&prompt, 1, output.data(), output.size(), nullptr), AM_STATUS_INVALID_ARGUMENT);
    error_.Clear();
    for (const bool null_prompt: {false, true}) {
        length = 99;
        EXPECT_EQ(Generate(null_prompt ? nullptr : &prompt, null_prompt ? 1 : 0, output.data(), output.size(), &length), AM_STATUS_INVALID_ARGUMENT);
        EXPECT_EQ(length, 0U);
        EXPECT_EQ(output, kSentinel);
        error_.Clear();
    }
    length = 99;
    EXPECT_EQ(Generate(&prompt, 1, nullptr, 1, &length), AM_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(length, 0U);
}

TEST_F(CAbiGenerate, ShortOutputReportsUpperBoundBeforeTokenValidation) {
    const uint32_t invalid_prompt = vocab_size_;
    auto output = kSentinel;
    size_t length = 99;
    EXPECT_EQ(Generate(&invalid_prompt, 1, output.data(), 2, &length), AM_STATUS_OUT_OF_RANGE);
    EXPECT_EQ(length, 3U);
    EXPECT_EQ(output, kSentinel);
    ExpectError(error_, AM_STATUS_OUT_OF_RANGE);
    EXPECT_NE(std::string(error_.message()).find("capacity"), std::string::npos);
}

TEST_F(CAbiGenerate, NullZeroOutputQueriesUpperBoundWithoutExecution) {
    constexpr uint32_t prompt = 7;
    size_t length = 0;
    EXPECT_EQ(Generate(&prompt, 1, nullptr, 0, &length), AM_STATUS_OUT_OF_RANGE);
    EXPECT_EQ(length, 3U);
    ExpectError(error_, AM_STATUS_OUT_OF_RANGE);
}

TEST_F(CAbiGenerate, ShortOutputThenRequestAllocationFailureProvesNoExecution) {
    ASSERT_TRUE(injection::MallocInterposerAvailable());
    constexpr uint32_t prompt = 7;
    auto output = kSentinel;
    size_t queried = 0;
    size_t failed_length = 99;
    injection::ScopedMallocFailure failure(0, 0, false, injection::MallocFailureTarget::kPosixMemalign);
    const auto short_code = am_session_generate(session_.get(), &prompt, 1, output.data(), 2, &queried, nullptr);
    const auto full_code = am_session_generate(session_.get(), &prompt, 1, output.data(), output.size(), &failed_length, nullptr);
    const auto result = failure.Stop();
    EXPECT_EQ(short_code, AM_STATUS_OUT_OF_RANGE);
    EXPECT_EQ(queried, 3U);
    EXPECT_TRUE(result.allocation_failed);
    EXPECT_EQ(full_code, AM_STATUS_RESOURCE_EXHAUSTED);
    EXPECT_EQ(failed_length, 0U);
    EXPECT_EQ(output, kSentinel);
    size_t length = 0;
    EXPECT_EQ(Generate(&prompt, 1, output.data(), output.size(), &length), AM_STATUS_OK);
    EXPECT_EQ(length, 3U);
}

TEST_F(CAbiGenerate, InvalidPromptAndEosLeaveOutputUnchanged) {
    auto output = kSentinel;
    const uint32_t invalid_prompt = vocab_size_;
    size_t length = 99;
    EXPECT_EQ(Generate(&invalid_prompt, 1, output.data(), output.size(), &length), AM_STATUS_OUT_OF_RANGE);
    EXPECT_EQ(length, 0U);
    EXPECT_EQ(output, kSentinel);
    error_.Clear();
    auto config = ct::SessionConfig();
    config.has_eos_token_id = 1;
    config.eos_token_id = vocab_size_;
    CreateSession(config);
    constexpr uint32_t prompt = 7;
    EXPECT_EQ(Generate(&prompt, 1, output.data(), output.size(), &length), AM_STATUS_OUT_OF_RANGE);
    EXPECT_EQ(length, 0U);
    EXPECT_EQ(output, kSentinel);
}

TEST_F(CAbiGenerate, PhysicalCapacityFailureDoesNotPublishOutput) {
    CreateSession(ct::SessionConfig(config_.max_tokens + 1));
    constexpr uint32_t prompt = 7;
    std::vector<uint32_t> output(config_.max_tokens + 1, kSentinel.front());
    const auto before = output;
    size_t length = 99;
    EXPECT_EQ(Generate(&prompt, 1, output.data(), output.size(), &length), AM_STATUS_OUT_OF_RANGE);
    EXPECT_EQ(length, 0U);
    EXPECT_EQ(output, before);
}

TEST_F(CAbiGenerate, ZeroGenerationStillValidatesPromptEosAndModelContext) {
    CreateSession(ct::SessionConfig(0));
    size_t length = 99;
    EXPECT_EQ(Generate(nullptr, 0, nullptr, 0, &length), AM_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(length, 0U);
    error_.Clear();
    const uint32_t invalid_prompt = vocab_size_;
    EXPECT_EQ(Generate(&invalid_prompt, 1, nullptr, 0, &length), AM_STATUS_OUT_OF_RANGE);
    EXPECT_EQ(length, 0U);
    error_.Clear();
    std::vector<uint32_t> prompt(model_context_ + 1, 7);
    EXPECT_EQ(Generate(prompt.data(), prompt.size(), nullptr, 0, &length), AM_STATUS_OUT_OF_RANGE);
    EXPECT_EQ(length, 0U);
    error_.Clear();
    auto config = ct::SessionConfig(0);
    config.has_eos_token_id = 1;
    config.eos_token_id = vocab_size_;
    CreateSession(config);
    constexpr uint32_t token = 7;
    EXPECT_EQ(Generate(&token, 1, nullptr, 0, &length), AM_STATUS_OUT_OF_RANGE);
    EXPECT_EQ(length, 0U);
}

TEST_F(CAbiGenerate, DisabledEosFieldIsIgnoredAndSessionConfigIsCopied) {
    auto config = ct::SessionConfig();
    config.eos_token_id = vocab_size_;
    CreateSession(config);
    config.max_new_tokens = 0;
    config.has_eos_token_id = 1;
    constexpr uint32_t prompt = 7;
    auto output = kSentinel;
    size_t length = 0;
    EXPECT_EQ(Generate(&prompt, 1, output.data(), output.size(), &length), AM_STATUS_OK);
    EXPECT_EQ(length, 3U);
}

TEST_F(CAbiGenerate, NullErrorOutputStillReturnsFailureCodeAndSuccess) {
    constexpr uint32_t prompt = 7;
    auto output = kSentinel;
    size_t length = 99;
    EXPECT_EQ(am_session_generate(session_.get(), &prompt, 1, output.data(), 2, &length, nullptr), AM_STATUS_OUT_OF_RANGE);
    EXPECT_EQ(length, 3U);
    EXPECT_EQ(output, kSentinel);
    EXPECT_EQ(am_session_generate(session_.get(), &prompt, 1, output.data(), output.size(), &length, nullptr), AM_STATUS_OK);
    EXPECT_EQ(length, 3U);
}

TEST_F(CAbiGenerate, RequestAllocationFailureClearsLengthAndCanRetry) {
    ASSERT_TRUE(injection::MallocInterposerAvailable());
    constexpr uint32_t prompt = 7;
    auto output = kSentinel;
    size_t length = 99;
    injection::ScopedMallocFailure failure(0);
    const auto code = Generate(&prompt, 1, output.data(), output.size(), &length);
    const auto result = failure.Stop();
    EXPECT_TRUE(result.allocation_failed);
    EXPECT_EQ(code, AM_STATUS_RESOURCE_EXHAUSTED);
    EXPECT_EQ(length, 0U);
    EXPECT_EQ(output, kSentinel);
    ExpectError(error_, AM_STATUS_RESOURCE_EXHAUSTED);
    error_.Clear();
    EXPECT_EQ(Generate(&prompt, 1, output.data(), output.size(), &length), AM_STATUS_OK);
    EXPECT_EQ(length, 3U);
}

TEST_F(CAbiGenerate, DiagnosticMessageAllocationFailureKeepsOriginalStatusAndUpperBound) {
    ASSERT_TRUE(injection::MallocInterposerAvailable());
    constexpr std::string_view message = "Output capacity is smaller than max_new_tokens";
    constexpr uint32_t prompt = 7;
    auto output = kSentinel;
    size_t length = 99;
    injection::BeginMallocCallCounting();
    injection::ScopedMallocFailure failure(message.size() + 1);
    const auto code = Generate(&prompt, 1, output.data(), 2, &length);
    const auto result = failure.Stop();
    const auto counts = injection::EndMallocCallCounting();
    EXPECT_TRUE(result.allocation_failed);
    EXPECT_EQ(code, AM_STATUS_OUT_OF_RANGE);
    EXPECT_EQ(length, 3U);
    EXPECT_EQ(output, kSentinel);
    EXPECT_EQ(error_.value, nullptr);
    EXPECT_EQ(counts.malloc_calls, counts.free_calls + 1);
}

class CAbiKVValidation : public ::testing::Test {
protected:
    void SetUp() override {
        ModelCompileOptions options;
        options.optimization.opt_level = 1;
        options.lowering.enable_packed_weights = false;
        options.lowering.selector.device_type = DeviceType::kCPU;
        options.lowering.selector.phase = ExecPhase::kBoth;
        auto compiled = ModelCompiler::LoadAndCompile(ct::Fixture(), options);
        ASSERT_TRUE(compiled.ok()) << compiled.status().ToString();
        artifact_.emplace(std::move(*compiled));
        const auto& config = artifact_->loaded_model->GetConfig();
        RuntimeOptions runtime_options;
        runtime_options.kv_cache = {.enable_manager = true,
                                    .num_layers = static_cast<size_t>(config.num_hidden_layers),
                                    .num_kv_heads = static_cast<size_t>(config.num_key_value_heads),
                                    .max_tokens = 8,
                                    .head_dim = static_cast<size_t>(config.head_dim),
                                    .kv_dtype = DataType::Float32(),
                                    .alignment = 64};
        RuntimeBuilder builder;
        builder.WithOptions(runtime_options);
        auto built = builder.TryBuild();
        ASSERT_TRUE(built.ok()) << built.status().ToString();
        runtime_.emplace(std::move(*built));
    }

    LoweredGraph::Builder Clone() const {
        const auto& graph = artifact_->graph;
        LoweredGraph::Builder builder;
        builder.steps.assign(graph.steps().begin(), graph.steps().end());
        builder.values.assign(graph.values().begin(), graph.values().end());
        builder.model_inputs.assign(graph.model_inputs().begin(), graph.model_inputs().end());
        builder.model_outputs.assign(graph.model_outputs().begin(), graph.model_outputs().end());
        builder.state_aliases.assign(graph.state_aliases().begin(), graph.state_aliases().end());
        return builder;
    }

    void ReplaceGraph(LoweredGraph::Builder builder) {
        // Keep copied port metadata in agreement with the modified value specs,
        // so rejection comes from the adapter's cross-object policy, not the
        // compiler artifact's structural trust boundary.
        for (auto& step: builder.steps) {
            for (size_t i = 0; i < step.binding.input_values.size(); ++i) {
                step.spec.input_specs[i] = builder.values[step.binding.input_values[i].index].spec;
            }
            for (size_t i = 0; i < step.binding.output_values.size(); ++i) {
                step.spec.output_specs[i] = builder.values[step.binding.output_values[i].index].spec;
            }
        }
        auto graph = std::move(builder).Build();
        ASSERT_TRUE(graph.ok()) << graph.status().ToString();
        artifact_->graph = std::move(*graph);
    }

    std::optional<Runtime> runtime_;
    std::optional<LoweredModelArtifact> artifact_;
};

TEST_F(CAbiKVValidation, AcceptsRealCompiledGraphIncludingStateAliases) {
    ASSERT_FALSE(artifact_->graph.state_aliases().empty());
    EXPECT_TRUE(c_api::detail::ValidateRuntimeModelKV(*runtime_, *artifact_).ok());
}

TEST_F(CAbiKVValidation, MissingManagerIsFailedPrecondition) {
    RuntimeBuilder builder;
    auto runtime = builder.TryBuild();
    ASSERT_TRUE(runtime.ok()) << runtime.status().ToString();
    const auto result = c_api::detail::ValidateRuntimeModelKV(*runtime, *artifact_);
    EXPECT_EQ(result.code(), StatusCode::kFailedPrecondition);
}

TEST_F(CAbiKVValidation, RuntimeDtypeMustMatchCompiledStateDtype) {
    const auto& config = artifact_->loaded_model->GetConfig();
    RuntimeOptions options;
    options.kv_cache = {.enable_manager = true,
                        .num_layers = static_cast<size_t>(config.num_hidden_layers),
                        .num_kv_heads = static_cast<size_t>(config.num_key_value_heads),
                        .max_tokens = 8,
                        .head_dim = static_cast<size_t>(config.head_dim),
                        .kv_dtype = DataType::BFloat(16),
                        .alignment = 64};
    RuntimeBuilder builder;
    builder.WithOptions(options);
    auto runtime = builder.TryBuild();
    ASSERT_TRUE(runtime.ok()) << runtime.status().ToString();
    EXPECT_EQ(c_api::detail::ValidateRuntimeModelKV(*runtime, *artifact_).code(), StatusCode::kInvalidArgument);
}

TEST_F(CAbiKVValidation, RejectsNonFirstLayerStateDtype) {
    auto builder = Clone();
    size_t changed = 0;
    for (auto& value: builder.values) {
        const auto* state = std::get_if<StateValue>(&value.payload);
        const auto* kv = state == nullptr ? nullptr : std::get_if<KVCacheStateBinding>(&state->binding);
        if (kv != nullptr && kv->decoder_layer_index == 1 && kv->slot == KVCacheSlot::kValue) {
            value.spec.dtype = DataType::BFloat(16);
            ++changed;
        }
    }
    ASSERT_GT(changed, 0U);
    ReplaceGraph(std::move(builder));
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    const auto result = c_api::detail::ValidateRuntimeModelKV(*runtime_, *artifact_);
    EXPECT_EQ(result.code(), StatusCode::kInvalidArgument);
}

TEST_F(CAbiKVValidation, RejectsNonFloat32AliasOutputEvenWhenRootIsFloat32) {
    auto builder = Clone();
    ASSERT_FALSE(builder.state_aliases.empty());
    builder.values[builder.state_aliases.back().output.index].spec.dtype = DataType::Float(16);
    ReplaceGraph(std::move(builder));
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    const auto result = c_api::detail::ValidateRuntimeModelKV(*runtime_, *artifact_);
    EXPECT_EQ(result.code(), StatusCode::kInvalidArgument);
}

TEST_F(CAbiKVValidation, RejectsMissingNonFirstLayerValueState) {
    auto builder = Clone();
    size_t changed = 0;
    for (auto& value: builder.values) {
        auto* state = std::get_if<StateValue>(&value.payload);
        const auto* kv = state == nullptr ? nullptr : std::get_if<KVCacheStateBinding>(&state->binding);
        if (kv != nullptr && kv->decoder_layer_index == 1 && kv->slot == KVCacheSlot::kValue) {
            state->binding = DecodeStateBinding{};
            ++changed;
        }
    }
    ASSERT_GT(changed, 0U);
    ReplaceGraph(std::move(builder));
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    EXPECT_EQ(c_api::detail::ValidateRuntimeModelKV(*runtime_, *artifact_).code(), StatusCode::kInvalidArgument);
}

TEST_F(CAbiKVValidation, RejectsAdditionalDuplicateRootState) {
    auto builder = Clone();
    const auto root = std::find_if(builder.values.begin(), builder.values.end(), [](const auto& value) {
        return std::holds_alternative<StateValue>(value.payload);
    });
    ASSERT_NE(root, builder.values.end());
    const auto copied = *root;
    builder.values.push_back(copied);
    ReplaceGraph(std::move(builder));
    ASSERT_FALSE(::testing::Test::HasFatalFailure());
    EXPECT_EQ(c_api::detail::ValidateRuntimeModelKV(*runtime_, *artifact_).code(), StatusCode::kInvalidArgument);
}

} // namespace
