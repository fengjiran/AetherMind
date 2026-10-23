#include "aethermind/base/alignment.h"
#include "aethermind/compiler/model_compiler.h"
#include "aethermind/inference/cpu_inference_options.h"
#include "aethermind/inference/executable_model.h"
#include "aethermind/inference/inference_session.h"
#include "aethermind/inference/model_execution_policy.h"
#include "aethermind/model/model_loader.h"
#include "aethermind/runtime/runtime_builder.h"
#include "c_api.h"
#include "c_api_runtime_internal.h"
#include "utils/overflow_check.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace aethermind;

struct RuntimeState {
    Runtime runtime;
    CpuInferencePreset preset;

    RuntimeState(Runtime runtime_value, CpuInferencePreset preset_value) noexcept
        : runtime(std::move(runtime_value)), preset(preset_value) {}
};

struct ModelState {
    // Reverse destruction releases resolved kernels before their backend owner.
    std::shared_ptr<RuntimeState> runtime;
    std::shared_ptr<const ExecutableModel> model;
};

void ClearError(am_error_handle* error) noexcept {
    if (error != nullptr) {
        *error = nullptr;
    }
}

am_status_code ReportError(am_error_handle* error, am_status_code code, const char* message) noexcept {
    if (error != nullptr) {
        *error = am_error_create(code, message);
    }
    return code;
}

am_status_code ReportStatus(am_error_handle* error, const Status& status) noexcept {
    return ReportError(error, ToAMStatusCode(status.code()), status.message().c_str());
}

template<class Config>
StatusOr<Config> CopyConfig(const Config* config, const char* name) {
    if (config == nullptr) {
        return Status::InvalidArgument(std::string(name) + " config must not be null");
    }
    // A rejected short prefix need not be backed by a complete Config object.
    uint32_t prefix[2];
    std::memcpy(prefix, config, sizeof(prefix));
    if (prefix[0] != sizeof(Config)) {
        return Status::InvalidArgument(std::string(name) + " config struct_size mismatch");
    }
    if ((prefix[1] >> 16U) != AM_ABI_VERSION_MAJOR ||
        (prefix[1] & 0xffffU) > AM_ABI_VERSION_MINOR) {
        return Status::Unimplemented(std::string(name) + " config ABI version is unsupported");
    }
    Config value;
    std::memcpy(&value, config, sizeof(value));
    return value;
}

StatusOr<size_t> ToSize(uint64_t value, const char* field, bool positive) {
    if (positive && value == 0) {
        return Status::InvalidArgument(std::string(field) + " must be positive");
    }
    size_t result;
    if (CheckNarrow(value, &result)) {
        return Status::OutOfRange(std::string(field) + " is not representable as size_t");
    }
    return result;
}

struct RuntimeConfiguration {
    RuntimeOptions options;
    CpuInferencePreset preset;
};

StatusOr<RuntimeConfiguration> ReadRuntimeConfiguration(const am_config_t* source) {
    AM_ASSIGN_OR_RETURN(const auto config, CopyConfig(source, "Runtime"));
    if (config.kv_dtype != AM_KV_DTYPE_FLOAT32) {
        return Status::InvalidArgument("Unknown KV dtype; required Float32");
    }
    CpuInferencePreset preset;
    switch (config.cpu_preset) {
        case AM_CPU_PRESET_O1_PLAIN:
            preset = CpuInferencePreset::kO1Plain;
            break;
        case AM_CPU_PRESET_O2_PACKED:
            preset = CpuInferencePreset::kO2Packed;
            break;
        default:
            return Status::InvalidArgument("Unknown CPU preset");
    }
    AM_ASSIGN_OR_RETURN(const size_t layers, ToSize(config.num_layers, "num_layers", true));
    AM_ASSIGN_OR_RETURN(const size_t heads, ToSize(config.num_kv_heads, "num_kv_heads", true));
    AM_ASSIGN_OR_RETURN(const size_t head_dim, ToSize(config.head_dim, "head_dim", true));
    AM_ASSIGN_OR_RETURN(const size_t capacity, ToSize(config.max_tokens, "max_tokens", true));
    // Reject byte/padding overflow before the manager rounds in element units;
    // wrapped geometry must not turn the ABI's Overflow into InvalidArgument.
    constexpr size_t kFp32Bytes = sizeof(uint32_t);
    if (head_dim > std::numeric_limits<size_t>::max() / kFp32Bytes) {
        return Status::Overflow("head_dim FP32 row byte count overflowed size_t");
    }
    auto aligned_row_bytes = AlignOffsetUp(head_dim * kFp32Bytes, 64);
    if (!aligned_row_bytes.ok()) {
        return std::move(aligned_row_bytes).status();
    }
    RuntimeOptions options;
    options.allocator = {.enable_cpu = true, .enable_cuda = false, .enable_cann = false};
    options.backend.enable_cpu = true;
    options.backend.enable_cuda = false;
    options.backend.enable_cann = false;
    options.kv_cache = {.enable_manager = true,
                        .num_layers = layers,
                        .num_kv_heads = heads,
                        .max_tokens = capacity,
                        .head_dim = head_dim,
                        .kv_dtype = DataType::Float32(),
                        .alignment = 64};
    return RuntimeConfiguration{options, preset};
}

StatusOr<GenerationConfig> ReadSessionConfiguration(const am_session_config_t* source) {
    AM_ASSIGN_OR_RETURN(const auto config, CopyConfig(source, "Session"));
    if (config.has_eos_token_id > 1) {
        return Status::InvalidArgument("has_eos_token_id must be zero or one");
    }
    AM_ASSIGN_OR_RETURN(const size_t limit, ToSize(config.max_new_tokens, "max_new_tokens", false));
    if (limit > std::numeric_limits<size_t>::max() / sizeof(uint32_t)) {
        return Status::Overflow("max_new_tokens output byte count overflowed size_t");
    }
    GenerationConfig generation{.max_new_tokens = limit};
    if (config.has_eos_token_id != 0) {
        generation.eos_token_id = config.eos_token_id;
    }
    return generation;
}

template<class Handle, class Function>
Handle CreateBoundary(am_error_handle* error, Function&& function) noexcept {
    ClearError(error);
    try {
        auto result = std::forward<Function>(function)();
        if (!result.ok()) {
            ReportStatus(error, result.status());
            return nullptr;
        }
        return result->release();
    } catch (const std::bad_alloc&) {
        ReportError(error, AM_STATUS_RESOURCE_EXHAUSTED, "");
    } catch (...) {
        ReportError(error, AM_STATUS_INTERNAL, "Unhandled C++ exception");
    }
    return nullptr;
}

} // namespace

struct am_runtime {
    std::shared_ptr<RuntimeState> owner;
};

struct am_model {
    std::shared_ptr<ModelState> owner;
};

struct am_session {
    // The session borrows Runtime and model state kept alive by owner.
    std::shared_ptr<ModelState> owner;
    GenerationConfig generation;
    InferenceSession session;

    am_session(std::shared_ptr<ModelState> owner_value,
               GenerationConfig generation_value,
               InferenceSession session_value) noexcept
        : owner(std::move(owner_value)),
          generation(generation_value),
          session(std::move(session_value)) {}
};

namespace aethermind::c_api::detail {

Status ValidateRuntimeModelKV(const Runtime& runtime, const LoweredModelArtifact& artifact) try {
    const auto* manager = runtime.GetKVCacheManager();
    if (manager == nullptr || !manager->is_initialized()) {
        return Status::FailedPrecondition("Runtime has no initialized KV cache manager");
    }
    if (artifact.loaded_model == nullptr) {
        return Status::InvalidArgument("Compiled artifact has no loaded model");
    }
    const auto& layout = manager->layout();
    AM_ASSIGN_OR_RETURN(const auto expected,
                        MakeFp32KVCacheOptions(artifact.loaded_model->GetConfig(), layout.max_tokens));
    if (layout.num_layers != expected.num_layers) {
        return Status::InvalidArgument("Runtime/model num_layers mismatch");
    }
    if (layout.num_kv_heads != expected.num_kv_heads) {
        return Status::InvalidArgument("Runtime/model num_kv_heads mismatch");
    }
    if (layout.head_dim != expected.head_dim) {
        return Status::InvalidArgument("Runtime/model head_dim mismatch");
    }
    const auto values = artifact.graph.values();
    if (expected.num_layers > values.size() / 2) {
        return Status::InvalidArgument("Compiled model is missing layer K/V state values");
    }
    std::vector<uint8_t> alias_outputs(values.size(), 0);
    for (const auto& alias: artifact.graph.state_aliases()) {
        if (alias.output.index >= values.size()) {
            return Status::InvalidArgument("Compiled KV state alias output is out of range");
        }
        alias_outputs[alias.output.index] = 1;
    }
    std::vector<uint8_t> layer_slots(expected.num_layers, 0);
    for (size_t index = 0; index < values.size(); ++index) {
        const auto* state = std::get_if<StateValue>(&values[index].payload);
        if (state == nullptr) {
            continue;
        }
        const auto* binding = std::get_if<KVCacheStateBinding>(&state->binding);
        if (binding == nullptr) {
            return Status::InvalidArgument("Compiled model has a non-KV state binding");
        }
        const size_t layer = binding->decoder_layer_index;
        if (layer >= expected.num_layers) {
            return Status::InvalidArgument("Compiled KV state layer is out of range");
        }
        if (values[index].spec.dtype != DataType::Float32()) {
            return Status::InvalidArgument("Compiled KV state dtype must be Float32 at layer " +
                                           std::to_string(layer));
        }
        uint8_t slot = 0;
        switch (binding->slot) {
            case KVCacheSlot::kKey:
                slot = 1;
                break;
            case KVCacheSlot::kValue:
                slot = 2;
                break;
            default:
                return Status::InvalidArgument("Compiled KV state slot is invalid");
        }
        if (alias_outputs[index] == 0) {
            if ((layer_slots[layer] & slot) != 0) {
                return Status::InvalidArgument("Compiled model has duplicate KV state roots at layer " +
                                               std::to_string(layer));
            }
            layer_slots[layer] |= slot;
        }
    }
    for (size_t layer = 0; layer < layer_slots.size(); ++layer) {
        if (layer_slots[layer] != 3) {
            return Status::InvalidArgument("Compiled model is missing K/V state at layer " +
                                           std::to_string(layer));
        }
    }
    if (layout.kv_dtype != DataType::Float32()) {
        return Status::InvalidArgument("Runtime/model kv_dtype mismatch; required Float32");
    }
    return Status::Ok();
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

} // namespace aethermind::c_api::detail

uint32_t am_get_abi_version(void) {
    return AM_ABI_VERSION;
}

am_runtime_t am_runtime_create(const am_config_t* config, am_error_handle* error) {
    return CreateBoundary<am_runtime_t>(error, [&]() -> StatusOr<std::unique_ptr<am_runtime>> {
        AM_ASSIGN_OR_RETURN(auto configuration, ReadRuntimeConfiguration(config));
        RuntimeBuilder builder;
        builder.WithOptions(configuration.options);
        AM_ASSIGN_OR_RETURN(auto runtime, builder.TryBuild());
        auto owner = std::make_shared<RuntimeState>(std::move(runtime), configuration.preset);
        auto handle = std::make_unique<am_runtime>();
        handle->owner = std::move(owner);
        return handle;
    });
}

void am_runtime_destroy(am_runtime_t runtime) {
    delete runtime;
}

am_model_t am_model_load(am_runtime_t runtime, const char* path, am_error_handle* error) {
    return CreateBoundary<am_model_t>(error, [&]() -> StatusOr<std::unique_ptr<am_model>> {
        if (runtime == nullptr || path == nullptr || *path == '\0') {
            return Status::InvalidArgument("Model loading requires a Runtime and nonempty path");
        }
        auto runtime_owner = runtime->owner;
        AM_ASSIGN_OR_RETURN(auto loaded, ModelLoader::Load(std::filesystem::path(path)));
        AM_RETURN_IF_ERROR(ValidateFp32ExecutionInputs(*loaded));
        AM_ASSIGN_OR_RETURN(const auto options, MakeCpuCompileOptions(runtime_owner->preset));
        AM_ASSIGN_OR_RETURN(auto artifact, ModelCompiler::Compile(std::move(loaded), options));
        AM_RETURN_IF_ERROR(c_api::detail::ValidateRuntimeModelKV(runtime_owner->runtime, artifact));
        AM_ASSIGN_OR_RETURN(auto prepared, PrepareExecutableModel(runtime_owner->runtime, std::move(artifact)));
        auto executable = std::make_shared<const ExecutableModel>(std::move(prepared));
        auto owner = std::make_shared<ModelState>(ModelState{std::move(runtime_owner), std::move(executable)});
        auto handle = std::make_unique<am_model>();
        handle->owner = std::move(owner);
        return handle;
    });
}

void am_model_unload(am_model_t model) {
    delete model;
}

am_session_t am_session_create(am_model_t model, const am_session_config_t* config,
                               am_error_handle* error) {
    return CreateBoundary<am_session_t>(error, [&]() -> StatusOr<std::unique_ptr<am_session>> {
        if (model == nullptr) {
            return Status::InvalidArgument("Session creation requires a model");
        }
        AM_ASSIGN_OR_RETURN(auto generation, ReadSessionConfiguration(config));
        auto owner = model->owner;
        AM_ASSIGN_OR_RETURN(auto session, InferenceSession::Create(owner->runtime->runtime, owner->model));
        return std::make_unique<am_session>(std::move(owner), std::move(generation), std::move(session));
    });
}

void am_session_destroy(am_session_t session) {
    delete session;
}

am_status_code am_session_generate(
        am_session_t session,
        const uint32_t* input_tokens, size_t input_len,
        uint32_t* output_tokens, size_t output_capacity, size_t* output_len,
        am_error_handle* error) {
    ClearError(error);
    if (output_len != nullptr) {
        *output_len = 0;
    }
    try {
        if (session == nullptr || output_len == nullptr || input_tokens == nullptr || input_len == 0 ||
            (output_tokens == nullptr && output_capacity != 0)) {
            return ReportError(error, AM_STATUS_INVALID_ARGUMENT, "Invalid generation handle, pointer, or length");
        }
        const size_t limit = session->generation.max_new_tokens;
        if (output_capacity < limit) {
            *output_len = limit;
            return ReportError(error, AM_STATUS_OUT_OF_RANGE, "Output capacity is smaller than max_new_tokens");
        }
        auto generated = session->session.Generate(std::span(input_tokens, input_len), session->generation);
        if (!generated.ok()) {
            return ReportStatus(error, generated.status());
        }
        if (generated->size() > limit) {
            return ReportError(error, AM_STATUS_INTERNAL, "Generated token count exceeds configured limit");
        }
        if (!generated->empty()) {
            std::memcpy(output_tokens, generated->data(), generated->size() * sizeof(uint32_t));
        }
        *output_len = generated->size();
        return AM_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return ReportError(error, AM_STATUS_RESOURCE_EXHAUSTED, "");
    } catch (...) {
        return ReportError(error, AM_STATUS_INTERNAL, "Unhandled C++ exception");
    }
}
