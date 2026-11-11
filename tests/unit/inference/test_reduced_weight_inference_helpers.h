#ifndef AETHERMIND_TEST_REDUCED_WEIGHT_INFERENCE_HELPERS_H
#define AETHERMIND_TEST_REDUCED_WEIGHT_INFERENCE_HELPERS_H

#include "aethermind/backend/backend_factory.h"
#include "inference/test_hf_fixed_trajectory_helpers.h"
#include "inference/test_malloc_interposer.h"
#include "model/test_utils.h"
#include "test_utils/test_resource_allocator.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace aethermind::test::reduced {
namespace numerical = aethermind::test::numerical;

struct CompatibilityCase {
    bool bfloat16;
    bool tied;
    int64_t kv_heads;
    const char* name;
    const char* source_hint = nullptr;
    DataType norm_dtype{};
};

struct CheckpointTensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<uint16_t> bits;
    bool norm = false;
    DataType dtype{};
    bool bfloat16_encoding = false;
};

struct CheckpointPair {
    std::filesystem::path source;
    std::filesystem::path reference;
    size_t source_payload_bytes = 0;
    size_t fp32_payload_bytes = 0;
    size_t matrix_payload_bytes = 0;
    size_t reduced_norm_source_bytes = 0;
    size_t converted_norm_bytes = 0;
};

// This oracle evaluates the format's value as a double, independently of the
// production integer-bit widening and Half/BFloat16 conversion helpers.
inline float DecodeCheckpointValue(uint16_t bits, bool bfloat16) {
    const int fraction_bits = bfloat16 ? 7 : 10;
    const int bias = bfloat16 ? 127 : 15;
    const unsigned exponent_mask = bfloat16 ? 0xFFU : 0x1FU;
    const unsigned fraction_mask = (1U << fraction_bits) - 1U;
    const unsigned exponent = (bits >> fraction_bits) & exponent_mask;
    const unsigned fraction = bits & fraction_mask;
    const double magnitude = exponent == 0
                                     ? std::ldexp(static_cast<double>(fraction), 1 - bias - fraction_bits)
                                     : std::ldexp(static_cast<double>((1U << fraction_bits) + fraction),
                                                  static_cast<int>(exponent) - bias - fraction_bits);
    return static_cast<float>((bits & 0x8000U) != 0 ? -magnitude : magnitude);
}

inline std::vector<CheckpointTensor> MakeCheckpointTensors(const CompatibilityCase& test_case) {
    constexpr int64_t kHidden = 8;
    constexpr int64_t kIntermediate = 12;
    constexpr int64_t kVocabulary = 32;
    constexpr int64_t kHeadDimension = 2;
    const int64_t kv_hidden = test_case.kv_heads * kHeadDimension;
    std::vector<CheckpointTensor> tensors;
    const auto append = [&](std::string name, std::vector<int64_t> shape, bool norm) {
        size_t count = 1;
        for (const int64_t dimension: shape) {
            count *= static_cast<size_t>(dimension);
        }
        const DataType dtype = norm && !test_case.norm_dtype.IsUndefined()
                                       ? test_case.norm_dtype
                                       : (test_case.bfloat16 ? DataType::BFloat(16) : DataType::Float(16));
        const bool bfloat16_encoding = dtype == DataType::BFloat(16) ||
                                       (dtype == DataType::Float32() && test_case.bfloat16);
        CheckpointTensor tensor{.name = std::move(name),
                                .shape = std::move(shape),
                                .bits = std::vector<uint16_t>(count),
                                .norm = norm,
                                .dtype = dtype,
                                .bfloat16_encoding = bfloat16_encoding};
        constexpr std::array<uint16_t, 11> kHalfValues{
                0x2800, 0x2C00, 0x3000, 0x3200, 0xA800, 0xAC00, 0xB000, 0xB200, 0x0000, 0x0001, 0x2E00};
        constexpr std::array<uint16_t, 11> kBfloatValues{
                0x3D00, 0x3D80, 0x3E00, 0x3E40, 0xBD00, 0xBD80, 0xBE00, 0xBE40, 0x0000, 0x0001, 0x3DC0};
        constexpr std::array<uint16_t, 3> kHalfNorm{0x3B00, 0x3C00, 0x3C80};
        constexpr std::array<uint16_t, 3> kBfloatNorm{0x3F60, 0x3F80, 0x3F90};
        const auto& values = bfloat16_encoding ? kBfloatValues : kHalfValues;
        const auto& norms = bfloat16_encoding ? kBfloatNorm : kHalfNorm;
        for (size_t index = 0; index < count; ++index) {
            tensor.bits[index] = norm ? norms[(index + tensors.size()) % norms.size()]
                                      : values[(index * 7 + index / 3 + tensors.size() * 5) % values.size()];
        }
        tensors.push_back(std::move(tensor));
    };
    append("model.embed_tokens.weight", {kVocabulary, kHidden}, false);
    append("model.norm.weight", {kHidden}, true);
    if (!test_case.tied) {
        append("lm_head.weight", {kVocabulary, kHidden}, false);
    }
    for (int64_t layer = 0; layer < 2; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer);
        append(prefix + ".input_layernorm.weight", {kHidden}, true);
        append(prefix + ".post_attention_layernorm.weight", {kHidden}, true);
        append(prefix + ".self_attn.q_proj.weight", {kHidden, kHidden}, false);
        append(prefix + ".self_attn.k_proj.weight", {kv_hidden, kHidden}, false);
        append(prefix + ".self_attn.v_proj.weight", {kv_hidden, kHidden}, false);
        append(prefix + ".self_attn.o_proj.weight", {kHidden, kHidden}, false);
        append(prefix + ".mlp.gate_proj.weight", {kIntermediate, kHidden}, false);
        append(prefix + ".mlp.up_proj.weight", {kIntermediate, kHidden}, false);
        append(prefix + ".mlp.down_proj.weight", {kHidden, kIntermediate}, false);
    }
    return tensors;
}

inline void WriteCheckpoint(const std::filesystem::path& directory,
                            const CompatibilityCase& test_case,
                            const std::vector<CheckpointTensor>& tensors, bool reference) {
    std::filesystem::create_directories(directory);
    const std::string dtype = reference ? "float32" : (test_case.bfloat16 ? "bfloat16" : "float16");
    const std::string hint = reference || test_case.source_hint == nullptr ? dtype : test_case.source_hint;
    WriteTextFile(directory / "config.json",
                  "{\"model_type\":\"llama\",\"architectures\":[\"LlamaForCausalLM\"],"
                  "\"hidden_size\":8,\"intermediate_size\":12,\"num_hidden_layers\":2,"
                  "\"num_attention_heads\":4,\"num_key_value_heads\":" +
                          std::to_string(test_case.kv_heads) +
                          ",\"vocab_size\":32,\"max_position_embeddings\":16,\"head_dim\":2,"
                          "\"rms_norm_eps\":0.00001,\"hidden_act\":\"silu\",\"tie_word_embeddings\":" +
                          (test_case.tied ? "true" : "false") +
                          (hint.empty() ? "}" : ",\"torch_dtype\":\"" + hint + "\"}"));
    std::string header = "{";
    std::vector<std::byte> payload;
    for (const auto& tensor: tensors) {
        if (header.size() != 1) {
            header += ',';
        }
        const size_t begin = payload.size();
        const bool fp32 = reference || tensor.dtype == DataType::Float32();
        const size_t element_bytes = fp32 ? sizeof(float) : sizeof(uint16_t);
        payload.resize(begin + tensor.bits.size() * element_bytes);
        for (size_t index = 0; index < tensor.bits.size(); ++index) {
            if (fp32) {
                const float value = DecodeCheckpointValue(tensor.bits[index], tensor.bfloat16_encoding);
                const uint32_t bits = std::bit_cast<uint32_t>(value);
                for (size_t byte = 0; byte < sizeof(bits); ++byte) {
                    payload[begin + index * element_bytes + byte] = static_cast<std::byte>((bits >> (byte * 8U)) & 0xFFU);
                }
            } else {
                const uint16_t bits = tensor.bits[index];
                payload[begin + index * element_bytes] = static_cast<std::byte>(bits & 0xFFU);
                payload[begin + index * element_bytes + 1] = static_cast<std::byte>(bits >> 8U);
            }
        }
        header += '"' + tensor.name + "\":{\"dtype\":\"" +
                  (fp32 ? "F32" : (tensor.dtype == DataType::BFloat(16) ? "BF16" : "F16")) + "\",\"shape\":[";
        for (size_t index = 0; index < tensor.shape.size(); ++index) {
            if (index != 0) {
                header += ',';
            }
            header += std::to_string(tensor.shape[index]);
        }
        header += "],\"data_offsets\":[" + std::to_string(begin) + ',' + std::to_string(payload.size()) + "]}";
    }
    header += '}';
    while ((sizeof(uint64_t) + header.size()) % 64 != 0) {
        header += ' ';
    }
    WriteSafetensorsFile(directory / "model.safetensors", header, payload);
}

inline CheckpointPair WriteCheckpointPair(const std::filesystem::path& root, const CompatibilityCase& test_case) {
    CheckpointPair pair{.source = root / "source", .reference = root / "reference"};
    const auto tensors = MakeCheckpointTensors(test_case);
    for (const auto& tensor: tensors) {
        const size_t source_bytes = tensor.bits.size() * static_cast<size_t>(tensor.dtype.nbytes());
        pair.source_payload_bytes += source_bytes;
        pair.fp32_payload_bytes += tensor.bits.size() * sizeof(float);
        if (!tensor.norm) {
            pair.matrix_payload_bytes += source_bytes;
        } else if (tensor.dtype != DataType::Float32()) {
            pair.reduced_norm_source_bytes += source_bytes;
            pair.converted_norm_bytes += tensor.bits.size() * sizeof(float);
        }
    }
    WriteCheckpoint(pair.source, test_case, tensors, false);
    WriteCheckpoint(pair.reference, test_case, tensors, true);
    return pair;
}

inline Runtime MakeCompatibilityRuntime(int64_t kv_heads,
                                        const std::shared_ptr<test::ResourceAllocationStats>& stats = {},
                                        std::unique_ptr<BackendFactory> factory = {}) {
    RuntimeOptions options;
    options.kv_cache = {.enable_manager = true,
                        .num_layers = 2,
                        .num_kv_heads = static_cast<size_t>(kv_heads),
                        .max_tokens = 8,
                        .head_dim = 2,
                        .kv_dtype = DataType::Float32(),
                        .alignment = 64};
    RuntimeBuilder builder;
    builder.WithOptions(options);
    if (factory != nullptr) {
        builder.RegisterBackendFactory(DeviceType::kCPU, std::move(factory));
    }
    if (stats != nullptr) {
        builder.RegisterCustomAllocatorProvider(DeviceType::kCPU, std::make_unique<test::ResourceRecordingProvider>(stats));
    }
    return builder.Build();
}

struct TrajectoryOptions {
    std::array<int64_t, 2> decode_inputs{17, 23};
    bool greedy_decode_inputs = false;
    std::array<MallocCallCounts, 2>* decode_allocations = nullptr;
};

inline StatusOr<numerical::CapturedTrajectory> ObserveCompatibilityTrajectory(Runtime& runtime,
                                                                              const ExecutableModel& model,
                                                                              const TrajectoryOptions& options = {}) {
    constexpr std::array<int64_t, 3> kPrompt{1, 7, 3};
    const auto& decode_inputs = options.decode_inputs;
    constexpr std::array<int64_t, 3> kPrefillRows{0, 1, 2};
    constexpr std::array<int64_t, 1> kDecodeRows{0};
    AM_ASSIGN_OR_RETURN(const auto prefill, model.plan(ExecPhase::kPrefill));
    AM_ASSIGN_OR_RETURN(const auto decode, model.plan(ExecPhase::kDecode));
    AM_ASSIGN_OR_RETURN(const auto prefill_weights, model.immutable_weight_bindings(ExecPhase::kPrefill));
    AM_ASSIGN_OR_RETURN(const auto decode_weights, model.immutable_weight_bindings(ExecPhase::kDecode));
    auto* manager = runtime.GetKVCacheManager();
    if (manager == nullptr) {
        return Status::Internal("Compatibility trajectory requires a KV manager");
    }
    AM_ASSIGN_OR_RETURN(auto view, manager->ReserveForSession(kPrompt.size(), decode_inputs.size()));
    numerical::SessionReservation reservation(*manager, view);
    const size_t workspace_bytes = std::max(prefill->total_workspace_bytes(), decode->total_workspace_bytes());
    const size_t alignment = std::max(prefill->workspace_alignment(), decode->workspace_alignment());
    Buffer workspace_storage;
    if (workspace_bytes != 0) {
        AM_ASSIGN_OR_RETURN(auto allocator, runtime.TryGetAllocator(Device::CPU()));
        AM_ASSIGN_OR_RETURN(workspace_storage, allocator->TryAllocate({.nbytes = workspace_bytes, .alignment = alignment}));
    }
    CpuWorkspaceArena workspace(workspace_bytes == 0 ? nullptr : workspace_storage.mutable_data(), workspace_bytes);
    WorkspaceArena* const workspace_ptr = workspace_bytes == 0 ? nullptr : &workspace;
    numerical::CapturedTrajectory snapshots;
    test::TestBuffer prompt(DataType::Int(64), {static_cast<int64_t>(kPrompt.size())});
    test::TestBuffer positions(DataType::Int(64), {static_cast<int64_t>(kPrompt.size())});
    std::memcpy(prompt.mutable_data(), kPrompt.data(), sizeof(kPrompt));
    for (size_t index = 0; index < kPrompt.size(); ++index) {
        static_cast<int64_t*>(positions.mutable_data())[index] = static_cast<int64_t>(index);
    }
    {
        AM_ASSIGN_OR_RETURN(auto context, numerical::PrepareContext(runtime, *prefill, *prefill_weights, prompt, positions, workspace_ptr, view));
        AM_RETURN_IF_ERROR(Executor::Execute(*prefill, context));
        AM_ASSIGN_OR_RETURN(snapshots[0], numerical::CaptureStage(context, *prefill, kPrefillRows, kPrompt.size(), view));
    }
    test::TestBuffer token(DataType::Int(64), {1});
    test::TestBuffer position(DataType::Int(64), {1});
    static_cast<int64_t*>(token.mutable_data())[0] = decode_inputs[0];
    static_cast<int64_t*>(position.mutable_data())[0] = static_cast<int64_t>(kPrompt.size());
    {
        AM_ASSIGN_OR_RETURN(auto context, numerical::PrepareContext(runtime, *decode, *decode_weights, token, position, workspace_ptr, view));
        for (size_t index = 0; index < decode_inputs.size(); ++index) {
            static_cast<int64_t*>(token.mutable_data())[0] = options.greedy_decode_inputs
                                                                     ? snapshots[index].tokens.back()
                                                                     : decode_inputs[index];
            static_cast<int64_t*>(position.mutable_data())[0] = static_cast<int64_t>(kPrompt.size() + index);
            workspace.Reset();
            const bool count_allocations = options.decode_allocations != nullptr && MallocInterposerAvailable();
            if (count_allocations) {
                BeginMallocCallCounting();
            }
            const Status status = Executor::Execute(*decode, context);
            if (count_allocations) {
                (*options.decode_allocations)[index] = EndMallocCallCounting();
            }
            AM_RETURN_IF_ERROR(status);
            AM_ASSIGN_OR_RETURN(snapshots[index + 1], numerical::CaptureStage(context, *decode, kDecodeRows, 1, view));
        }
    }
    AM_RETURN_IF_ERROR(reservation.Release());
    return snapshots;
}

} // namespace aethermind::test::reduced

#endif
