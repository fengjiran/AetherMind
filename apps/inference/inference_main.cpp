#include "inference_runner.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <limits>
#include <new>
#include <spdlog/sinks/stdout_sinks.h>
#include <spdlog/spdlog.h>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace aethermind;
using namespace aethermind::apps::inference;

constexpr char kUsage[] =
        "Usage: aethermind_infer --model <hf-directory>\n"
        "                        --tokens <comma-separated-uint32>\n"
        "                        --context-tokens <positive-integer>\n"
        "                        --max-new-tokens <nonnegative-integer>\n"
        "                        [--eos-token-id <uint32>]\n"
        "                        [--preset o1-plain|o2-packed]\n"
        "       aethermind_infer --help\n"
        "\n"
        "Generate new Token IDs with CPU FP32 Llama Dense inference.\n"
        "The default preset is o1-plain. Output is one JSON integer array.\n"
        "Options use separate flag/value arguments; duplicates are rejected.\n";

enum class Option {
    kModel,
    kTokens,
    kContextTokens,
    kMaxNewTokens,
    kEosTokenId,
    kPreset,
};

struct FlagSpec {
    std::string_view name;
    Option option;
    bool required;
};

constexpr std::array kFlags{
        FlagSpec{"--model", Option::kModel, true},
        FlagSpec{"--tokens", Option::kTokens, true},
        FlagSpec{"--context-tokens", Option::kContextTokens, true},
        FlagSpec{"--max-new-tokens", Option::kMaxNewTokens, true},
        FlagSpec{"--eos-token-id", Option::kEosTokenId, false},
        FlagSpec{"--preset", Option::kPreset, false},
};

template<std::unsigned_integral T>
StatusOr<T> ParseUnsigned(std::string_view value, std::string_view flag) {
    if (value.empty() || std::ranges::any_of(value, [](char character) {
            return character < '0' || character > '9';
        })) {
        return Status::InvalidArgument(std::string(flag) + " requires an unsigned decimal integer");
    }

    T number = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), number);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
        return Status::InvalidArgument(std::string(flag) + " integer is out of range");
    }
    return number;
}

StatusOr<std::vector<uint32_t>> ParseTokens(std::string_view value) {
    std::vector<uint32_t> tokens;
    size_t begin = 0;
    for (;;) {
        const size_t comma = value.find(',', begin);
        const size_t end = comma == std::string_view::npos ? value.size() : comma;
        AM_ASSIGN_OR_RETURN(const uint32_t token, ParseUnsigned<uint32_t>(value.substr(begin, end - begin), "--tokens"));
        tokens.push_back(token);
        if (comma == std::string_view::npos) {
            return tokens;
        }
        begin = comma + 1;
    }
}

StatusOr<RunOptions> ParseArguments(int argc, char* argv[]) {
    RunOptions options;
    std::array<bool, kFlags.size()> seen{};
    for (int index = 1; index < argc; ++index) {
        const std::string_view flag(argv[index]);
        const auto spec = std::ranges::find_if(kFlags, [flag](const FlagSpec& candidate) {
            return candidate.name == flag;
        });
        if (spec == kFlags.end()) {
            return Status::InvalidArgument("Unknown option: " + std::string(flag));
        }
        const size_t option_index = static_cast<size_t>(spec->option);
        if (seen[option_index]) {
            return Status::InvalidArgument("Duplicate option: " + std::string(flag));
        }
        if (index + 1 >= argc) {
            return Status::InvalidArgument("Missing value for " + std::string(flag));
        }
        const std::string_view value(argv[++index]);
        if (value.empty() || value.starts_with("--")) {
            return Status::InvalidArgument("Missing value for " + std::string(flag));
        }

        switch (spec->option) {
            case Option::kModel:
                options.model_dir = value;
                break;
            case Option::kTokens: {
                AM_ASSIGN_OR_RETURN(auto tokens, ParseTokens(value));
                options.prompt_tokens = std::move(tokens);
                break;
            }
            case Option::kContextTokens: {
                AM_ASSIGN_OR_RETURN(const size_t capacity, ParseUnsigned<size_t>(value, flag));
                if (capacity == 0) {
                    return Status::InvalidArgument("--context-tokens must be positive");
                }
                options.context_tokens = capacity;
                break;
            }
            case Option::kMaxNewTokens: {
                AM_ASSIGN_OR_RETURN(const size_t limit, ParseUnsigned<size_t>(value, flag));
                options.generation.max_new_tokens = limit;
                break;
            }
            case Option::kEosTokenId: {
                AM_ASSIGN_OR_RETURN(const uint32_t eos, ParseUnsigned<uint32_t>(value, flag));
                options.generation.eos_token_id = eos;
                break;
            }
            case Option::kPreset:
                if (value == "o1-plain") {
                    options.preset = CpuInferencePreset::kO1Plain;
                } else if (value == "o2-packed") {
                    options.preset = CpuInferencePreset::kO2Packed;
                } else {
                    return Status::InvalidArgument("--preset must be o1-plain or o2-packed");
                }
                break;
        }
        seen[option_index] = true;
    }

    for (const auto& flag: kFlags) {
        if (flag.required && !seen[static_cast<size_t>(flag.option)]) {
            return Status::InvalidArgument("Missing required option: " + std::string(flag.name));
        }
    }
    return options;
}

void ReportStatus(const Status& status, RunStage stage) noexcept {
    std::fputs("aethermind_infer: ", stderr);
    std::fputs(ToString(stage), stderr);
    std::fputs(": ", stderr);
    const std::string_view code = StatusCodeName(status.code());
    std::fwrite(code.data(), 1, code.size(), stderr);
    const auto& message = status.message();
    if (status.code() == StatusCode::kResourceExhausted && message.empty()) {
        std::fputs(": memory allocation failed\n", stderr);
        return;
    }
    if (!message.empty()) {
        std::fputs(": ", stderr);
        std::fwrite(message.data(), 1, message.size(), stderr);
    }
    std::fputc('\n', stderr);
}

void ReportException(const char* kind, const char* message) noexcept {
    std::fputs("aethermind_infer: ", stderr);
    std::fputs(kind, stderr);
    std::fputs(": ", stderr);
    std::fputs(message, stderr);
    std::fputc('\n', stderr);
}

std::string SerializeTokens(const std::vector<uint32_t>& tokens) {
    std::string output;
    output.push_back('[');
    bool first = true;
    for (const uint32_t token: tokens) {
        if (!first) {
            output.push_back(',');
        }
        first = false;
        // This buffer holds every uint32_t, including its ten-digit maximum.
        std::array<char, std::numeric_limits<uint32_t>::digits10 + 1> digits{};
        const auto result = std::to_chars(digits.data(), digits.data() + digits.size(), token);
        output.append(digits.data(), static_cast<size_t>(result.ptr - digits.data()));
    }
    output += "]\n";
    return output;
}

int RunCommandLine(int argc, char* argv[]) {
    // Help is the sole successful stdout response that is not a generation JSON.
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        if (std::fputs(kUsage, stdout) == EOF || std::fflush(stdout) != 0) {
            std::fputs("aethermind_infer: failed to write stdout\n", stderr);
            return 1;
        }
        return 0;
    }

    auto parsed = ParseArguments(argc, argv);
    if (!parsed.ok()) {
        ReportStatus(parsed.status(), RunStage::kParameters);
        return 2;
    }
    RunStage failure_stage = RunStage::kNone;
    auto generated = RunInference(*parsed, &failure_stage);
    if (!generated.ok()) {
        ReportStatus(generated.status(), failure_stage);
        return failure_stage == RunStage::kParameters ? 2 : 1;
    }

    // Complete serialization before the first write so allocation failures
    // cannot publish a partial result. OS-level write failures can still be partial.
    const std::string output = SerializeTokens(*generated);
    if (std::fwrite(output.data(), 1, output.size(), stdout) != output.size() ||
        std::fflush(stdout) != 0) {
        std::fputs("aethermind_infer: failed to write stdout\n", stderr);
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        auto logger = spdlog::stderr_logger_mt("aethermind_infer");
        spdlog::set_default_logger(std::move(logger));
    } catch (const std::bad_alloc&) {
        std::fputs("aethermind_infer: logger initialization: RESOURCE_EXHAUSTED\n", stderr);
        return 1;
    } catch (...) {
        std::fputs("aethermind_infer: logger initialization failed\n", stderr);
        return 1;
    }

    try {
        return RunCommandLine(argc, argv);
    } catch (const std::bad_alloc&) {
        std::fputs("aethermind_infer: RESOURCE_EXHAUSTED: memory allocation failed\n", stderr);
    } catch (const std::filesystem::filesystem_error& error) {
        ReportException("filesystem error", error.what());
    } catch (const std::exception& error) {
        ReportException("INTERNAL", error.what());
    } catch (...) {
        std::fputs("aethermind_infer: INTERNAL: unknown exception\n", stderr);
    }
    return 1;
}
