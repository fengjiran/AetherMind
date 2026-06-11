#include "aethermind/model/formats/hf/hf_safetensors_index.h"
#include "aethermind/model/formats/hf/hf_json_reader.h"
#include "aethermind/model/formats/hf/hf_utils.h"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace aethermind {
namespace {

Status ValidateShardFilename(std::string_view filename) {
    if (filename.empty()) {
        return Status::InvalidArgument("Safetensors index shard filename must not be empty");
    }

    if (filename.find("://") != std::string_view::npos) {
        return Status::InvalidArgument("Safetensors index shard filename must not contain URI schemes");
    }

    const std::filesystem::path path(filename);
    if (path.is_absolute() || path.has_root_path()) {
        return Status::InvalidArgument("Safetensors index shard filename must be relative");
    }

    for (const auto& component: path) {
        if (component == "..") {
            return Status::InvalidArgument("Safetensors index shard filename must not contain '..'");
        }
    }
    return Status::Ok();
}

class SafetensorsIndexParser final : public hf::HfJsonReader {
public:
    explicit SafetensorsIndexParser(std::string_view input) noexcept
        : HfJsonReader(input) {}

    struct Result {
        std::unordered_map<std::string, std::string> weight_map{};// NOLINT(readability-identifier-naming)
        std::optional<uint64_t> total_size{};                     // NOLINT(readability-identifier-naming)
    };

    StatusOr<Result> Parse() {
        AM_RETURN_IF_ERROR(Expect('{', "at start of safetensors index"));

        Result result;
        bool has_weight_map = false;

        SkipWhitespace();
        if (!TryConsume('}')) {
            bool has_metadata = false;
            while (true) {
                const auto key = ParseString();
                if (!key.ok()) {
                    return key.status();
                }

                AM_RETURN_IF_ERROR(Expect(':', "after safetensors index key"));
                if (*key == "weight_map") {
                    if (has_weight_map) {
                        return Status::InvalidArgument("Safetensors index contains duplicate weight_map field");
                    }

                    has_weight_map = true;
                    auto parsed_weight_map = ParseWeightMap();
                    if (!parsed_weight_map.ok()) {
                        return parsed_weight_map.status();
                    }
                    result.weight_map = std::move(*parsed_weight_map);
                } else if (*key == "metadata") {
                    if (has_metadata) {
                        return Status::InvalidArgument("Safetensors index contains duplicate metadata field");
                    }

                    has_metadata = true;
                    auto total_size = ParseMetadata();
                    if (!total_size.ok()) {
                        return total_size.status();
                    }
                    result.total_size = *total_size;
                } else {
                    AM_RETURN_IF_ERROR(SkipValue());
                }

                SkipWhitespace();
                if (TryConsume('}')) {
                    break;
                }
                AM_RETURN_IF_ERROR(Expect(',', "between safetensors index fields"));
            }
        }

        SkipWhitespace();
        if (!AtEnd()) {
            return Status::InvalidArgument("Safetensors index contains trailing JSON content");
        }

        if (!has_weight_map) {
            return Status::InvalidArgument("Safetensors index is missing required weight_map field");
        }

        if (result.weight_map.empty()) {
            return Status::InvalidArgument("Safetensors index weight_map must not be empty");
        }
        return result;
    }

private:
    StatusOr<std::optional<uint64_t>> ParseMetadata() {
        AM_RETURN_IF_ERROR(Expect('{', "at start of safetensors metadata"));

        std::optional<uint64_t> total_size;
        SkipWhitespace();
        if (!TryConsume('}')) {
            while (true) {
                const auto key = ParseString();
                if (!key.ok()) {
                    return key.status();
                }

                AM_RETURN_IF_ERROR(Expect(':', "after safetensors metadata key"));
                if (*key == "total_size") {
                    if (total_size.has_value()) {
                        return Status::InvalidArgument("Safetensors index metadata contains duplicate total_size field");
                    }
                    auto parsed_total_size = ParseTotalSize();
                    if (!parsed_total_size.ok()) {
                        return parsed_total_size.status();
                    }
                    total_size = *parsed_total_size;
                } else {
                    AM_RETURN_IF_ERROR(SkipValue());
                }

                SkipWhitespace();
                if (TryConsume('}')) {
                    break;
                }
                AM_RETURN_IF_ERROR(Expect(',', "between safetensors metadata entries"));
            }
        }
        return total_size;
    }

    StatusOr<uint64_t> ParseTotalSize() {
        SkipWhitespace();
        if (!AtEnd() && input_[position_] == '"') {
            const auto value = ParseString();
            if (!value.ok()) {
                return value.status();
            }
            uint64_t parsed = 0;
            const auto [ptr, ec] = std::from_chars(value->data(), value->data() + value->size(), parsed);
            if (ec != std::errc{} || ptr != value->data() + value->size()) {
                return Status::InvalidArgument("Safetensors index metadata total_size string is not a valid unsigned integer");
            }
            return parsed;
        }
        return ParseUInt64();
    }

    StatusOr<std::unordered_map<std::string, std::string>> ParseWeightMap() {
        AM_RETURN_IF_ERROR(Expect('{', "at start of safetensors weight_map"));

        std::unordered_map<std::string, std::string> weight_map;
        SkipWhitespace();
        if (!TryConsume('}')) {
            while (true) {
                const auto tensor_name = ParseString();
                if (!tensor_name.ok()) {
                    return tensor_name.status();
                }

                AM_RETURN_IF_ERROR(Expect(':', "after safetensors weight_map tensor name"));
                const auto shard_filename = ParseString();
                if (!shard_filename.ok()) {
                    return shard_filename.status();
                }

                AM_RETURN_IF_ERROR(ValidateShardFilename(*shard_filename));
                if (!weight_map.emplace(*tensor_name, *shard_filename).second) {
                    return Status::InvalidArgument(
                            "Safetensors index weight_map contains duplicate tensor name: '" + *tensor_name + "'");
                }

                SkipWhitespace();
                if (TryConsume('}')) {
                    break;
                }
                AM_RETURN_IF_ERROR(Expect(',', "between safetensors weight_map entries"));
            }
        }
        return weight_map;
    }
};

}// namespace

StatusOr<HfSafetensorsIndex> HfSafetensorsIndex::Load(const std::filesystem::path& index_path) {
    auto text = hf::ReadFileText(index_path);
    if (!text.ok()) {
        return text.status();
    }

    auto parsed = SafetensorsIndexParser(*text).Parse();
    if (!parsed.ok()) {
        return parsed.status().WithMessage(
                hf::FormatPathMessage(parsed.status().message(), index_path));
    }
    return HfSafetensorsIndex(std::move(parsed->weight_map), parsed->total_size);
}

std::vector<std::string> HfSafetensorsIndex::UniqueShardFilenames() const {
    std::unordered_set<std::string> seen;
    std::vector<std::string> shards;
    shards.reserve(weight_map_.size());
    for (const auto& shard: weight_map_ | std::views::values) {
        if (seen.insert(shard).second) {
            shards.push_back(shard);
        }
    }
    std::ranges::sort(shards);
    return shards;
}

}// namespace aethermind
