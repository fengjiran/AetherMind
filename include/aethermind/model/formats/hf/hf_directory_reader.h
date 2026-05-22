#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_DIRECTORY_READER_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_DIRECTORY_READER_H

#include "aethermind/base/status.h"
#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/model/raw_weight.h"

#include <filesystem>
#include <utility>

namespace aethermind {

/// HF safetensors layout detected from a model directory.
///
/// The reader supports either the canonical single-file layout or the sharded
/// layout described by `model.safetensors.index.json`. Mixed layouts are rejected
/// during inspection so later loading code can rely on one active weight source.
enum class HfDirectoryLayout {
    kUnknown = 0,
    kSingleSafetensors,
    kShardedSafetensors,
};

/// Resolved paths and layout metadata for a Hugging Face model directory.
///
/// The descriptor is produced by `HfDirectoryReader::InspectDirectory` after
/// validating the directory shape. Only the paths relevant to `layout` are set;
/// callers should use `IsSingleFile()` or `IsSharded()` before reading the
/// corresponding safetensors path.
struct HfDirectoryDescriptor {
    HfDirectoryLayout layout = HfDirectoryLayout::kUnknown;
    std::filesystem::path model_dir{};
    std::filesystem::path config_path{};
    std::filesystem::path safetensors_path{};
    std::filesystem::path safetensors_index_path{};

    AM_NODISCARD bool IsSingleFile() const noexcept {
        return layout == HfDirectoryLayout::kSingleSafetensors;
    }

    AM_NODISCARD bool IsSharded() const noexcept {
        return layout == HfDirectoryLayout::kShardedSafetensors;
    }
};

/// Reads Hugging Face model metadata and raw safetensors weights from a directory.
///
/// Instances are immutable after construction and borrow no external resources.
/// File I/O is performed by `ParseConfig()` and `LoadRawWeightTable()`, which
/// return `StatusOr` values instead of throwing for expected validation or format
/// failures.
class HfDirectoryReader {
public:
    /// Opens and validates a Hugging Face model directory.
    ///
    /// @return A reader bound to the inspected directory, or an error if the
    /// directory layout is missing, ambiguous, or invalid.
    AM_NODISCARD static StatusOr<HfDirectoryReader> Open(
            const std::filesystem::path& model_dir);

    /// Inspects a directory without loading model configuration or weights.
    ///
    /// The inspection validates the presence and regular-file status of
    /// `config.json` plus exactly one supported safetensors layout.
    AM_NODISCARD static StatusOr<HfDirectoryDescriptor> InspectDirectory(
            const std::filesystem::path& model_dir);

    AM_NODISCARD const HfDirectoryDescriptor& GetDirDesc() const noexcept {
        return dir_desc_;
    }

    /// Parses `config.json` into the subset of HF configuration used by Phase 1.
    ///
    /// Unknown JSON fields are skipped for forward compatibility with upstream
    /// Hugging Face model configs.
    AM_NODISCARD StatusOr<HfModelConfig> ParseConfig() const;

    /// Loads raw tensor views from the inspected safetensors layout.
    ///
    /// For sharded models, the index must agree with every shard file and every
    /// indexed tensor must be present exactly once.
    AM_NODISCARD StatusOr<RawWeightTable> LoadRawWeightTable() const;

private:
    explicit HfDirectoryReader(HfDirectoryDescriptor dir_desc) noexcept
        : dir_desc_(std::move(dir_desc)) {}

    HfDirectoryDescriptor dir_desc_{};
};

}// namespace aethermind

#endif
