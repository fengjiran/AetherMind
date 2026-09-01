#ifndef AETHERMIND_BACKEND_CPU_CPU_CAPABILITIES_H
#define AETHERMIND_BACKEND_CPU_CPU_CAPABILITIES_H

/// @file cpu_capabilities.h
/// @brief CPU architecture and feature capability model.
///
/// Defines `CpuArchitecture`, `CpuFeature`, the allocation-free
/// `CpuFeatureSet` bitset, kernel requirements, feature policy, and the
/// immutable `CpuCapabilities` snapshot used by CPU kernel selection.
#include "aethermind/base/macros.h"
#include "utils/hash.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>

namespace aethermind {

/// @brief CPU architecture family represented by a `CpuCapabilities` snapshot.
enum class CpuArchitecture : uint8_t {
    kUnknown = 0,
    kX86_64,
    kAArch64
};

/// @brief Atomic CPU execution capabilities.
///
/// A feature never implies another feature: kernel descriptors declare every
/// instruction-set requirement explicitly. Features are partitioned by
/// architecture; a single snapshot never mixes x86-64 and AArch64 features.
enum class CpuFeature : uint8_t {
    // x86-64
    kSse41 = 0,
    kAvx,
    kAvx2,
    kFma,
    kF16c,
    kAvx512F,
    kAvx512Bw,
    kAvx512Vnni,
    kAvxVnni,
    kAvx512Bf16,
    kAmxTile,
    kAmxInt8,
    kAmxBf16,

    // AArch64
    kNeon,
    kFp16,
    kDotProd,
    kI8mm,
    kBf16,
    kSve,
    kSve2,

    kCount
};

/// @brief Fixed-size, allocation-free set of `CpuFeature` values.
///
/// Stores features as a bitset over two 64-bit words, avoiding heap
/// allocation. All operations are `noexcept` and `constexpr` where possible.
/// The set is value-typed and freely copyable.
class CpuFeatureSet {
public:
    static constexpr size_t kWordCount = 2;

    constexpr CpuFeatureSet() noexcept = default;

    /// @brief Creates a set from an initializer list.
    ///
    /// @param features Features to include.
    /// @return Set containing exactly the listed features.
    static constexpr CpuFeatureSet From(std::initializer_list<CpuFeature> features) noexcept {
        CpuFeatureSet result;
        for (const CpuFeature feature: features) {
            result.Enable(feature);
        }
        return result;
    }

    /// @brief Checks whether the set contains a feature.
    ///
    /// @param feature Feature to test.
    /// @return True if `feature` is present, false otherwise.
    AM_NODISCARD constexpr bool Contains(CpuFeature feature) const noexcept {
        const auto index = static_cast<size_t>(feature);
        return index < static_cast<size_t>(CpuFeature::kCount) &&
               (words_[index / 64U] & (uint64_t{1} << (index % 64U))) != 0;
    }

    /// @brief Checks whether the set contains all features of another set.
    ///
    /// @param required Feature set that must be fully contained.
    /// @return True if every feature in `required` is present.
    AM_NODISCARD constexpr bool ContainsAll(const CpuFeatureSet& required) const noexcept {
        for (size_t index = 0; index < words_.size(); ++index) {
            if ((words_[index] & required.words_[index]) != required.words_[index]) {
                return false;
            }
        }
        return true;
    }

    /// @brief Checks whether the set is empty.
    ///
    /// @return True if no feature is present.
    AM_NODISCARD constexpr bool empty() const noexcept {
        return std::ranges::all_of(words_, [](uint64_t word) { return word == 0; });
    }

    /// @brief Enables a feature in the set.
    ///
    /// @param feature Feature to add. Values outside `CpuFeature::kCount`
    ///        are ignored.
    constexpr void Enable(CpuFeature feature) noexcept {
        if (const auto index = static_cast<size_t>(feature);
            index < static_cast<size_t>(CpuFeature::kCount)) {
            words_[index / 64U] |= uint64_t{1} << (index % 64U);
        }
    }

    /// @brief Returns the intersection of two feature sets.
    ///
    /// @param other Other set to intersect with.
    /// @return Set containing features present in both operands.
    AM_NODISCARD constexpr CpuFeatureSet Intersect(const CpuFeatureSet& other) const noexcept {
        CpuFeatureSet result;
        for (size_t index = 0; index < words_.size(); ++index) {
            result.words_[index] = words_[index] & other.words_[index];
        }
        return result;
    }

    /// @brief Returns the set difference.
    ///
    /// @param other Set whose features are removed from this set.
    /// @return Set containing features present in this set but not in `other`.
    AM_NODISCARD constexpr CpuFeatureSet Difference(const CpuFeatureSet& other) const noexcept {
        CpuFeatureSet result;
        for (size_t index = 0; index < words_.size(); ++index) {
            result.words_[index] = words_[index] & ~other.words_[index];
        }
        return result;
    }

    /// @brief Computes a hash for the feature set.
    ///
    /// @return Size-based hash suitable for unordered containers.
    AM_NODISCARD constexpr size_t Hash() const noexcept {
        size_t seed = 0;
        for (const uint64_t word: words_) {
            seed = hash_combine(seed, static_cast<size_t>(word));
        }
        return seed;
    }

    friend constexpr bool operator==(const CpuFeatureSet& lhs,
                                     const CpuFeatureSet& rhs) noexcept = default;

private:
    std::array<uint64_t, kWordCount> words_{};
};

/// @brief Runtime policy that can only reduce usable features.
///
/// `disabled_features` are removed from the usable set; `required_features`
/// must remain in the resulting effective set, otherwise policy application
/// fails. A policy never enables a feature absent from the usable set.
struct CpuFeaturePolicy {
    CpuFeatureSet disabled_features{};
    CpuFeatureSet required_features{};
};

/// @brief Returns the string representation of a CPU architecture.
///
/// @param architecture Architecture to stringify.
/// @return Human-readable name; "Unknown" for unrecognized values.
AM_NODISCARD const char* ToString(CpuArchitecture architecture) noexcept;

/// @brief Returns the string representation of a CPU feature.
///
/// @param feature Feature to stringify.
/// @return Human-readable name; "Unknown" for out-of-range values.
AM_NODISCARD const char* ToString(CpuFeature feature) noexcept;

/// @brief Returns the string representation of a feature set.
///
/// @param features Feature set to stringify as `{F1, F2}`.
/// @return Formatted set string; "{}" for an empty set.
AM_NODISCARD std::string ToString(const CpuFeatureSet& features);

/// @brief Immutable snapshot of CPU capabilities for the current process.
///
/// Holds the architecture, raw hardware features (diagnostics only), usable
/// features permitted by hardware and OS, effective features after policy,
/// and thread-local SVE vector length. The snapshot is value-typed and
/// produced by `DetectCpuCapabilities`; kernel selection must use
/// `effective_features` exclusively.
struct CpuCapabilities {
    CpuArchitecture architecture = CpuArchitecture::kUnknown;
    /// Raw hardware capabilities, retained only for diagnostics.
    CpuFeatureSet hardware_features{};
    /// Features that hardware and the current OS/process context permit.
    CpuFeatureSet usable_features{};
    /// Usable features after `CpuFeaturePolicy` has restricted them. Kernel
    /// selection must use this set exclusively.
    CpuFeatureSet effective_features{};
    /// SVE vector length for the thread that created this snapshot, in bytes.
    /// It is not a kernel-selection feature because it may be thread-local.
    uint32_t sve_vector_bytes = 0;
};

} // namespace aethermind
#endif
