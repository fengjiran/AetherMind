#ifndef AETHERMIND_BACKEND_CPU_CPU_CAPABILITIES_H
#define AETHERMIND_BACKEND_CPU_CPU_CAPABILITIES_H

#include "aethermind/backend/backend_capabilities.h"
#include "aethermind/base/macros.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>

namespace aethermind {

/// CPU architecture family represented by a CpuCapabilities snapshot.
enum class CpuArchitecture : uint8_t {
    kUnknown = 0,
    kX86_64,
    kAArch64,
};

/// Atomic CPU execution capabilities. A feature never implies another
/// feature: kernel descriptors declare every instruction-set requirement.
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

    kCount,
};

/// Fixed-size, allocation-free set of CpuFeature values.
class CpuFeatureSet {
public:
    static constexpr size_t kWordCount = 2;

    constexpr CpuFeatureSet() noexcept = default;

    static CpuFeatureSet From(std::initializer_list<CpuFeature> features) noexcept {
        CpuFeatureSet result;
        for (const CpuFeature feature: features) {
            result.Enable(feature);
        }
        return result;
    }

    constexpr bool Contains(CpuFeature feature) const noexcept {
        const size_t index = static_cast<size_t>(feature);
        return index < static_cast<size_t>(CpuFeature::kCount) &&
               (words_[index / 64U] & (uint64_t{1} << (index % 64U))) != 0;
    }

    constexpr bool ContainsAll(const CpuFeatureSet& required) const noexcept {
        for (size_t index = 0; index < words_.size(); ++index) {
            if ((words_[index] & required.words_[index]) != required.words_[index]) {
                return false;
            }
        }
        return true;
    }

    constexpr bool empty() const noexcept {
        for (const uint64_t word: words_) {
            if (word != 0) {
                return false;
            }
        }
        return true;
    }

    constexpr void Enable(CpuFeature feature) noexcept {
        const size_t index = static_cast<size_t>(feature);
        if (index < static_cast<size_t>(CpuFeature::kCount)) {
            words_[index / 64U] |= uint64_t{1} << (index % 64U);
        }
    }

    constexpr CpuFeatureSet Intersect(const CpuFeatureSet& other) const noexcept {
        CpuFeatureSet result;
        for (size_t index = 0; index < words_.size(); ++index) {
            result.words_[index] = words_[index] & other.words_[index];
        }
        return result;
    }

    constexpr CpuFeatureSet Difference(const CpuFeatureSet& other) const noexcept {
        CpuFeatureSet result;
        for (size_t index = 0; index < words_.size(); ++index) {
            result.words_[index] = words_[index] & ~other.words_[index];
        }
        return result;
    }

    constexpr size_t Hash() const noexcept {
        size_t seed = 0;
        for (const uint64_t word: words_) {
            seed ^= static_cast<size_t>(word) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
        }
        return seed;
    }

    friend constexpr bool operator==(const CpuFeatureSet& lhs,
                                     const CpuFeatureSet& rhs) noexcept = default;

private:
    std::array<uint64_t, kWordCount> words_{};
};

struct CpuKernelRequirements {
    CpuFeatureSet all_of{};

    friend constexpr bool operator==(const CpuKernelRequirements& lhs,
                                     const CpuKernelRequirements& rhs) noexcept = default;
};

/// Runtime policy can only reduce the features reported as usable by the OS.
struct CpuFeaturePolicy {
    CpuFeatureSet disabled_features{};
    CpuFeatureSet required_features{};
};

AM_NODISCARD const char* ToString(CpuArchitecture architecture) noexcept;
AM_NODISCARD const char* ToString(CpuFeature feature) noexcept;
AM_NODISCARD std::string ToString(const CpuFeatureSet& features);

struct CpuCapabilities {
    BackendCapabilities base{
            .device_type = DeviceType::kCPU};
    bool supports_inline_execution = true;

    CpuArchitecture architecture = CpuArchitecture::kUnknown;
    /// Raw hardware capabilities, retained only for diagnostics.
    CpuFeatureSet hardware_features{};
    /// Features that hardware and the current OS/process context permit.
    CpuFeatureSet usable_features{};
    /// Usable features after CpuFeaturePolicy has restricted them. Kernel
    /// selection must use this set exclusively.
    CpuFeatureSet effective_features{};
    /// SVE vector length for the thread that created this snapshot, in bytes.
    /// It is not a kernel-selection feature because it may be thread-local.
    uint32_t sve_vector_bytes = 0;
};

} // namespace aethermind
#endif
