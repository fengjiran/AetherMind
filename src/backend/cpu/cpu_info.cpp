#include "aethermind/backend/cpu/cpu_info.h"

#include <array>
#include <cstdint>
#include <string>

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#if defined(__linux__)
#include <asm/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#elif defined(__aarch64__) && defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#include <sys/prctl.h>
#elif defined(__aarch64__) && defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace aethermind {
namespace {

constexpr std::array<const char*, static_cast<size_t>(CpuFeature::kCount)> kCpuFeatureNames = {
        "SSE4.1",
        "AVX",
        "AVX2",
        "FMA",
        "F16C",
        "AVX512F",
        "AVX512BW",
        "AVX512VNNI",
        "AVXVNNI",
        "AVX512BF16",
        "AMX_TILE",
        "AMX_INT8",
        "AMX_BF16",
        "NEON",
        "FP16",
        "DOTPROD",
        "I8MM",
        "BF16",
        "SVE",
        "SVE2",
};

} // namespace

const char* ToString(CpuArchitecture architecture) noexcept {
    switch (architecture) {
        case CpuArchitecture::kX86_64:
            return "x86_64";
        case CpuArchitecture::kAArch64:
            return "AArch64";
        case CpuArchitecture::kUnknown:
        default:
            return "Unknown";
    }
}

const char* ToString(CpuFeature feature) noexcept {
    const auto index = static_cast<size_t>(feature);
    if (index >= kCpuFeatureNames.size()) {
        return "Unknown";
    }
    return kCpuFeatureNames[index];
}

std::string ToString(const CpuFeatureSet& features) {
    std::string result{"{"};
    bool first = true;
    for (size_t index = 0; index < static_cast<size_t>(CpuFeature::kCount); ++index) {
        const auto feature = static_cast<CpuFeature>(index);
        if (!features.Contains(feature)) {
            continue;
        }

        if (!first) {
            result += ", ";
        }
        result += ToString(feature);
        first = false;
    }
    result += '}';
    return result;
}

namespace cpu {
namespace {

#if defined(__x86_64__) || defined(_M_X64)

struct CpuidRegs {
    uint32_t eax{};
    uint32_t ebx{};
    uint32_t ecx{};
    uint32_t edx{};
};

CpuidRegs ReadCpuid(uint32_t leaf, uint32_t subleaf = 0) noexcept {
    CpuidRegs regs;
#if defined(_MSC_VER)
    int data[4]{};
    __cpuidex(data, static_cast<int>(leaf), static_cast<int>(subleaf));
    regs.eax = static_cast<uint32_t>(data[0]);
    regs.ebx = static_cast<uint32_t>(data[1]);
    regs.ecx = static_cast<uint32_t>(data[2]);
    regs.edx = static_cast<uint32_t>(data[3]);
#else
    __cpuid_count(leaf, subleaf, regs.eax, regs.ebx, regs.ecx, regs.edx);
#endif
    return regs;
}

uint64_t ReadXcr0() noexcept {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    uint32_t eax = 0;
    uint32_t edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<uint64_t>(edx) << 32U) | eax;
#endif
}

bool HasBit(uint32_t value, uint32_t bit) noexcept {
    return (value & (uint32_t{1} << bit)) != 0;
}

bool RequestAmxTileDataPermission() noexcept {
#if defined(__linux__) && defined(ARCH_REQ_XCOMP_PERM) && defined(SYS_arch_prctl)
    constexpr unsigned long kXfeatureXtiledata = 18;
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, kXfeatureXtiledata) == 0;
#else
    return false;
#endif
}

CpuCapabilities DetectX86Capabilities() noexcept {
    CpuCapabilities capabilities;
    capabilities.architecture = CpuArchitecture::kX86_64;

    const CpuidRegs leaf0 = ReadCpuid(0);
    const uint32_t max_leaf = leaf0.eax;
    if (max_leaf < 1) {
        return capabilities;
    }

    const CpuidRegs leaf1 = ReadCpuid(1);
    const bool has_osxsave = HasBit(leaf1.ecx, 27);
    const bool has_avx = HasBit(leaf1.ecx, 28);
    const bool has_fma = HasBit(leaf1.ecx, 12);
    const bool has_f16c = HasBit(leaf1.ecx, 29);
    const bool has_avx_state = has_osxsave && (ReadXcr0() & 0x6U) == 0x6U;

    if (HasBit(leaf1.ecx, 19)) {
        capabilities.hardware_features.Enable(CpuFeature::kSse41);
        capabilities.usable_features.Enable(CpuFeature::kSse41);
    }
    if (has_avx) {
        capabilities.hardware_features.Enable(CpuFeature::kAvx);
    }
    if (has_fma) {
        capabilities.hardware_features.Enable(CpuFeature::kFma);
    }
    if (has_f16c) {
        capabilities.hardware_features.Enable(CpuFeature::kF16c);
    }
    if (has_avx && has_avx_state) {
        capabilities.usable_features.Enable(CpuFeature::kAvx);
        if (has_fma) {
            capabilities.usable_features.Enable(CpuFeature::kFma);
        }
        if (has_f16c) {
            capabilities.usable_features.Enable(CpuFeature::kF16c);
        }
    }

    if (max_leaf < 7) {
        return capabilities;
    }

    const CpuidRegs leaf7 = ReadCpuid(7, 0);
    const CpuidRegs leaf7_1 = leaf7.eax >= 1 ? ReadCpuid(7, 1) : CpuidRegs{};
    const bool has_avx2 = HasBit(leaf7.ebx, 5);
    const bool has_avx512f = HasBit(leaf7.ebx, 16);
    const bool has_avx512bw = HasBit(leaf7.ebx, 30);
    const bool has_avx512vnni = HasBit(leaf7.ecx, 11);
    const bool has_avx512bf16 = HasBit(leaf7_1.eax, 5);
    const bool has_avxvnni = HasBit(leaf7_1.eax, 4);
    const bool has_amx_tile = HasBit(leaf7.edx, 24);
    const bool has_amx_int8 = HasBit(leaf7.edx, 25);
    const bool has_amx_bf16 = HasBit(leaf7.edx, 22);

    const auto record_hardware = [&](bool present, CpuFeature feature) {
        if (present) {
            capabilities.hardware_features.Enable(feature);
        }
    };
    record_hardware(has_avx2, CpuFeature::kAvx2);
    record_hardware(has_avx512f, CpuFeature::kAvx512F);
    record_hardware(has_avx512bw, CpuFeature::kAvx512Bw);
    record_hardware(has_avx512vnni, CpuFeature::kAvx512Vnni);
    record_hardware(has_avx512bf16, CpuFeature::kAvx512Bf16);
    record_hardware(has_avxvnni, CpuFeature::kAvxVnni);
    record_hardware(has_amx_tile, CpuFeature::kAmxTile);
    record_hardware(has_amx_int8, CpuFeature::kAmxInt8);
    record_hardware(has_amx_bf16, CpuFeature::kAmxBf16);

    if (has_avx && has_avx_state) {
        if (has_avx2) {
            capabilities.usable_features.Enable(CpuFeature::kAvx2);
        }
        if (has_avxvnni) {
            capabilities.usable_features.Enable(CpuFeature::kAvxVnni);
        }
    }

    // AVX-512 and AMX state require separate XSAVE components. AMX also
    // needs Linux to grant XTILEDATA permission to this execution context.
    const uint64_t xcr0 = has_osxsave ? ReadXcr0() : 0;
    const bool has_avx512_state = (xcr0 & 0xE6U) == 0xE6U;
    if (has_avx && has_avx512_state && has_avx512f) {
        capabilities.usable_features.Enable(CpuFeature::kAvx512F);
        if (has_avx512bw) {
            capabilities.usable_features.Enable(CpuFeature::kAvx512Bw);
        }
        if (has_avx512vnni) {
            capabilities.usable_features.Enable(CpuFeature::kAvx512Vnni);
        }
        if (has_avx512bf16) {
            capabilities.usable_features.Enable(CpuFeature::kAvx512Bf16);
        }
    }

    const bool has_amx_permission = has_amx_tile && RequestAmxTileDataPermission();
    const uint64_t amx_xcr0 = has_osxsave ? ReadXcr0() : 0;
    const bool has_amx_state = (amx_xcr0 & (uint64_t{1} << 17U)) != 0 &&
                               (amx_xcr0 & (uint64_t{1} << 18U)) != 0;
    if (has_amx_permission && has_amx_state) {
        capabilities.usable_features.Enable(CpuFeature::kAmxTile);
        if (has_amx_int8) {
            capabilities.usable_features.Enable(CpuFeature::kAmxInt8);
        }
        if (has_amx_bf16) {
            capabilities.usable_features.Enable(CpuFeature::kAmxBf16);
        }
    }

    return capabilities;
}

#endif

#if defined(__aarch64__)

void RecordAArch64Feature(CpuCapabilities& capabilities,
                          bool present,
                          CpuFeature feature) noexcept {
    if (present) {
        capabilities.hardware_features.Enable(feature);
        capabilities.usable_features.Enable(feature);
    }
}

CpuCapabilities DetectAArch64Capabilities() noexcept {
    CpuCapabilities capabilities;
    capabilities.architecture = CpuArchitecture::kAArch64;

#if defined(__linux__)
    const unsigned long hwcap = getauxval(AT_HWCAP);
#if defined(AT_HWCAP2)
    const unsigned long hwcap2 = getauxval(AT_HWCAP2);
#else
    const unsigned long hwcap2 = 0;
#endif
#if defined(HWCAP_ASIMD)
    RecordAArch64Feature(capabilities, (hwcap & HWCAP_ASIMD) != 0, CpuFeature::kNeon);
#else
    RecordAArch64Feature(capabilities, true, CpuFeature::kNeon);
#endif
#if defined(HWCAP_ASIMDHP)
    RecordAArch64Feature(capabilities, (hwcap & HWCAP_ASIMDHP) != 0, CpuFeature::kFp16);
#endif
#if defined(HWCAP_ASIMDDP)
    RecordAArch64Feature(capabilities, (hwcap & HWCAP_ASIMDDP) != 0, CpuFeature::kDotProd);
#endif
#if defined(HWCAP_I8MM)
    RecordAArch64Feature(capabilities, (hwcap & HWCAP_I8MM) != 0, CpuFeature::kI8mm);
#elif defined(HWCAP2_I8MM)
    RecordAArch64Feature(capabilities, (hwcap2 & HWCAP2_I8MM) != 0, CpuFeature::kI8mm);
#endif
#if defined(HWCAP2_BF16)
    RecordAArch64Feature(capabilities, (hwcap2 & HWCAP2_BF16) != 0, CpuFeature::kBf16);
#endif
#if defined(HWCAP_SVE)
    const bool has_sve = (hwcap & HWCAP_SVE) != 0;
    if (has_sve) {
        capabilities.hardware_features.Enable(CpuFeature::kSve);
#if defined(PR_SVE_GET_VL) && defined(PR_SVE_VL_LEN_MASK)
        const int vector_length = prctl(PR_SVE_GET_VL);
        if (vector_length >= 0) {
            capabilities.usable_features.Enable(CpuFeature::kSve);
            capabilities.sve_vector_bytes = static_cast<uint32_t>(
                    vector_length & PR_SVE_VL_LEN_MASK);
        }
#else
        capabilities.usable_features.Enable(CpuFeature::kSve);
#endif
    }
#endif
#if defined(HWCAP2_SVE2)
    if ((hwcap2 & HWCAP2_SVE2) != 0) {
        capabilities.hardware_features.Enable(CpuFeature::kSve2);
        if (capabilities.usable_features.Contains(CpuFeature::kSve)) {
            capabilities.usable_features.Enable(CpuFeature::kSve2);
        }
    }
#endif
#elif defined(__APPLE__)
    RecordAArch64Feature(capabilities, true, CpuFeature::kNeon);
    const auto has_sysctl_feature = [](const char* name) {
        int value = 0;
        size_t value_size = sizeof(value);
        return sysctlbyname(name, &value, &value_size, nullptr, 0) == 0 && value == 1;
    };
    RecordAArch64Feature(capabilities,
                         has_sysctl_feature("hw.optional.arm.FEAT_FP16"),
                         CpuFeature::kFp16);
    RecordAArch64Feature(capabilities,
                         has_sysctl_feature("hw.optional.arm.FEAT_DotProd"),
                         CpuFeature::kDotProd);
    RecordAArch64Feature(capabilities,
                         has_sysctl_feature("hw.optional.arm.FEAT_I8MM"),
                         CpuFeature::kI8mm);
    RecordAArch64Feature(capabilities,
                         has_sysctl_feature("hw.optional.arm.FEAT_BF16"),
                         CpuFeature::kBf16);
#else
    RecordAArch64Feature(capabilities, true, CpuFeature::kNeon);
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    RecordAArch64Feature(capabilities, true, CpuFeature::kFp16);
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    RecordAArch64Feature(capabilities, true, CpuFeature::kDotProd);
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    RecordAArch64Feature(capabilities, true, CpuFeature::kI8mm);
#endif
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    RecordAArch64Feature(capabilities, true, CpuFeature::kBf16);
#endif
#if defined(__ARM_FEATURE_SVE)
    RecordAArch64Feature(capabilities, true, CpuFeature::kSve);
#endif
#if defined(__ARM_FEATURE_SVE2)
    RecordAArch64Feature(capabilities, true, CpuFeature::kSve2);
#endif
#endif

    return capabilities;
}

#endif

CpuCapabilities DetectUsableCapabilities() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
    return DetectX86Capabilities();
#elif defined(__aarch64__)
    return DetectAArch64Capabilities();
#else
    return {};
#endif
}

} // namespace

StatusOr<CpuCapabilities> ApplyCpuFeaturePolicy(
        CpuCapabilities capabilities,
        const CpuFeaturePolicy& policy) noexcept {
    if (!capabilities.hardware_features.ContainsAll(capabilities.usable_features)) {
        return Status::InvalidArgument(
                "CpuCapabilities usable_features must be a hardware feature subset");
    }

    capabilities.effective_features =
            capabilities.usable_features.Difference(policy.disabled_features);
    if (!capabilities.effective_features.ContainsAll(policy.required_features)) {
        return Status::FailedPrecondition(
                "CPU feature policy requires unsupported effective features: " +
                ToString(policy.required_features.Difference(
                        capabilities.effective_features)));
    }
    return capabilities;
}

StatusOr<CpuCapabilities> DetectCpuCapabilities(
        const CpuFeaturePolicy& policy) noexcept {
    return ApplyCpuFeaturePolicy(DetectUsableCapabilities(), policy);
}

} // namespace cpu
} // namespace aethermind
