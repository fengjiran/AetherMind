#include "aethermind/backend/cpu/cpu_info.h"

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#elif defined(__aarch64__) && defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#elif defined(__aarch64__) && defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace aethermind {
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

#endif

CpuFeatures DetectPhysicalFeatures() noexcept {
    CpuFeatures features;

#if defined(__x86_64__) || defined(_M_X64)
    const CpuidRegs leaf0 = ReadCpuid(0);
    const uint32_t max_leaf = leaf0.eax;

    if (max_leaf >= 1) {
        const CpuidRegs leaf1 = ReadCpuid(1);
        features.has_sse4_1 = HasBit(leaf1.ecx, 19);

        const bool has_osxsave = HasBit(leaf1.ecx, 27);
        const bool has_avx = HasBit(leaf1.ecx, 28);
        const uint64_t xcr0 = has_osxsave ? ReadXcr0() : 0;
        const bool has_avx_state = (xcr0 & 0x6U) == 0x6U;
        const bool has_avx512_state = (xcr0 & 0xE6U) == 0xE6U;
        const bool has_amx_state = (xcr0 & (uint64_t{1} << 17U)) != 0 &&
                                   (xcr0 & (uint64_t{1} << 18U)) != 0;

        if (max_leaf >= 7) {
            const CpuidRegs leaf7 = ReadCpuid(7, 0);
            const CpuidRegs leaf7_subleaf1 = leaf7.eax >= 1 ? ReadCpuid(7, 1) : CpuidRegs{};

            features.has_avx2 = has_avx && has_avx_state && HasBit(leaf7.ebx, 5);
            features.has_avx512f = has_avx && has_avx512_state && HasBit(leaf7.ebx, 16);
            features.has_vnni = (features.has_avx512f && HasBit(leaf7.ecx, 11)) ||
                                (has_avx && has_avx_state && HasBit(leaf7_subleaf1.eax, 4));
            features.has_amx = has_osxsave && has_amx_state && HasBit(leaf7.edx, 24) && HasBit(leaf7.edx, 25);
        }
    }
#elif defined(__aarch64__)
#if defined(__linux__)
    const unsigned long hwcap = getauxval(AT_HWCAP);
#if defined(HWCAP_ASIMD)
    features.has_neon = (hwcap & HWCAP_ASIMD) != 0;
#else
    features.has_neon = true;
#endif
#if defined(HWCAP_SVE)
    features.has_sve = (hwcap & HWCAP_SVE) != 0;
#endif
#if defined(HWCAP_ASIMDDP)
    features.has_dotprod = (hwcap & HWCAP_ASIMDDP) != 0;
#endif
#elif defined(__APPLE__)
    features.has_neon = true;

    int dotprod = 0;
    size_t dotprod_size = sizeof(dotprod);
    if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &dotprod, &dotprod_size, nullptr, 0) == 0) {
        features.has_dotprod = dotprod == 1;
    }
#else
    features.has_neon = true;
#if defined(__ARM_FEATURE_SVE)
    features.has_sve = true;
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    features.has_dotprod = true;
#endif
#endif
#endif

    return features;
}

}// namespace

const CpuFeatures& GetCpuFeatures() noexcept {
    static const CpuFeatures features = DetectPhysicalFeatures();
    return features;
}

}// namespace cpu
}// namespace aethermind
