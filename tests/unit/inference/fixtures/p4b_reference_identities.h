#ifndef AETHERMIND_TEST_P4B_REFERENCE_IDENTITIES_H
#define AETHERMIND_TEST_P4B_REFERENCE_IDENTITIES_H

// Pinned to the P4-B calibration fixture produced by tools/calibrate_p4b_reference.py.
#include <array>
#include <cstdint>
#include <string_view>

namespace aethermind::test::p4b {
struct FileIdentity {
    std::string_view path;
    uint64_t bytes;
    uint64_t fnv1a64;
};
inline constexpr std::array kFileIdentities{
        FileIdentity{"gqa-tail/config.json", 594ULL, 0xa5dc87d009b9e8eeULL},
        FileIdentity{"gqa-tail/model.safetensors", 23056ULL, 0x74d3116fec721966ULL},
        FileIdentity{"gqa-tail/oracle.safetensors", 4712ULL, 0xf7cc29fc47376eadULL},
        FileIdentity{"mha-near-limit/config.json", 726ULL, 0xab971d9f44982610ULL},
        FileIdentity{"mha-near-limit/model.safetensors", 2066816ULL, 0xa212880af9131b0dULL},
        FileIdentity{"mha-near-limit/oracle.safetensors", 1990644ULL, 0x5d40c7cafb2754e2ULL},
        FileIdentity{"mha-tied/config.json", 725ULL, 0xa389ffe6d243bb03ULL},
        FileIdentity{"mha-tied/model.safetensors", 1042736ULL, 0x3624f193d296e5b9ULL},
        FileIdentity{"mha-tied/oracle.safetensors", 644900ULL, 0xb887a77e3add6f7eULL},
        FileIdentity{"mha-untied/config.json", 726ULL, 0xab971d9f44982610ULL},
        FileIdentity{"mha-untied/model.safetensors", 2066816ULL, 0xa212880af9131b0dULL},
        FileIdentity{"mha-untied/oracle.safetensors", 644900ULL, 0x6f5c121733d583b5ULL},
        FileIdentity{"mqa-tail/config.json", 594ULL, 0x4cdc9b76c6de40b1ULL},
        FileIdentity{"mqa-tail/model.safetensors", 7536ULL, 0xd79f067110b63194ULL},
        FileIdentity{"mqa-tail/oracle.safetensors", 2928ULL, 0x9382214d0f172481ULL},
};
} // namespace aethermind::test::p4b
#endif
