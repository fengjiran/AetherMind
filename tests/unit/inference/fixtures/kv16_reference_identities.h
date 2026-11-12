#ifndef AETHERMIND_TEST_KV16_REFERENCE_IDENTITIES_H
#define AETHERMIND_TEST_KV16_REFERENCE_IDENTITIES_H

// Generated from the reviewed SHA-256 manifest by prepare_kv16_reference.py.
// The direct GTest path checks size/FNV to catch absent or changed assets even
// when the separate SHA-256 preparation/verification step was not invoked.
#include <array>
#include <cstdint>
#include <string_view>

namespace aethermind::test::kv16 {
struct FileIdentity {
    std::string_view path;
    uint64_t bytes;
    uint64_t fnv1a64;
};
inline constexpr std::array kFileIdentities{
        FileIdentity{"gqa-tail/wbf16/config.json", 594ULL, 0xa5dc87d009b9e8eeULL},
        FileIdentity{"gqa-tail/wbf16/derivation.json", 1279ULL, 0xd5116412d381d5dfULL},
        FileIdentity{"gqa-tail/wbf16/model.safetensors", 23072ULL, 0xc9ae2915c5e1f139ULL},
        FileIdentity{"gqa-tail/wbf16/oracle-kvbf16.safetensors", 4712ULL, 0x5efb8343d1c21302ULL},
        FileIdentity{"gqa-tail/wbf16/oracle-kvf16.safetensors", 4712ULL, 0x407772d48dfbb156ULL},
        FileIdentity{"gqa-tail/wf16/config.json", 594ULL, 0xa5dc87d009b9e8eeULL},
        FileIdentity{"gqa-tail/wf16/derivation.json", 1278ULL, 0x67a71f60ccc01cbaULL},
        FileIdentity{"gqa-tail/wf16/model.safetensors", 23056ULL, 0x74d3116fec721966ULL},
        FileIdentity{"gqa-tail/wf16/oracle-kvbf16.safetensors", 4712ULL, 0xc1f0c6fd8602dacdULL},
        FileIdentity{"gqa-tail/wf16/oracle-kvf16.safetensors", 4712ULL, 0x841a3568f0e45eeeULL},
        FileIdentity{"mha-near-limit/wbf16/config.json", 726ULL, 0xab971d9f44982610ULL},
        FileIdentity{"mha-near-limit/wbf16/derivation.json", 1279ULL, 0x3c5801210f68e1bfULL},
        FileIdentity{"mha-near-limit/wbf16/model.safetensors", 2066832ULL, 0xe27cbf62cefbd441ULL},
        FileIdentity{"mha-near-limit/wbf16/oracle-kvbf16.safetensors", 1990644ULL, 0x3b40558b36e2185aULL},
        FileIdentity{"mha-near-limit/wbf16/oracle-kvf16.safetensors", 1990644ULL, 0x9f53d54de35eacd6ULL},
        FileIdentity{"mha-near-limit/wf16/config.json", 726ULL, 0xab971d9f44982610ULL},
        FileIdentity{"mha-near-limit/wf16/derivation.json", 1278ULL, 0xc1cfd74a2136a9c9ULL},
        FileIdentity{"mha-near-limit/wf16/model.safetensors", 2066816ULL, 0xa212880af9131b0dULL},
        FileIdentity{"mha-near-limit/wf16/oracle-kvbf16.safetensors", 1990644ULL, 0x4c1c3657968b46c2ULL},
        FileIdentity{"mha-near-limit/wf16/oracle-kvf16.safetensors", 1990644ULL, 0x7e6952a75ff156eaULL},
        FileIdentity{"mha-tied/wbf16/config.json", 725ULL, 0xa389ffe6d243bb03ULL},
        FileIdentity{"mha-tied/wbf16/derivation.json", 1257ULL, 0x6b9e4cc60f104736ULL},
        FileIdentity{"mha-tied/wbf16/model.safetensors", 1042752ULL, 0xdf8d936741d23c93ULL},
        FileIdentity{"mha-tied/wbf16/oracle-kvbf16.safetensors", 644900ULL, 0xc0d1a77b2a7d8ca0ULL},
        FileIdentity{"mha-tied/wbf16/oracle-kvf16.safetensors", 644900ULL, 0x9c8c8600d155fc6bULL},
        FileIdentity{"mha-tied/wf16/config.json", 725ULL, 0xa389ffe6d243bb03ULL},
        FileIdentity{"mha-tied/wf16/derivation.json", 1256ULL, 0x3ef24378bc5edffdULL},
        FileIdentity{"mha-tied/wf16/model.safetensors", 1042736ULL, 0x3624f193d296e5b9ULL},
        FileIdentity{"mha-tied/wf16/oracle-kvbf16.safetensors", 644900ULL, 0x8dcf1dfe416337a2ULL},
        FileIdentity{"mha-tied/wf16/oracle-kvf16.safetensors", 644900ULL, 0xc59b13dcf91b9c71ULL},
        FileIdentity{"mha-untied/wbf16/config.json", 726ULL, 0xab971d9f44982610ULL},
        FileIdentity{"mha-untied/wbf16/derivation.json", 1279ULL, 0x3c5801210f68e1bfULL},
        FileIdentity{"mha-untied/wbf16/model.safetensors", 2066832ULL, 0xe27cbf62cefbd441ULL},
        FileIdentity{"mha-untied/wbf16/oracle-kvbf16.safetensors", 644900ULL, 0xabb9e38372f748d0ULL},
        FileIdentity{"mha-untied/wbf16/oracle-kvf16.safetensors", 644900ULL, 0xba788145349007c6ULL},
        FileIdentity{"mha-untied/wf16/config.json", 726ULL, 0xab971d9f44982610ULL},
        FileIdentity{"mha-untied/wf16/derivation.json", 1278ULL, 0xc1cfd74a2136a9c9ULL},
        FileIdentity{"mha-untied/wf16/model.safetensors", 2066816ULL, 0xa212880af9131b0dULL},
        FileIdentity{"mha-untied/wf16/oracle-kvbf16.safetensors", 644900ULL, 0xae11e19353ce2b53ULL},
        FileIdentity{"mha-untied/wf16/oracle-kvf16.safetensors", 644900ULL, 0xc1afd32ab3f138a4ULL},
        FileIdentity{"mqa-tail/wbf16/config.json", 594ULL, 0x4cdc9b76c6de40b1ULL},
        FileIdentity{"mqa-tail/wbf16/derivation.json", 869ULL, 0x47374f6ddd597d9fULL},
        FileIdentity{"mqa-tail/wbf16/model.safetensors", 7544ULL, 0xe3901febf5641d95ULL},
        FileIdentity{"mqa-tail/wbf16/oracle-kvbf16.safetensors", 2928ULL, 0x7fb3f14d063cab66ULL},
        FileIdentity{"mqa-tail/wbf16/oracle-kvf16.safetensors", 2928ULL, 0xd978dd564449a605ULL},
        FileIdentity{"mqa-tail/wf16/config.json", 594ULL, 0x4cdc9b76c6de40b1ULL},
        FileIdentity{"mqa-tail/wf16/derivation.json", 868ULL, 0x723d3e9b52448a38ULL},
        FileIdentity{"mqa-tail/wf16/model.safetensors", 7536ULL, 0xd79f067110b63194ULL},
        FileIdentity{"mqa-tail/wf16/oracle-kvbf16.safetensors", 2928ULL, 0x3c95bcad2806bad0ULL},
        FileIdentity{"mqa-tail/wf16/oracle-kvf16.safetensors", 2928ULL, 0x06ea989b1c449601ULL},
};
} // namespace aethermind::test::kv16
#endif
