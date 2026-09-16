#ifndef AETHERMIND_BACKEND_CPU_IDENTITY_PACKING_H
#define AETHERMIND_BACKEND_CPU_IDENTITY_PACKING_H

/// @file identity_packing.h
/// @brief CPU identity-packing recipe contract.
///
/// The Phase 1 CPU prepacker copies logical Float32 weights without a layout
/// transformation. Producers attach this recipe to the artifact and packed
/// kernel consumers validate the same recipe before interpreting its storage.
/// Keeping these constants here prevents the two sides from drifting when
/// tiled or blocked CPU packing recipes are added later.

#include <cstddef>
#include <string_view>

namespace aethermind::cpu {

inline constexpr std::string_view kCpuIdentityPackingLayout = "cpu_identity";
inline constexpr size_t kCpuIdentityPackingAlignment = 64;

} // namespace aethermind::cpu

#endif // AETHERMIND_BACKEND_CPU_IDENTITY_PACKING_H
