#ifndef AETHERMIND_CONTAINER_DETAIL_FIBONACCI_HASH_H
#define AETHERMIND_CONTAINER_DETAIL_FIBONACCI_HASH_H

#include <cstddef>
#include <cstdint>

namespace aethermind::details {

/// @brief Maps a hash to the high bits of a 64-bit Fibonacci product.
/// @param hash_value Hash to mix.
/// @param fib_shift Number of low product bits to discard.
/// @return The bucket index; shifts of 64 or more select the single bucket zero.
/// @note The product remains 64-bit on platforms with a 32-bit size_t.
constexpr size_t FibonacciHash(size_t hash_value, uint32_t fib_shift) noexcept {
    if (fib_shift >= 64) {
        return 0;
    }
    constexpr uint64_t kCoefficient = 11400714819323198485ULL;
    return static_cast<size_t>((kCoefficient * static_cast<uint64_t>(hash_value)) >> fib_shift);
}

} // namespace aethermind::details

#endif // AETHERMIND_CONTAINER_DETAIL_FIBONACCI_HASH_H
