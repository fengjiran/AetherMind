#ifndef AETHERMIND_CPU_ALLOCATOR_H
#define AETHERMIND_CPU_ALLOCATOR_H

#include "aethermind/base/status.h"
#include "aethermind/memory/buffer.h"
#include "allocator.h"

#include <cstddef>

namespace aethermind {

/// @brief Allocates aligned CPU storage, returning null for zero bytes.
/// @param nbytes Number of bytes to allocate.
/// @return Storage that must be released with free_cpu, or null for zero bytes.
/// @throws std::bad_alloc if the system cannot allocate the requested storage.
void* alloc_cpu(size_t nbytes);

void free_cpu(void* data);

/// @brief Allocates an owning CPU Buffer with an explicit alignment contract.
///
/// Unlike CPUAllocator::Allocate, this helper does not select a THP policy or
/// initialize the allocation. It reports allocation and Buffer metadata
/// failures through Status rather than exceptions.
///
/// @param nbytes Visible byte count. Zero creates a valid zero-byte Buffer.
/// @param alignment Required non-zero power-of-two alignment that is also a
///        multiple of sizeof(void*).
/// @return An owning Buffer, or a status for an invalid alignment, allocation
///         failure, unexpected posix_memalign failure, or metadata OOM.
StatusOr<Buffer> TryAllocateAlignedCpuBuffer(
        size_t nbytes, size_t alignment) noexcept;

class CPUAllocator final : public Allocator {
public:
    explicit CPUAllocator(Device device) : device_(device) {}

    /// @brief Allocates an owning CPU Buffer, including a valid zero-byte Buffer.
    /// @param nbytes Number of visible bytes in the Buffer.
    /// @return Buffer owning aligned storage and its metadata.
    /// @throws std::bad_alloc if storage or Buffer metadata allocation fails.
    Buffer Allocate(size_t nbytes) override;

    AM_NODISCARD Device device() const noexcept override;

private:
    Device device_;
};

class CPUAllocatorProvider final : public AllocatorProvider {
public:
    std::unique_ptr<Allocator> CreateAllocator(Device device) override;
};

} // namespace aethermind

#endif // AETHERMIND_CPU_ALLOCATOR_H
