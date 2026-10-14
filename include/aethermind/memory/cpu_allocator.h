#ifndef AETHERMIND_CPU_ALLOCATOR_H
#define AETHERMIND_CPU_ALLOCATOR_H

#include "aethermind/memory/allocator.h"
#include "aethermind/memory/buffer.h"

#include <cstddef>
#include <memory>

namespace aethermind {

/// @brief CPU-specific storage policy applied to every allocation.
///
/// These knobs describe how CPU storage is obtained, which is an allocator
/// concern rather than a caller concern, so they stay out of the generic
/// AllocationRequest contract.
struct CPUAllocatorPolicy {
    /// Alignment floor used when a request states no minimum. Zero selects the
    /// build's default cache-line alignment.
    size_t default_alignment = 0;
    /// Overwrite fresh storage with a junk pattern, so an uninitialized read
    /// surfaces instead of silently reusing previously freed bytes.
    bool junk_fill = true;
    /// Allow transparent huge pages above the THP size threshold, subject to
    /// the THP_MEM_ALLOC_ENABLE environment gate.
    bool allow_thp = true;
};

/// @brief Allocates owning CPU storage through the counted allocation entry point.
///
/// Storage comes from posix_memalign, so a Buffer always satisfies the
/// alignment its handle declares.
class CPUAllocator final : public Allocator {
public:
    /// @param device CPU device this allocator reports.
    /// @param policy Storage policy. Defaults preserve the historical behavior
    ///        of junk filling and env-gated THP.
    /// @param state Ledger to charge, or null to create a private one.
    explicit CPUAllocator(Device device,
                          CPUAllocatorPolicy policy = {},
                          std::shared_ptr<AllocationState> state = nullptr);

    AM_NODISCARD Device device() const noexcept override;

    /// @brief Returns the storage policy this allocator applies.
    AM_NODISCARD const CPUAllocatorPolicy& policy() const noexcept {
        return policy_;
    }

protected:
    /// @brief Obtains aligned CPU storage for `request`.
    ///
    /// A request alignment below `sizeof(void*)` is promoted rather than
    /// rejected, because posix_memalign cannot express it and the request states
    /// a floor. Zero selects the policy default. THP-eligible sizes are
    /// page-aligned and advised, which may return a stronger alignment than
    /// requested; the handle always declares what it actually guarantees.
    StatusOr<MemoryHandle> AllocateStorage(const AllocationRequest& request) override;

private:
    Device device_;
    CPUAllocatorPolicy policy_;
};

class CPUAllocatorProvider final : public AllocatorProvider {
public:
    std::unique_ptr<Allocator> CreateAllocator(
            Device device, std::shared_ptr<AllocationState> state) override;
};

} // namespace aethermind

#endif // AETHERMIND_CPU_ALLOCATOR_H
