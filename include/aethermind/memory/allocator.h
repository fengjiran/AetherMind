/// \file
/// Memory allocation abstraction, accounting state, and registry.
///
/// This file defines the core memory allocation interfaces for AetherMind:
/// - AllocationTag: opaque classification of one allocation for statistics
/// - AllocationRequest: what a caller needs, without saying how to get it
/// - AllocationState: shared accounting for every Buffer an allocator produces
/// - Allocator: abstract entry point that accounts, then delegates storage
/// - AllocatorProvider: factory interface for creating Allocator instances
/// - AllocatorRegistry: registry for allocator providers and instances
/// - MemoryReportingInfoBase: interface for thread-local memory usage reporting
///
/// The allocation model uses Buffer as the primary memory container, which owns
/// the allocated memory through MemoryHandle. Allocators are registered per
/// DeviceType and instantiated per Device.

#ifndef AETHERMIND_MEMORY_ALLOCATOR_H
#define AETHERMIND_MEMORY_ALLOCATOR_H

#include "aethermind/base/device.h"
#include "aethermind/base/status.h"
#include "buffer.h"
#include "utils/thread_local_debug_info.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace aethermind {

/// @brief Opaque classification of one allocation, used only to group statistics.
///
/// The memory layer never interprets an id beyond grouping; the meaning is
/// owned by whoever allocated. Ids below `kCustomBase` are reserved for
/// AetherMind's own storage classes and must not be reused by downstream code.
/// Ids from `kCustomBase` up belong to custom allocators and consumers, which
/// must partition that range among themselves.
class AllocationTag {
public:
    constexpr AllocationTag() noexcept = default;

    /// @brief Wraps a raw id. Only needed to claim a custom-range id.
    constexpr explicit AllocationTag(uint16_t id) noexcept : id_(id) {}

    AM_NODISCARD constexpr uint16_t id() const noexcept {
        return id_;
    }

    friend constexpr bool operator==(AllocationTag lhs, AllocationTag rhs) noexcept {
        return lhs.id_ == rhs.id_;
    }

    /// @brief First id a downstream consumer may claim.
    static constexpr uint16_t kCustomBase = 1024;

    /// @brief Storage with no declared class; the default for a plain request.
    static constexpr AllocationTag General() noexcept {
        return AllocationTag{kGeneral};
    }

    /// @brief Backing storage of a per-request activation arena.
    static constexpr AllocationTag Activation() noexcept {
        return AllocationTag{kActivation};
    }

    /// @brief Static KV cache key/value plane storage.
    static constexpr AllocationTag KVCache() noexcept {
        return AllocationTag{kKVCache};
    }

    /// @brief Packed weight artifact storage.
    static constexpr AllocationTag PackedWeight() noexcept {
        return AllocationTag{kPackedWeight};
    }

    /// @brief Backing storage of an execution workspace arena.
    static constexpr AllocationTag Workspace() noexcept {
        return AllocationTag{kWorkspace};
    }

private:
    static constexpr uint16_t kGeneral = 0;
    static constexpr uint16_t kActivation = 1;
    static constexpr uint16_t kKVCache = 2;
    static constexpr uint16_t kPackedWeight = 3;
    static constexpr uint16_t kWorkspace = 4;

    uint16_t id_ = kGeneral;
};

/// @brief What a caller needs from an allocator, without saying how to get it.
///
/// Callers state a visible size, a minimum alignment, and a classification.
/// Capacity planning, padding policy, and the physical allocation strategy all
/// stay with the allocator and the owning module: `nbytes` is the byte count
/// the resulting Buffer reports, so any layout padding a module needs must
/// already be included in it.
struct AllocationRequest {
    /// Visible byte count of the resulting Buffer. May be zero.
    size_t nbytes = 0;

    /// Minimum alignment, or zero to accept the allocator's default policy.
    /// A non-zero value must be a power of two. It is a floor, not an exact
    /// request: the returned Buffer may declare a stronger alignment, and
    /// callers must read the declared value rather than assume this one.
    size_t alignment = 0;

    /// Classification used only for statistics.
    AllocationTag tag{};
};

/// @brief Whole-ledger counters at one instant.
///
/// These describe managed backing storage only. Memory-mapped raw weights,
/// ordinary container allocations, per-Buffer metadata, and libc-internal
/// overhead are all excluded, so this is not a process RSS bound.
struct AllocationStats {
    /// Bytes currently charged and not yet released.
    size_t live_bytes = 0;
    /// Allocations currently charged and not yet released.
    size_t live_allocations = 0;
    /// Highest `live_bytes` observed since the state was created.
    size_t peak_live_bytes = 0;
    /// Buffers constructed and successfully committed to this ledger.
    size_t total_allocations = 0;
    /// Releases that returned capacity to the ledger.
    size_t total_releases = 0;
    /// Storage-hook failures reported as ResourceExhausted.
    size_t system_allocation_failures = 0;
};

/// @brief Per-tag counters at one instant. See AllocationStats for scope.
struct AllocationTagStats {
    size_t live_bytes = 0;
    size_t live_allocations = 0;
    size_t peak_live_bytes = 0;
    size_t total_allocations = 0;
};

/// @brief Shared accounting for every Buffer produced from one allocation state.
///
/// One state is shared by all allocators of a Runtime, including allocators
/// created after a provider is replaced, so a Buffer outliving its allocator
/// still charges and refunds the same ledger. Each released Buffer's accounting
/// record holds a strong reference, which keeps the state alive until the last
/// Buffer goes away; the state never owns those records.
///
/// Thread-safety: every entry point is safe to call concurrently. Distinct
/// Buffer copies may be destroyed on different threads, so releases are
/// synchronized with acquisitions. Backing storage allocation and release run
/// outside the mutex; inserting a first-seen tag may allocate counter metadata.
class AllocationState {
public:
    AllocationState() = default;
    AllocationState(const AllocationState&) = delete;
    AllocationState& operator=(const AllocationState&) = delete;
    AllocationState(AllocationState&&) = delete;
    AllocationState& operator=(AllocationState&&) = delete;

    /// @brief Creates a shared state. Allocators created without one use a
    ///        private state, so they still account consistently.
    AM_NODISCARD static std::shared_ptr<AllocationState> Create();

    /// @brief Charges `bytes` to `tag`.
    ///
    /// Strongly exception-safe: a first-seen tag inserts its counter entry
    /// before any total is updated, so a failure leaves nothing charged.
    /// @throws std::bad_alloc when a new tag entry cannot be created.
    void Acquire(size_t bytes, AllocationTag tag);

    /// @brief Returns `bytes` charged to `tag`.
    ///
    /// Never inserts, so it cannot allocate; safe to call from a deleter.
    /// @pre Matches a successful Acquire that has not already been released.
    void Release(size_t bytes, AllocationTag tag) noexcept;

    /// @brief Counts one storage-hook failure reported as ResourceExhausted.
    void RecordSystemFailure() noexcept;

    /// @brief Returns a consistent snapshot of the whole-ledger counters.
    AM_NODISCARD AllocationStats Snapshot() const noexcept;

    /// @brief Returns a consistent snapshot of one tag's counters.
    AM_NODISCARD AllocationTagStats TagSnapshot(AllocationTag tag) const noexcept;

private:
    mutable std::mutex mutex_;
    std::unordered_map<uint16_t, AllocationTagStats> tags_;
    AllocationStats totals_{};
};

/// Abstract interface for device-specific memory allocation.
///
/// Allocators produce Buffer objects that own allocated memory via MemoryHandle.
/// Each Allocator is associated with a specific Device (e.g., CPU, CUDA).
///
/// Allocation runs through one public entry point so that accounting happens
/// exactly once per Buffer, no matter which subclass provides the storage:
/// `TryAllocate` validates the request, calls the protected `AllocateStorage`
/// hook, then attaches an accounting record that refunds the shared
/// AllocationState when the last Buffer copy is released.
///
/// Implementations must ensure:
/// - A zero-byte request returns a valid Buffer with nbytes() == 0
/// - The returned MemoryHandle owns its storage and declares the alignment it
///   actually guarantees, which must satisfy the request's minimum
/// - AllocateStorage reports failure through Status; only std::bad_alloc is
///   tolerated as an exception, and TryAllocate converts it
///
/// \note Allocator instances are owned by AllocatorRegistry. Do not create
///       Allocators directly; use AllocatorRegistry::GetAllocator().
class Allocator {
public:
    virtual ~Allocator() = default;

    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;

    /// @brief Allocates a Buffer satisfying `request` and charges the ledger.
    ///
    /// @param request Visible size, minimum alignment, and classification.
    /// @return An owning Buffer, or a status: InvalidArgument for a non-zero
    ///         alignment that is not a power of two, ResourceExhausted with an
    ///         empty message when the system cannot satisfy the request or the
    ///         accounting record cannot be created, and whatever the storage
    ///         hook reports otherwise. Invalid hook ownership, device, or
    ///         alignment is reported as Internal. Every failure releases storage
    ///         already obtained without updating successful-allocation counters
    ///         or peaks; storage-hook exhaustion updates its failure counter.
    /// @note Does not throw; allocation failures arrive as a status.
    StatusOr<Buffer> TryAllocate(const AllocationRequest& request) noexcept;

    /// @brief Allocates a Buffer of the specified size using the default policy.
    ///
    /// Equivalent to `TryAllocate({.nbytes = nbytes})`, so it charges the same
    /// ledger and cannot bypass accounting.
    ///
    /// @param nbytes Number of bytes to allocate. May be zero.
    /// @return A Buffer owning the allocated memory.
    /// @throws std::bad_alloc when the system cannot satisfy the request.
    /// @throws Error when the storage hook reports a non-exhaustion failure.
    Buffer Allocate(size_t nbytes);

    /// Returns the Device this allocator is associated with.
    /// \return The target Device for allocations from this allocator.
    AM_NODISCARD virtual Device device() const noexcept = 0;

    /// @brief Returns the ledger every Buffer from this allocator charges.
    ///
    /// The returned state stays valid while any such Buffer is alive, because
    /// each Buffer's accounting record holds its own strong reference.
    AM_NODISCARD const std::shared_ptr<AllocationState>& allocation_state() const noexcept {
        return state_;
    }

protected:
    /// @param state Ledger to charge, or null to create a private one.
    explicit Allocator(std::shared_ptr<AllocationState> state);

    /// @brief Obtains owning storage for `request`.
    ///
    /// The returned handle must own its memory, must carry this allocator's
    /// device, and must declare the alignment it actually guarantees — at least
    /// the request's minimum, and the allocator's default when the request
    /// states none. This is the only device-specific step; accounting, request
    /// validation, and Buffer construction belong to the base class.
    ///
    /// @param request Validated request. `alignment` may be zero, meaning the
    ///        implementation's default policy applies.
    /// @return An owning handle, or a status describing the failure.
    virtual StatusOr<MemoryHandle> AllocateStorage(const AllocationRequest& request) = 0;

private:
    std::shared_ptr<AllocationState> state_;
};

/// Factory interface for creating Allocator instances.
///
/// AllocatorProviders are registered with AllocatorRegistry per DeviceType.
/// When GetAllocator is called for a Device, the corresponding Provider creates
/// an Allocator instance specific to that Device's index.
///
/// \note Implementations should cache device-specific state (e.g., CUDA streams)
///       per Device index if needed.
class AllocatorProvider {
public:
    virtual ~AllocatorProvider() = default;

    /// Creates an Allocator for the specified Device.
    ///
    /// \param device The Device to create an allocator for.
    /// \param state Ledger the new allocator must charge. The registry passes
    ///        the same state to every allocator it creates, including after a
    ///        provider is replaced, so implementations must forward it to the
    ///        Allocator base rather than substituting their own.
    /// \return A new Allocator instance for the device.
    /// \throws RuntimeError if the device type is unsupported or creation fails.
    virtual std::unique_ptr<Allocator> CreateAllocator(
            Device device, std::shared_ptr<AllocationState> state) = 0;
};

/// Global registry for allocator providers and cached allocator instances.
///
/// AllocatorRegistry manages:
/// 1. Provider registration: One provider per DeviceType (CPU, CUDA, etc.)
/// 2. Instance caching: One Allocator per Device (type + index combination)
/// 3. Accounting state: One AllocationState shared by every allocator it creates
///
/// Call flow:
/// - RegisterProvider: Called during initialization to register providers
/// - GetAllocator: Called during runtime to obtain allocators for tensor allocation
///
/// \note This registry is not thread-safe. If concurrent access is needed,
///       external synchronization is required. The mutex_ member is commented
///       out pending multi-threaded runtime support. The AllocationState it
///       hands out is separately synchronized, because Buffers may be released
///       from any thread.
class AllocatorRegistry {
public:
    /// Creates a registry with a fresh, empty AllocationState.
    AllocatorRegistry();

    /// @brief Creates a registry that charges `state`.
    ///
    /// @param state Ledger to share with every allocator created here. A null
    ///        state is replaced by a fresh one, so the registry always has one.
    explicit AllocatorRegistry(std::shared_ptr<AllocationState> state);

    /// Registers an allocator provider for a DeviceType.
    ///
    /// \param type The device type (kCPU, kCUDA, etc.). Must not be kUndefined.
    /// \param provider The provider to register. Must not be null.
    /// \throws RuntimeError if type is kUndefined or a provider is already registered.
    void RegisterProvider(DeviceType type, std::unique_ptr<AllocatorProvider> provider);

    /// Sets or replaces an allocator provider for a DeviceType.
    ///
    /// Unlike RegisterProvider, this allows replacing an existing provider.
    /// Existing allocator instances for the DeviceType are cleared. The shared
    /// AllocationState is kept, so Buffers allocated before the replacement
    /// still refund the ledger this registry reports.
    ///
    /// \param type The device type. Must not be kUndefined.
    /// \param provider The provider to set. Must not be null.
    void SetProvider(DeviceType type, std::unique_ptr<AllocatorProvider> provider);

    /// Returns the Allocator for a Device, creating it if necessary.
    ///
    /// If an allocator for the device already exists in the cache, returns it.
    /// Otherwise, creates a new allocator via the registered provider.
    ///
    /// \param device The Device to get an allocator for.
    /// \return Reference to the cached Allocator for this device.
    /// \throws RuntimeError if no provider is registered for the device type.
    Allocator& GetAllocator(Device device);

    /// @brief Returns the Allocator for a Device, or a status when unavailable.
    ///
    /// Performs the same lookup and caching as GetAllocator, but reports a
    /// missing provider as FailedPrecondition instead of aborting. Use this
    /// wherever the caller can propagate a Status.
    ///
    /// @param device The Device to get an allocator for.
    /// @return Borrowed pointer to the cached Allocator, owned by this registry.
    /// @note Not noexcept: a provider's CreateAllocator may throw, and this
    ///       propagates rather than converting it.
    StatusOr<Allocator*> TryGetAllocator(Device device);

    /// @brief Returns whether a provider is registered for the device type.
    AM_NODISCARD bool HasProvider(DeviceType type) const noexcept {
        return providers_.contains(type);
    }

    /// @brief Returns the ledger shared by every allocator this registry creates.
    AM_NODISCARD const std::shared_ptr<AllocationState>& allocation_state() const noexcept {
        return state_;
    }

private:
    std::shared_ptr<AllocationState> state_;
    std::unordered_map<DeviceType, std::unique_ptr<AllocatorProvider>> providers_;
    std::unordered_map<Device, std::unique_ptr<Allocator>> instances_;
};

// An interface for reporting thread local memory usage per device
class MemoryReportingInfoBase : public DebugInfoBase {
public:
    /**
     * \brief Reports memory usage.
     * \param ptr The pointer to the memory block.
     *
     * \param alloc_size The size of the memory block.
     *
     * \param total_allocated The total allocated memory.
     *
     * \param total_reserved The total reserved memory.
     *
     * \param device The device type.
     */
    virtual void reportMemoryUsage(void* ptr,
                                   int64_t alloc_size,
                                   size_t total_allocated,
                                   size_t total_reserved,
                                   Device device) = 0;

    virtual void reportOutOfMemory(int64_t alloc_size,
                                   size_t total_allocated,
                                   size_t total_reserved,
                                   Device device);

    virtual bool memoryProfilingEnabled() const = 0;
};

bool memoryProfilingEnabled();

} // namespace aethermind

#endif // AETHERMIND_MEMORY_ALLOCATOR_H
