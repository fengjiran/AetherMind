/// \file
/// Implementation of Allocator, AllocationState, and AllocatorRegistry.
///
/// Provides:
/// - The single accounting allocation entry point and its release record
/// - Synchronized ledger counters for AllocationState
/// - Provider registration and instance caching for AllocatorRegistry
/// - Thread-local memory profiling state check via memoryProfilingEnabled()

#include "aethermind/memory/allocator.h"
#include "aethermind/base/alignment.h"
#include "aethermind/base/error.h"
#include "utils/logging.h"

#include <format>
#include <memory>
#include <new>
#include <utility>

namespace aethermind {
namespace {

/// @brief Ownership payload behind a counted Buffer's MemoryHandle.
///
/// The handle's deleter owns this record, so it outlives the Allocator and the
/// Runtime that created the storage: it borrows neither, and keeps the ledger
/// alive itself. That is what makes releasing a surviving Buffer copy safe
/// after its allocator is gone, including after a provider replacement.
struct ReleaseRecord {
    std::shared_ptr<AllocationState> state;
    /// Owns the real bytes and knows how to free them.
    MemoryHandle storage;
    size_t charged_bytes = 0;
    AllocationTag tag{};
};

/// @brief Frees the storage, then refunds the ledger, then destroys the record.
///
/// Storage is released first so the counters never report capacity as free
/// while the bytes are still resident. Runs from a noexcept deleter, so it must
/// neither allocate nor throw; deleting the record only frees.
void ReleaseAccountedStorage(void* deleter_ctx, void* ptr) noexcept {
    UNUSED(ptr);
    auto* record = static_cast<ReleaseRecord*>(deleter_ctx);
    record->storage.reset();
    record->state->Release(record->charged_bytes, record->tag);
    delete record;
}

/// @brief Refunds a charge unless the caller takes ownership of it.
///
/// Covers the window between charging the ledger and installing the record that
/// will refund it, so a failure there cannot leak accounted capacity.
class ChargeGuard {
public:
    ChargeGuard(AllocationState& state, size_t bytes, AllocationTag tag) noexcept
        : state_(&state), bytes_(bytes), tag_(tag) {}

    ~ChargeGuard() {
        if (state_ != nullptr) {
            state_->Release(bytes_, tag_);
        }
    }

    ChargeGuard(const ChargeGuard&) = delete;
    ChargeGuard& operator=(const ChargeGuard&) = delete;

    void Dismiss() noexcept {
        state_ = nullptr;
    }

private:
    AllocationState* state_;
    size_t bytes_;
    AllocationTag tag_;
};

} // namespace

std::shared_ptr<AllocationState> AllocationState::Create() {
    return std::make_shared<AllocationState>();
}

void AllocationState::Acquire(size_t bytes, AllocationTag tag) {
    const std::lock_guard<std::mutex> lock(mutex_);
    // Insert before touching any total: a first-seen tag is the only step that
    // can allocate, so doing it first keeps Acquire strongly exception-safe.
    auto& [live_bytes, live_allocations, peak_live_bytes, total_allocations] = tags_[tag.id()];
    live_bytes += bytes;
    ++live_allocations;
    ++total_allocations;
    if (live_bytes > peak_live_bytes) {
        peak_live_bytes = live_bytes;
    }

    totals_.live_bytes += bytes;
    ++totals_.live_allocations;
    ++totals_.total_allocations;
    if (totals_.live_bytes > totals_.peak_live_bytes) {
        totals_.peak_live_bytes = totals_.live_bytes;
    }
}

void AllocationState::Release(size_t bytes, AllocationTag tag) noexcept {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (const auto it = tags_.find(tag.id()); it != tags_.end()) {
        auto& entry = it->second;
        entry.live_bytes -= bytes;
        --entry.live_allocations;
    }

    totals_.live_bytes -= bytes;
    --totals_.live_allocations;
    ++totals_.total_releases;
}

void AllocationState::RecordSystemFailure() noexcept {
    const std::lock_guard<std::mutex> lock(mutex_);
    ++totals_.system_allocation_failures;
}

AllocationStats AllocationState::Snapshot() const noexcept {
    const std::lock_guard<std::mutex> lock(mutex_);
    return totals_;
}

AllocationTagStats AllocationState::TagSnapshot(AllocationTag tag) const noexcept {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto it = tags_.find(tag.id());
    return it == tags_.end() ? AllocationTagStats{} : it->second;
}

Allocator::Allocator(std::shared_ptr<AllocationState> state)
    : state_(state != nullptr ? std::move(state) : AllocationState::Create()) {}

StatusOr<Buffer> Allocator::TryAllocate(const AllocationRequest& request) noexcept try {
    if (request.alignment != 0 && !IsValidAlignment(request.alignment)) {
        return Status::InvalidArgument(
                "Allocation alignment must be zero or a power of two");
    }

    auto storage = AllocateStorage(request);
    if (!storage.ok()) {
        if (storage.status().code() == StatusCode::kResourceExhausted) {
            state_->RecordSystemFailure();
        }
        return storage.status();
    }

    MemoryHandle owned = std::move(storage).value();
    if (!owned) {
        return Status::Internal(
                "Allocator storage hook returned an empty handle");
    }

    // Metadata first: the pointer, device, and declared alignment outlive the
    // move of `owned` into the release record.
    void* const data = owned.get();
    const Device device = owned.device();
    const size_t alignment = owned.alignment();

    // A zero-byte Buffer still reserves one byte from the system, so the ledger
    // reports what the process actually holds rather than the visible size.
    const size_t charged_bytes = request.nbytes == 0 ? 1 : request.nbytes;

    state_->Acquire(charged_bytes, request.tag);
    ChargeGuard guard{*state_, charged_bytes, request.tag};
    auto* record = new ReleaseRecord{state_, std::move(owned), charged_bytes,
                                     request.tag};
    guard.Dismiss();

    // If Buffer metadata allocation throws, the handle parameter is destroyed
    // during unwinding, which runs the deleter: storage is freed and the charge
    // is refunded.
    return Buffer{request.nbytes,
                  MemoryHandle(data, record, &ReleaseAccountedStorage, device,
                               alignment)};
} catch (const std::bad_alloc&) {
    // Error diagnostics may allocate; reporting OOM must not allocate again.
    return Status::ResourceExhausted({});
}

Buffer Allocator::Allocate(size_t nbytes) {
    auto result = TryAllocate({.nbytes = nbytes});
    if (result.ok()) {
        return std::move(result).value();
    }

    const Status& status = result.status();
    if (status.code() == StatusCode::kResourceExhausted) {
        throw std::bad_alloc();
    }
    AM_THROW(RuntimeError) << "Allocation of " << nbytes << " bytes failed: "
                           << status.ToString();
}

void MemoryReportingInfoBase::reportOutOfMemory(int64_t alloc_size,
                                                size_t total_allocated,
                                                size_t total_reserved,
                                                Device device) {
    (void) alloc_size;
    (void) total_allocated;
    (void) total_reserved;
    (void) device;
}

AllocatorRegistry::AllocatorRegistry() : state_(AllocationState::Create()) {}

AllocatorRegistry::AllocatorRegistry(std::shared_ptr<AllocationState> state)
    : state_(state != nullptr ? std::move(state) : AllocationState::Create()) {}

void AllocatorRegistry::RegisterProvider(DeviceType type, std::unique_ptr<AllocatorProvider> provider) {
    AM_CHECK(provider != nullptr, "Allocator provider cannot be null");
    AM_CHECK(type != DeviceType::kUndefined, "Cannot register provider for kUndefined device type");
    AM_CHECK(!providers_.contains(type), "Allocator provider already registered for device type: {}", DeviceType2Str(type).c_str());
    providers_[type] = std::move(provider);
}

void AllocatorRegistry::SetProvider(DeviceType type, std::unique_ptr<AllocatorProvider> provider) {
    AM_CHECK(provider != nullptr, "Allocator provider cannot be null");
    AM_CHECK(type != DeviceType::kUndefined, "Cannot set provider for kUndefined device type");

    providers_[type] = std::move(provider);

    // Clear cached instances for the replaced provider's device type.
    // New instances will be created on next GetAllocator call. The shared
    // AllocationState is deliberately kept: Buffers from the cleared instances
    // may still be alive and must refund the ledger this registry reports.
    std::erase_if(instances_, [type](const auto& pair) {
        return pair.first.type() == type;
    });
}

Allocator& AllocatorRegistry::GetAllocator(Device device) {
    auto result = TryGetAllocator(device);
    AM_CHECK(result.ok(), "Failed to get allocator for device {}: {}",
             device.ToString().c_str(), result.status().ToString().c_str());
    return *result.value();
}

StatusOr<Allocator*> AllocatorRegistry::TryGetAllocator(Device device) {
    // TODO: Enable mutex for thread-safe access once multi-threaded runtime is supported.
    // std::lock_guard<std::mutex> lock(mutex_);
    auto it = instances_.find(device);
    if (it != instances_.end()) {
        return it->second.get();
    }

    const auto type = device.type();
    if (!providers_.contains(type)) {
        return Status::FailedPrecondition(std::format(
                "No allocator provider registered for device type: {}",
                DeviceType2Str(type)));
    }

    auto allocator = providers_[type]->CreateAllocator(device, state_);
    if (allocator == nullptr) {
        return Status::Internal(std::format(
                "Allocator provider failed to create allocator for device: {}",
                device.ToString()));
    }
    if (allocator->device() != device) {
        return Status::Internal(std::format(
                "Created allocator device mismatch: expected {}, got {}",
                device.ToString(), allocator->device().ToString()));
    }

    auto [new_it, inserted] = instances_.emplace(device, std::move(allocator));
    return new_it->second.get();
}

bool memoryProfilingEnabled() {
    auto* reporter_ptr = static_cast<MemoryReportingInfoBase*>(
            ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE));
    return reporter_ptr && reporter_ptr->memoryProfilingEnabled();
}


} // namespace aethermind
