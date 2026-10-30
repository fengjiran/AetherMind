#ifndef AETHERMIND_TEST_RESOURCE_ALLOCATOR_H
#define AETHERMIND_TEST_RESOURCE_ALLOCATOR_H

#include "aethermind/memory/allocator.h"
#include "aethermind/memory/cpu_allocator.h"

#include <limits>
#include <memory>
#include <vector>

namespace aethermind::test {

struct ResourceAllocationStats {
    std::vector<AllocationRequest> requests;
    size_t fail_on_call = std::numeric_limits<size_t>::max();
};

class ResourceRecordingAllocator final : public Allocator {
public:
    explicit ResourceRecordingAllocator(std::shared_ptr<ResourceAllocationStats> stats)
        : stats_(std::move(stats)), cpu_(Device::CPU()) {}
    Device device() const noexcept override { return Device::CPU(); }

protected:
    StatusOr<MemoryHandle> AllocateStorage(const AllocationRequest& request) override {
        const size_t index = stats_->requests.size();
        stats_->requests.push_back(request);
        if (index == stats_->fail_on_call) return Status::ResourceExhausted({});
        AM_ASSIGN_OR_RETURN(auto backing, cpu_.TryAllocate(request));
        auto owner = std::make_unique<Buffer>(std::move(backing));
        auto* data = owner->mutable_data();
        const auto alignment = owner->alignment();
        return MemoryHandle(data, owner.release(), [](void* context, void*) noexcept { delete static_cast<Buffer*>(context); }, Device::CPU(), alignment);
    }

private:
    std::shared_ptr<ResourceAllocationStats> stats_;
    CPUAllocator cpu_;
};

class ResourceRecordingProvider final : public AllocatorProvider {
public:
    explicit ResourceRecordingProvider(std::shared_ptr<ResourceAllocationStats> stats)
        : stats_(std::move(stats)) {}
    std::unique_ptr<Allocator> CreateAllocator(Device) override {
        return std::make_unique<ResourceRecordingAllocator>(stats_);
    }

private:
    std::shared_ptr<ResourceAllocationStats> stats_;
};

} // namespace aethermind::test
#endif
