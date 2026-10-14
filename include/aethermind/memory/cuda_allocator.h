#ifndef AETHERMIND_CUDA_ALLOCATOR_H
#define AETHERMIND_CUDA_ALLOCATOR_H

#include "allocator.h"

#include <memory>

namespace aethermind {

class CUDAAllocatorProvider final : public AllocatorProvider {
public:
    std::unique_ptr<Allocator> CreateAllocator(
            Device device, std::shared_ptr<AllocationState> state) override;
};

} // namespace aethermind

#endif // AETHERMIND_CUDA_ALLOCATOR_H
