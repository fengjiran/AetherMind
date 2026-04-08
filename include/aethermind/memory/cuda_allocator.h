#ifndef AETHERMIND_CUDA_ALLOCATOR_H
#define AETHERMIND_CUDA_ALLOCATOR_H

#include "allocator.h"

namespace aethermind {

class CUDAAllocatorProvider final : public AllocatorProvider {
public:
    std::unique_ptr<Allocator> CreateAllocator(Device device) override;
};

}// namespace aethermind

#endif// AETHERMIND_CUDA_ALLOCATOR_H
