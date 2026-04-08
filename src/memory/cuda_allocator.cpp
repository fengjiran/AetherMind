#include "aethermind/memory/cuda_allocator.h"

namespace aethermind {

std::unique_ptr<Allocator> CUDAAllocatorProvider::CreateAllocator(Device device) {
    (void) device;
    return nullptr;
}

}// namespace aethermind
