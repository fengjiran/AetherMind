#include "aethermind/memory/cuda_allocator.h"
#include "utils/logging.h"

namespace aethermind {

std::unique_ptr<Allocator> CUDAAllocatorProvider::CreateAllocator(Device device) {
    AM_CHECK(false, "CUDA allocator is not implemented in this build configuration. "
                    "Device: {}",
             device.ToString().c_str());
    return nullptr;
}

}// namespace aethermind
