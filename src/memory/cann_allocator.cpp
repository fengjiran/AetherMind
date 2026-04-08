#include "aethermind/memory/cann_allocator.h"

namespace aethermind {

std::unique_ptr<Allocator> CANNAllocatorProvider::CreateAllocator(Device device) {
    (void) device;
    return nullptr;
}

}// namespace aethermind
