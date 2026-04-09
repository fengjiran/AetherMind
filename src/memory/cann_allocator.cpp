#include "aethermind/memory/cann_allocator.h"
#include "utils/logging.h"

namespace aethermind {

std::unique_ptr<Allocator> CANNAllocatorProvider::CreateAllocator(Device device) {
    AM_CHECK(false, "CANN allocator is not implemented in this build configuration. "
                    "Device: {}",
             device.ToString().c_str());
    return nullptr;
}

}// namespace aethermind
