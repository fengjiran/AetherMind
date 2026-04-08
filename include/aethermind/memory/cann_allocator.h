#ifndef AETHERMIND_CANN_ALLOCATOR_H
#define AETHERMIND_CANN_ALLOCATOR_H

#include "allocator.h"

namespace aethermind {

class CANNAllocatorProvider final : public AllocatorProvider {
public:
    std::unique_ptr<Allocator> CreateAllocator(Device device) override;
};

}// namespace aethermind

#endif// AETHERMIND_CANN_ALLOCATOR_H
