//
// Created by 赵丹 on 25-1-23.
//

#include "memory/allocator.h"

namespace aethermind {

REGISTER_ALLOCATOR(DeviceType::kUndefined, UndefinedAllocator);

}// namespace aethermind
