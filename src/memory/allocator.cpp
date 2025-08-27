//
// Created by 赵丹 on 25-1-23.
//

#include "memory/allocator.h"

namespace aethermind {

REGISTER_ALLOCATOR(DeviceType::kUndefined, UndefinedAllocator);

bool memoryProfilingEnabled() {
    auto* reporter_ptr = static_cast<MemoryReportingInfoBase*>(
            ThreadLocalDebugInfo::get(DebugInfoKind::PROFILER_STATE));
    return reporter_ptr && reporter_ptr->memoryProfilingEnabled();
}


}// namespace aethermind
