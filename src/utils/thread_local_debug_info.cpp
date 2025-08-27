//
// Created by 赵丹 on 2025/8/27.
//
#include "utils/thread_local_debug_info.h"
#include "utils/thread_local.h"

namespace aethermind {

DEFINE_TLS_STATIC(std::shared_ptr<ThreadLocalDebugInfo>, tls_debug_info);
#define debug_info (tls_debug_info.get())

DebugInfoBase* ThreadLocalDebugInfo::get(DebugInfoKind kind) {
    ThreadLocalDebugInfo* cur = debug_info.get();
    while (cur) {
        if (cur->kind_ == kind) {
            return cur->info_.get();
        }
        cur = cur->parent_info_.get();
    }
    return nullptr;
}


}// namespace aethermind