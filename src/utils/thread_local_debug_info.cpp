//
// Created by 赵丹 on 2025/8/27.
//
#include "utils/thread_local_debug_info.h"
#include "utils/thread_local.h"

#include <glog/logging.h>

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

std::shared_ptr<ThreadLocalDebugInfo> ThreadLocalDebugInfo::current() {
    return debug_info;
}

void ThreadLocalDebugInfo::_forceCurrentDebugInfo(std::shared_ptr<ThreadLocalDebugInfo> info) {
    debug_info = std::move(info);
}

void ThreadLocalDebugInfo::_push(DebugInfoKind kind, std::shared_ptr<DebugInfoBase> info) {
    auto prev_info = debug_info;
    debug_info = std::make_shared<ThreadLocalDebugInfo>();
    debug_info->parent_info_ = prev_info;
    debug_info->kind_ = kind;
    debug_info->info_ = std::move(info);
}

std::shared_ptr<DebugInfoBase> ThreadLocalDebugInfo::_pop(DebugInfoKind kind) {
    CHECK(debug_info && debug_info->kind_ == kind) << "Expected debug info of type " << static_cast<int>(kind);
    auto res = debug_info;
    debug_info = debug_info->parent_info_;
    return res->info_;
}

std::shared_ptr<DebugInfoBase> ThreadLocalDebugInfo::_peek(DebugInfoKind kind) {
    CHECK(debug_info && debug_info->kind_ == kind) << "Expected debug info of type " << static_cast<int>(kind);
    return debug_info->info_;
}

DebugInfoGuard::DebugInfoGuard(DebugInfoKind kind, std::shared_ptr<DebugInfoBase> info) {
    if (info) {
        prev_info_ = debug_info;
        ThreadLocalDebugInfo::_push(kind, std::move(info));
        active_ = true;
    }
}

DebugInfoGuard::DebugInfoGuard(std::shared_ptr<ThreadLocalDebugInfo> info) {
    if (info) {
        prev_info_ = std::move(debug_info);
        debug_info = std::move(info);
        active_ = true;
    }
}

DebugInfoGuard::~DebugInfoGuard() {
    if (active_) {
        debug_info = prev_info_;
    }
}

}// namespace aethermind