//
// Created by 赵丹 on 2025/8/27.
//

#ifndef AETHERMIND_UTILS_THREAD_LOCAL_DEBUG_INFO_H
#define AETHERMIND_UTILS_THREAD_LOCAL_DEBUG_INFO_H

#include <memory>

namespace aethermind {

enum class DebugInfoKind : uint8_t {
    PRODUCER_INFO = 0,
    MOBILE_RUNTIME_INFO,
    PROFILER_STATE,
    INFERENCE_CONTEXT,// for inference usage
    PARAM_COMMS_INFO,

    TEST_INFO,  // used only in tests
    TEST_INFO_2,// used only in tests
};

class DebugInfoBase {
public:
    DebugInfoBase() = default;
    virtual ~DebugInfoBase() = default;
};

class ThreadLocalDebugInfo {
public:
    static DebugInfoBase* get(DebugInfoKind kind);

    // Get current ThreadLocalDebugInfo
    static std::shared_ptr<ThreadLocalDebugInfo> current();

    // Internal, use DebugInfoGuard/ThreadLocalStateGuard
    static void _forceCurrentDebugInfo(std::shared_ptr<ThreadLocalDebugInfo> info);

    // Push debug info struct of a given kind
    static void _push(DebugInfoKind kind, std::shared_ptr<DebugInfoBase> info);

    // Pop debug info, throws in case the last pushed
    // debug info is not of a given kind
    static std::shared_ptr<DebugInfoBase> _pop(DebugInfoKind kind);

    // Peek debug info, throws in case the last pushed debug info is not of the
    // given kind
    static std::shared_ptr<DebugInfoBase> _peek(DebugInfoKind kind);

private:
    DebugInfoKind kind_;
    std::shared_ptr<DebugInfoBase> info_;
    std::shared_ptr<ThreadLocalDebugInfo> parent_info_;

    friend class DebugInfoGuard;
};

// DebugInfoGuard is used to set debug information,
// ThreadLocalDebugInfo is semantically immutable, the values are set
// through the scope-based guard object.
// Nested DebugInfoGuard adds/overrides existing values in the scope,
// restoring the original values after exiting the scope.
// Users can access the values through the ThreadLocalDebugInfo::get() call;
class DebugInfoGuard {
public:
    DebugInfoGuard(DebugInfoKind kind, std::shared_ptr<DebugInfoBase> info);

    explicit DebugInfoGuard(std::shared_ptr<ThreadLocalDebugInfo> info);

    ~DebugInfoGuard();

    DebugInfoGuard(const DebugInfoGuard&) = delete;
    DebugInfoGuard(DebugInfoGuard&&) = delete;
    DebugInfoGuard& operator=(const DebugInfoGuard&) = delete;
    DebugInfoGuard& operator=(DebugInfoGuard&&) = delete;

private:
    bool active_ = false;
    std::shared_ptr<ThreadLocalDebugInfo> prev_info_ = nullptr;
};

}// namespace aethermind

#endif//AETHERMIND_UTILS_THREAD_LOCAL_DEBUG_INFO_H
