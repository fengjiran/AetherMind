//
// Created by 赵丹 on 2025/8/27.
//

#ifndef AETHERMIND_THREAD_LOCAL_DEBUG_INFO_H
#define AETHERMIND_THREAD_LOCAL_DEBUG_INFO_H

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

private:
    DebugInfoKind kind_;
    std::shared_ptr<DebugInfoBase> info_;
    std::shared_ptr<ThreadLocalDebugInfo> parent_info_;
};

}// namespace aethermind

#endif//AETHERMIND_THREAD_LOCAL_DEBUG_INFO_H
