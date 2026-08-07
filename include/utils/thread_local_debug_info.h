#ifndef AETHERMIND_UTILS_THREAD_LOCAL_DEBUG_INFO_H
#define AETHERMIND_UTILS_THREAD_LOCAL_DEBUG_INFO_H

/// @file
/// @brief Scope-based storage for thread-local debug information.
///
/// Debug information is represented as an immutable linked stack. A
/// `DebugInfoGuard` installs a node for the current thread and restores the
/// previous node when its scope ends.

#include <memory>

namespace aethermind {

/// @brief Categories used to identify thread-local debug information.
enum class DebugInfoKind : uint8_t {
    /// Information associated with a producer.
    PRODUCER_INFO = 0,
    /// Information associated with the mobile runtime.
    MOBILE_RUNTIME_INFO,
    /// Profiler state.
    PROFILER_STATE,
    /// Information associated with an inference context.
    INFERENCE_CONTEXT,
    /// Parameter communication information.
    PARAM_COMMS_INFO,

    /// Test-only debug information category.
    TEST_INFO,
    /// Second test-only debug information category.
    TEST_INFO_2,
};

/// @brief Polymorphic base for values stored in the debug-information stack.
///
/// Derived types own the payload-specific state. The stack stores each value
/// through `std::shared_ptr<DebugInfoBase>` so nested guards can retain the
/// previous scope safely.
class DebugInfoBase {
public:
    DebugInfoBase() = default;
    virtual ~DebugInfoBase() = default;
};

/// @brief Provides access to the current thread's debug-information stack.
///
/// Each thread maintains an independent stack. Nodes are immutable after they
/// are pushed; nesting is represented by `parent_info_`. The class does not
/// synchronize access across threads.
class ThreadLocalDebugInfo {
public:
    /// @brief Finds the nearest value of a given category in the current stack.
    /// @param kind Category to find.
    /// @return A borrowed pointer to the nearest matching value, or null if no
    ///         value of that category is active. The pointer is valid only
    ///         while the corresponding shared ownership remains alive.
    static DebugInfoBase* get(DebugInfoKind kind);

    /// @brief Returns the current stack node for this thread.
    /// @return A shared pointer to the current node, or null when the stack is
    ///         empty.
    static std::shared_ptr<ThreadLocalDebugInfo> current();

    /// @brief Replaces the current thread's stack node.
    /// @param info New current node; null clears the stack.
    /// @note Internal operation for scope guards and state restoration. Prefer
    ///       `DebugInfoGuard` for normal use.
    static void _forceCurrentDebugInfo(std::shared_ptr<ThreadLocalDebugInfo> info);

    /// @brief Pushes a debug-information value onto the current thread's stack.
    /// @param kind Category of the value.
    /// @param info Shared ownership of the value to push.
    /// @note Internal operation for `DebugInfoGuard`.
    static void _push(DebugInfoKind kind, std::shared_ptr<DebugInfoBase> info);

    /// @brief Removes and returns the top value of the current thread's stack.
    /// @param kind Expected category of the top value.
    /// @return Shared ownership of the removed value.
    /// @note A category mismatch or an empty stack triggers `AM_CHECK` and
    ///       terminates the process; this operation does not throw.
    static std::shared_ptr<DebugInfoBase> _pop(DebugInfoKind kind);

    /// @brief Returns the top value without removing it.
    /// @param kind Expected category of the top value.
    /// @return Shared ownership of the top value.
    /// @note A category mismatch or an empty stack triggers `AM_CHECK` and
    ///       terminates the process; this operation does not throw.
    static std::shared_ptr<DebugInfoBase> _peek(DebugInfoKind kind);

private:
    DebugInfoKind kind_;
    std::shared_ptr<DebugInfoBase> info_;
    std::shared_ptr<ThreadLocalDebugInfo> parent_info_;

    friend class DebugInfoGuard;
};

/// @brief Installs thread-local debug information for a lexical scope.
///
/// A guard either pushes a new category/value pair or installs a previously
/// captured stack node. Nested guards can override values temporarily; the
/// destructor restores the exact previous stack. A null payload or stack node
/// creates an inactive guard.
/// @note A guard affects only the thread on which it is constructed and must be
///       destroyed on that same thread.
class DebugInfoGuard {
public:
    /// @brief Pushes a value for the duration of this guard's scope.
    /// @param kind Category of the value.
    /// @param info Value to expose. A null pointer creates an inactive guard.
    DebugInfoGuard(DebugInfoKind kind, std::shared_ptr<DebugInfoBase> info);

    /// @brief Installs a previously captured stack for the duration of a scope.
    /// @param info Stack node to install. A null pointer creates an inactive
    ///             guard.
    explicit DebugInfoGuard(std::shared_ptr<ThreadLocalDebugInfo> info);

    /// @brief Restores the stack that was active before construction.
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

#endif// AETHERMIND_UTILS_THREAD_LOCAL_DEBUG_INFO_H
