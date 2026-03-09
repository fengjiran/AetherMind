# C++ Commenting Guidelines

## Goals

Comments must improve understanding, not repeat the code.

Use comments to explain:

- intent
- assumptions
- invariants
- ownership and lifetime
- thread-safety expectations
- non-obvious tradeoffs
- performance constraints
- reasons for unusual code

Do **not** use comments to mechanically translate obvious code into English.

------

## General Rules

- Prefer fewer, higher-value comments over many low-value comments.
- Write comments in clear English.
- Keep comments accurate and update them when code changes.
- If a comment becomes misleading, fix or remove it immediately.
- Prefer explaining **why** over explaining **what**.
- Prefer explaining **constraints** over explaining **syntax**.
- Delete stale comments and commented-out code instead of keeping them.

------

## Comment Placement

### Public API declarations

Use documentation comments for public classes, structs, enums, functions, methods, and important type aliases declared in header files.

Preferred styles:

- `///`

These comments should describe:

- purpose
- parameters
- return value
- ownership/lifetime expectations when relevant
- thread-safety when relevant
- exception/error behavior when relevant
- complexity or performance guarantees when relevant

Example:

```cpp
/// Parses a configuration file and returns the loaded options.
///
/// @param path Path to the configuration file.
/// @return Parsed configuration object.
/// @throws std::runtime_error if the file cannot be read or parsed.
Config parse_config(const std::string& path);
```

------

### Internal implementation comments

Use `//` for implementation details inside `.cc` / `.cpp` files.

This is the default style for local comments.

Example:

```cpp
// Keep the previous buffer alive until all async callbacks complete.
pending_buffers_.push_back(std::move(buffer));
```

------

### Block comments

Avoid `/* ... */` for normal implementation comments.

Use block comments only for:

- Doxygen-style API docs
- copyright/license headers
- rare multi-paragraph explanations that do not fit well as line comments

------

## What to Document

### 1. Intent

Document the purpose of non-obvious code.

Good:

```cpp
// Reserve slot 0 as an invalid handle sentinel.
handles_.push_back({});
```

Bad:

```cpp
// Push back an empty element.
handles_.push_back({});
```

------

### 2. Invariants

Document invariants that must remain true.

Examples:

```cpp
// `free_list_` never contains duplicate indices.
// `head_` is null iff the queue is empty.
// Elements in `ready_` are always sorted by deadline.
```

------

### 3. Ownership and lifetime

Document ownership transfer, borrowing, and lifetime assumptions.

Examples:

```cpp
// Ownership is transferred to the scheduler.
scheduler_->enqueue(std::move(task));
// Borrowed pointer. The caller must ensure `ctx` outlives this parser.
Parser(Context* ctx);
// `view` points into `storage_`; do not mutate `storage_` while it is in use.
std::string_view current_view() const;
```

------

### 4. Thread-safety expectations

Document locking requirements and concurrency assumptions.

Examples:

```cpp
// Must be called with `mutex_` held.
void unlink_node(Node* node);
// Read-only after initialization; no synchronization required.
std::atomic<bool> started_{false};
// Safe because worker threads are joined before destruction continues.
```

------

### 5. Error handling and exceptional behavior

Document non-obvious failure modes.

Examples:

```cpp
/// Returns false on validation failure; does not throw.
bool validate(const Request& req);
// Ignore EINTR and retry because the operation is logically idempotent.
```

------

### 6. Performance-sensitive choices

Document surprising or intentionally non-ideal code when performance or scale justifies it.

Examples:

```cpp
// Linear scan is intentional here: the list is capped at <= 8 entries.
// Avoid `std::function` here to keep the hot path allocation-free.
// Keep this layout packed to improve cache locality during traversal.
```

------

### 7. Workarounds and compatibility constraints

Document platform quirks, compiler limitations, ABI constraints, file format compatibility, and protocol requirements.

Examples:

```cpp
// Keep this field order stable: it is part of the on-disk binary format.
// Work around MSVC miscompilation in version 19.3x.
```

------

## What Not to Comment

Do not comment code that is already obvious.

Bad:

```cpp
// Increment index.
++index;
// Return success.
return true;
// Constructor.
Widget::Widget() = default;
```

Do not leave commented-out code in the repository.

Bad:

```cpp
// old implementation
// process_legacy(data);
```

Use version control instead.

Do not add comments that merely restate names.

Bad:

```cpp
int timeout_ms;  // timeout in ms
```

Unless the unit or semantics are genuinely non-obvious and important.

------

## API Documentation Rules

Use documentation comments for all public APIs and important protected extension points.

Each documented API should answer the relevant subset of these questions:

- What does it do?
- What are the preconditions?
- What are the postconditions?
- Who owns the returned object or referenced memory?
- Is it thread-safe?
- Can it block?
- Can it throw? If not, how are errors reported?
- Are there complexity or performance guarantees?

Example:

```cpp
/// Inserts a value if the key is not already present.
///
/// Returns true if insertion occurred, false if the key already existed.
/// This method is not thread-safe.
/// Average complexity is O(1).
bool insert(std::string key, Value value);
```

------

## Class and Struct Comments

Document a class or struct when its role, lifecycle, invariants, or synchronization model are non-trivial.

A class comment should describe:

- its responsibility
- its ownership model
- whether it is thread-safe
- important invariants
- any usage restrictions

Example:

```cpp
/// Thread-safe bounded queue for producer-consumer workloads.
///
/// Multiple producers and consumers are supported.
/// `push` blocks when the queue is full.
/// `pop` blocks when the queue is empty.
/// Destruction requires that no threads are blocked inside the queue.
template <typename T>
class BoundedQueue {
    ...
};
```

------

## Enum Comments

Document enums when values are protocol-visible, persisted, or semantically non-obvious.

Example:

```cpp
enum class LogPolicy {
    Disabled,   ///< Never emit log records.
    ErrorsOnly, ///< Emit only error records.
    Verbose     ///< Emit all diagnostic records.
};
```

------

## Member Comments

Comment data members only when the meaning is not obvious from the name and type, or when invariants/locking rules matter.

Good:

```cpp
// Protected by `mutex_`.
std::vector<Job> pending_;
// Immutable after construction.
const std::size_t capacity_;
```

Bad:

```cpp
// Pending jobs.
std::vector<Job> pending_;
```

------

## Function Comments

Comment functions when:

- behavior is not obvious
- side effects matter
- preconditions/postconditions matter
- blocking/synchronization matters
- ownership/lifetime matters
- the algorithm is subtle
- the function intentionally violates a common expectation

Inside function bodies, comment only the tricky parts.

Preferred pattern:

```cpp
// Fast path: avoid taking the lock when the cache is already initialized.
if (ready_.load(std::memory_order_acquire)) {
    return value_;
}
```

------

## Templates and Generic Code

Document template assumptions explicitly.

Examples:

```cpp
/// `Fn` must be invocable as `bool(const T&)`.
template <class T, class Fn>
std::vector<T> filter(const std::vector<T>& input, Fn&& fn);
// Requires `T` to be nothrow-move-constructible for the strong guarantee.
```

------

## Concurrency Comments

For concurrent code, comments must clearly state:

- which mutex protects which data
- which operations may block
- whether atomics are synchronization-only or carry state meaning
- destruction requirements
- callback/thread affinity assumptions

Examples:

```cpp
// Protected by `sessions_mu_`.
std::unordered_map<SessionId, Session> sessions_;
// Called only on the IO thread.
void on_readable();
// Release store publishes initialized state to worker threads.
initialized_.store(true, std::memory_order_release);
```

------

## Memory and Resource Management Comments

For manual memory management, intrusive structures, pools, arenas, and custom allocators, comments must describe the ownership model and destruction expectations.

Examples:

```cpp
// Nodes are arena-allocated and remain valid until the arena is reset.
// `next_` is non-owning; node lifetime is managed by `List`.
Node* next_ = nullptr;
```

------

## TODO / FIXME / HACK / NOTE Policy

Use structured markers sparingly and intentionally.

### TODO

Use for planned future work that is safe to defer.

Format:

```cpp
// TODO(username): Support batched eviction once metrics are available.
```

### FIXME

Use for known incorrect behavior that should be fixed.

Format:

```cpp
// FIXME(username): This fails for empty input because the parser assumes one token.
```

### HACK

Use only for intentional ugly workarounds.

Format:

```cpp
// HACK(username): Keep the extra copy to avoid a use-after-free in the legacy callback path.
```

### NOTE

Use for important non-actionable context.

Format:

```cpp
// NOTE: This field is serialized and must remain backward-compatible.
```

Rules:

- Every TODO/FIXME/HACK should be actionable.
- Prefer linking an issue number when available.
- Remove markers once resolved.
- Do not use TODO for vague wishes.

Bad:

```cpp
// TODO: improve this
```

Good:

```cpp
// TODO(richard): Replace linear scan with indexed lookup if handle count exceeds 64.
```

------

## Language Style

- Use English for all comments unless the project explicitly requires another language.
- Write full sentences for public API docs and important design comments.
- Short inline comments may be sentence fragments if clear.
- Avoid jokes, sarcasm, and personal notes in comments.
- Avoid ambiguous words like "just", "obviously", or "simply".

------

## Formatting Rules

- Keep comments close to the code they describe.
- Keep line length consistent with project style.
- Use one space after `//`.
- Prefer sentence case.
- For multi-line `//` comments, align them consistently.

Example:

```cpp
// Keep the old state alive until the async flush completes.
// The callback may still read from `buffer`.
```

------

## Review Standard for Comments

When adding or reviewing comments, check:

- Does the comment explain something non-obvious?
- Is it still true?
- Does it describe intent, invariants, or constraints?
- Would the code be clearer if renamed/refactored instead of commented?
- Is the comment at the right level of abstraction?
- Is it consistent with nearby comments and project terminology?

If a comment only compensates for bad naming or unclear structure, prefer improving the code first.

------

## Examples

### Good examples

```cpp
// Skip index 0 because it is reserved as an invalid handle sentinel.
// Protected by `mutex_`.
std::deque<Task> pending_;
/// Returns a borrowed view into internal storage.
/// The returned view becomes invalid after the next call to `append`.
std::string_view view() const;
// Linear search is intentional: the array never grows beyond 8 elements.
```

### Bad examples

```cpp
// Add item to vector.
items.push_back(item);
// This is a mutex.
std::mutex mutex_;
// TODO: optimize
// old code
// foo();
```

------

## Summary

Use comments to document:

- why the code exists
- what assumptions it relies on
- what must remain true
- how ownership, lifetime, and concurrency work
- why a non-obvious design or performance tradeoff is justified

Do not comment obvious code.
Do not keep stale comments.
Do not use comments as a substitute for clear naming and good structure.