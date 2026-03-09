# C++ Commenting Guidelines V2

> AetherMind Project - High-Performance C++ System Code Documentation Standards

---

## 1. Core Principles

### 1.1 Purpose of Comments

Comments must **enhance understanding**, not repeat the code.

**Must Explain**:
- **Intent** - Why this code exists
- **Assumptions** - Preconditions that must hold
- **Invariants** - Conditions that must always remain true
- **Ownership and Lifetime** - Who owns resources and when they are released
- **Thread-Safety Expectations** - Rules for concurrent access
- **Performance Constraints** - Hot paths, complexity guarantees
- **Non-Obvious Tradeoffs** - Why choose solution A over B
- **Mathematical/Algorithmic Rationale** - Proof of correctness

**Prohibited**:
- ❌ Mechanically translating code into English
- ❌ Obvious comments (`// increment i`)
- ❌ Stale comments (not updated when code changes)
- ❌ Commented-out code (use version control)

### 1.2 Golden Rules

- **Less is More** - High-value comments over many low-value comments
- **Explain Why** - Over explaining what
- **Explain Constraints** - Over explaining syntax
- **Fix Immediately** - Correct or remove comments when they become inaccurate

---

## 2. Comment Styles and Placement

### 2.1 Public API Declarations (Header Files)

**Use** `///` triple-slash Doxygen style.

**Required Tags** (must include for public APIs):

| Tag | Description | Required |
|-----|-------------|----------|
| `@brief` | Short summary | ✅ Required |
| `@param` | Parameter description (with direction `[in]`, `[out]`, `[in,out]`) | ✅ Required |
| `@tparam` | Template parameter semantic constraints | ✅ Required for templates |
| `@return` | Return value description | ✅ Required for non-void |
| `@pre` | Preconditions | ⚠️ Required for non-trivial |
| `@post` | Postconditions | ⚠️ Required for non-trivial |
| `@throws` | Exception specification | ⚠️ Required if may throw |
| `@note` | Important implementation details | ⚠️ When needed |
| `@see` | Related functions/documentation | ⚠️ When needed |

**AetherMind-Specific Tags**:

| Tag | Description | Example |
|-----|-------------|---------|
| `Thread Safety:` | Concurrent access rules | `Thread Safety: Thread-safe, internally synchronized.` |
| `Blocking:` | Blocking behavior | `Blocking: Non-blocking, spinlock-based.` |
| `Complexity:` | Time/space complexity | `Complexity: O(1) time, O(n) space.` |
| `Memory Order:` | Memory ordering requirements | `Memory Order: Requires acquire semantics.` |

**Complete Example**:

```cpp
/// Maps a requested size to its corresponding Size Class index.
///
/// Uses TCMalloc-style size classing:
/// - Small objects (≤1024B): Table lookup O(1)
/// - Large objects (>1024B): Bitwise operations O(1)
///
/// @param size Requested allocation size in bytes. Must be > 0.
/// @return Zero-based Size Class index; returns SIZE_MAX if size > MAX_TC_SIZE
/// @pre size > 0
/// @post Return value ∈ [0, kNumSizeClasses) ∪ {SIZE_MAX}
///
/// Thread Safety: Thread-safe, pure function, no state.
/// Complexity: O(1) time, O(1) space.
/// Memory Order: No atomic operations, pure computation.
AM_ALWAYS_INLINE constexpr static size_t Index(size_t size) noexcept;

/// Retrieves a Span of the specified size from CentralCache.
///
/// If CentralCache has no available Span, allocates a new Span from PageCache.
///
/// @param size Object size in bytes. Must be <= MAX_TC_SIZE.
/// @return Pointer to Span; returns nullptr on failure
/// @pre size > 0 && size <= MAX_TC_SIZE
/// @post Return value != nullptr ⇒ Span::obj_size == size
/// @throws Does not throw; returns nullptr on failure
///
/// Thread Safety: Thread-safe, internally holds span_list_lock_.
/// Blocking: May block (Mutex).
Span* GetOneSpan(size_t size) noexcept;
```

---

### 2.2 Templates and Concepts (C++20)

**Template parameters must document semantic constraints**, not just type names.

**Required**:
- `@tparam` - Semantic requirements for each template parameter
- `requires` clause explanation - Why these constraints are needed
- Compile-time behavior - What happens during instantiation

**Example**:

```cpp
/// Thread-safe object pool.
///
/// @tparam T Must satisfy:
///   - `std::default_initializable<T>`
///   - `sizeof(T) >= sizeof(void*)` (for intrusive list)
///   - `noexcept(std::is_nothrow_constructible_v<T>)`
/// @tparam Capacity Pool capacity, must be power of 2 and ≤ 65536
///
/// @note For non-trivially-destructible types, explicit Destroy() call is required.
template<typename T, size_t Capacity>
    requires std::default_initializable<T> && 
             (Capacity > 0) && ((Capacity & (Capacity - 1)) == 0)
class ObjectPool {
    // ...
};

/// Compile-time computation of Size Class lookup table.
///
/// @tparam Func Must be callable as `size_t(size_t)`, return value is valid index.
/// @tparam Size Table size, must be `MAX_TC_SIZE + 1`.
///
/// @pre Func is a pure function (no side effects, same input → same output).
/// @post Returned array satisfies: for all s ∈ [0, Size), result[s] == Func(s).
template<typename Func, size_t Size>
consteval auto GenerateLookupTable(Func&& fn) {
    std::array<uint8_t, Size> table{};
    for (size_t i = 0; i < Size; ++i) {
        table[i] = static_cast<uint8_t>(fn(i));
    }
    return table;
}
```

---

### 2.3 Macro Definitions

**Public macros must document**:
- Functionality description
- Parameter evaluation count (prevent side-effect traps)
- Usage context requirements
- Side-effect warnings
- ABI/API impact

**Example**:

```cpp
/// Branch prediction hint - marks condition as likely true.
///
/// @note Parameter `expr` is evaluated exactly once. Do not pass expressions with side effects.
/// @note Only valid in boolean contexts.
/// @note Used only as optimization hint, does not affect correctness.
///
/// @param expr Boolean expression.
/// @return Value of expr (preserved).
///
/// Thread Safety: N/A (compile-time macro).
///
/// Example:
/// ```cpp
/// if (AM_LIKELY(ptr != nullptr)) {
///     // Hot path
/// }
/// ```
#define AM_LIKELY(expr) (__builtin_expect(!!(expr), 1))

/// Assertion macro - checks preconditions.
///
/// @note In Release builds, failed assertions call abort().
/// @note Conditional expression is evaluated exactly once.
/// @param condition Must be boolean-convertible.
/// @param ... Format message (printf-style).
///
/// Thread Safety: Thread-safe (no internal state).
#define AM_CHECK(condition, ...) \
    do { /* ... */ } while (false)
```

---

### 2.4 Internal Implementation Comments (.cpp files)

**Use** `//` double-slash.

**Comment Focus**:
- Complex algorithm steps
- Non-obvious optimizations
- Locking rules (what locks must be held)
- Temporary state explanations

**Example**:

```cpp
void Span::FreeObject(void* ptr) {
    // Calculate offset within the Span
    size_t offset = static_cast<char*>(ptr) - data_base_ptr_;
    
    // Mark object as free (set corresponding bit in bitmap to 1)
    size_t idx = offset / obj_size_;
    bitmap_[idx / 64] |= (1ULL << (idx % 64));
    
    // Decrement use count; if zero, return to PageCache
    if (--use_count_ == 0) {
        // At this point must hold CentralCache::span_list_lock_
        PageCache::GetInstance().ReleaseSpan(this);
    }
}
```

---

## 3. Documentation Content Standards

### 3.1 Intent

Document the purpose of non-obvious code.

**Good**:
```cpp
// Reserve slot 0 as an invalid handle sentinel.
handles_.push_back({});

// Use linear scan: the list is capped at <= 8 entries.
```

**Bad**:
```cpp
// Push back an empty element.  ❌ Obvious
handles_.push_back({});
```

---

### 3.2 Invariants

Document conditions that must always remain true.

```cpp
// Invariants:
// - `free_list_` never contains duplicate indices.
// - `head_` is null iff the queue is empty.
// - Elements in `ready_` are always sorted by deadline.
// - Span count: allocated_spans_ + free_spans_ == total_spans_
```

---

### 3.3 Ownership and Lifetime

```cpp
// Ownership is transferred to the scheduler.
scheduler_->enqueue(std::move(task));

// Borrowed pointer. The caller must ensure `ctx` outlives this parser.
Parser(Context* ctx);

// `view` points into `storage_`; do not mutate `storage_` while it is in use.
std::string_view current_view() const;
```

---

### 3.4 Thread Safety (Critical for AetherMind)

**Must Explicitly State**:
- Which mutex protects which data
- What locks must be held on entry
- Memory order rationale for atomic operations
- Callback/thread affinity assumptions

```cpp
// Protected by `sessions_mu_`.
std::unordered_map<SessionId, Session> sessions_;

// Called only on the IO thread.
void on_readable();

// Release store publishes initialized state to worker threads.
initialized_.store(true, std::memory_order_release);

// Lock ordering: must acquire mutex_a_ before mutex_b_
void transfer_resources(Resource* from, Resource* to);
```

**Memory Order Must Document Rationale**:
```cpp
// Acquire: synchronize-with the initialization thread's release store.
// This ensures we see all writes made before initialized_ = true.
if (initialized_.load(std::memory_order_acquire)) {
    use_data();
}

// Relaxed: only a counter, no synchronization needed.
counter_.fetch_add(1, std::memory_order_relaxed);
```

---

### 3.5 Complex Algorithms and Mathematics

**Must Include**:
- High-level algorithm idea
- Loop invariants
- Error bounds (for numerical computations)
- Reference sources (papers/algorithm books)

```cpp
// Calculates the size class index using logarithmic stepped mapping.
//
// Algorithm:
//   1. Find MSB of (size - 1) using std::bit_width (BSR instruction).
//   2. Group by power-of-2 intervals (128-256, 256-512, ...).
//   3. Each group divided into 4 steps (kStepsPerGroup = 4).
//
// Loop invariant: For each size s, the returned idx satisfies
//   Size(idx - 1) < s ≤ Size(idx)
//
// Complexity: O(1) time, O(1) space.
// Reference: TCMalloc size class algorithm (https://...)
constexpr size_t CalculateIndex(size_t size) noexcept {
    // ...
}

// Monotonic stack invariant: indices in `stack_` are strictly increasing
// and corresponding values are strictly decreasing. This guarantees O(n)
// total pops across the full scan.
//
// Proof: Each element is pushed once and popped at most once.
for (size_t i = 0; i < n; ++i) {
    while (!stack_.empty() && values_[stack_.back()] < values_[i]) {
        stack_.pop_back();
    }
    stack_.push_back(i);
}
```

---

### 3.6 Performance-Sensitive Tradeoffs

**Must Document**:
- Why this tradeoff is made
- Workload assumptions (if relevant)
- Measured performance data (if non-obvious)

```cpp
// Linear scan is intentional here: the list is capped at <= 8 entries
// (benchmarked: binary search is slower for n <= 8 due to branch misprediction).

// Avoid std::function here to keep the hot path allocation-free.
// Benchmark: 15ns vs 45ns for std::function.

// Keep this layout packed (no padding between fields) to improve
// cache locality during traversal. Measured: 12% improvement on
// BM_IterateSpans.
struct alignas(64) Bucket {
    SpinLock lock;
    TransferCache tc;
    SpanList spans;
};
```

---

### 3.7 File-Level Comments

**Every header file** should include at the top:

```cpp
//
// ammalloc/size_class.h
//
// Size Class grading system.
//
// Responsible for mapping arbitrary memory requests to fixed-size buckets,
// controlling internal fragmentation to < 12.5%.
// Uses TCMalloc-style size classing:
//   - Small objects: Table lookup O(1)
//   - Large objects: Bitwise operations O(1)
//
// Primary exports:
//   - SizeClass::Index() - Size → Index
//   - SizeClass::Size() - Index → Size
//   - SizeClass::RoundUp() - Round up to bucket size
//
// Thread Safety: All methods are pure functions, thread-safe.
// ABI Stability: Bucket sizes and counts are compile-time constants,
//                do not affect ABI.
//

#ifndef AETHERMIND_AMMALLOC_SIZE_CLASS_H
#define AETHERMIND_AMMALLOC_SIZE_CLASS_H
// ...
```

---

## 4. TODO / FIXME / HACK / NOTE Policy

**Structured markers**, use sparingly and intentionally.

| Marker | Purpose | Format |
|--------|---------|--------|
| `TODO(username)` | Planned work, can be deferred | `// TODO(richard): Support batched eviction once metrics are available.` |
| `FIXME(username)` | Known incorrect behavior, must fix | `// FIXME(richard): This fails for empty input because the parser assumes one token.` |
| `HACK(username)` | Intentionally ugly workaround | `// HACK(richard): Keep the extra copy to avoid a use-after-free in the legacy callback path.` |
| `NOTE:` | Important non-actionable context | `// NOTE: This field is serialized and must remain backward-compatible.` |

**Rules**:
- Every TODO/FIXME/HACK must be actionable (specific, concrete).
- Prefer linking Issue numbers: `// TODO(#123): ...`.
- Remove markers immediately once resolved.
- ❌ Don't use TODO for vague wishes: `// TODO: improve this`.

---

## 5. AetherMind-Specific Requirements

### 5.1 Memory Allocator Comments

**Must Document**:
- Recursion safety (whether may trigger malloc)
- Bootstrapping constraints (whether depends on other allocator initialization)

```cpp
// Self-bootstrapping: Uses ObjectPool for metadata allocation.
// Does NOT call am_malloc (avoids recursion).
Span* AllocSpan(size_t pages);

// Thread-safety: Lock-free. Uses thread-local storage only.
// Does not touch global state.
void* ThreadCache::Allocate(size_t size);
```

### 5.2 Concurrency Primitive Comments

**Must Explicitly State**:
- Lock ordering (prevent deadlocks)
- Memory order selection rationale
- Progress guarantees (lock-free / wait-free / blocking)

```cpp
// Lock hierarchy: PageCache::mutex_ > CentralCache::span_list_lock_
// Must acquire in this order everywhere.

// Lock-free: Uses compare-and-swap loop. Wait-free for readers.
std::atomic<size_t> free_count_{0};

// Blocking: May block waiting for Span availability.
Span* WaitForSpan(size_t size);
```

---

## 6. Review Checklist

When adding or reviewing comments, check:

- [ ] Does it explain something non-obvious?
- [ ] Is it still accurate?
- [ ] Does it describe intent, invariants, or constraints?
- [ ] Could the code be made clearer through renaming/refactoring instead of commenting?
- [ ] Is the comment at the right level of abstraction?
- [ ] Is it consistent with nearby comments and project terminology?
- [ ] Does the public API include required Doxygen tags?
- [ ] Does concurrent code clearly specify locks and memory order?
- [ ] Does the performance optimization explain rationale or evidence?

If a comment only compensates for bad naming or unclear structure, **prefer improving the code first**.

---

## 7. Example Summary

### Good Examples ✅

```cpp
// Skip index 0 because it is reserved as an invalid handle sentinel.
handles_.push_back({});

// Protected by `mutex_`.
std::deque<Task> pending_;

/// Returns a borrowed view into internal storage.
/// The returned view becomes invalid after the next call to `append`.
std::string_view view() const;

// Linear search is intentional: the array never grows beyond 8 elements.
```

### Bad Examples ❌

```cpp
// Add item to vector.  ❌ Obvious
items.push_back(item);

// This is a mutex.  ❌ Naming already explains
std::mutex mutex_;

// TODO: optimize  ❌ Too vague

// old code  ❌ Delete, don't comment out
// foo();
```

---

**Document Version**: 2.0  
**Last Updated**: 2026-03-09  
**Scope**: AetherMind C++ Project  
**Based on**: Google C++ Style Guide + High-Performance Systems Programming Best Practices
