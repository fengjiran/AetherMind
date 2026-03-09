# C++ Commenting Guidelines V2

> AetherMind 项目专用 - 高性能 C++ 系统代码注释规范

---

## 1. 核心原则

### 1.1 注释的目标

注释必须**提升理解**，而非重复代码。

**必须解释**:
- **意图 (Intent)** - 为什么存在这段代码
- **假设 (Assumptions)** - 依赖的前置条件
- **不变量 (Invariants)** - 必须始终保持为真的条件
- **所有权与生命周期** - 谁拥有资源，何时释放
- **线程安全期望** - 并发访问规则
- **性能约束** - 热路径、复杂度保证
- **非显而易见的权衡** - 为什么选择方案 A 而非 B
- **数学/算法依据** - 算法正确性证明

**禁止**:
- ❌ 机械地将代码翻译成英语
- ❌ 显而易见的注释 (`// 递增 i`)
- ❌ 过时的注释（代码更新时未同步）
- ❌ 注释掉的代码（使用版本控制）

### 1.2 黄金法则

- **少即是多** - 高价值注释优于大量低价值注释
- **说清为什么** - 优于说什么
- **说清约束** - 优于说语法
- **立即修正** - 注释与代码不符时立刻修正或删除

---

## 2. 注释风格与位置

### 2.1 公共 API 声明（头文件）

**使用** `///` 三斜杠 Doxygen 风格。

**必需标签**（公共 API 必须包含）：

| 标签 | 说明 | 必需性 |
|------|------|--------|
| `@brief` | 简短摘要 | ✅ 必须 |
| `@param` | 参数说明（含方向 `[in]`, `[out]`, `[in,out]`） | ✅ 必须 |
| `@tparam` | 模板参数语义约束 | ✅ 模板必须 |
| `@return` | 返回值说明 | ✅ 非 void 必须 |
| `@pre` | 前置条件 | ⚠️ 非平凡时必须 |
| `@post` | 后置条件 | ⚠️ 非平凡时必须 |
| `@throws` | 异常说明 | ⚠️ 可能抛出时必须 |
| `@note` | 重要实现细节 | ⚠️ 需要时 |
| `@see` | 相关函数/文档 | ⚠️ 需要时 |

**AetherMind 专属标签**:

| 标签 | 说明 | 示例 |
|------|------|------|
| `Thread Safety:` | 并发访问规则 | `Thread Safety: 线程安全，内部同步` |
| `Blocking:` | 是否阻塞 | `Blocking: 非阻塞，自旋锁实现` |
| `Complexity:` | 时间/空间复杂度 | `Complexity: O(1) 时间，O(n) 空间` |
| `Memory Order:` | 内存序要求 | `Memory Order: 需要 acquire 语义` |

**完整示例**:

```cpp
/// 将请求大小映射到对应的 Size Class 索引。
///
/// 使用 TCMalloc 风格的分级策略：
/// - 小对象 (≤1024B)：查表 O(1)
/// - 大对象 (>1024B)：位运算 O(1)
///
/// @param size 请求的分配大小（字节）。必须 > 0。
/// @return Size Class 的零基索引；若 size > MAX_TC_SIZE 返回 SIZE_MAX
/// @pre size > 0
/// @post 返回值 ∈ [0, kNumSizeClasses) ∪ {SIZE_MAX}
/// 
/// Thread Safety: 线程安全，纯函数，无状态。
/// Complexity: O(1) 时间，O(1) 空间。
/// Memory Order: 无原子操作，纯计算。
AM_ALWAYS_INLINE constexpr static size_t Index(size_t size) noexcept;

/// 从 CentralCache 获取指定大小的 Span。
///
/// 如果 CentralCache 没有可用 Span，将从 PageCache 分配新 Span。
///
/// @param size 对象大小（字节）。必须 <= MAX_TC_SIZE。
/// @return 指向 Span 的指针；失败返回 nullptr
/// @pre size > 0 && size <= MAX_TC_SIZE
/// @post 返回值 != nullptr ⇒ Span::obj_size == size
/// @throws 不抛出；失败时返回 nullptr
///
/// Thread Safety: 线程安全，内部持有 span_list_lock_。
/// Blocking: 可能阻塞（Mutex）。
Span* GetOneSpan(size_t size) noexcept;
```

---

### 2.2 模板与概念（C++20）

**模板参数必须文档化语义约束**，而非仅类型名。

**必需**:
- `@tparam` - 每个模板参数的语义要求
- `requires` 子句的说明 - 为什么需要这些约束
- 编译期行为 - 实例化时会做什么

**示例**:

```cpp
/// 线程安全的对象池。
///
/// @tparam T 必须满足：
///   - `std::default_initializable<T>`
///   - `sizeof(T) >= sizeof(void*)`（用于 intrusive list）
///   - `noexcept(std::is_nothrow_constructible_v<T>)`
/// @tparam Capacity 池容量，必须是 2 的幂且 ≤ 65536
///
/// @note 对于非平凡析构类型，需要显式调用 Destroy()。
template<typename T, size_t Capacity>
    requires std::default_initializable<T> && 
             (Capacity > 0) && ((Capacity & (Capacity - 1)) == 0)
class ObjectPool {
    // ...
};

/// 编译期计算 Size Class 查找表。
///
/// @tparam Func 必须可调用为 `size_t(size_t)`，返回值为有效索引。
/// @tparam Size 表大小，必须是 `MAX_TC_SIZE + 1`。
///
/// @pre Func 是纯函数（无副作用，相同输入相同输出）。
/// @post 返回的数组满足：对所有 s ∈ [0, Size)，result[s] == Func(s)。
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

### 2.3 宏定义

**公共宏必须文档化**:
- 功能说明
- 参数求值次数（防副作用陷阱）
- 使用上下文要求
- 副作用警告
- ABI/API 影响

**示例**:

```cpp
/// 分支预测提示 - 标记条件很可能为真。
///
/// @note 参数 `expr` 会被求值一次。不要传入带副作用的表达式。
/// @note 仅在布尔上下文中有效。
/// @note 仅作为优化提示，不影响正确性。
///
/// @param expr 布尔表达式。
/// @return expr 的值（保持原值）。
///
/// Thread Safety: 不适用（编译时宏）。
///
/// 示例:
/// ```cpp
/// if (AM_LIKELY(ptr != nullptr)) {
///     // 热路径
/// }
/// ```
#define AM_LIKELY(expr) (__builtin_expect(!!(expr), 1))

/// 断言宏 - 检查前置条件。
///
/// @note 在 Release 构建中，失败的断言会调用 abort()。
/// @note 条件表达式会被求值一次。
/// @param condition 必须为布尔可转换。
/// @param ... 格式化消息（printf 风格）。
///
/// Thread Safety: 线程安全（内部无状态）。
#define AM_CHECK(condition, ...) \
    do { /* ... */ } while (false)
```

---

### 2.4 内部实现注释（.cpp 文件）

**使用** `//` 双斜杠。

**注释重点**:
- 复杂算法步骤
- 非显而易见的优化
- 锁定规则（必须持有什么锁）
- 临时状态说明

**示例**:

```cpp
void Span::FreeObject(void* ptr) {
    // 计算在 Span 内的偏移量
    size_t offset = static_cast<char*>(ptr) - data_base_ptr_;
    
    // 将对象标记为空闲（bitmap 对应位设为 1）
    size_t idx = offset / obj_size_;
    bitmap_[idx / 64] |= (1ULL << (idx % 64));
    
    // 递减使用计数；如果为 0，需要归还到 PageCache
    if (--use_count_ == 0) {
        // 此时必须持有 CentralCache::span_list_lock_
        PageCache::GetInstance().ReleaseSpan(this);
    }
}
```

---

## 3. 文档内容规范

### 3.1 意图（Intent）

文档化非显而易见的代码目的。

**好**:
```cpp
// Reserve slot 0 as an invalid handle sentinel.
handles_.push_back({});

// Use linear scan: the list is capped at <= 8 entries.
```

**坏**:
```cpp
// Push back an empty element.  ❌ 显而易见
handles_.push_back({});
```

---

### 3.2 不变量（Invariants）

文档化必须始终保持为真的条件。

```cpp
// Invariants:
// - `free_list_` never contains duplicate indices.
// - `head_` is null iff the queue is empty.
// - Elements in `ready_` are always sorted by deadline.
// - Span count: allocated_spans_ + free_spans_ == total_spans_
```

---

### 3.3 所有权与生命周期

```cpp
// Ownership is transferred to the scheduler.
scheduler_->enqueue(std::move(task));

// Borrowed pointer. The caller must ensure `ctx` outlives this parser.
Parser(Context* ctx);

// `view` points into `storage_`; do not mutate `storage_` while it is in use.
std::string_view current_view() const;
```

---

### 3.4 线程安全（AetherMind 关键）

**必须明确说明**:
- 哪些 mutex 保护哪些数据
- 调用时必须持有什么锁
- 原子操作的内存序意图
- 回调/线程亲和性假设

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

**内存序必须注释理由**:
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

### 3.5 复杂算法与数学

**必须包含**:
- 高层算法思想
- 循环不变量
- 误差边界（数值计算）
- 参考来源（论文/算法书）

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

### 3.6 性能敏感的权衡

**必须文档化**:
- 为什么做这种权衡
- 工作负载假设（如果相关）
- 实测性能数据（如果非直观）

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

### 3.7 文件级注释

**每个头文件**应在顶部包含：

```cpp
//
// ammalloc/size_class.h
//
// Size Class 分级系统。
//
// 负责将任意内存请求映射到固定大小的桶，控制内部碎片率 < 12.5%。
// 使用 TCMalloc 风格的分级策略：
//   - 小对象：查表 O(1)
//   - 大对象：位运算 O(1)
//
// 主要导出：
//   - SizeClass::Index() - 大小 -> 索引
//   - SizeClass::Size() - 索引 -> 大小
//   - SizeClass::RoundUp() - 向上取整
//
// Thread Safety: 所有方法都是纯函数，线程安全。
// ABI Stability: 桶大小和数量是编译期常量，不影响 ABI。
//

#ifndef AETHERMIND_AMMALLOC_SIZE_CLASS_H
#define AETHERMIND_AMMALLOC_SIZE_CLASS_H
// ...
```

---

## 4. TODO / FIXME / HACK / NOTE 政策

**结构化标记**，谨慎使用。

| 标记 | 用途 | 格式 |
|------|------|------|
| `TODO(username)` | 计划的工作，可以推迟 | `// TODO(richard): Support batched eviction once metrics are available.` |
| `FIXME(username)` | 已知的错误行为，必须修复 | `// FIXME(richard): This fails for empty input because the parser assumes one token.` |
| `HACK(username)` | 故意丑陋的变通方案 | `// HACK(richard): Keep the extra copy to avoid a use-after-free in the legacy callback path.` |
| `NOTE:` | 重要的非行动性上下文 | `// NOTE: This field is serialized and must remain backward-compatible.` |

**规则**:
- 每个 TODO/FIXME/HACK 必须是可执行的（具体、可操作）。
- 优先链接 Issue 编号：`// TODO(#123): ...`。
- 解决后立即删除标记。
- ❌ 不要用 TODO 表示模糊的愿望：`// TODO: improve this`。

---

## 5. 针对 AetherMind 的特殊规定

### 5.1 内存分配器注释

**必须注释**:
- 递归安全性（是否可能触发 malloc）
- 自举约束（是否依赖其他分配器初始化）

```cpp
// Self-bootstrapping: Uses ObjectPool for metadata allocation.
// Does NOT call am_malloc (avoids recursion).
Span* AllocSpan(size_t pages);

// Thread-safety: Lock-free. Uses thread-local storage only.
// Does not touch global state.
void* ThreadCache::Allocate(size_t size);
```

### 5.2 并发原语注释

**必须明确**:
- 锁顺序（防死锁）
- 内存序选择理由
- 进度保证（lock-free / wait-free / blocking）

```cpp
// Lock hierarchy: PageCache::mutex_ > CentralCache::span_list_lock_
// Must acquire in this order everywhere.

// Lock-free: Uses compare-and-swap loop. Wait-free for readers.
std::atomic<size_t> free_count_{0};

// Blocking: May block waiting for Span availability.
Span* WaitForSpan(size_t size);
```

---

## 6. 审查标准

添加或审查注释时，检查：

- [ ] 是否解释了非显而易见的内容？
- [ ] 是否仍然准确？
- [ ] 是否描述了意图、不变量或约束？
- [ ] 代码是否可以通过重命名/重构变得更清晰，而非添加注释？
- [ ] 注释是否处于正确的抽象层次？
- [ ] 是否与附近注释和项目术语一致？
- [ ] 公共 API 是否包含必需的 Doxygen 标签？
- [ ] 并发代码是否明确了锁和内存序？
- [ ] 性能优化是否说明了理由或证据？

如果注释只是为了弥补糟糕的命名或不清晰的结构，**优先改进代码**。

---

## 7. 示例总结

### 好示例 ✅

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

### 坏示例 ❌

```cpp
// Add item to vector.  ❌ 显而易见
items.push_back(item);

// This is a mutex.  ❌ 命名已说明
std::mutex mutex_;

// TODO: optimize  ❌ 太模糊

// old code  ❌ 删除，不要注释掉
// foo();
```

---

**文档版本**: 2.0  
**最后更新**: 2026-03-09  
**适用范围**: AetherMind C++ 项目  
**基于**: Google C++ Style Guide + 高性能系统编程最佳实践
