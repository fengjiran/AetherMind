# ammalloc Code Review

## 架构总评

三层缓存架构（ThreadCache → CentralCache → PageCache → PageAllocator）实现完整，设计思路清晰，深度借鉴 TCMalloc。代码注释质量极高（双语、含数学模型和设计决策原因）。`consteval` IILE 查表、TTAS SpinLock、bitmap 对象管理、4 层 Radix Tree 等核心组件实现正确且高效。

以下按严重程度排序列出发现的问题。

---

## 🔴 P0 — 违反核心约束 / 潜在崩溃

### 1. `HugePageCache` 使用了 `std::vector` — 违反自举约束

**文件**: `page_allocator.cpp:66`

```cpp
std::vector<void*> cache_;  // ← 违反 ammalloc 核心约束
```

`HugePageCache` 位于 `PageAllocator` 最底层。`std::vector` 的 `reserve(16)` / `push_back()` 会触发系统 `malloc`。如果 `am_malloc` 已通过 `LD_PRELOAD` 或 alias 替换了系统 `malloc`，这里会导致**无限递归 → 栈溢出**。

**修复建议**: 替换为定长栈数组：
```cpp
void* cache_[kMaxCacheSize];
size_t count_{0};
```
`HugePageCacheSize` 最大值由 `RuntimeConfig` 控制（默认 16），完全可以用定长数组。

---

### 2. `Span::FreeObject` 无 double-free 防护 — use_count 下溢

**文件**: `span.cpp:98-106`

```cpp
void Span::FreeObject(void* ptr) {
    // ... 直接设置 bitmap 位，无检查
    bitmap[bitmap_idx] |= (1ULL << bit_pos);
    --use_count;  // ← 如果已释放，下溢为 SIZE_MAX
}
```

`test_span.cpp:DoubleFreeCorruption` 测试已经**证实了此 bug**：double-free 导致 `use_count` 回绕到 `SIZE_MAX`，之后 `AllocObject()` 永远返回 `nullptr`（因为 `use_count >= capacity`）。

**修复建议**: 检查 bitmap 位状态：
```cpp
void Span::FreeObject(void* ptr) {
    // ...
    uint64_t bit = 1ULL << bit_pos;
    AM_DCHECK(!(bitmap[bitmap_idx] & bit), "double free detected");
    bitmap[bitmap_idx] |= bit;
    --use_count;
}
```

---

### 3. `GetOneSpan` Release 模式下的空指针解引用

**文件**: `central_cache.cpp:303-305`

```cpp
auto* span = PageCache::GetInstance().AllocSpan(page_num, size);
AM_DCHECK(span != nullptr);  // ← Release 下编译为空！
span->Init(size);             // ← OOM 时空指针解引用
```

`AM_DCHECK` 在 Release 构建（`NDEBUG`）下是空操作。如果 `AllocSpan` 返回 `nullptr`（OOM），下一行直接崩溃。

**修复建议**:
```cpp
auto* span = PageCache::GetInstance().AllocSpan(page_num, size);
if (!span) return nullptr;
span->Init(size);
```

---

## 🟠 P1 — 并发 / 正确性风险

### 4. `CreateThreadCache` 的全局互斥锁不必要

**文件**: `ammalloc.cpp:27-31`

```cpp
static std::mutex tc_init_mtx;
std::lock_guard<std::mutex> lock(tc_init_mtx);
if (pTLSThreadCache) {       // ← pTLSThreadCache 是 thread_local
    return pTLSThreadCache;   //   只有当前线程能访问它
}
```

`pTLSThreadCache` 是 `thread_local`，只有当前线程能读写。这个互斥锁**序列化了所有线程的 ThreadCache 创建**，在大量线程同时启动时造成不必要的竞争。

**修复建议**: 移除锁，或者如果意图是保护 `PageAllocator::SystemAlloc` 的并发调用，那应该在 `SystemAlloc` 内部保护。

---

### 5. `CentralCache::ReleaseListToSpans` 的 unlock-relock 窗口

**文件**: `central_cache.cpp:177-183`

```cpp
lock.unlock();
PageCache::GetInstance().ReleaseSpan(span);  // 持有 PageCache 锁
lock.lock();  // ← 重新获取 bucket 锁
// 继续处理 local_ptrs[i+1..] 中的对象
```

在 unlock/relock 窗口期间，其他线程可能：
- 将新的 Span 加入此 bucket 的 SpanList
- 修改同一 Span 的状态

虽然后续的 `PageMap::GetSpan()` 查找是 lock-free 的，但释放后剩余 `local_ptrs` 中的对象若指向已被其他线程回收/重新分配的 Span，会导致**静默数据损坏**。

当前逻辑在实践中大概率安全（因为 Span 元数据通过 ObjectPool 管理，不会真正被 free），但建议在此处添加注释或断言，确认 Span 指针在 relock 后仍然有效。

---

### 6. TLS 析构顺序与单例生命周期

**文件**: `ammalloc.cpp:50-58`

```cpp
struct ThreadCacheCleaner {
    ~ThreadCacheCleaner() {
        pTLSThreadCache->ReleaseAll();  // 调用 CentralCache::GetInstance()
        ReleaseThreadCache(pTLSThreadCache);
    }
};
```

如果主线程退出时 `CentralCache` 的静态单例已被销毁（static destruction order），`ReleaseAll()` 内部访问 `CentralCache::GetInstance()` 是 UB。

`CentralCache`、`PageCache` 都是 Meyers singleton（函数内 `static`），C++ 保证它们按构造的逆序析构。但 `thread_local` 变量的析构时机相对于函数局部 `static` 的析构顺序在标准中是**实现定义**的。

**建议**: 将 `CentralCache` / `PageCache` 改为 leaky singleton（不析构），或在 `ThreadCacheCleaner` 中添加存活检查。

---

## 🟡 P2 — 功能缺失 / 健壮性

### 7. `PageHeapScavenger` 完全未实现

**文件**: `page_heap_scavenger.cpp` — 仅 13 行，`Start()` 为空函数体

意味着归还给 PageCache 的空闲 Span **永远不会被 MADV_DONTNEED** 释放物理内存。长时间运行的服务会持续消耗 RSS，即使工作负载已下降。

`Span` 结构体中已预留了 `last_used_time_ms` 和 `is_committed` 字段（span.h:40-41），`PageCache::ReleaseSpan` 也设置了 `last_used_time_ms`（page_cache.cpp:339）。基础设施就绪，缺少的仅是扫描循环。

---

### 8. 缺少 `am_realloc` / `am_calloc` / `am_memalign`

`ammalloc.h` 仅暴露 `am_malloc` 和 `am_free`。作为 malloc 替代品，缺少：
- `am_realloc` — 必须有，否则无法替换系统 malloc
- `am_calloc` — 零初始化分配
- `am_memalign` / `am_aligned_alloc` — 对齐分配

---

### 9. `am_free` 不验证指针属于 data 区域

**文件**: `ammalloc.cpp:128`

```cpp
auto* span = PageMap::GetSpan(ptr);
```

只检查了 `ptr` 落在某个 Span 管理的页范围内，但没有验证 `ptr >= span->data_base_ptr`。如果用户误传了指向 bitmap 区域的指针，`FreeObject` 会计算出错误的 `global_obj_idx`，静默损坏 bitmap。

---

### 10. `ObjectPool` 的 `Delete` 不检查对象归属

**文件**: `page_allocator.h:125-131`

```cpp
void Delete(T* obj) {
    obj->~T();
    auto* header = reinterpret_cast<FreeHeader*>(obj);
    header->next = free_list_;
    free_list_ = header;
}
```

不验证 `obj` 是否真的由此 `ObjectPool` 分配。传入错误指针会损坏 free list。在 Debug 模式下添加范围检查会有价值。

---

## 🟢 P3 — 代码质量 / 微优化建议

### 11. `SpinLock` 自旋阈值硬编码

**文件**: `spin_lock.h:51` — `spin_cnt > 2000` 硬编码，注释已标注应作为 `RuntimeConfig` 配置项。建议兑现此 TODO。

### 12. `Span::Init` 对齐硬编码为 16

**文件**: `span.cpp:26`

```cpp
data_start = (data_start + 16 - 1) & ~(16 - 1);
```

应使用 `SystemConfig::ALIGNMENT`（也是 16，但语义更清晰）。

### 13. `FreeList` 成员初始化不一致

**文件**: `central_cache.h:22`

```cpp
constexpr FreeList() noexcept : head_(nullptr), size_(0), max_size_(1) {}
```

`head_` 类型是 `FreeBlock*`，`size_` 和 `max_size_` 是 `uint32_t`。`size_` 使用 `uint32_t` 而 `max_size()` 返回 `size_t`，在比较时会发生隐式提升。建议统一类型。

### 14. 测试中的 `thread_local ThreadCache cache` 模式

多个测试用例使用 `thread_local ThreadCache cache`。由于 GoogleTest 在同一线程中运行所有 TEST_F，多个测试共享同一个 `thread_local` 实例，可能导致测试间状态泄漏。建议在每个测试内使用栈上分配（通过 `PageAllocator::SystemAlloc` + placement new）并在结尾清理。

---

## 📊 汇总

| 严重级 | 数量 | 概览 |
|--------|------|------|
| 🔴 P0 | 3 | `std::vector` 自举违反、double-free 无防护、Release 空指针 |
| 🟠 P1 | 3 | 不必要的互斥锁、unlock-relock 窗口、TLS 析构顺序 |
| 🟡 P2 | 4 | Scavenger 未实现、缺 realloc/calloc、free 指针不验证、ObjectPool 不验证 |
| 🟢 P3 | 4 | 硬编码阈值、对齐常量、类型不一致、测试 TLS 共享 |

## 👍 亮点

- `consteval` IILE 编译期查表（`SizeClass`）— 零运行时开销，O(1) 查找
- TTAS SpinLock 实现教科书级别，内存序正确
- 4 层 Radix Tree 读路径全 lock-free，写路径受 PageCache 锁保护
- TransferCache 预取策略设计精巧——摊还 mutex 开销
- `static_assert` 广泛用于编译期验证 SizeClass 映射
- 注释质量高，含设计理由、性能分析、数学公式
