# PageHeapScavenger Code Review

**审查日期**: 2026-03-03  
**审查文件**:
- `ammalloc/include/ammalloc/page_heap_scavenger.h`
- `ammalloc/src/page_heap_scavenger.cpp`

---

## 1. 整体架构评估

| 方面 | 状态 | 说明 |
|------|------|------|
| C++20 特性使用 | ✅ | `std::jthread`, `std::stop_token`, `std::condition_variable_any` |
| 后台线程生命周期 | ✅ | Start/Stop 接口清晰，析构自动停止 |
| 锁策略 | ⚠️ | 分段加锁正确，但有改进空间 |
| 链表操作 | ❌ | **存在 Bug** |
| 与 PageCache 集成 | ✅ | 友元关系已声明，可访问内部状态 |

---

## 2. 发现的问题

### 🔴 P0 — 严重问题

#### 1. 链表构建不完整

**位置**: `page_heap_scavenger.cpp:78-84`

```cpp
if (!head) {
    head = cur;
    tail = cur;
} else {
    tail->next = cur;  // ❌ 只设置 next
    tail = cur;        // ❌ 未设置 cur->prev = nullptr
                       // ❌ 未设置 cur->next = nullptr
}
```

**问题分析**:
- 构建临时链表时只设置 `tail->next = cur`，未重置 `cur->prev` 和 `cur->next`
- 虽然 `erase()` 已将 `cur->prev/next` 设为 `nullptr`，但代码依赖此假设
- 临时链表遍历 (`cur = cur->next`) 可能因指针混乱导致越界访问

**修复建议**:

```cpp
if (!head) {
    head = cur;
    tail = cur;
} else {
    tail->next = cur;
    tail = cur;
}
// 确保链表终止
if (tail) {
    tail->next = nullptr;  // 添加这行
}
// cur->prev 已由 erase() 设为 nullptr，但建议显式确保
cur->prev = nullptr;   // 添加这行
```

---

### 🟡 P1 — 中等问题

#### 2. 缺少 is_used 检查

**位置**: `page_heap_scavenger.cpp:71-76`

```cpp
if (!cur->is_committed) {
    cur = next;
    continue;
}

if (now - cur->last_used_time_ms >= kIdleThresholdMs) {
```

**问题**: 只检查 `is_committed`，未检查 `is_used`。理论上 Scavenger 只处理空闲 Span，但应显式断言或检查。

**修复建议**:

```cpp
// 添加检查确保只处理空闲 Span
if (cur->is_used) {
    spdlog::error("Scavenger found used span in free list!");
    cur = next;
    continue;
}
```

---

#### 3. 锁的粒度优化空间

**位置**: `page_heap_scavenger.cpp:66-88`

当前实现在锁内遍历整个链表，每次 `erase` 都修改链表结构。可考虑使用 `pop_front` 优化。

**优化建议**:

```cpp
// 更高效的实现：只处理链表头部
Span* to_process = nullptr;
{
    std::lock_guard<std::mutex> lock(page_cache.GetMutex());
    while (auto* span = page_cache.span_lists_[i].pop_front()) {
        if (span->is_committed && 
            now - span->last_used_time_ms >= kIdleThresholdMs) {
            span->next = to_process;
            to_process = span;
        } else {
            // 不符合条件，放回链表
            page_cache.span_lists_[i].push_back(span);
            break;  // 链表 LIFO，后面的更新
        }
    }
}
```

---

### 🟢 P2 — 轻微问题

#### 4. 拼写错误

**位置**: `page_heap_scavenger.cpp:93`

```cpp
void* star_ptr = cur->GetStartAddr();  // star_ptr -> start_ptr
```

---

#### 5. 时间戳更新缺失

**位置**: `page_heap_scavenger.cpp:95-98`

```cpp
if (madvise(start_ptr, size, MADV_DONTNEED) == 0) {
    cur->is_committed = false;
    // ❌ 未更新 last_used_time_ms
    release_bytes += size;
}
```

**建议**: 释放后更新时间戳，便于后续统计和调试。

---

## 3. 正确性验证

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 友元关系 | ✅ | `friend class PageHeapScavenger;` 已声明在 page_cache.h:197 |
| MADV_DONTNEED 使用 | ✅ | 正确释放物理内存，保留 VMA |
| 锁内外操作分离 | ✅ | madvise 在锁外执行，符合最佳实践 |
| 时间戳更新 | ⚠️ | 释放后未更新 `last_used_time_ms` |
| is_committed 标记 | ✅ | 释放后置为 `false` |
| 日志输出 | ✅ | 使用 spdlog 记录释放信息 |

---

## 4. 改进建议

### 建议 1: 添加统计信息

```cpp
struct ScavengerStats {
    std::atomic<uint64_t> total_released_bytes{0};
    std::atomic<uint64_t> total_scavenge_passes{0};
    std::atomic<uint64_t> total_spans_processed{0};
    std::atomic<uint64_t> total_madvise_failures{0};
};
```

### 建议 2: 自适应扫描间隔

```cpp
// 根据释放效率动态调整扫描间隔
if (release_bytes == 0) {
    // 没有释放内存，增加间隔
    sleep_ms = std::min(sleep_ms * 2, kMaxScavengeIntervalMs);
} else {
    // 有内存释放，恢复默认间隔
    sleep_ms = kScavengeIntervalMs;
}
```

### 建议 3: 配置化参数

```cpp
// RuntimeConfig 添加:
size_t scavenge_interval_ms = 1000;
size_t scavenge_idle_threshold_ms = 10000;
```

### 建议 4: 修复后的完整代码

```cpp
void PageHeapScavenger::ScavengeOnePass() {
    auto now = GetCurrentTimeMs();
    auto& page_cache = PageCache::GetInstance();
    size_t release_bytes = 0;
    size_t spans_processed = 0;

    for (size_t i = PageConfig::MAX_PAGE_NUM; i > 0; --i) {
        Span* head = nullptr;
        Span* tail = nullptr;
        
        {
            std::lock_guard<std::mutex> lock(page_cache.GetMutex());
            auto* cur = page_cache.span_lists_[i].begin();
            while (cur != page_cache.span_lists_[i].end()) {
                auto* next = cur->next;
                
                // 安全检查
                if (cur->is_used) {
                    spdlog::error("Scavenger: used span {} in free list", 
                                  static_cast<void*>(cur));
                    cur = next;
                    continue;
                }
                
                if (!cur->is_committed) {
                    cur = next;
                    continue;
                }

                if (now - cur->last_used_time_ms >= kIdleThresholdMs) {
                    page_cache.span_lists_[i].erase(cur);
                    
                    // 正确构建临时链表
                    cur->prev = nullptr;  // 确保
                    if (!head) {
                        head = cur;
                        tail = cur;
                    } else {
                        tail->next = cur;
                        tail = cur;
                    }
                    spans_processed++;
                }
                cur = next;
            }
        }

        // 确保链表终止
        if (tail) {
            tail->next = nullptr;
        }

        // 锁外执行 madvise
        auto* cur = head;
        while (cur) {
            auto* next = cur->next;
            void* start_ptr = cur->GetStartAddr();  // 修复拼写
            size_t size = cur->page_num << SystemConfig::PAGE_SHIFT;
            
            if (madvise(start_ptr, size, MADV_DONTNEED) == 0) {
                cur->is_committed = false;
                cur->last_used_time_ms = now;  // 更新时间戳
                release_bytes += size;
            } else {
                spdlog::warn("madvise MADV_DONTNEED failed for span {}", 
                             static_cast<void*>(cur));
            }
            cur = next;
        }

        // 挂回链表
        if (head) {
            std::lock_guard<std::mutex> lock(page_cache.GetMutex());
            cur = head;
            while (cur) {
                auto* next = cur->next;
                page_cache.span_lists_[i].push_back(cur);
                cur = next;
            }
        }
    }

    if (release_bytes > 0) {
        spdlog::info("Scavenger: released {} MB from {} spans", 
                     release_bytes >> 20, spans_processed);
    }
}
```

---

## 5. 结论

| 评估维度 | 评分 | 说明 |
|----------|------|------|
| 功能完整性 | ⭐⭐⭐⭐☆ | 核心功能实现正确 |
| 代码正确性 | ⭐⭐⭐☆☆ | **链表操作有 Bug** |
| 性能优化 | ⭐⭐⭐⭐☆ | 锁分离正确，可进一步优化 |
| C++20 使用 | ⭐⭐⭐⭐⭐ | jthread/stop_token 使用得当 |
| 与现有架构集成 | ⭐⭐⭐⭐⭐ | 正确使用 PageCache 友元接口 |

**当前状态**: 🟡 **需修复后使用**

**必须修复**:
1. 链表构建时设置 `tail->next = nullptr` 和 `cur->prev = nullptr`
2. 添加 `is_used` 安全检查
3. 修复 `star_ptr` 拼写错误

**建议增强**:
1. 添加统计信息
2. 参数配置化
3. 优化扫描算法
