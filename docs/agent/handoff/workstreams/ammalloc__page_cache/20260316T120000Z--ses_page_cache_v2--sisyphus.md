---
kind: handoff
schema_version: "1.1"
created_at: 2026-03-16T12:00:00Z
session_id: ses_page_cache_v2
task_id: task_span_v2_refactor
module: ammalloc
submodule: page_cache
slug: null
agent: sisyphus
status: active
memory_status: not_needed
supersedes: null
closed_at: null
closed_reason: null
---

# PageCache Span v2 64B 重构 Handoff

## 目标

将 `Span` 结构体从 **112 字节压缩到 64 字节（单缓存行）**，消除 False Sharing，减少 Cache Miss，提升多线程并发性能。

## 当前状态

**阶段**: 设计审核通过，等待实施
**阻塞点**: 需用户显式批准后方可编码（Per `ammalloc/AGENTS.md` 硬性约束）

### 已完成工作

1. **代码审查** (`docs/reviews/code_review/20260315_page_cache_code_review.md`)
   - 修复 P0 问题：`PageMap::GetSpan` 缺少 `i0` 边界检查

2. **测试用例补充** (13/13 通过)
   - 增量测试 5 个 + 失败路径测试 3 个
   - 全部 `PageCacheTest.*` 通过

3. **性能基准测试**
   - `benchmark_page_cache.cpp` 已修复段错误
   - `compare_benchmark_json.py` 对比脚本
   - 覆盖：单线程、多线程竞争、PageMap 查询

4. **架构分析决策**
   - ✅ HugePageCache 不加 CPUPause
   - ✅ In-band bitmap 性能中性
   - ✅ 拒绝 bitfield+union 方案（实现相关、RMW 隐患）
   - ✅ 确定字段精简方案

## 涉及文件

### 需修改的源文件

| 文件路径 | 变更类型 | 关键修改点 |
|----------|----------|-----------|
| `ammalloc/include/ammalloc/span.h` | 重写 | Span 结构体 112B→64B；添加 GetBitmap()/GetDataBasePtr()/GetBitmapNum()；flags 位域替代 is_used/is_committed |
| `ammalloc/src/span.cpp` | 修改 | Init(): bitmap→GetBitmap()；计算并设置 obj_offset；AllocObject()/FreeObject(): data_base_ptr→GetDataBasePtr() |
| `ammalloc/src/central_cache.cpp` | 修改 | ReleaseListToSpans(): bitmap/data_base_ptr 清理 → ResetObjectMetadata()；Reset(): 同上 |
| `ammalloc/src/page_cache.cpp` | 修改 | is_used → IsUsed()/SetUsed()；is_committed → IsCommitted()/SetCommitted() |
| `ammalloc/src/page_heap_scavenger.cpp` | 修改 | is_used → IsUsed()；is_committed → IsCommitted() |
| `ammalloc/src/ammalloc.cpp` | 修改 | obj_size 类型 size_t→uint32_t（注意符号扩展） |
| `tests/unit/test_page_cache.cpp` | 修改 | is_used → IsUsed()；is_committed → IsCommitted() |

## 已确认接口与不变量

### Span v2 64B 最终布局（已批准）

```cpp
struct alignas(SystemConfig::CACHE_LINE_SIZE) Span {
    // 1. 链表指针 (16B)
    Span* next{nullptr};            // 8B
    Span* prev{nullptr};            // 8B
    
    // 2. 核心寻址与状态 (16B)
    uint64_t start_page_idx{0};     // 8B: 保持全宽度，支持 sentinel (max)
    uint32_t page_num{0};           // 4B: 最大 40 亿页
    uint16_t flags{0};              // 2B: is_used, is_committed 等标志位
    uint16_t reserved_{0};          // 2B: 预留（size_class 已删除）
    
    // 3. 对象分配元数据 (16B)
    uint32_t obj_size{0};           // 4B: 最大 4GB 对象
    uint32_t capacity{0};           // 4B: 最大 40 亿个对象
    uint32_t use_count{0};          // 4B: 当前使用量
    uint32_t scan_cursor{0};        // 4B: 扫描游标
    
    // 4. 杂项与冷数据 (16B)
    uint32_t obj_offset{0};         // 4B: 替代 data_base_ptr
    uint32_t padding{0};            // 4B: 对齐填充
    uint64_t last_used_time_ms{0};  // 8B: Scavenger 使用的时间戳
    // total = 64B

    // 标志位定义
    enum FlagBits : uint16_t {
        kUsed      = 1u << 0,
        kCommitted = 1u << 1,
    };

    // 内联计算方法
    [[nodiscard]] AM_ALWAYS_INLINE void* GetPageBaseAddr() const noexcept {
        return reinterpret_cast<void*>(start_page_idx << SystemConfig::PAGE_SHIFT);
    }

    [[nodiscard]] AM_ALWAYS_INLINE uint64_t* GetBitmap() const noexcept {
        return static_cast<uint64_t*>(GetPageBaseAddr());
    }

    [[nodiscard]] AM_ALWAYS_INLINE size_t GetBitmapNum() const noexcept {
        return (capacity + 63) >> 6;
    }

    [[nodiscard]] AM_ALWAYS_INLINE void* GetDataBasePtr() const noexcept {
        return static_cast<char*>(GetPageBaseAddr()) + obj_offset;
    }

    // 标志位访问器
    [[nodiscard]] AM_ALWAYS_INLINE bool IsUsed() const noexcept {
        return flags & kUsed;
    }
    AM_ALWAYS_INLINE void SetUsed(bool v) noexcept {
        if (v) flags |= kUsed; else flags &= ~kUsed;
    }

    [[nodiscard]] AM_ALWAYS_INLINE bool IsCommitted() const noexcept {
        return flags & kCommitted;
    }
    AM_ALWAYS_INLINE void SetCommitted(bool v) noexcept {
        if (v) flags |= kCommitted; else flags &= ~kCommitted;
    }

    // Cleanup helper
    AM_ALWAYS_INLINE void ResetObjectMetadata() noexcept {
        obj_offset = 0;
    }
};
```

### 字段迁移对照表

| 原字段 | 新状态 | 替代方式 |
|--------|--------|----------|
| `bitmap` (8B) | 删除 | `GetBitmap()` 内联计算 |
| `data_base_ptr` (8B) | 删除 | `GetDataBasePtr()` PageBase + obj_offset |
| `bitmap_num` (8B) | 删除 | `GetBitmapNum()` 计算 |
| `is_used` | 删除 | `flags & kUsed` |
| `is_committed` | 删除 | `flags & kCommitted` |
| `size_t page_num` | 降级 | `uint32_t` |
| `size_t obj_size` | 降级 | `uint32_t` |
| `size_t capacity` | 降级 | `uint32_t` |
| `size_t use_count` | 降级 | `uint32_t` |
| `size_t scan_cursor` | 降级 | `uint32_t` |

### 关键调用点迁移

```cpp
// span->bitmap          → span->GetBitmap()
// span->data_base_ptr   → span->GetDataBasePtr()
// span->bitmap_num      → span->GetBitmapNum()
// span->is_used = true  → span->SetUsed(true)
// if (span->is_used)    → if (span->IsUsed())
// span->is_committed    → span->IsCommitted()/SetCommitted()
// span->bitmap = nullptr → // 无需操作（计算得出）
// span->data_base_ptr = nullptr → span->ResetObjectMetadata()
```

## 阻塞点

**必须获得用户显式批准后方可开始编码**（Per `ammalloc/AGENTS.md` 硬性约束："如果需要生成或者修改源文件，先给出具体的实现方案供我审核，审核通过之后再开始写代码"）

当前状态：**等待用户说"继续"、"批准实施"或"是"**

## 推荐下一步

用户批准后，按以下顺序实施：

1. **修改 `span.h`**（核心）
   - 重写 Span 结构体为 64B 布局
   - 添加 GetBitmap/GetDataBasePtr/GetBitmapNum 方法
   - 添加 IsUsed/SetUsed/IsCommitted/SetCommitted/ResetObjectMetadata 方法

2. **修改 `span.cpp`**
   - `Init()`: 使用 GetBitmap() 替代 bitmap，计算并设置 obj_offset
   - `AllocObject()`: GetBitmap() + GetDataBasePtr()
   - `FreeObject()`: GetBitmap() + GetDataBasePtr()

3. **修改 `central_cache.cpp`**
   - `ReleaseListToSpans()`: bitmap/data_base_ptr = nullptr → ResetObjectMetadata()
   - `Reset()`: 同上

4. **修改 `page_cache.cpp`**
   - 所有 `is_used` → `IsUsed()`/`SetUsed()`
   - 所有 `is_committed` → `IsCommitted()`/`SetCommitted()`

5. **修改 `page_heap_scavenger.cpp`**
   - `is_used`/`is_committed` → 访问器方法

6. **修改 `ammalloc.cpp`**
   - `span->obj_size` 类型适配（size_t → uint32_t）

7. **修改 `test_page_cache.cpp`**
   - 测试中断言适配新访问器

8. **验证**
   - 构建：`cmake --build build --target ammalloc aethermind_unit_tests -j`
   - 单元测试：`./build/tests/unit/aethermind_unit_tests --gtest_filter=PageCache.*`
   - 基准测试：`./build/tests/benchmark/aethermind_benchmark --benchmark_filter="BM_PageCache.*|BM_PageMap.*"`
   - 对比：`python3 tools/compare_benchmark_json.py span_v1.json span_v2.json`
   - TSAN：`cmake -S . -B build-tsan -DENABLE_TSAN=ON && cmake --build build-tsan --target aethermind_unit_tests && ./build-tsan/tests/unit/aethermind_unit_tests --gtest_filter=PageCache.*`

## 验证方式

### 构建命令
```bash
cmake --build build --target ammalloc aethermind_unit_tests -j
```

### 单元测试
```bash
./build/tests/unit/aethermind_unit_tests --gtest_filter=PageCache.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=CentralCache.*
./build/tests/unit/aethermind_unit_tests --gtest_filter=ThreadCache.*
```

### 性能基准（回归门禁）
```bash
./build/tests/benchmark/aethermind_benchmark --benchmark_out=span_v2.json \
    --benchmark_filter="BM_PageCache.*|BM_PageMap.*"
python3 tools/compare_benchmark_json.py span_v1.json span_v2.json
```
**门禁**: 单线程退化 >5% 或多线程 >8% 视为失败

### TSAN 验证
```bash
cmake -S . -B build-tsan -DENABLE_TSAN=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF
cmake --build build-tsan --target aethermind_unit_tests -j
./build-tsan/tests/unit/aethermind_unit_tests --gtest_filter=PageCache.*
```

## 参考

- **代码审查报告**: `docs/reviews/code_review/20260315_page_cache_code_review.md`
- **模块约束**: `ammalloc/AGENTS.md`
- **注释规范**: `docs/guides/cpp_comment_guidelines.md`
- **产品需求**: `docs/products/aethermind_prd.md`
