# SizeClass 详细设计文档

> ammalloc 内存分配器的尺寸分级系统设计

---

## 1. 设计目标与背景

### 1.1 问题背景

内存分配器面临的核心挑战：**如何高效地将任意大小的内存请求映射到有限数量的固定尺寸桶中？**

- **内部碎片**: 分配的内存块大于请求的大小，造成浪费
- **外部碎片**: 空闲内存分散，无法满足大内存请求
- **管理开销**: 过多的桶数量会增加管理成本（内存、查找时间）

### 1.2 设计目标

| 目标 | 指标 | 说明 |
|------|------|------|
| **低内部碎片** | < 12.5% | 每个桶的浪费不超过 12.5% |
| **快速查找** | O(1) | 尺寸到桶索引的映射时间恒定 |
| **合理桶数量** | < 100 | 控制 ThreadCache 的 FreeList 数量 |
| **缓存友好** | 局部性好 | 常用尺寸集中在前几个桶 |

### 1.3 设计参考

本设计基于 **Google TCMalloc** 的尺寸分级策略：
- 小对象（≤128B）：线性映射，8字节对齐
- 大对象（>128B）：对数步进，每 2^k 区间分成 4 步

---

## 2. 算法设计

### 2.1 尺寸分级策略

```
┌─────────────────────────────────────────────────────────────┐
│                     尺寸分级总览                             │
├──────────────┬─────────────────┬────────────┬───────────────┤
│   范围       │    对齐粒度      │   桶数量   │    索引计算   │
├──────────────┼─────────────────┼────────────┼───────────────┤
│  1-128 B     │     8 B         │    16      │  (size-1)>>3  │
│  129-256 B   │    32 B         │     4      │  数学计算     │
│  257-512 B   │    64 B         │     4      │  数学计算     │
│  513-1024 B  │   128 B         │     4      │  数学计算     │
│  ...         │   ...           │   ...      │  数学计算     │
│  16KB-32KB   │    4 KB         │     4      │  数学计算     │
└──────────────┴─────────────────┴────────────┴───────────────┘
```

### 2.2 小对象映射（1-128B）

**策略**: 线性映射，8字节对齐

```
Size Range    Index    Calculation
──────────    ─────    ───────────
[1, 8]        0        (size-1) >> 3
[9, 16]       1        (size-1) >> 3
[17, 24]      2        (size-1) >> 3
...
[121, 128]   15        (size-1) >> 3
```

**内部碎片**: 最大 7/8 = 12.5%

**优点**: 计算简单，O(1)，分支预测友好

### 2.3 大对象映射（>128B）

**策略**: 对数步进（Logarithmic Stepped Mapping）

#### 数学模型

对于大小为 `size`（>128B）的对象：

```
1. 确定所在的 2 的幂次组：
   msb = bit_width(size - 1) - 1
   
2. 计算组索引：
   group_idx = msb - 7  （因为 2^7 = 128）
   
3. 计算该组的基础索引：
   base_idx = 16 + (group_idx << kStepShift)
   
4. 计算组内偏移：
   shift = msb - kStepShift
   group_offset = ((size - 1) >> shift) & (kStepsPerGroup - 1)
   
5. 最终索引：
   index = base_idx + group_offset
```

#### 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `kStepShift` | 2 | 每 2^2 = 4 步一个组 |
| `kStepsPerGroup` | 4 | 每个 2^k 区间分成 4 步 |
| `MAX_TC_SIZE` | 32KB | ThreadCache 处理的最大尺寸 |

#### 示例计算

**示例 1**: size = 200B

```
msb = bit_width(199) - 1 = 7

// 落在 128-256 区间（第 0 大组）
group_idx = 7 - 7 = 0
base_idx = 16 + (0 << 2) = 16

shift = 7 - 2 = 5
group_offset = (199 >> 5) & 3 = 6 & 3 = 2

index = 16 + 2 = 18
```

对应桶的最大尺寸：192B（内部碎片 4%）

**示例 2**: size = 320B

```
msb = bit_width(319) - 1 = 8

// 落在 256-512 区间（第 1 大组）
group_idx = 8 - 7 = 1
base_idx = 16 + (1 << 2) = 20

shift = 8 - 2 = 6
group_offset = (319 >> 6) & 3 = 4 & 3 = 0

index = 20 + 0 = 20
```

对应桶的最大尺寸：320B（刚好匹配）

### 2.4 内部碎片分析

#### 小对象（≤128B）

```
最大内部碎片 = 对齐粒度 - 1 = 7B
相对碎片率 = 7 / 8 = 12.5%
```

#### 大对象（>128B）

```
每组的步长 = (2^(k+1) - 2^k) / 4 = 2^(k-2)
最大内部碎片 = 步长 - 1 ≈ 2^(k-2)
相对碎片率 = 2^(k-2) / 2^k = 25%

实际最大碎片率出现在组边界：
- 129B 映射到 160B 桶：碎片率 (160-129)/129 = 24%
- 257B 映射到 320B 桶：碎片率 (320-257)/257 = 24.5%
```

**平均内部碎片率**: 约 12.5%

---

## 3. 架构设计

### 3.1 类层次结构

```
namespace aethermind {
    namespace details {
        // 内部计算函数
        constexpr size_t CalculateIndex(size_t size) noexcept;
        constexpr size_t CalculateSize(size_t idx) noexcept;
    }
    
    class SizeClass {
    public:
        // 核心接口
        static size_t Index(size_t size) noexcept;     // Size -> Index
        static size_t Size(size_t idx) noexcept;       // Index -> Size
        static size_t RoundUp(size_t size) noexcept;   // 向上取整到桶尺寸
        
        // 批量策略
        static size_t CalculateBatchSize(size_t size) noexcept;
        static size_t GetMovePageNum(size_t size) noexcept;
        
    private:
        // 编译期表
        static auto small_index_table_;  // 小对象快速查表
        static auto size_table_;         // 全尺寸查表
    };
}
```

### 3.2 分层设计 rationale

| 层级 | 职责 | 理由 |
|------|------|------|
| `details::` | 数学计算 | 分离算法实现，便于测试和验证 |
| `SizeClass::` | 公共接口 + 优化 | 封装实现细节，提供优化后的访问 |
| 编译期表 | 运行时加速 | 用空间换时间，热路径 O(1) |

### 3.3 快速路径 vs 慢速路径

```
SizeClass::Index(size)
    │
    ├── [size ≤ 128B] AM_LIKELY ──→ 查表 small_index_table_[size] ──→ O(1)
    │
    └── [size > 128B] ────────────→ 计算 details::CalculateIndex(size) ──→ O(1)
```

**设计决策**:
- 小对象（高频）走查表，消除分支和计算
- 大对象（低频）走计算，节省内存（不需要 32KB 的表）

---

## 4. 性能优化

### 4.1 编译期计算

```cpp
// 使用 C++20 consteval IIFE 生成查找表
constexpr static auto size_table_ = []() consteval {
    std::array<uint32_t, kNumSizeClasses> size_table{};
    for (size_t idx = 0; idx < kNumSizeClasses; ++idx) {
        size_table[idx] = static_cast<uint32_t>(details::CalculateSize(idx));
    }
    return size_table;
}();
```

**优势**:
- 零运行时开销
- 保证表的正确性（编译期验证）
- 缓存友好（连续内存）

### 4.2 分支预测提示

```cpp
AM_ALWAYS_INLINE constexpr static size_t Index(size_t size) noexcept {
    if (size > SizeConfig::MAX_TC_SIZE) AM_UNLIKELY {
        return std::numeric_limits<size_t>::max();
    }
    
    if (size <= SizeConfig::kSmallSizeThreshold) AM_LIKELY {
        return small_index_table_[size];
    }
    
    return details::CalculateIndex(size);
}
```

**使用**:
- `AM_LIKELY`: 标注热路径（小对象）
- `AM_UNLIKELY`: 标注冷路径（超大对象、错误）

### 4.3 强制内联

```cpp
AM_ALWAYS_INLINE static constexpr size_t Size(size_t idx) noexcept;
AM_ALWAYS_INLINE static constexpr size_t RoundUp(size_t size) noexcept;
```

**理由**: 这些函数在分配/释放热路径上被频繁调用，内联消除函数调用开销。

### 4.4 位运算优化

```cpp
// 使用 std::bit_width (C++20) 替代循环查找 MSB
int msb = std::bit_width(size - 1) - 1;

// 使用位运算替代除法/乘法
size_t page_num = (total_bytes + PAGE_SIZE - 1) >> PAGE_SHIFT;  // 代替 / PAGE_SIZE
```

---

## 5. 批量策略设计

### 5.1 CalculateBatchSize

**目标**: 确定 ThreadCache 和 CentralCache 之间一次传输多少个对象

**策略**: 反比于对象大小

```cpp
static constexpr size_t CalculateBatchSize(size_t size) noexcept {
    // 基础计算: 32KB / size
    size_t batch = SizeConfig::MAX_TC_SIZE / size;
    
    // 下限: 至少 2 个（利用缓存局部性）
    if (batch < 2) batch = 2;
    
    // 上限: 最多 512 个（防止 CentralCache 被抽空）
    if (batch > kMaxBatchSize) batch = kMaxBatchSize;
    
    return batch;
}
```

**示例**:
- 8B 对象: 32768 / 8 = 4096 → clamp 到 512
- 1KB 对象: 32768 / 1024 = 32
- 32KB 对象: 32768 / 32768 = 1 → 提升到 2

### 5.2 GetMovePageNum

**目标**: 确定 CentralCache 从 PageCache 申请多少页

**策略**: 一个 Span 满足约 8 个 batch 请求

```cpp
static constexpr size_t GetMovePageNum(size_t size) noexcept {
    size_t batch_num = CalculateBatchSize(size);
    size_t total_objs = batch_num << 3;  // 8 个 batch
    size_t total_bytes = total_objs * size;
    
    // 小对象优化: 至少 32KB（8 页）
    if (total_bytes < 32 * 1024) {
        total_bytes = 32 * 1024;
    }
    
    size_t page_num = total_bytes >> PAGE_SHIFT;
    
    // 边界检查
    if (page_num < 1) page_num = 1;
    if (page_num > PageConfig::MAX_PAGE_NUM) page_num = PageConfig::MAX_PAGE_NUM;
    
    return page_num;
}
```

**理由**: 减少 PageCache 全局锁的竞争频率

---

## 6. 验证与测试

### 6.1 编译期验证

```cpp
// 验证小对象边界
static_assert(details::CalculateSize(0) == 8);
static_assert(details::CalculateSize(15) == 128);

// 验证大对象第 0 组
static_assert(details::CalculateSize(16) == 160);  // 128 + 32
static_assert(details::CalculateSize(17) == 192);  // 160 + 32
static_assert(details::CalculateSize(19) == 256);  // 组边界

// 验证互逆性（近似）
static_assert(details::CalculateIndex(129) == 16);
static_assert(details::CalculateIndex(160) == 16);  // 同桶

// Round-trip 检查
static_assert(SizeClass::Index(SizeClass::Size(20)) == 20);
```

### 6.2 运行时测试

```cpp
TEST(SizeClassTest, ComprehensiveRoundTrip) {
    for (size_t sz = 1; sz <= SizeConfig::MAX_TC_SIZE; ++sz) {
        size_t idx = SizeClass::Index(sz);
        size_t bucket_size = SizeClass::Size(idx);
        
        // 验证: bucket_size >= sz
        EXPECT_GE(bucket_size, sz);
        
        // 验证: 前一个桶的尺寸 < sz
        if (idx > 0) {
            EXPECT_LT(SizeClass::Size(idx - 1), sz);
        }
    }
}
```

### 6.3 内部碎片统计

```cpp
TEST(SizeClassTest, FragmentationAnalysis) {
    double total_waste = 0;
    size_t total_alloc = 0;
    
    for (size_t sz = 1; sz <= 32768; ++sz) {
        size_t bucket = SizeClass::Size(SizeClass::Index(sz));
        total_waste += (bucket - sz);
        total_alloc += bucket;
    }
    
    double avg_fragmentation = total_waste / total_alloc;
    EXPECT_LT(avg_fragmentation, 0.125);  // < 12.5%
}
```

---

## 7. 使用示例

### 7.1 基本使用

```cpp
// 获取尺寸的桶索引
size_t idx = SizeClass::Index(100);     // idx = 12
size_t idx = SizeClass::Index(200);     // idx = 18

// 获取桶的最大尺寸
size_t size = SizeClass::Size(12);      // size = 104
size_t size = SizeClass::Size(18);      // size = 192

// 向上取整到桶尺寸
size_t aligned = SizeClass::RoundUp(150);  // aligned = 160
```

### 7.2 在分配器中的使用

```cpp
void* am_malloc(size_t size) {
    // 1. 获取 size class
    size_t idx = SizeClass::Index(size);
    if (idx == std::numeric_limits<size_t>::max()) {
        // 超大对象，直接 mmap
        return PageAllocator::SystemAlloc(size);
    }
    
    // 2. 对齐到桶尺寸
    size_t aligned_size = SizeClass::Size(idx);
    
    // 3. 获取 batch size（用于 ThreadCache 补充）
    size_t batch = SizeClass::CalculateBatchSize(aligned_size);
    
    // 4. 从 ThreadCache 分配...
}
```

### 7.3 在 CentralCache 中的使用

```cpp
Span* CentralCache::GetOneSpan(size_t size) {
    // 1. 计算需要多少页
    size_t page_num = SizeClass::GetMovePageNum(size);
    
    // 2. 从 PageCache 申请 Span
    Span* span = PageCache::GetInstance().AllocSpan(page_num, size);
    
    // 3. 初始化 Span...
}
```

---

## 8. 关键设计决策

### 8.1 为什么使用 8 字节对齐（小对象）？

- **硬件原因**: x86-64 架构的最小对齐要求
- **内存效率**: 8 字节是 `void*` 的大小，覆盖大多数小对象
- **碎片平衡**: 12.5% 的最大碎片是可接受的

### 8.2 为什么大对象使用 4 步/组？

- **碎片控制**: 4 步提供 25% 的粒度，平均碎片 ~12.5%
- **桶数量控制**: 32KB 范围只需 72 个桶
- **计算简单**: 4 步 = 2^2，位运算友好

### 8.3 为什么小对象用查表，大对象用计算？

| 方案 | 内存占用 | 查找时间 | 缓存友好性 |
|------|----------|----------|------------|
| 全查表（32KB） | 32KB | O(1) | 一般 |
| 全计算 | 0 | O(1) | 好 |
| **混合（当前）** | 128B + 288B | O(1) | **好** |

**权衡**: 小对象查表（消除分支），大对象计算（节省内存）

### 8.4 为什么 batch size 反比于对象大小？

- **小对象**: 传输 512 个 8B 对象 = 4KB，摊销锁开销
- **大对象**: 传输 2 个 16KB 对象 = 32KB，防止 ThreadCache 囤积
- **内存平衡**: ThreadCache 总占用约 32KB-64KB

---

## 9. 性能特征

### 9.1 时间复杂度

| 操作 | 复杂度 | 实际开销 |
|------|--------|----------|
| `Index(size ≤ 128)` | O(1) | 1 次数组访问 |
| `Index(size > 128)` | O(1) | 若干位运算 |
| `Size(idx)` | O(1) | 1 次数组访问 |
| `RoundUp(size)` | O(1) | Index + Size |

### 9.2 空间占用

| 表 | 大小 | 说明 |
|----|------|------|
| `small_index_table_` | 129 bytes | size 0-128 的索引 |
| `size_table_` | 288 bytes | 72 个 size class × 4 bytes |
| **总计** | **417 bytes** | 代码段，只读 |

### 9.3 缓存行为

- `small_index_table_`: 适合 L1 cache（129B）
- `size_table_`: 适合 L1 cache（288B）
- 无动态内存分配

---

## 10. 扩展性考虑

### 10.1 修改对齐粒度

如果要改为 16 字节对齐：

```cpp
// 修改 kSmallSizeThreshold 相关的计算
// 桶数量从 16 变为 8
// 需要重新生成 static_assert 验证
```

### 10.2 支持更大尺寸

如果要支持 64KB 对象：

```cpp
// 修改 SizeConfig::MAX_TC_SIZE
// kNumSizeClasses 会自动计算
// 重新验证 static_assert
```

### 10.3 调整步数

如果要改为 8 步/组（更细粒度）：

```cpp
// 修改 kStepShift = 3
// 桶数量增加，碎片降低
// 需要权衡 ThreadCache 内存
```

---

## 11. 参考资料

1. **TCMalloc**: `https://goog-perftools.sourceforge.net/doc/tcmalloc.html`
2. **jemalloc**: `https://jemalloc.net/`
3. **C++ 标准**: `std::bit_width` (C++20)
4. **内存分配器设计**: "Scalable Memory Allocation" - Google

---

**文档版本**: 1.0  
**最后更新**: 2026-03-07  
**设计状态**: 已实施并验证  
**相关文件**: `ammalloc/include/ammalloc/size_class.h`
