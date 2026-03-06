# ammalloc 全面测试与基准测试方案

## 1. 执行摘要

### 1.1 目标
构建 ammalloc 的综合测试与基准测试套件，实现以下目标：
- 验证单线程和多线程场景下的功能正确性
- 与 std::malloc、TCMalloc 和 jemalloc 进行公平的性能对比
- 测量内存效率（RSS、碎片化、页面释放行为）
- 建立可复现的方法论，用于持续回归检测

### 1.2 成功标准
- [ ] 所有正确性测试在 TSan（ThreadSanitizer）下通过
- [ ] 基准测试结果具有统计显著性（延迟测试的变异系数 CV < 5%）
- [ ] 跨分配器对比产生可操作的洞察
- [ ] CI 集成可防止 > 10% 的性能回归

### 1.3 约束与风险
| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| HugePageCache 使用 std::vector（P0 bug） | 拦截模式下崩溃 | 在全拦截测试前修复 |
| 自举递归 | 死锁 | 初始阶段实现 in_malloc 保护或坚持使用 API 模式 |
| 跨分配器污染 | 结果偏差 | 使用独立进程，用 dladdr 验证 |

---

## 2. 技术架构

### 2.1 对比模式

#### 模式 A：API 模式（主要，立即实施）
直接函数指针分派到分配器特定 API。

```cpp
template<typename AllocFn, typename FreeFn>
class AllocatorHarness {
    AllocFn alloc_;
    FreeFn free_;
public:
    void* allocate(size_t n) { return alloc_(n); }
    void deallocate(void* p) { free_(p); }
};

// 实例化
using AmmallocHarness = AllocatorHarness<am_malloc, am_free>;
using StdHarness = AllocatorHarness<std::malloc, std::free>;
using JemallocHarness = AllocatorHarness<je_malloc, je_free>;  // 或拦截时的 malloc
using TcmallocHarness = AllocatorHarness<tc_malloc, tc_free>;  // 或拦截时的 malloc
```

**优点：**
- 安全，无自举问题
- 精确控制代码路径
- 可在单个二进制文件中运行所有分配器（顺序测试）

**缺点：**
- 遗漏 C++ new/delete 路径
- "其他堆"分配仍使用系统分配器

#### 模式 B：全拦截模式（次要，未来）
在构建时将分配器链接到独立的可执行文件。

```cmake
add_executable(bench_ammalloc ...)  # 链接 ammalloc
add_executable(bench_system ...)     # 链接 libc malloc
add_executable(bench_jemalloc ...)   # 链接 -ljemalloc
add_executable(bench_tcmalloc ...)   # 链接 -ltcmalloc
```

**前置条件：**
1. 移除 HugePageCache 中的 std::vector
2. 实现递归安全的初始化
3. 提供完整的 malloc/free/calloc/realloc/posix_memalign/aligned_alloc
4. 覆盖 C++ new/delete 操作符

### 2.2 测试框架设计原则

**防污染措施：**
- 在计时前预分配所有测试结构（vector、queue）
- 尽可能使用栈上的固定大小数组
- 避免在计时区段使用 std::string、iostreams、日志
- 使用 `benchmark::DoNotOptimize()` 和 `benchmark::ClobberMemory()`

**统计严谨性：**
- 固定 RNG 种子（可复现）
- 预热阶段（填充线程缓存）
- 最少 5 次重复
- 报告：均值、标准差、CV%、95% CI

---

## 3. 测试分类

### 3.1 正确性测试（不计时）

#### 3.1.1 单线程
| 测试用例 | 描述 | 验证方式 |
|----------|------|----------|
| BasicAllocFree | 8B-32KB 所有 size class | 读写模式验证 |
| ZeroSize | malloc(0) 行为 | 跨分配器一致性 |
| NullFree | free(nullptr) | 不能崩溃 |
| DoubleFree | 检测损坏 | Bitmap 状态检查 |
| LargeAlloc | >32KB、>512KB、>1MB | 直接 mmap 路径 |
| Alignment | alignof(max_align_t) 合规性 | 指针对齐验证 |

#### 3.1.2 多线程
| 测试用例 | 线程模式 | 验证方式 |
|----------|----------|----------|
| ConcurrentAllocFree | N 线程，独立 | 无数据竞争（TSan） |
| CrossThreadFree | 在 T0 分配，在 T1 释放 | TransferCache 正确性 |
| ProducerConsumer | 环形缓冲区模式 | 无 ABA 问题 |
| BurstAllocation | 同步启动 | 可扩展性测量 |

### 3.2 性能微基准测试

#### 3.2.1 延迟测试（单线程）

| 基准测试 | 大小 | 窗口 | 测量内容 |
|----------|------|------|----------|
| Churn_FastPath | 8B、64B、256B、4KB | 1 | 纯 alloc+free 延迟 |
| Churn_TC_Steady | 8B、64B | 256 | ThreadCache 热路径 |
| Churn_CC_Pressure | 8B、64B | 1024 | CentralCache 交互 |
| Churn_Deep | 8B、64B | 2000+ | Span 分配压力 |
| RandomSize | 1B-32KB 均匀分布 | 1024 | Size class 分布 |
| LargeObject | 64KB、512KB、1MB | 8 | 直接 mmap 开销 |

#### 3.2.2 吞吐量测试（多线程）

| 基准测试 | 大小 | 批次 | 线程数 | 指标 |
|----------|------|------|--------|------|
| MT_Parallel | 8B、64B、1KB | 1000 | 1,2,4,8,16,32,64 | Ops/sec，挂钟时间 |
| MT_Contended | 64B | 1000 | 1,2,4,8,16 | 锁竞争效应 |
| MT_Random | 混合 | 1000 | 16 | 真实工作负载 |

#### 3.2.3 生命周期模式

| 模式 | 描述 | 分配器压力 |
|------|------|------------|
| LIFO | 栈顺序 | ThreadCache 最优 |
| FIFO | 队列顺序 | CentralCache 轮换 |
| Random | 随机释放顺序 | 碎片化压力 |
| Generational | 年轻/老对象 | 页面释放行为 |

### 3.3 内存效率测试

#### 3.3.1 指标

| 指标 | 定义 | 测量方式 |
|------|------|----------|
| RSS | 常驻内存集大小 | `/proc/self/statm`（Linux） |
| Peak RSS | 最大常驻内存 | `getrusage(RUSAGE_SELF)` |
| 内部碎片化 | `usable - requested` | `malloc_usable_size()` |
| 外部碎片化 | `allocated - used` | 存活字节数 / 提交字节数 |
| 页面释放率 | 随时间 RSS 减少 | 时间序列采样 |

#### 3.3.2 测试场景

```cpp
void BM_MemoryLifecycle(benchmark::State& state) {
    // 阶段 1：基线 RSS
    // 阶段 2：达到峰值（分配 N 个对象）
    // 阶段 3：碎片化（随机释放 50%）
    // 阶段 4：重新分配（尝试大分配）
    // 阶段 5：完全释放（释放所有）
    // 阶段 6：稳态（测量页面释放）
    
    state.counters["rss_baseline_mb"] = baseline;
    state.counters["rss_peak_mb"] = peak;
    state.counters["rss_final_mb"] = final;
    state.counters["fragmentation_pct"] = frag;
}
```

### 3.4 PageHeapScavenger 测试

| 测试 | 描述 |
|------|------|
| ScavengerStartup | 验证线程在首次慢路径启动 |
| IdlePageRelease | 测量 MADV_DONTNEED 有效性 |
| ReleaseLatency | 从释放到 RSS 减少的时间 |
| PressureResponse | 高负载 vs 空闲行为 |

---

## 4. 实施计划

### 4.1 阶段 1：基础（第 1 周）

**任务：**
1. [ ] 修复 HugePageCache std::vector → 固定数组
2. [ ] 实现带 RSS 跟踪的 BM_MemoryFootprint
3. [ ] 添加 jemalloc/TCMalloc API 包装器
4. [ ] 创建正确性测试套件（单线程）

**交付物：**
- `tests/benchmark/benchmark_memory_efficiency.cpp`
- `tests/unit/test_allocator_correctness.cpp`
- 用于 jemalloc/tcmalloc 检测的 CMake 集成

### 4.2 阶段 2：完整性（第 2 周）

**任务：**
1. [ ] 实现跨线程释放测试
2. [ ] 添加生命周期模式基准测试
3. [ ] 创建 PageHeapScavenger 效率测试
4. [ ] 构建自动化对比报告

**交付物：**
- `tests/benchmark/benchmark_lifecycle_patterns.cpp`
- `tests/benchmark/benchmark_page_release.cpp`
- `tools/generate_comparison_report.py`

### 4.3 阶段 3：CI 集成（第 3 周）

**任务：**
1. [ ] GitHub Actions 基准测试工作流
2. [ ] 性能回归检测
3. [ ] PR 上的自动化报告生成

**交付物：**
- `.github/workflows/benchmark.yml`
- 性能仪表板（GitHub Pages 或制品）

### 4.4 阶段 4：全拦截（未来）

**前置条件：** 自举安全的 ammalloc

**任务：**
1. [ ] 实现完整的 malloc 家族
2. [ ] 添加 new/delete 覆盖
3. [ ] 为每个分配器构建独立可执行文件
4. [ ] 实现 dladdr 验证

---

## 5. 配置标准

### 5.1 jemalloc 配置

```bash
# 高吞吐量
export MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"

# 低内存
export MALLOC_CONF="background_thread:true,tcache_max:4096,dirty_decay_ms:5000,muzzy_decay_ms:5000"
```

### 5.2 TCMalloc 配置

```bash
export TCMALLOC_RELEASE_RATE=10.0
export TCMALLOC_MAX_PER_CPU_CACHE_SIZE=33554432  # 32MB
```

### 5.3 ammalloc 配置

```bash
export AM_ENABLE_SCAVENGER=1
export AM_SCAVENGER_INTERVAL_MS=1000
export AM_TC_SIZE=32768
```

---

## 6. 数据收集与报告

### 6.1 输出格式

**JSON（原始数据）：**
```json
{
  "benchmark": "BM_Churn_8_1",
  "allocator": "ammalloc",
  "mean_ns": 3.52,
  "stddev_ns": 0.07,
  "cv_pct": 1.96,
  "rss_peak_mb": 45.2,
  "throughput_ops_per_sec": 284090909
}
```

**TSV（分析）：**
```tsv
benchmark	allocator	mean_ns	stddev_ns	cv_pct	rss_peak_mb	speedup_vs_std
BM_Churn_8_1	ammalloc	3.52	0.07	1.96	45.2	1.91
BM_Churn_8_1	std	6.71	0.07	1.04	52.1	1.00
```

**Markdown（报告）：**
```markdown
## 摘要

ammalloc 在小对象分配上比 std::malloc 快 1.91 倍。

| 指标 | ammalloc | std | jemalloc | tcmalloc |
|------|----------|-----|----------|----------|
| 延迟 (ns) | 3.52 | 6.71 | 4.12 | 3.89 |
| RSS 峰值 (MB) | 45.2 | 52.1 | 48.7 | 47.3 |
```

### 6.2 可视化

- 延迟分布（箱线图）
- 吞吐量扩展曲线（线程数 vs ops/sec）
- 内存使用随时间变化（生命周期图）
- 加速比热图（大小 vs 线程数）

---

## 7. 待解决问题

1. **平台范围**：官方数字仅限 Linux？macOS 作为次要？
2. **jemalloc 集成**：前缀 API（`je_malloc`）vs 拦截构建？
3. **CI 深度**：CI 中是否使用 PMU 计数器或仅限手动？
4. **可接受回归阈值**：5%、10% 还是大小相关？

---

## 8. 验收标准

### 8.1 功能性
- [ ] 所有正确性测试在启用 TSan 下通过
- [ ] ASan 未检测到内存泄漏
- [ ] 跨线程释放无数据竞争

### 8.2 性能
- [ ] 延迟测试基准 CV < 5%（预热后）
- [ ] 多线程测试 CV < 10%
- [ ] 性能差异具有统计显著性（p < 0.05）

### 8.3 可比性
- [ ] 所有分配器的工作负载相同
- [ ] 每个分配器的配置有文档记录
- [ ] 内存指标的基线减法

### 8.4 自动化
- [ ] CI 在每个 PR 上运行基准测试
- [ ] > 10% 减速的回归警报
- [ ] 自动化报告生成和归档

---

## 9. 附录

### 9.1 文件结构

```
tests/
├── benchmark/
│   ├── benchmark_memory_pool.cpp          # 现有
│   ├── benchmark_memory_efficiency.cpp    # 新增
│   ├── benchmark_lifecycle_patterns.cpp   # 新增
│   ├── benchmark_page_release.cpp         # 新增
│   └── CMakeLists.txt
├── unit/
│   ├── test_allocator_correctness.cpp     # 新增
│   └── ...
├── correctness/                           # 新增目录
│   ├── test_single_thread.cpp
│   ├── test_multi_thread.cpp
│   └── test_cross_thread.cpp
tools/
├── plot_ammalloc_benchmark.py             # 现有
├── generate_comparison_report.py          # 新增
└── run_benchmark_suite.sh                 # 新增
```

### 9.2 参考

1. `docs/ammalloc_benchmark_rigorous_20260303.md` - 现有方法论
2. `docs/allocator_background_thread_research.md` - jemalloc/tcmalloc 研究
3. `docs/review.md` - 已知问题（HugePageCache P0）
4. `docs/ammalloc_todo_list.md` - Bootstrap 递归修复
5. jemalloc TUNING.md: https://github.com/jemalloc/jemalloc/blob/dev/TUNING.md
6. TCMalloc Tuning: https://google.github.io/tcmalloc/tuning.html
7. Google Benchmark Best Practices: https://github.com/google/benchmark

---

*方案版本: 1.0*
*创建时间: 2026-03-06*
*状态: 决策完成（等待自审）*
