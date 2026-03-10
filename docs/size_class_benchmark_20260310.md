# SizeClass 性能基准测试方案与结果（2026-03-10）

## 1. 测试目标

评估 `ammalloc/include/ammalloc/size_class.h` 的核心路径性能，确认以下目标是否成立：

- `Index` / `Size` / `RoundUp` / `CalculateBatchSize` / `GetMovePageNum` 为 O(1)
- 热路径调用延迟维持在纳秒级
- 随机输入场景下结果稳定，不被常量折叠误导

---

## 2. 测试环境

- 日期：`2026-03-10`
- 平台：`Linux x86_64 (WSL2)`
- 内核：`6.6.87.2-microsoft-standard-WSL2`
- CPU（benchmark 输出）：`16 X 3686.4 MHz`
- 可执行文件：`build/tests/benchmark/aethermind_benchmark`

---

## 3. 测试方案

### 3.1 基准项

来自 `tests/benchmark/benchmark_size_class.cpp`：

- `BM_SizeClass_Index_Small`
- `BM_SizeClass_Index_Large`
- `BM_SizeClass_Size`
- `BM_SizeClass_RoundUp_Small`
- `BM_SizeClass_RoundUp_Large`
- `BM_SizeClass_CalculateBatchSize`
- `BM_SizeClass_GetMovePageNum`

### 3.2 输入策略（关键）

为避免固定循环输入导致的常量折叠/不真实优化，本轮基准改为：

- 每次迭代生成一次确定性伪随机输入（xorshift64*）
- Small 场景输入范围：`[1, 128]`
- Large 场景输入范围：`[129, SizeConfig::MAX_TC_SIZE]`
- 使用 `benchmark::DoNotOptimize(...)` 保护输出及随机状态

### 3.3 执行命令

```bash
cmake --build build --target aethermind_benchmark -j$(nproc)
./build/tests/benchmark/aethermind_benchmark \
  --benchmark_filter="BM_SizeClass.*" \
  --benchmark_min_time=0.2s \
  --benchmark_repetitions=5 \
  --benchmark_report_aggregates_only=true
```

---

## 4. 测试结果（5 次聚合）

| Benchmark | Mean CPU Time | Median | Stddev | CV |
|---|---:|---:|---:|---:|
| BM_SizeClass_Index_Small | 1.87 ns | 1.85 ns | 0.071 ns | 3.78% |
| BM_SizeClass_Index_Large | 3.37 ns | 3.41 ns | 0.167 ns | 4.96% |
| BM_SizeClass_Size | 2.13 ns | 2.11 ns | 0.069 ns | 3.24% |
| BM_SizeClass_RoundUp_Small | 1.83 ns | 1.89 ns | 0.125 ns | 6.85% |
| BM_SizeClass_RoundUp_Large | 3.18 ns | 3.15 ns | 0.055 ns | 1.73% |
| BM_SizeClass_CalculateBatchSize | 3.36 ns | 3.38 ns | 0.087 ns | 2.60% |
| BM_SizeClass_GetMovePageNum | 4.25 ns | 4.21 ns | 0.176 ns | 4.13% |

---

## 5. 结论

- 结果显示 SizeClass 核心路径在随机输入下仍保持 **纳秒级**，满足性能预期。
- 7 个基准项 CV 约在 `1.73% ~ 6.85%`，波动可接受，结果稳定。
- `Index_Large` / `RoundUp_Large` / `GetMovePageNum` 虽略高于 small 路径，但仍明显处于热路径可接受区间。

综合判断：`SizeClass` 当前实现性能达标，不构成 allocator 热路径瓶颈。

---

## 6. 复现实验建议

- 保持 `Release` 构建与相同 benchmark 参数。
- 测试前尽量降低系统负载，避免后台任务干扰。
- 若需要跨版本比较，固定 CPU 亲和性并导出 JSON 报告。
