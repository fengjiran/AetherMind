# 静态 KV 布局与字节规划验证

- 日期：2026-10-29
- 范围：[08 号方案](../improvement-plan/08-kv-cache-and-attention-capability-evolution.md) 的静态 KV 纯规划子项；预算限额、资源汇总和 CPU dtype 扩展不在本批范围。
- 源码基线：`1b4eecc4` 加本轮未提交变更；没有暂存或提交。开始时已有的 CMake 空白行修改保持原样。
- 环境：Linux x86_64 / WSL、GCC 14.2.0、CMake 3.28.3。Release 使用 baseline-only 构建，Debug 和 ASAN/UBSAN 使用现有 AVX2 配置；本批未修改 kernel 或 ISA 策略。

## 1. 实现与合同

- [PlanKVCacheLayout](../../include/aethermind/kv_cache/kv_cache_layout.h) 成功路径不分配内存，统一计算 padded head、token/head/layer strides，并验证两 plane 总字节可表示。
- 元素粒度采用 `alignment / gcd(alignment, element_bytes)`，复用 checked alignment；支持完整元素的行 padding，不假定 element bytes 是 2 的幂。既有 FP32/FP16/BF16 物理布局保持一致。
- `KVCacheLayout::TotalBytes()` 复用单 plane 校验并检查乘 2 溢出，手工 layout 的合法 stride gaps 计入结果。
- `ElementBytes()` 对未初始化、scalable 或保留 lane 编码返回 0，planner/Validate 返回 InvalidArgument，不调用会触发断言的 fixed-lane 查询。诊断 OOM 返回空消息 ResourceExhausted。
- [KVCacheManager::Init](../../src/kv_cache/kv_cache_manager.cpp) 保持原参数接口，先释放旧 storage/reservation、保留 generation，再复用 planner 并分配 K/V；两块均成功才发布 storage。

物理布局测试覆盖定长多 lane dtype，不构成 CPU kernel 或整模型精度支持。没有增加预算字段、Runtime 资源统计或 Paged KV。

## 2. 新增回归

[规划测试](../../tests/unit/kv_cache/test_kv_cache_layout_planning.cpp) 共 13 个用例：

- 8 组独立常量几何，覆盖 F32/F16/BF16、多个 KV heads、padding、alignment 小于元素宽度及 3/12 字节元素；检查所有 strides、plane/total bytes 和末元素 offset。
- 零维度、非法 alignment、undefined/scalable/reserved lane；padding、token/head/layer/plane/total bytes 的溢出和可表示边界。
- 手工 layout 的 stride gaps、无效布局和两 plane 总和溢出。
- 记录型 allocator 的两次实际请求与规划一致；第二 plane 自定义 OOM 清理和重试；planner overflow 的 re-init 清旧状态及旧 view 永不复活。
- 成功规划与 TotalBytes 的 malloc-family 调用为零；planner/TotalBytes 诊断分配失败的 noexcept 状态合同。

后两项依赖 malloc interposer，已逐项标记；sanitizer 适用集合排除它们，没有将 skip 计为通过。新增分类由既有 CMake contract 验证。

## 3. 实际结果

| 配置 | KV/Layout/Manager/Runtime 聚焦集合 | 生产准备、生成与独立 HF 固定轨迹 |
|---|---:|---:|
| Release / baseline | 83/83 | 42/42 |
| Debug | 83/83 | 42/42 |
| ASAN/UBSAN/LSAN | 66/66 | 34/34，含 2 个资产 fixture |

所有列明集合均零失败、零 disabled、零 skip。Release 先执行单个 geometry 用例，再扩大集合。Debug/Release 的逐例 JSON 已核对 RUN/COMPLETED；ASAN 的 inventory 与 JUnit 使用既有严格 checker 验证发现、执行和确切名称集合一致，not_run 为 0。

ASAN 使用 `halt_on_error=1:detect_leaks=1`，UBSAN 使用 `halt_on_error=1:print_stacktrace=1`，没有新增 suppression，未出现 sanitizer 报告。普通构建负责 interposer 的真实零分配及诊断 OOM 证明。

Debug 初次构建曾误启动重复任务，中断后造成对象文件缺失；终止重复任务后顺序重建单测及 CLI 成功，表中结果均来自之后的完整产物。该构建调度失败不作为源码失败或通过证据。

## 4. 复现与产物

已构建 `AetherMind`、`aethermind_unit_tests` 和 `aethermind_infer`。以下从仓库根运行，Debug 使用 `build-debug`，Release 使用 `build/validation/validation-baseline`：

```bash
contract_filter='KVCacheLayoutPlanning*.*:KVLayoutContract.*:KVCacheManager.*:KVCacheSessionView.*:KVCacheCommitTest.*:Validation/KVCacheManagerInitFailure.*:Allocation/KVCacheManagerInitAllocationFailure.*:RuntimeBuilderTryBuild.*'
production_filter='PreparationPaths/TinyLlamaExecution.*:LoadAndPrepareExecutableModel.*:LoadAndPrepareExecutableModelPath.*:NumericalPresets/HfFixedTrajectory.*'
TMPDIR=/tmp AETHERMIND_REQUIRED_VALIDATION=1 \
  ./build/validation/validation-baseline/tests/unit/aethermind_unit_tests --gtest_filter="$contract_filter"
TMPDIR=/tmp AETHERMIND_REQUIRED_VALIDATION=1 \
  ./build/validation/validation-baseline/tests/unit/aethermind_unit_tests --gtest_filter="$production_filter"
```

ASAN 通过 CTest 的同类名称过滤并使用 `-LE '^interposer$'`；精确执行记录及过滤表达式保存在产物的 JUnit/inventory 中。

本轮原始产物位于 gitignored `build/validation/kv-layout-planning-20261029/`：构建日志、Debug/Release JSON 与日志、ASAN JUnit/inventory、`summary.json` 的源码 SHA256、`policy-delta.json` 的确切新增名称。

## 5. 冻结清单与边界

冻结策略只新增本批 13 个测试，sanitizer 适用集合排除 2 项 interposer：baseline/AVX production 为 425，ASAN production 为 329，TSAN applicable 为 251；numerical/ISA 及 CLI/benchmark 方法集合未变。

baseline 与 ASAN 的实际完整过滤 inventory 已逐项核对原集合加新增名称。AVX/TSAN 的策略增量来自审核后的无条件注册项，本轮没有执行其完整矩阵或重新声明它们通过；统一驱动后续会实际重建并校验。

本轮没有运行全量单测、完整七 profile 验收或性能 benchmark；不据此宣称预算已闭环、跨平台支持、并发支持或性能收益。08 号方案保持 In Progress。
