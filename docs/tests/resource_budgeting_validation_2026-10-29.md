# 分阶段资源预算验证

- 日期：2026-10-29
- 范围：[08 号方案](../improvement-plan/08-kv-cache-and-attention-capability-evolution.md) 的第二步：KV/workspace 限额、activation 预检、packing 大小查询、模型与请求 payload 汇总。
- 基线：`4f5c0cef` 加本轮未提交变更；本轮未暂存、未提交。
- 环境：Linux x86_64 / WSL、GCC 14.2.0、CMake 3.28.3。Release 使用 validation-baseline；Debug 使用 build-debug；ASAN/UBSAN 使用 validation-asan-ubsan。
- 使用方式与 scope：[资源预算指南](../guides/resource-budgeting.md)。

## 1. 实现边界

1. RuntimeBuilder 在获取 KV allocator 前，复用 PlanKVCacheLayout/TotalBytes 检查 max_bytes；Runtime 按值持有 workspace limit，Session 不因 enable_workspace_manager=false 绕过限额。
2. PlanExecutionBindings 冻结受控 concrete layout/metadata，不分配 activation Buffer；MaterializeExecutionBindings 验证 plan identity 后物化。旧 PrepareExecutionBindings 复用同一管线。
3. Session 以 workspace + max(prefill activation, decode activation) 检查请求 cap，保持两个 arena 顺序存活；只读 reservation 校验复用原有容量/lifecycle/epoch 规则，预算拒绝前不 reserve 或 commit。N=0 保持原行为。
4. CPU packing query 与实际 packing 共享验证和 layout 公式；batch 在任一 artifact 分配前检查查询上界，完成后核对实际 bytes/alignment。有限 cap 不接受未知 query；无限制路径兼容 legacy backend，并标记 planning completeness。
5. RawStorage 可选 footprint 描述完整 retained backing，按 shared owner 合并 view ranges。模型汇总 separately 记录 mapped/owned/external、inline constant payload 和 unique packed Buffer bytes；共享 owner 下不同 constant vector 对象分别计入。
6. max_model_payload_bytes 只约束 known owned raw + inline constants + packed；max_packed 只约束对应 batch。未知 raw metadata 不阻止仅设置 packed cap，默认无限制 Prepare 不因未被 graph 消费的特殊 raw role 新增失败。

这些是当前模型/请求的 visible tensor payload 准入，不是进程 RSS、allocator rounded capacity 或同 Runtime 多模型全局 live-byte ledger。KV cap 只约束 RuntimeBuilder，裸 Manager.Init 不自动继承。workspace 使用现有冻结 plan 的需求；没有新增 shape-dependent workspace 协议、activation liveness、Paged KV、并发或精度能力。

## 2. 新增与相关回归

新增 40 个 GTest，覆盖：

- KV limit 边界、invalid/overflow、分配/allocator creation 前拒绝、disabled KV、失败注册消费与重试；workspace limit 的值所有权与 Runtime move。
- binding planning 的精确 arena、metadata 局部生命周期、输入值更新、错 plan/moved-from 拒绝、writable external 输出和分配失败重试。
- source shared-owner/range union、未知 footprint、冲突/不足/overflow、完整 HF mmap 字节和 metadata OOM。
- identity/B-panel query 与实际 storage 一致、不可读 payload 地址的 metadata-only 查询、invalid/overflow、全 batch cap、legacy query、query/actual mismatch、共享 Buffer 去重。
- 真实 O1/plain 与 O2/packed 的请求预算边界、workspace cap、N=0、物理容量/busy 错误顺序、部分失败重试；模型/packed cap、unknown source、常量计数和 unlimited 低层 Prepare 兼容。

两个新增诊断 OOM 用例依赖 malloc interposer，已精确分类；它们在 Debug/Release 实际执行，在 sanitizer 集合排除。

| 配置 | 新增集合 | 原有相关回归 |
|---|---:|---:|
| Release / baseline | 40/40 | 359/359 |
| Debug | 40/40 | 359/359 |
| ASAN/UBSAN/LSAN | 39/39，38 个 GTest + 1 个 fixture | 326/326，324 个 GTest + 2 个 fixture |

最终列明集合零失败、零 disabled、零 skip。Debug/Release 逐例 JSON 核对 RUN/COMPLETED；ASAN 的实际 filtered inventory 与 JUnit 经既有 verify_ctest_report.py 校验确切名称、发现、执行和结果一致，not_run 为 0。CTest 分类契约和指南 C++ 示例的 syntax-only 编译均通过。

ASAN_OPTIONS 为 halt_on_error=1:detect_leaks=1；UBSAN_OPTIONS 为 halt_on_error=1:print_stacktrace=1。没有新增 suppression；最终运行没有 sanitizer 报告。原有回归包含 preparation/bindings、packing、KV/Runtime、C ABI 和独立 HF 固定轨迹，覆盖既有 Decode 零分配相关合同；不把 sanitizer 的 interposer 排除当成零分配证据。

## 3. 暴露并修复的旧 fixture 越界

首次扩大 ASAN 回归时，WeightPacking.LoweredDrivenPrepackAndResolve 报 heap-buffer-overflow。旧 fixture 只分配 512 bytes，却声明 32×8×sizeof(float)=1024 bytes 的 embedding，并将 norm views 放在该区域内。普通构建未捕获这个错误。

本批将 backing 改为 embedding + 两个 norm 的完整容量，使用不重叠、按真实字节计算的 offsets。没有放宽生产检查或 sanitizer。修复后先单例 ASAN 验证，再完整重跑表中相关集合。该用例已加入 validation 固定门禁。初次失败保留在 asan-regression.log，最终成功使用独立的 asan-regression-final.*，不覆盖原失败。

初次新增测试构建还修正了 recipe namespace 和 LoadedModel 完整类型 include；失败/后续构建日志均保留。

## 4. 命令与产物

最窄用例先执行 ExecutionBindingPlanning.PlansWithoutStorageAndMaterializesExactArena，再扩大到新增集合：

```bash
new_filter='RuntimeResourceLimits.*:ExecutionBindingPlanning*.*:ModelWeightStorageSummary*.*:CpuWeightPackingRequirements.*:PackingResourceLimits.*:ModelResourceLimits.*:InferenceResourceLimits.*'
TMPDIR=/tmp AETHERMIND_REQUIRED_VALIDATION=1 \
  ./build/validation/validation-baseline/tests/unit/aethermind_unit_tests --gtest_filter="$new_filter"
```

ASAN 使用真实 CTest 名称过滤、-LE '^interposer$'、--parallel 4 和 --output-junit；最终完整 filtered inventory 已保存。日志/JUnit 保存每个实际执行用例。

原始产物位于 gitignored build/validation/resource-budgeting-20261029/：各阶段构建日志、Debug/Release JSON、ASAN 初次失败与最终 JUnit/inventory、summary.json 的二进制/cache/策略身份、source.patch 与新增源码快照、policy-delta.json 的确切名称增量。

## 5. 冻结策略与兼容

先从保留的原始集合与第一步 13 项增量重建旧集合，验证 HEAD 的 count/SHA256；再只接受 40 个新用例和一项旧 packing 门禁提升。baseline/AVX production 为 466，ASAN production 为 368，TSAN applicable 为 290；numerical/ISA、CLI 和 benchmark 集合未变。baseline/ASAN 的实际完整 inventory 已核对；AVX/TSAN 仅按审核后的无条件注册更新，本轮不声明重新执行它们的完整矩阵。

RawStorage、Backend、Runtime 以及相关 C++ 类型增加合同或字段，直接 C++ consumer 必须统一重新编译。原函数入口保留为同一实现链的 overload，不构成稳定 C++ binary ABI 承诺。C ABI v1 的声明、struct 和默认 FP32 行为没有改变，新的 cap 目前由 C++ options 配置。

本轮没有重跑全量单测、完整七 profile 矩阵或性能 benchmark。没有吞吐收益、全进程内存硬限、其他 dtype 或并发支持声明；第三步 FP16/BF16 compatibility mode 尚未实施。
