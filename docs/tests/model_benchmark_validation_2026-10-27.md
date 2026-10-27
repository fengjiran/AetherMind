# 模型 benchmark 第五批基线验收

- **状态**: Current
- **日期**: 2026-10-27
- **关联方案**: [13 号方案 Q4](../improvement-plan/13-deepinfer-quality-and-validation.md)、[算子工作流](../guides/operator-development-workflow.md)
- **执行基线**: `ff2f94797033938e8e5d7c2e39810904142f67fd` 加第三、第四及本批工作树补丁；dirty，未暂存、未提交。
- **机器观测**: [desktop-54h5mmi 模型记录](benchmarks/models/desktop-54h5mmi-model.md)

## 1. 实现范围与职责

新增 Linux 开发工具 [model_benchmark.cpp](../../tests/benchmark/models/model_benchmark.cpp)、[进程驱动](../../tools/run_model_benchmark.py) 和 [报告校验/统计器](../../tools/verify_model_benchmark.py)。工具在既有 BUILD_BENCHMARKS 下构建；模型 target 与 Google Benchmark main 分开，支持 BUILD_TESTS=OFF，不链接 GTest 或 Google Benchmark。原有 Google Benchmark 依赖仍随该构建选项配置，其 target 已另行构建/发现验证。

计时分为 ModelLoader::Load、ModelCompiler::Compile、RuntimeBuilder::TryBuild、PrepareExecutableModel，内部 prepared Prefill/逐步 Decode，以及公开 InferenceSession::Generate。内部阶段复用现有公开低层构建块；观察逻辑留在 benchmark，未增加 inference 调试接口、平行产品 runner 或 kernel 选择策略。

每个 measured sample 使用 fresh Runtime/model，Runtime 在 Prepare 之后不移动。KV reservation 的 RAII 生命周期覆盖输入、workspace 和 context；所有 context 在释放 reservation 前销毁。Decode context 只绑定一次，各步骤更新稳定 token/position storage。完整构建及运行说明见 [模型 benchmark 指南](../../tests/benchmark/models/README.md)。

## 2. 测量与报告合同

- timing 使用 Release/non-sanitized 构建及默认 allocator；profile 和 allocations 使用独立进程，所有 ns 字段为 null，不进入性能统计。allocations 仅统计 Runtime allocator 在 Execute 区间的请求，不是全局 malloc/heap 计数。
- Prefill/Decode Execute-only 包含 workspace reset 和模型 Argmax，排除 token 读取、反馈、统计及序列化。内部 first-token-request 从 reservation/setup 开始；公开 Generate 单独包含其请求准备、执行、输出与清理。两者不是同一次请求的阶段分解，不能相加重建端到端耗时。
- 实际输出 G>0 时，第一 token 来自 Prefill，Decode 为 G-1 步，最终 committed KV 为 P+G-1；EOS 包含在输出，分母使用实际数量。零生成不执行内部 Prefill/Decode；每步保存输入/输出 token、KV 前后位置。
- 全部 numbered repetitions 必须存在，sample ID 连续且唯一；轨迹重复一致、internal/public 相同。错误数量、EOS 后继续、错位、NaN/Inf、配置混杂及缺指标组均失败。
- Decode 按 kv_before 分组；连续 Decode Execute 总时间另报，不冒充完整 loop latency。未定义的吞吐使整个指标组为 null，原始时间和全部 n 保留。p90 至少 10 个样本、p99 至少 100 个样本，均只称 observed percentile。
- 驱动在执行前读取/hash 模型，因此标记 pre_read_by_identity_collection，不是冷文件缓存启动。每 sample 固定先 internal、后 public Generate；后者已经使用过模型页。所有比较沿相同顺序。
- 模型身份包括 config、根层权重/index 和所有 index 引用的嵌套 shard；拒绝缺文件、逃逸和 symlink。记录源码 patch、实际 executable/library、build、CPU/affinity/OS/内存及环境身份，并在执行后检查变化。
- 环境合同覆盖 14 项输入，包括 THP_MEM_ALLOC_ENABLE、loader 和线程设置；缺失或非空 timing loader injection 被拒绝。未设置变量记为空字符串，默认行为由记录的库/源码身份限定。
- raw weight bytes 按 data/size 去重，packed storage 按 Buffer implementation 去重；KV/workspace 为容量，RSS 为截至 sample 检查点的累计进程 high-water。它们不是可直接相加的独立 RSS 类别。

## 3. 正式结果

| 构建/检查集合 | 项数 | 结果 |
|---|---:|---|
| benchmark-model 正式 Release target | 1 | 构建成功，无 GTest/Google Benchmark 链接依赖 |
| benchmark-model-baseline 正式 Release target | 1 | 构建成功，排除可选 AVX2/FMA 实现 |
| 真实 CLI 集成：AVX2 / baseline / ASAN+UBSAN+LSAN | 各 14 | 全部通过；sanitizer 诊断无报告 |
| stdlib 报告合同 | 64 | 全部通过 |
| stdlib 文件身份合同 | 13 | 全部通过 |
| 原有生产/lifecycle CTest：baseline | 408 | 全部通过，0 failed/skipped |
| 原有生产/lifecycle CTest：ASAN+UBSAN+LSAN | 314 | 全部通过，0 failed/skipped，无 sanitizer 报告 |
| CMake target/preset 合同 | 2 | 真实配置与 compile/link 隔离通过 |
| build metadata/Linux 适用性探针 | 3 | 通过 |
| 原有 aethermind_benchmark | 1 | 构建及 benchmark_list_tests 成功 |

CLI 集成包括四组已冻结 HF 数值 fixture × O1/O2、重复请求、warmup/sample 数量、EOS、零/单 token、context/vocab/参数拒绝、三模式隔离及失败归档。三种构建复用同一组检查，不能相加为不同功能用例。sanitizer 运行 profile/allocations 诊断；其 timing 统计明确拒绝。

正式采集保存 23 个独立进程、119 个 measured samples：21 个 timing 进程共 115 个样本，profile/allocations 各一进程、各两样本。P=3 的 O1/O2 以 A1/B1/A2/B2/A3/B3/A4/B4 顺序采集，每进程 warmup=1、repetitions=7；P=31/127/511/2046 另作两 preset 扫描，覆盖生成上限 0/1/3/8、Prefill/Decode EOS 和恰到 context=2048。没有删除最慢 repetition。

同实现的 Generate process-median 最大相对离散观测约为 O1 4.046%、O2 6.906%；该值是本机该 workload 的观测，不是自动回退阈值。全部 A/A medians、pairwise differences 和 O1/O2 原始对照保留，结论为 Needs More Data，未宣称 O2 加速或调整 production priority。具体机器数字只保存在上方机器记录及 raw artifacts。

## 4. 审核发现与关闭

最终审核在正式交付前发现并修复两项采集身份缺口：

1. THP_MEM_ALLOC_ENABLE 会影响大块 CPU allocation 的 alignment/madvise，初版未记录该输入。现由 runner/checker 共享必需环境键；补了缺字段、不同 THP 配置不能比较及 loader injection 负测。
2. HF loader 支持 index 引用嵌套 shard，初版仅根层枚举可能漏掉实际权重。现逐个收集 weight_map 目标，并增加直接文件身份回归。真实 nested-only、nested+unused-root 两个模型均 Load/Compile/Generate 成功，报告包含实际 nested shard，输出与外部 oracle 一致。

首轮 23 个进程保留在 20261027T152658Z_ff2f94797033_desktop-54h5mmi_baseline-suite-67564a2d，qualification.json 标记配置身份不完整，不用于最终基线。修复后重新执行合同、CLI/sanitizer 和正式采集；没有修改旧报告让它们通过新合同。

## 5. 复现与证据

```bash
cmake --preset benchmark-model
cmake --build --preset benchmark-model
cmake --preset benchmark-model-baseline
cmake --build --preset benchmark-model-baseline

python3 tests/unit/inference/test_verify_model_benchmark.py -q
python3 tests/unit/inference/test_run_model_benchmark.py -q
python3 tests/unit/inference/test_model_benchmark_cli.py \
  --binary build/validation/benchmark-model/tests/benchmark/aethermind_model_benchmark \
  --fixtures .models/inference-numerical -v

python3 tools/run_model_benchmark.py \
  --binary build/validation/benchmark-model/tests/benchmark/aethermind_model_benchmark \
  --model .models/inference-numerical/mha-untied/model \
  --prompt 1,7,3 --max-new-tokens 8 --preset o1-plain \
  --warmup 1 --repetitions 7 --mode timing
```

本次使用 FETCHCONTENT_SOURCE_DIR_SPDLOG / FETCHCONTENT_SOURCE_DIR_GOOGLE_BENCHMARK 复用既有源码；未引入第三方依赖。ASAN 复用 validation-asan-ubsan，额外设置 BUILD_BENCHMARKS=ON 后只构建 model target；detect_leaks=1，没有 suppression。原有 correctness profiles 不自动采集性能，仅纳入两个无模型资产依赖的 stdlib 合同。

最终 raw suite 为 benchmark-results/models/20261027T153722Z_ff2f94797033_desktop-54h5mmi_baseline-suite-a7fa449c；collection.json 保存全部命令、顺序和每个 report 路径。每个 report 所在目录保存原始 stdout/stderr、完整 repetitions、machine/source/binary/model 身份和 summary。suite 的 o1-noise.json、o2-noise.json、preset-observations.json 保存全部对照。

gitignored build/validation/model-artifacts 保存构建/CLI/CTest/JUnit、CMake 探针、13/64 契约结果、真实 sharded 模型结果、采集脚本及 run-manifest.json；各 benchmark preset 保存 compile_commands、CMakeCache 和 build 日志。独立 probe 的 50 项资格检查保留在 build/validation/model-benchmark-probe，与正式 CLI 的 14 项检查范围不同。

格式和 git diff --check 通过。文档检查前后仍为既有 2 个失效链接、29 个 warning，本批未新增诊断。

## 6. 未验证范围

这是 tiny random Llama FP32、standard RoPE 的采集基线，不是代表性大模型性能或质量验收。WSL 未提供 governor/turbo 的完整可见性，未隔离宿主后台负载；没有裸机核心类别/SMT/NUMA 收益证据。当前各 timing process 仅 3 或 7 repetitions，不报告 p90/p99。

没有测冷启动、PMU 归因、稳定置信区间、自动 regression gate、其他精度/模型家族或质量 corpus；没有运行 TSAN。未验收 Ninja Multi-Config、其他 compiler/OS 或物理无 AVX 平台。Q4 的候选优化推广与模型质量评测、Q5 的架构/交付门禁仍待推进。
