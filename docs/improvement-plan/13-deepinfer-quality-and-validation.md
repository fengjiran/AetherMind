# 借鉴 DeepInfer 的生产验证与质量实施方案

- **状态**: In Progress
- **版本**: 1.1
- **日期**: 2026-10-26
- **产品边界**: [当前产品 PRD](../products/aethermind_prd.md)
- **架构边界**: [AGENTS.md](../../AGENTS.md) §2.1
- **关联计划**: [02 工程质量](02-engineering-quality-system.md)、[06 系统能力路线图](06-system-capability-evolution-roadmap.md)、[10 产品推理入口](10-upper-level-runtime-assembly.md)
- **关联模块**: model / graph / compiler / execution / inference / backend / 测试与构建 / 应用入口

本文将 DeepInfer 中适合当前产品的验证方法，细化为 AetherMind 的实施工作包。质量属性、风险台账、证据等级与 build profiles 以 02 号提案为准；算子、fusion、packing 与性能实验继续遵循[算子开发与优化工作流](../guides/operator-development-workflow.md)。本文不建立第二套质量规范、kernel registry 或模型运行入口。

§1 保留 2026-10-22 的静态核对快照。当前实施进度以 §4 对应工作包为准；2026-10-26 已完成 Q1 的 sliding_window 准入子项。其余拟新增报告格式、数值工具、CTest labels 和模型 benchmark 不能据此视为已实现。

## 1. 现状分析

### 1.1 DeepInfer 的可借鉴内容与适用边界

以下依据来自 DeepSeek 官方仓库于 2026-10-22 查阅的文档与源码，链接指向可变化的 main。后续采用具体实现或重现实验时，须另记实际 commit、依赖和硬件；本次没有在其目标硬件上运行验证。

| 已查阅的官方依据 | 可借鉴机制 | AetherMind 的适用范围 |
|---|---|---|
| [README](https://github.com/deepseek-ai/DeepInfer)、[部署 Q&A](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md) | 明确模型、硬件、功能开关及公开代码与完整生产服务之间的边界 | 首批仅覆盖 CPU、同步单请求、Llama Dense、FP32、静态 KV、Token IDs I/O；完整服务的 KV 存储和负载均衡未完整开源，不能视为可直接复用的交付物 |
| [离线回归](https://github.com/deepseek-ai/DeepInfer/blob/main/tests/models/test_offline_inference.py)、[长上下文回归](https://github.com/deepseek-ai/DeepInfer/blob/main/tests/models/test_offline_inference_long.py) | 真实引擎运行、平台/workload golden、样本完整性及有限数值检查 | 复用现有 HF→Generate 链，建立固定 fixture 和短/长上下文回归；不引入 DP rank 维度 |
| [YoCo 等价验证](https://github.com/deepseek-ai/DeepInfer/blob/main/examples/yoco/README.md) | metadata 比较与固定 teacher-forcing 轨迹的 logprob/top-k 比较分开 | 用于 graph rewrite、packing、prepared kernel 和完整模型的分层差分；不引入 YoCo 或 SWA 语义 |
| [benchmark](https://github.com/deepseek-ai/DeepInfer/blob/main/examples/benchmark.py)、[执行说明](https://github.com/deepseek-ai/DeepInfer/blob/main/examples/README.md)、[sweep](https://github.com/deepseek-ai/DeepInfer/blob/main/examples/sweep/run_sweep.sh) | Prefill/Decode 独立测量、预热、profiling 与稳态测量隔离、配置和原始计数量留存 | 增加 CPU 模型路径测量，统一计时边界；DeepInfer 的投机 Decode 吞吐含满接受口径，不能直接解释为实际有效输出吞吐 |
| [CONTRIBUTING](https://github.com/deepseek-ai/DeepInfer/blob/main/CONTRIBUTING.md) | 测试自动发现，平台相关与耗时验证分组 | 建立 CTest labels 与适用 build profiles；不据此推断公开仓库所有 CI 都已充分验证 |
| [eval](https://github.com/deepseek-ai/DeepInfer/blob/main/eval/README.md) | 生成原始结果与评分分离 | 将数值 conformance 与模型质量评测分开；评分复用保存的输出，文本/tokenizer 适配留在外部评测工具 |

DeepInfer 当前面向 DeepSeek-V4.1-Flash 和 SM100/Ascend 950 的优化不能直接转换为 CPU 收益。异步 batch scheduler、EP/CP、MoE、投机解码、分离式部署、远端 KV 和 batch invariance 不属于本文实施范围。未来模型、精度和 CUDA 扩展分别沿 [09](09-qwen-and-moe-architecture-support.md)、[11](11-fp16-bf16-execution-support.md)、[12](12-cuda-backend-execution-support.md) 验收。

### 1.2 AetherMind 已有资产与增量缺口

下表的“已有”指实现和测试源码存在，本次没有重新执行这些测试。

| 当前事实 | 源码依据 | 增量工作 |
|---|---|---|
| HF→Prepare→Session→Generate 测试显式覆盖 O1/plain、O2/packed，并检查重复输出和 Runtime 不匹配 | [生产链测试](../../tests/unit/inference/test_load_prepare_executable_model.cpp) | 固定 fixture 来源，将测试纳入可重复 profiles；产品入口沿 10 号提案实施 |
| 已有 scalar reference、logits/KV 对照、多步 Decode、分配失败恢复和已准备 Decode loop 零分配测试 | [direct Prefill/Decode](../../tests/unit/inference/test_direct_prefill_decode.cpp)、[Prefill 测试](../../tests/unit/inference/test_prefill.cpp) | 扩大轨迹/边界覆盖，保存可定位差异的报告；零分配结论严格限定到被测区间 |
| 完整 plan 和所有 layer 写入成功后，LayerRunner 才调用公开 CommitAppend | [LayerRunner](../../src/execution/layer_runner.cpp)、[KV view](../../src/kv_cache/kv_cache_session_view.cpp) | 保留该合同，将已有失败和 stale-view 测试纳入 lifecycle 集合 |
| Linear benchmark 已区分 prepared hot、streaming、binding specialization，并先做 reference 校验 | [Linear benchmark](../../tests/benchmark/cpu_kernels/benchmark_cpu_linear.cpp) | 补充模型级准备、Prefill、Decode 和 Generate 测量 |
| 顶层只有全局 TSAN 开关，未接入 CTest 与标准 ASAN/UBSAN profile；benchmark 依赖无条件配置 | [根 CMake](../../CMakeLists.txt)、[单测 CMake](../../tests/unit/CMakeLists.txt) | 按 02 的 target-scoped profiles 落实自动发现和构建变体；依赖条件化与 10 共用实现 |
| 库按源文件启用 AVX2/FMA，单测却给整个 target 加 ISA flags | [库 CMake](../../src/CMakeLists.txt)、[单测 CMake](../../tests/unit/CMakeLists.txt) | 隔离 SIMD 测试 translation units，补 baseline CPU 与有效 ISA 受限的 dispatch 验证 |
| activation 已限制为 silu；HF reader 跳过未知字段，而配置中尚无 sliding_window 的解析/拒绝合同 | [validator](../../src/model/formats/hf/hf_model_validator.cpp)、[reader](../../src/model/formats/hf/hf_directory_reader.cpp)、[config](../../include/aethermind/model/formats/hf/hf_model_config.h) | 对已知会改变语义的配置 fail closed，保持无关 metadata 的兼容读取 |
| 源码仍编入单个 shared target，尚无 install/export 和正式模型/session C ABI 闭环 | [库 CMake](../../src/CMakeLists.txt)、[C API](../../include/c_api.h)、[10 号提案](10-upper-level-runtime-assembly.md) | 先建立依赖检查；安装/consumer 与 C ABI 分别验收 |

## 2. 目标架构

### 2.1 职责、所有权与接口边界

| 所在位置 | 拟承担职责 | 必须保留的边界 |
|---|---|---|
| model | HF 行为字段解析、家族准入与构图语义 | 不把配置 schema 复制到 CLI、测试报告生成器或 Runtime |
| graph / operators / compiler | 语义、rewrite、lowering 及 artifact 结构校验 | semantic rewrite、fusion、layout transformation、weight packing 分别验证；compiler 不查询 kernel registry |
| backend / execution / kv_cache | recipe/layout、kernel resolve、prepared binding、执行与 KV 状态 | PackingLayout 继续标识不可变物理格式；CommitAppend 不证明写入完成 |
| inference / 应用入口 | 生产准备与同步 Generate 编排；CPU preset、配置适配与工具 I/O 沿 10 实施 | 不新增平行 runner；产品工具只消费 public headers；不向 Generate 增加测试模式、logits 回调或 teacher-forcing API |
| tests / 开发验证工具 | 独立 oracle、固定轨迹、tensor snapshot、golden 比较和故障注入 | 测试 helper 归 tests；需要内部观察的工具不冒充 public consumer，不成为库的运行依赖 |
| 构建与报告工具 | CTest 发现、profile、事实收集、capability 证据关联 | 不新建 runtime capability registry；声明来源、测试执行状态与成熟度分别记录 |

请求期 ownership 保持现状：Runtime 在 Prepare 前安置于最终地址，长于 ExecutableModel 和 Session；请求拥有 activation/workspace storage，context 借用其切片及 KV view。测试 snapshot 必须复制到自有存储，并在 value 被复用、workspace reset 或请求释放前完成读取。观测不能延长生产 tensor 的隐含 lifetime。

### 2.2 能力与验证矩阵

复用 06 的 capability matrix，按“模型家族 × weight/activation/KV dtype × device × preset × phase × 有效 ISA”关联证据。首批范围如下：

| 组合 | 当前证据状态 | 本方案目标 |
|---|---|---|
| Llama Dense，W32/A32/KV32，CPU，O1/plain | 有生产链及 reference 测试资产，本次未重跑 | 默认产品入口、短/长上下文、数值/确定性/lifecycle 全链路验证 |
| 相同模型与精度，O2/packed | 有独立参数化测试资产，不自动获得性能承诺 | 与 O1/reference 对照并单独记录 recipe、数值和性能 |
| baseline CPU / AVX2 等执行路径 | 库已有 ISA 分离，测试目标 flags 尚需收口 | 分开构建与实际执行；compiler 能接受 flags 不代表机器具备 ISA |
| FP16/BF16、INT8/INT4、其他家族、GPU | 不由本文新增支持声明 | 由各 capability 专题闭环后复用验证资产 |

每条机器可读证据记录拟包含：schema 版本、组合身份、模型/fixture checksum、git commit 与 dirty 状态、build profile、实际 CPU features、测试命令/filter、期望与实际 case 数、pass/fail/skip/not-run、失败阶段、结果 artifact/checksum。结果关联 02 的证据等级；成熟度仍需按对应验收条件审核，不能从一个 pass 自动推导 Supported 或 Production-ready。

声明从 operator 契约、kernel descriptors、构建选项与既有符号校验，执行状态来自真实运行。测试缺失、未执行、硬件不适用、fixture 缺失和执行失败必须区分；发布必需 case 的 skip/缺失不能计为通过。该记录是测试输出，不是新的 public ABI 或运行时决策接口。

### 2.3 生产路径与数值诊断路径

```text
固定来源的 HF fixture + Token IDs + 显式 CPU preset
  ├─ 公开路径：Load → Compile → Runtime/TryBuild → Prepare → CreateSession → Generate
  │            └─ token/EOS/错误状态/生命周期/产品计时
  └─ 测试路径：同一 fixture → prepared plan + 固定输入 token 轨迹
               └─ Prefill 与每步 Decode → logits/KV snapshot → 独立 oracle 比较
```

公开路径的对象构造顺序与失败边界以 10 为准：可提前确定的模型/请求错误在 KV 大块分配前返回；backend、packing 与 kernel 可执行性由 Prepare 判定。现有要求 Runtime 已存在的 LoadAndPrepareExecutableModel 便利入口继续独立覆盖，不能据上图改写其接口。

诊断路径复用现有 direct Prefill/Decode 测试基础，在每步输入相同的既定 token 与绝对 position，避免自由生成分叉后无法定位初始误差。它不代替公开 Generate 集成测试，也不使用观测后的运行数据生成自己的正确性期望值。

### 2.4 Golden、独立 oracle 与数值合同

分三类资产维护：

1. **微型独立 oracle**：保留 scalar reference，覆盖 logits、KV 和 watermark；新增随机/边界用例使用固定 seed，不与 optimized 路径共享被测计算实现。
2. **固定模型轨迹**：记录 prompt、continuation、逐步 logits 或参考 token 的 logprob/top-k、关键 KV 样本及明确的 sample 坐标；较大模型完整 tensor 可放受校验的外部 artifact。哈希或摘要可做快速定位，但不能替代浮点误差检查。
3. **公开 Generate 行为**：检查完整输出 token、EOS、输出长度、重复调用、错误状态和失败后可恢复性；按现有请求生命周期检查 KV 释放。

数值比较必须事先冻结 max absolute/relative error、非有限值策略、必要的 top-k/top-1 检查及近似路径的质量容限。同一模型、输入、构建和有效执行配置的确定性回归精确比较 token；不同 ISA 的 reduction/FMA 不承诺 bitwise equality。接近 Argmax 决策边界的样本应记录 top-1/top-2 margin，保留 token 差异诊断，不能事后放宽容差或删除样本。

独立外部实现可通过离线脚本生成参考数据，须固定实现版本、模型 revision、dtype、RoPE 和 position 语义；Python/框架依赖仅用于开发验证，不进入核心库。fixture 来源、获取方式、许可证和文件校验沿 10 的约定；不假设被忽略的 .models 目录随仓库交付。

Golden 更新工具必须先输出 candidate 和逐样本 diff，由评审确认数值变化与原因后再更新基线；普通 CI 不自动覆盖期望值。报告必须检测缺样本、重复 sample ID、轨迹错位和非有限值。相同实现历史输出的自回归 golden 只证明未变化；生成 token 上的 decode_ppl 不等同于固定验证语料的标准 perplexity，也不能单独证明语义正确或模型质量。

### 2.5 等价验证的分层合同

| 层次 | 必须检查 | 失败定位 |
|---|---|---|
| Graph rewrite / fusion | 输入输出及端口、shape/dtype、residual、alias、consumer、layer boundary、权重角色；仅比较语义等价的配置 | pass/节点/端口与首次不同的 value |
| Packing / materialization | logical weight 映射、recipe/layout 身份、padding、tail、alignment、checked bytes 与 owner lifetime | weight role、recipe、逻辑坐标与存储边界 |
| Prepared operator | registered/resolved kernel + prepared params 对独立 oracle；目标 ISA 与 fallback 均覆盖 | selector、kernel、shape、元素坐标与误差 |
| 完整模型 | 固定轨迹下 Prefill/Decode logits、KV 内容、watermark 与 token；O1/plain、O2/packed 分别检查 | 首个不同的阶段/步/层/元素及上下文 |

Metadata 通过不代表数值通过；O1/O2 互相一致也不能代替独立 oracle。诊断观测若因 liveness 或 fusion 无法读到中间值，可在测试构图/执行辅助层增加受控观测，须标明其对计划与内存的影响；该运行不能作为无观测生产性能证据。

### 2.6 模型级性能测量

统计方法、交错 A/B、噪声 floor、机器元数据与算子 raw artifact 规范引用算子工作流，不复制或降低其门禁。模型级结果拟按同一 run-id 规则写入 gitignored benchmark-results/models 目录；摘要进入现有验证文档或 CI artifact，禁止仅保留最好一轮。

| 被测区间 | 计时与计数边界 | 必须报告 |
|---|---|---|
| 准备期 | 加载、编译、packing/Prepare 分项；声明文件页缓存与预热状态 | 耗时、峰值内存、raw/packed 权重 bytes；共享/映射存储避免重复计数 |
| Prefill | 已有模型准备完成，从本次请求执行起到首 token；另列含请求准备的首 token 延迟 | prompt 长度、处理 token 数、耗时及 tokens/s；两种边界使用不同字段 |
| Decode | Prefill 后已有明确 KV 长度，计量实际执行的 Decode 步 | 每步延迟分布、实际输出 token 数、上下文区间、KV/workspace bytes、被测区间分配计数 |
| 完整 Generate | 输入 Token IDs 到返回生成 Token IDs，包含请求准备与清理；模型准备另列 | 总耗时、实际生成长度、EOS、进程/请求内存口径 |

首批固定单请求，扫 prompt 长度、生成长度和已有 KV 长度；不借此引入 batching。首 token 通常来自 Prefill，不能计入 Decode 执行步数；EOS 提前停止时分母使用实际输出数量，不使用 max_new_tokens。每次至少明确样本数；不足以估计 p99 的样本不得报告稳定 p99。

Decode 上下文随生成增长。固定上下文实验应在计时区外用独立 reservation 重建相同前缀，或以相同上下文区间交错比较；不能跨不同 KV 长度直接归因 kernel 收益。所有 reset/重建成本须明确排除或包含，并另报真实连续生成结果。

预热、分配/packing、测量、profiling 分段执行；若 profiler 改变后续稳态，应分进程运行。日志、trace 和数值 snapshot 不进入无观测热路径计时。除有注明的纯计算实验外，正常生产必需的工作必须包含在端到端结果中。

优化先由 profiling 建立瓶颈假设：Prefill 检查 GEMM 复用，Decode 检查 GEMV/权重流量与 dispatch，长上下文检查 Attention/KV 读取。区分 logical bytes 与实际层级流量，通过 Roofline/Amdahl 和实测支持候选方向；达到数值和 production-path 门禁后，才可按 workload 推广 candidate。O2/packed 不预设优于 O1/plain。

## 3. 方案与备选

推荐在既有模块与测试上增量实施：用 10 的唯一产品入口验证公开 API，用内部测试诊断数值，用 02 的 profiles 承载验证，按算子工作流记录性能。这使每个新增工具都有明确消费者，也能复用已有 reference、OOM 与 KV 测试。

| 备选 | 主要问题 | 裁决 |
|---|---|---|
| 直接移植 DeepInfer 的调度、GPU runtime 和分布式 KV | workload、硬件和产品范围不同，且部分服务基础设施未完整公开 | 留给对应 capability 专题，不纳入本方案 |
| 只检查最终文本/token 或单个 PPL golden | 中间数值错误可能被 Argmax 掩盖；自由生成分叉降低定位能力 | 同时使用固定轨迹、逐步数值与公开行为回归 |
| 在 Generate 中加入 teacher-forcing、debug callback 和可变 logits 输出 | 测试需求扩大 public API、lifetime 与热路径合同 | 保持测试/开发工具边界 |
| 新建 capability runtime registry 或一套 benchmark 规则 | 重复已有权威，形成声明、dispatch 与文档漂移 | 关联既有事实与运行结果，复用现行规范 |
| 立即拆分所有 CMake targets 并重写测试框架 | 改动面大，可能影响静态 kernel 注册及既有行为 | 先依赖检查与 profiles，再根据实际风险拆分 |

## 4. 实施步骤

以下编号仅为工作包追踪，不定义产品阶段、API 或模块身份；实施状态在各工作包中单独记录。语义修正、入口、验证基础设施和性能优化分别提交，避免将收益验证与 correctness 修复混为一项。

### 4.1 Q1：模型准入与公开入口证据（P0）

- **准入子项已实现（2026-10-26）**：model 显式解析 sliding_window，缺失/null 保持当前全注意力语义；正整数因不支持滑动窗口而拒绝，0/负数因非法值而拒绝，非法类型、小数/指数、非法后缀和整数溢出在解析时拒绝。覆盖 reader→validator→loader、直接 BuildModelGraph/BuildLlamaDense 与 runner 的 Runtime 分配前失败边界，沿用已有 activation 与 RoPE 权威。实现和限定验证证据见 [第一批验收记录](../tests/sliding_window_admission_validation_2026-10-26.md)。本子项不表示 Q1 全部退出条件或 Q2–Q5 已完成。
- 按 10 实施 apps/inference 下的独立入口，默认 O1/plain，O2/packed 显式选择；完成模型和请求失败边界、Runtime 稳定地址、stderr/stdout 与退出码合同。
- 固定 fixture 来源、checksum、命令和预期样本集合；复用已有 LoadAndPrepareExecutableModel/Generate 测试。输出第一版能力证据记录，尚未运行的组合明确标记。

退出条件：不支持的已知语义在规定阶段被拒绝；关闭测试与 benchmark 后能单独构建/运行入口；合法 fixture 通过公开全链路，失败用例清理正确。FP16/BF16、量化和其他家族不能因 parser 接受而出现在支持列表。

### 4.2 Q2：CTest、sanitizer 与 ISA 基础设施（P0）

- 根 CMake 接入 CTest 和 GoogleTest 发现，按 02 的 target-scoped helper 实施 warnings、ASAN+UBSAN、独立 TSAN；ASAN/UBSAN 与 TSAN 不合并，第三方不继承项目 Werror。
- 建立 smoke、numerical、lifecycle、long-context、performance labels，明确 build profile 与 test label 是两个维度。缺数据的本地可选测试可报告 skip；发布必需 fixture 由 CI 准备，缺失即失败。
- 移除单测 target 全局 AVX2/FMA flags，仅对需要的 translation units 启用；baseline 构建配合已有有效 feature 控制验证 fallback，另用具备硬件的节点验证 AVX2。
- benchmark 依赖条件化复用 10 的修改；接入现有 Decode 零分配、OOM、stale-view 与 KV 失败测试，不重建平行用例。

退出条件：发布必需测试被发现且实际执行，case 数与期望相符；ASAN/UBSAN 聚焦集合和独立 TSAN 适用集合真实运行；ISA 不适用与未验证分别报告。sanitizer 与 malloc interposer 的兼容性需专门确认，不能把 interposer 不可用时的 skip 记为零分配证明。

### 4.3 Q3：数值与优化等价回归（P1）

- 在 tests 下提取共享 fixture/轨迹/snapshot helper；读取元数据和 snapshot 的分配不计入 Decode 零分配区间。
- 先扩展 tiny scalar oracle，再加入带固定 revision 的小型真实模型及外部参考数据；覆盖多 token Prefill、至少两步 Decode、GQA/MHA/MQA 中实际准入的形状、非 SIMD 整除尺寸和上下文上限附近。
- 分别检查等价 rewrite、packing、prepared operator 与完整模型；增加 candidate golden 生成/比较工具和缺样本、NaN/Inf、位置错位的负面测试。
- 保留同配置重复生成，跨 ISA 比较执行预先冻结的数值合同。覆盖首 token EOS、Decode EOS、连续请求与失败后重试。

退出条件：O1/plain 与 O2/packed 均有独立 oracle 和公开 Generate 证据；人为扰动一个权重/position/报告样本能触发相应失败并定位首次差异；诊断不改变公开 API 或生产热路径。

### 4.4 Q4：模型性能与质量评测（P1）

- 新增模型级 benchmark，优先组合既有公开构建块；内部阶段诊断与完整 Generate 分开命名，不能把内部路径称为公开 consumer。
- 保存完整配置、实际 token/step/时间、模型与机器身份、全部 repetitions 和 raw artifact；增加 profile 与计时隔离、EOS 分母及上下文区间校验。
- 先建立同机噪声 floor 与趋势报告，再为可靠 workload 配置 regression gate；不得使用跨机器绝对数字或全局固定回退百分比。
- 可选模型质量慢测采用生成与评分分离，固定 tokenized corpus/参考实现；报告缺样本即无效。tokenizer/文本适配在外部，评分依赖不进入核心库。

退出条件：准备、Prefill、Decode、Generate 有不同计时边界与可复现命令；结果不足以估计分位数或存在配置混杂时拒绝收益结论；至少一个候选优化完成同机 A/B、数值、prepared 与完整模型验收后，才调整 production 选择策略。

### 4.5 Q5：架构约束与交付验证（P2）

- 先检查违反 AGENTS 模块依赖的 include/链接，覆盖执行层纯数据头等允许例外；用小范围故意违规验证规则能检出，避免基于目录名误报。
- 若再拆内部 targets，保护匿名静态注册的可达性，比较拆分前后 descriptor 集合和公开全链路；按需要选择 OBJECT targets 或显式注册。
- install/export、独立 C++ consumer smoke 由 02 承载；后续版本化 C ABI 沿 10 实施，增加独立 C consumer 与异常/错误/缓冲区/销毁顺序验证。

退出条件：依赖违规可机械检测；安装产物在源码树外可消费，测试/benchmark 不是正式 consumer 的链接依赖；C++ 入口通过不自动标记 C ABI 已交付。

### 4.6 依赖与实施批次

| 批次 | 工作包 | 顺序与可并行内容 |
|---|---|---|
| 首批 | Q1 + Q2 | 模型准入修正、产品入口、CTest/sanitizer/ISA 分开提交；基础设施可并行，公开入口验收不等待优化 |
| 第二批 | Q3 + Q4 基线 | 数值 helper 可先做；完整模型比较与 benchmark 依赖 fixture/可执行组合；质量慢测可选 |
| 第三批 | Q4 候选优化 + Q5 | 性能推广依赖数值和可重复基线；依赖检查可提前，安装/C ABI 不阻塞前述正确性回归 |

本方案不推翻 06/08 中现有 P0 资源与 KV dtype 工作，也不以完成全部 Q5 为新增 kernel 或数值回归的前置。

## 5. 风险与依赖

| 风险类别 | 风险 | 控制与失败边界 |
|---|---|---|
| Correctness | 历史 golden 或两个同源实现一致，却共同计算错误 | 独立 oracle + 语义/数值/公开行为分层；golden 更新需差异证据 |
| Correctness | 模型配置被忽略，或缺数据/skip 被当作支持 | model 中拒绝已知行为改变字段；证据记录检测完整性，成熟度独立审核 |
| Ownership / memory | snapshot 读取已复用 storage，Runtime 移动，失败后 KV 可见状态错误 | snapshot 在有效期内复制；保留现有 RAII 和稳定地址；完整 plan/layer 成功后才提交 |
| Failure semantics | 将逻辑提交误解为整个请求的物理回滚 | kernel/plan 失败不发布新的 watermark，可能已有未提交写入；完整执行后输出校验失败不承诺回滚已提交位置，Generate 失败按请求 RAII 释放 reservation |
| Concurrency | sanitizer 组合或 interposer 冲突导致测试失真 | TSAN 单独配置，仅验证实际线程/全局状态合同；不据单线程主链测试宣称并发 Session 支持 |
| Portability | 测试本身带 AVX2，硬件不适用却被视为 fallback 通过 | flags 按 translation unit 隔离；分别记录编译、feature 控制与物理硬件执行证据 |
| Performance | profiler/logging/snapshot、页缓存或 KV 长度改变读数 | 固定工作量与上下文，分开计时/profile，记录冷暖边界与真实端到端开销 |
| Maintenance | 13 与 02/06/10/算子工作流形成多个权威 | 13 只细化实施；稳定规范仍归指南，功能与 API 归原专题，来源和结果分别更新 |
| Delivery | fixture 不随仓库提供、外部 reference 版本漂移、旧二进制运行 | fixture/checksum 与参考实现版本固定；干净 CI 准备并构建本 commit，记录实际 executable 身份 |

## 6. 验收标准

| ID | 验收要求 | 归属 |
|---|---|---|
| V1 | 已知不支持的语义配置显式拒绝；模型、请求与 Prepare 失败阶段符合 10 的边界 | Q1 |
| V2 | O1/plain、O2/packed 分别通过合法 FP32 HF→Generate；来源、checksum、命令可追溯 | Q1/Q3 |
| V3 | CTest 发现与实际执行数量匹配；发布必需 case 无隐含 skip；warnings/sanitizer profiles 有真实结果 | Q2 |
| V4 | baseline CPU 与目标 ISA 的编译、dispatch 和硬件运行证据区分；受测二进制不受全局 AVX2 flags 污染 | Q2 |
| V5 | 固定轨迹下逐步 logits/KV/watermark 对独立 oracle；rewrite/packing/kernel/model 各层有对应定位 | Q3 |
| V6 | golden 更新先生成 candidate/diff；缺样本、错误位置、非有限值及人为数值扰动均能使验证失败 | Q3 |
| V7 | 同配置重复生成、EOS、连续请求、OOM/stale view/执行失败恢复及已准备 Decode 零分配具有适用证据 | Q2/Q3 |
| V8 | 准备、Prefill、Decode、Generate 边界明确；实际 token 分母、上下文区间、样本量和机器配置完整 | Q4 |
| V9 | 性能结论符合算子工作流；profiling 与计时隔离；数值与完整模型门禁通过后才推广优化 | Q4 |
| V10 | 能力声明、测试执行状态和成熟度分离；不从加载、枚举、descriptor 或局部 pass 推导端到端支持 | Q1–Q4 |
| V11 | 架构依赖检查能检测实际违规；若拆 targets，kernel descriptor 与生产路径无意外丢失 | Q5 |
| V12 | C++ 安装 consumer 和后续 C ABI consumer 分别验收；核心库不依赖外部评测框架或测试工具 | Q5 |

每个工作包保存变更 commit、构建/运行命令、实际覆盖范围、失败与未运行项、raw artifact 引用；按 02 的证据等级评审。文档检查通过仅证明文档约束，不代表 V1–V12 已通过。所有要求完成或由明确的新专题接管并记录范围裁决后，才能将本提案标记为 Implemented。
