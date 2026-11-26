# CPU GEMM 算子内多线程并行方案

**状态**：Draft，CPU-08，待复审。**版本**：0.2（2026-11-26）。

首个实验选用 FP32 identity Linear 的单 token Decode，测量多核读取权重节省的时间能否覆盖每次投影的派发与同步开销。线程数、启用阈值和加速比均待实测。[CPU 算子路线图](../operators/cpu-operator-evolution.md#34-线程与资源规划) · [GEMM 单线程基线与布局状态](../operators/gemm/README.md)

## 1. 生产入口与执行边界

首版沿用已注册的 FP32 Linear、现有 plain/identity 权重格式及串行 microkernel。计划构建时按现有规则解析 descriptor 和 recipe。拟在冷路径按实际形状、stride、线程预算确定分片；需要额外 workspace 的范围按 §3.2 先规划容量，再绑定地址。执行时从 kernel 调用上下文借用窄的同步并行接口，由 GEMM driver 派发。调用线程继续按原顺序执行计划，单次 Generate 同步返回。[Linear 到 GEMM 的适配](../../src/backend/cpu/kernels/linear/linear_a32_optimized.cpp) · [prepared kernel 调用入口](../../src/execution/kernel_invoker.cpp) · [产品边界](../product.md)

## 2. 输出 tile 分片

对逻辑矩阵乘法 `C[M,N] = A[M,K] × B[K,N]`，每个输出 tile 交给一个线程写入，由它按原内核顺序完成全部 K 归约。线程之间共享只读输入与权重，一次 GEMM 在所有分片结束后同步一次。首版采用静态分片，无需跨线程合并部分和。

| 场景 | 分片方法 | 选择依据 |
|---|---|---|
| Decode，M=1 | 按 N 方向的完整列 tile 数均分为连续区间，末段处理尾列 | 权重沿输出通道分工。实际线程数取单 token 延迟进入带宽平台期的最小值 |
| 短 Prefill，M 行块少 | 优先切 N，每个线程处理所领列区间内的全部行块 | 同一个 B panel 可跨行复用，完整行块可继续走 blocked driver |
| 长 Prefill | N 块充足时先按列分工。列块不足时再沿完整 M 行块拆成矩形 | 比较增加 M 分片带来的负载均衡收益与多个线程重读或重复打包同一 B panel 的成本 |

例如 `M=1, N=4096, K=4096`，若选择四个参与线程，各线程负责连续 1024 列，每列完成 4096 项归约。一般形状按 microkernel 的列宽确定边界，结合输出首地址、元素宽度和行 stride 检查相邻线程是否反复写同一缓存行。可用线程数上限取线程预算与独立 tile 数的较小值。

**先按完整形状选 driver，再划分其外层 tile 循环。** 现有 FP32 AVX2 driver 在 `M=1..8` 与 `M>=9` 采用不同路径。若把九行拆成几个小矩阵再各自选路，会改变 blocked 路径、累加方式和 panel 复用。分片应沿用完整问题的 tile 分解、数值安全检查与尾块处理。后续接入 packed-B 和融合投影时保留原权重的逻辑维度、分段列偏移及输出 stride，覆盖起点落在 panel 内部的切片。[GEMM 参数、布局与 driver 契约](../../src/backend/cpu/kernels/gemm/gemm_internal.h)

设完整列 tile 数为 Q，参与线程数为 P。对非空完整块取 P ≤ Q，按商和余数分配，各片的完整块数相差至多一块，最后一片另处理尾列；只有尾列时走串行。采集每片的块数、尾列数和执行时间，报告最慢片与平均片的差值。短 Prefill 沿 N 切分时，P 个线程可能各读一遍 A，逻辑读取量可达单次遍历的 P 倍。分别记录逻辑字节与可测的 cache miss、内存流量；若 A 重读抵消 B panel 复用收益，该形状改试 M 分片或保留串行。

## 3. 同步接口与资源契约

### 3.1 从 Runtime 到 GEMM driver

本提案将线程预算归入 `ExecutionRuntimeOptions::intra_op_threads`（拟新增），默认 T=1，零值无效。T 包含调用线程。首版按已验证的物理核集合取预算，每核一个执行线程；超过集合容量的显式配置在创建 Runtime 时报告错误。SMT sibling 作为后续独立实验。T=1 走现有串行入口。T>1 时 Runtime 负责启动 T−1 个 worker；部分启动失败时先停止并 join 已启动线程，再返回初始化错误。宿主串行使用 Runtime，销毁时唤醒并 join 所有 worker。[现有 Runtime 配置](../../include/aethermind/runtime/runtime_options.h)

下列 C++ 签名是拟新增的窄接口。共享契约只依赖基础类型；线程实现归 runtime，backend 无需包含 Runtime。区间按**完整问题中编号的输出 tile**计数，半开区间互不重叠且合起来覆盖全部输出；编号到 M/N 矩形的映射由已选 driver 定义。

```cpp
struct TileRange {
    size_t begin;
    size_t end;
};

struct TileTask {
    const void* args;
    StatusCode (*run)(const void* args, size_t slot, TileRange range) noexcept;
};

// IntraOpExecutor 的同步成员。ranges[0] 由调用线程执行。
Status RunTiles(const TileTask& task, std::span<const TileRange> ranges) noexcept;
```

- **传递与借用。** Session 在创建 ExecutionContext 时传入 Runtime 所有的 executor 借用句柄；LayerRunner 构造 KernelContext 时转交可空句柄。空句柄沿用串行 driver。具体 ranges 由 prepared bindings 在冷路径准备；每次调用的参数、权重、输出和 workspace 在 RunTiles 返回前保持有效。计划与 kernel 均不持有池的所有权。[ExecutionContext](../../include/aethermind/execution/execution_context.h) · [LayerRunner](../../src/execution/layer_runner.cpp) · [KernelContext](../../include/aethermind/backend/kernel_context.h)
- **发布与汇合。** Runtime 在建池时为每个 worker 预建可复用的任务槽、状态码和发布／完成代数。不同 worker 的槽分别按缓存行隔离。调用线程检查分片数量与容量后，将描述和范围写入本轮活动槽，以 release 发布新代数并唤醒对应 worker。worker 以 acquire 确认代数、执行回调、写入状态，再以 release 发布完成代数并通知调用线程。等待均按代数循环检查，调用线程以 acquire 观察全部活动槽完成后才复用。首轮采用 C++20 atomic wait/notify，空派发实测包含唤醒与汇合。
- **再入与错误。** 调用线程和 worker 在回调范围内均用 RAII 维护 thread-local 深度。深度非零时，在当前线程按序处理内层任务的 ranges，使用内层调用自己的参数与 scratch，避免重入外层任务槽。正常路径按 slot 索引借用本次调用的独立 scratch。空输出直接成功；同一轮出现多个错误时，汇合后按最小失败分片索引将状态码转为 Status。普通错误终止本次执行。W16 数值回退请求单独写入对应分片的预建标志；所有分片完成且无普通错误时，才以 reference 覆盖整份输出。释放 arena、返回请求或提交 KV 均晚于最后一次 worker 访问。[W16 全量重算入口](../../src/backend/cpu/kernels/linear/linear_a32_optimized.cpp)

### 3.2 CPU-07 前置清单

| 责任与时点 | 带新增 scratch 的 Prefill 硬前置 | 验收 |
|---|---|---|
| CPU backend 在冷路径计算每片需求 | 对选定 driver 给出共享字节数、每个参与线程的 panel／累加缓冲大小和对齐。以包含 caller 的 P 计算总量：对齐的共享前缀 + P × 对齐的私有切片大小，检查对齐、乘法、加法溢出 | P=1、上限和溢出边界可独立复算 |
| execution 提供形状与 P，runtime 规划容量 | 在请求资源查询、arena 分配前解析实际形状或已验证的上界，把含线程维度的结果折算成每步 WorkspaceRequirement，再由 planner 布置 offset。查询与执行采用同一预算和切法 | 同时核对 workspace cap 与请求 payload cap；必要时降低 P 重算完整需求。串行也超限时沿用预算错误 |
| Session 持有 arena，execution 绑定地址 | 校验已规划容量、基址对齐与切片边界，按 slot 绑定至各自区间并保持到汇合。池的常驻元数据单独计量 | 检查各片互不覆盖、容量恰好／不足、OOM 传播和稳态零分配 |

当前 planner 在建计划时接收已确定的每步 requirement。上述按形状与 P 解析需求是 CPU-07 的增量。直读 FP32 Decode 的任务槽由池预建，算子新增 scratch 为零，可与 CPU-07 同期推进。Prefill 的串行 profiling、切分对照和现有栈缓冲审计也可先做；使用额外 arena 切片的生产路径须通过表中三项。workspace lifetime 复用和 activation liveness 可独立排期。[现有 workspace 契约](../../include/aethermind/base/workspace_types.h) · [计划构建](../../src/execution/execution_plan_builder.cpp) · [workspace planner](../../src/runtime/workspace_planner.cpp)

## 4. 实施顺序

| 阶段 | 实施内容 | 进入下一阶段的证据 |
|---|---|---|
| O2 串行基线 | 冻结当前 FP32 Linear 的 descriptor、recipe、实际形状和模型，分别测 prepared Decode、Prefill 及 Generate | 记录同机 A/A 噪声，确认选中的投影在目标负载中占用的时间 |
| FP32 Decode | 接通常驻线程池和同步接口，在 M=1 identity Linear 上按 N 分片。第一组实验轮转总量大于 LLC 的权重，并设置 hot 对照 | 验证单线程直通、失败收敛、稳态零分配与 TSAN，找到 streaming 延迟的带宽饱和点 |
| FP32 Prefill | 在已选 driver 内分配 N tile，随后对列块不足的形状试验 M/N 二维分片 | 分开验证短、长 prompt，比较重复打包成本、workspace 峰值、prepared 耗时和模型 Prefill 延迟 |
| 扩展 consumer 与精度 | 按生产 profile 接入 QKV、GateUp，再覆盖 W16 数值回退。B-panel 候选沿用其独立布局验收 | 分别验证融合输出段、权重偏移、精度与准备成本，每个新增范围单独作启用决定 |

### 4.1 采集机与依赖顺序

| 采集目标 | 拓扑与身份 | 用途 |
|---|---|---|
| 54H5MMI | Intel Core Ultra 9 285H，记录为 16 核、SMT off，WSL2，guest affinity 仅具咨询性 | 复用 FP32 identity 的单线程基线、验证采集脚本；虚拟化数据仅作本机诊断。[机器记录](../operators/gemm/benchmarks/54h5mmi-gemm.md) |
| 首轮启用验收机（待落实） | 目标规格为 Linux 裸机、单 socket、同一 NUMA node 至少 4 个可独占的同构物理核，支持 AVX2/FMA。实机型号、主机名和可用核数在采集前登记 | 在相同核集合比较 T=1/2/4，再按实际预算扩展，完成 TSAN、频率受控的 scaling 与模型验收 |

首版多线程仅在已验证的同构 CPU 集合上启用。Runtime 在建池与选核前核实 topology、允许的 affinity mask、集合中各核的 ISA 及 OS 向量状态，将其交集与 CPU feature policy 共同用于 descriptor 资格判断。worker 固定到该集合；进入执行的调用线程亦须验证或约束 affinity，临时收窄的掩码在退出时恢复。CPU 热插拔、运行时掩码变化或虚拟化使约束无法证明时，停止使用该并行计划，按重新核实的有效 ISA 准备串行路径。多 socket 的权重 placement、first-touch 归 Runtime 与模型准备阶段，跨 node 的性能单独立项。[GEMM threading 与 NUMA](../operators/gemm/cpu-gemm-optimization.md#47-threading-与-numa)

先完成 O2、接口审查和目标机登记，再在该机采集 Decode 启用证据。Prefill 共用已验证的池与失败处理，新增 scratch 另受 §3.2 约束。与 11 号精度工作共享裸机时，为其 AVX-512 真机补证保留独立窗口；与 13 号验证工作复用模型、正确性 fixture 和构建记录。性能窗口独占主机，TSAN、其他基准和模型验证错峰执行。裸机尚未落实时可推进接口、测试与 WSL2 诊断，启用决定等待裸机证据。[11 号精度方案](11-fp16-bf16-execution-support.md) · [13 号验证方案](13-deepinfer-quality-and-validation.md)

## 5. 验收与启用条件

**首轮止损规则。** 对每个目标 shape、P 与缓存状态，以墙钟测串行整算子和相同 P 的空回调派发至汇合。若稳定测得空派发已达到或超过串行时长，该组合记为 Rejected，停止调阈值。若完整路径的配对收益落入同组 A/A 噪声，记为 Needs More Data 并结束该轮，沿用串行。若所有目标 Decode 组合均触发止损，暂停 Decode 推广；新的同步机制或负载证据可作为独立实验重新立项。负结果和原始数据同样归入对应机器记录。

1. **正确性与确定性。** 用独立 reference 对照，覆盖空 M/N、零 K 写正零、奇数尺寸、tile 边界、支持的 stride、非有限值和数值回退。固定线程数重复运行。对保持相同指令和累加顺序的路径做单线程与多线程逐位对照。注入 worker 失败验证返回后无迟到写入，运行 TSAN，并在全进程 malloc-family 检查中覆盖 prepared Decode。
2. **派发阈值。** 测不同 T 的空派发与完成屏障耗时，扫描 1、2、4 等参与线程数直至目标机器预算。Decode 分 hot 与 streaming 记录延迟和带宽；Prefill 按 M/N/K、线程数和切法记录耗时与吞吐。先在单 socket 上测饱和点，记录 affinity、SMT 和有效 ISA。小任务或收益落在噪声内的形状继续使用串行路径。
3. **生产收益。** 在同机同配置下交错运行独立进程，用原串行实现分别对照新实现 T=1 和候选线程数。依次测 registered/prepared Linear、ExecutionPlan、真实 Generate 的 Prefill 延迟和 Decode ms/token，分开记录准备与稳态。启用范围须同时满足正确性与资源门禁、T=1 无可辨回退、目标线程数的配对收益超过该组 A/A 噪声。阈值按已验证的机器、phase、dtype、布局和形状记录。

**频率与功耗对照。** 在裸机上为 T=1 和 T>1 使用相同的 governor、turbo、包功耗限制、亲和性策略和预热条件。优先固定双方均可持续的 P-state，逐进程核实计时窗口的有效频率，并记录包能耗或平均功耗、温度及节流状态。无法锁频时按活动核记录有效频率和包功耗；仅对频率区间可比且无节流的样本解释并行 scaling。另以部署默认 DVFS 单列报告实际端到端墙钟延迟，保留所有样本及分组理由。每个 T 采同配置 A/A 与空派发，原实现对新 T=1 单独成对。WSL2 的预热频率漂移和不可控迁核会造成跨进程偏移；缺少可核实的频率或拓扑时，结论限于诊断。[噪声与跨进程偏移](../operators/gemm/cpu-gemm-optimization.md#681-噪声-floor-是机器属性必须按机器按组量化)

每轮 O4 实验按工作流保存 commit、构建配置、实际 descriptor/recipe、线程与拓扑、命令、配对样本和 raw 路径，将决定追加到对应算子的单机基准记录。[算子开发与优化工作流](../guides/operator-development-workflow.md) · [GEMM 优化提案](../operators/gemm/cpu-gemm-optimization.md)

## 修订记录

| 版本 | 日期 | 变更 |
|---|---|---|
| 0.2 | 2026-11-26 | 根据首轮评审补充同步接口、CPU-07 依赖、ISA 与拓扑约束、采集顺序、DVFS 对照和止损规则，并补列尾块不均与 A 重读的测量项。[首轮评审](14-cpu-gemm-intra-op-parallelism-review.md) |
