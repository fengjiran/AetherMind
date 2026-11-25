# CPU GEMM 算子内多线程并行方案

**状态**：Draft，属于 CPU-08。首个实验选用 FP32 identity Linear 的单 token Decode，测量多核读取权重节省的时间能否覆盖每次投影的派发与同步开销。线程数、启用阈值和加速比均待实测。[CPU 算子路线图](../operators/cpu-operator-evolution.md#34-线程与资源规划) · [GEMM 单线程基线与布局状态](../operators/gemm/README.md)

## 1. 生产入口与执行边界

首版沿用已注册的 FP32 Linear、现有 plain/identity 权重格式及串行 microkernel。计划构建时按现有规则解析 descriptor 和 recipe。绑定时利用实际形状、stride、线程预算和 workspace 确定分片。执行时从 kernel 调用上下文借用窄的同步并行接口，由 GEMM driver 派发。调用线程继续按原顺序执行计划，单次 Generate 同步返回。[Linear 到 GEMM 的适配](../../src/backend/cpu/kernels/linear/linear_a32_optimized.cpp) · [prepared kernel 调用入口](../../src/execution/kernel_invoker.cpp) · [产品边界](../product.md)

## 2. 输出 tile 分片

对逻辑矩阵乘法 `C[M,N] = A[M,K] × B[K,N]`，每个输出 tile 交给一个线程写入，由它按原内核顺序完成全部 K 归约。线程之间共享只读输入与权重，一次 GEMM 在所有分片结束后同步一次。首版采用静态分片，无需跨线程合并部分和。

| 场景 | 分片方法 | 选择依据 |
|---|---|---|
| Decode，M=1 | 按 N 方向的完整列 tile 数均分为连续区间，末段处理尾列 | 权重沿输出通道分工。实际线程数取单 token 延迟进入带宽平台期的最小值 |
| 短 Prefill，M 行块少 | 优先切 N，每个线程处理所领列区间内的全部行块 | 同一个 B panel 可跨行复用，完整行块可继续走 blocked driver |
| 长 Prefill | N 块充足时先按列分工。列块不足时再沿完整 M 行块拆成矩形 | 比较增加 M 分片带来的负载均衡收益与多个线程重读或重复打包同一 B panel 的成本 |

例如 `M=1, N=4096, K=4096`，若选择四个参与线程，各线程负责连续 1024 列，每列完成 4096 项归约。一般形状按 microkernel 的列宽确定边界，结合输出首地址、元素宽度和行 stride 检查相邻线程是否反复写同一缓存行。可用线程数上限取线程预算与独立 tile 数的较小值。

**先按完整形状选 driver，再划分其外层 tile 循环。** 现有 FP32 AVX2 driver 在 `M=1..8` 与 `M>=9` 采用不同路径。若把九行拆成几个小矩阵再各自选路，会改变 blocked 路径、累加方式和 panel 复用。分片应沿用完整问题的 tile 分解、数值安全检查与尾块处理。后续接入 packed-B 和融合投影时保留原权重的逻辑维度、分段列偏移及输出 stride，覆盖起点落在 panel 内部的切片。[GEMM 参数、布局与 driver 契约](../../src/backend/cpu/kernels/gemm/gemm_internal.h)

## 3. 线程、临时存储与失败处理

- **生命周期与线程预算。** 建议 Runtime 统一配置参与线程数，暂称 `intra_op_threads=T`，默认 1。调用线程计入 T，Runtime 创建时启动 T−1 个常驻 worker，部分启动失败时回收已启动的线程。T=1 直接进入串行路径。Runtime 销毁前等待 worker 退出。共享 Runtime 仍由宿主串行调用。
- **派发与同步。** 预建任务描述、worker 状态和完成同步对象。调用线程发布一次分片任务并计算其中一片，等待所有已派发分片完成后读取状态。worker 内再次进入同一并行接口时走串行路径，避免嵌套等待和超额占核。选用的 ISA 对调用线程及各 worker 可运行的 CPU 均须有效。
- **scratch 预算。** Decode 从权重直读路径开始。Prefill 逐个核对临时 panel 和累加缓冲的容量，按对齐后的每线程大小乘实际参与线程数检查溢出、上限和可用 workspace。需要新增 scratch 的路径先接通 CPU-07 的 sizing、预算查询及绑定，再启用并行。每个参与线程使用独立切片，容量仅够较小线程数时在绑定阶段缩减并行度。prepared Decode 稳态包括所有 worker 的派发与汇合，须保持零分配。
- **错误与回退。** worker 报告失败后，调用线程等所有已派发任务结束再返回状态、重用 arena 或发起整算子回退。W16 的 `TryRunGemm*` 可在写入部分结果后要求 reference 重算，此时必须先汇合再覆盖整份输出。请求侧沿用现有 KV 提交边界，确保失败传播和资源回收晚于最后一次 worker 访问。

## 4. 实施顺序

| 阶段 | 实施内容 | 进入下一阶段的证据 |
|---|---|---|
| O2 串行基线 | 冻结当前 FP32 Linear 的 descriptor、recipe、实际形状和模型，分别测 prepared Decode、Prefill 及 Generate | 记录同机 A/A 噪声，确认选中的投影在目标负载中占用的时间 |
| FP32 Decode | 接通常驻线程池和同步接口，在 M=1 identity Linear 上按 N 分片。第一组实验轮转总量大于 LLC 的权重，并设置 hot 对照 | 验证单线程直通、失败收敛、稳态零分配与 TSAN，找到 streaming 延迟的带宽饱和点 |
| FP32 Prefill | 在已选 driver 内分配 N tile，随后对列块不足的形状试验 M/N 二维分片 | 分开验证短、长 prompt，比较重复打包成本、workspace 峰值、prepared 耗时和模型 Prefill 延迟 |
| 扩展 consumer 与精度 | 按生产 profile 接入 QKV、GateUp，再覆盖 W16 数值回退。B-panel 候选沿用其独立布局验收 | 分别验证融合输出段、权重偏移、精度与准备成本，每个新增范围单独作启用决定 |

## 5. 验收与启用条件

1. **正确性与确定性。** 用独立 reference 对照，覆盖空 M/N、零 K 写正零、奇数尺寸、tile 边界、支持的 stride、非有限值和数值回退。固定线程数重复运行。对保持相同指令和累加顺序的路径做单线程与多线程逐位对照。注入 worker 失败验证返回后无迟到写入，运行 TSAN，并在全进程 malloc-family 检查中覆盖 prepared Decode。
2. **派发阈值。** 测不同 T 的空派发与完成屏障耗时，扫描 1、2、4 等参与线程数直至目标机器预算。Decode 分 hot 与 streaming 记录延迟和带宽；Prefill 按 M/N/K、线程数和切法记录耗时与吞吐。先在单 socket 上测饱和点，记录 affinity、SMT 和有效 ISA。小任务或收益落在噪声内的形状继续使用串行路径。
3. **生产收益。** 在同机同配置下交错运行独立进程，用原串行实现分别对照新实现 T=1 和候选线程数。依次测 registered/prepared Linear、ExecutionPlan、真实 Generate 的 Prefill 延迟和 Decode ms/token，分开记录准备与稳态。启用范围须同时满足正确性与资源门禁、T=1 无可辨回退、目标线程数的配对收益超过该组 A/A 噪声。阈值按已验证的机器、phase、dtype、布局和形状记录。

每轮 O4 实验按工作流保存 commit、构建配置、实际 descriptor/recipe、线程与拓扑、命令、配对样本和 raw 路径，将决定追加到对应算子的单机基准记录。[算子开发与优化工作流](../guides/operator-development-workflow.md) · [GEMM 优化提案](../operators/gemm/cpu-gemm-optimization.md)
