# CPU Linear 的 W16/A32 优化方案

本方案落实 CPU-02；现有 FP32 GEMM 生产接入归 CPU-03，流程沿用 [算子工作流](../../guides/operator-development-workflow.md)。状态：已实现（Core Ultra 9 285H / GCC 14.2 / WSL2 的限定验收）；退出证据见 [Linear 机器记录](benchmarks/desktop-54h5mmi-linear.md#cpu-02-w16a32-生产优化与-cpu-03-关联闭环) 与 [GEMM 机器记录](../gemm/benchmarks/54h5mmi-gemm.md#cpu-03-fp32-gemm-生产闭环与布局-promotion-决策)。

## 合同与职责

- A/output 为 F32，W 为 scalar F16 或 BF16；plain 与 identity-packed 的权重物理格式不变。布局、alias、范围与 F32 自然对齐继续由已有 params builder 验证；W16 支持 byte-safe 非自然对齐读取。
- `GemmArgs` 保持 byte 指针与元素 stride；精度由独立的 F16/BF16 entry 决定。原 reference 保持 K 升序、分离 F32 乘加，独立于 optimized 实现与 expected 数据。
- optimized 保留完整合法合同：非单位列 stride、zero-size/zero-K 走原 reference；SIMD tail 不越界。input 不窄化，累加为 F32；FMA、lane reduction 与 blocking 的重排按已有冻结预算对独立 oracle 验收，不承诺逐位等同 reference。
- 热路径不分配、不查 registry、不 pack 全模型权重。QKV/GateUp 复用 Linear engine，并保留原有独立输出与 composite owner/recipe 校验；不把多个 projection 调用声称为跨 projection 的 input-tile 复用。

## 实现

1. Decode/small-M 用 K 维 8-lane SIMD、两行与四个输出的 register block。F16 用 F16C，BF16 用整数 widening。单行 GEMV 为各输出并行计算幅值上界，随后合并并走原 safety guard。tail 用 byte-safe scalar load。
2. Prefill 用共享 F32 6×16 microkernel，KC=512。W16 按 8×8 小块 widening 并转置到连续 N lanes。完整 M 行数超过 48 且 K 至少为 512 时直接读 A，临时 B panel 在最多 252 行内复用。
3. direct-A 路径的 B panel 使用 32 KiB 固定栈 scratch，短 blocked 路径的 A/B panel 合计 128 KiB。后续并行化须将调用帧计入 worker scratch 规划。
4. 使用输入/权重的幅值上界约束 F32 overflow 安全域；不满足时完整重算 reference。非有限输出也回退，防止 fast path 在特殊值上丢失原 reference 的分类语义。安全检查成本计入 production benchmark。
5. descriptor 只声明必要 ISA；BF16 不依赖 F16C，F16 须同时具备 AVX2/FMA/F16C。编译禁用与 runtime feature policy 都须有 fallback 回归。

## 验证与 promotion

- 覆盖两种 W16、plain/identity、Prefill/Decode、非 SIMD 整齐维度、非自然对齐权重、正 stride、output guards、zero-size、NaN/Inf 与 overflow fallback。
- 既有 W16/KV16 模型轨迹、O1/O2 fusion、Generate、错误恢复、owner lifetime 与 Decode malloc-family 门禁继续适用；ASan/UBSan 与真实无特定 ISA 配置分别记账。
- registered/prepared、ExecutionPlan 与 Generate 分开测量；hot/streaming、准备/packing-inclusive 与稳态执行分开报告。保存优化前二进制和源码补丁，同机独立进程交错 A/B，按 CPU-01 的本机噪声判断收益。
- 不引入 W16 B-panel。FP32 B-panel 由 CPU-03 对 identity optimized 单独比较；缺少稳定跨形状收益与 packing 摊销证据时保持 candidate。

## CPU-02 退出结果

F16/BF16 plain/identity、Decode/small-M/Prefill blocking、QKV/GateUp engine 复用、边界与 fallback、模型 Generate/零分配、适用 sanitizer 与 ISA 编译变体均已取得对应证据。本机主 workload 的 prepared 与公开 Generate 同合同 A/B 稳定受益；fusion 自身性能仍 Needs More Data。其他机器、native ISA 与产品 7B/4k 门禁独立验收，不因本工作包关闭而扩大支持承诺。
