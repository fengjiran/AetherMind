# CPU Linear

当前生产路径包含 F32 plain/identity 的 portable scalar 与 AVX2/FMA driver，以及 W16/A32 plain/identity 的 SIMD widening、Decode/small-M 与 Prefill blocking。F16 需要 AVX2/FMA/F16C，BF16 需要 AVX2/FMA；不满足编译或运行条件时 W16 保留 reference。Linear、LM head、QKV 和 GateUp 复用同一计算 engine，支持 CPU O1/plain 与 O2/identity-packed Llama Dense。

input 为 rank 至少 1 的 F32 activation，weight 为 `[N,K]` scalar F16/BF16，output 最后轴为 N，前导轴跟随 input。prepared args 使用 byte weight pointer、元素单位 strides 和已校验的 dtype。合法 rowwise strides、rank flatten、地址范围与 alias 由 binding-time helpers 验证。零 M/N 为无操作，零 K 写正零。

W16 plain/identity-packed Linear、QKV 和 GateUp 在准备期校验实际读写的 F32 地址满足 `alignof(float)`，不依赖 view 的 alignment 标签。零 K 仍校验 output，不要求未读取的 input 对齐；空输出不要求对齐。W16 weight 保持 byte-safe 读取，不新增 SIMD 对齐要求。边界回归见 [W16 alignment tests](../../../tests/unit/backend/cpu/kernels/test_cpu_w16_alignment.cpp)。

reference 以 `memcpy` 读取 W16，按 K 升序执行独立的 F32 product 和 add，TU 禁用 FMA contraction。输入 activation 的 F32 精度参与乘法。F32 reference 的 double 累加与 W16 F32 累加分别通过自己的数值合同和独立 oracle 验证。

O2 identity/composite artifacts 保留 W16 元素；QKV/GateUp 复用 Linear driver，当前仍逐 projection 调用，不宣称已实现跨 projection 的 input-tile 复用。W16 optimized 的 FMA、lane reduction 与 blocking 按冻结预算独立验收；zero-size、非单位列 stride、极端幅值与非有限结果完整回退 reference。固定 A/B tile 栈存储为 128 KiB，加上调用帧；不持久化 F32 矩阵。方案见 [Linear 优化](linear-optimization.md)，单机证据见 [本机记录](benchmarks/desktop-54h5mmi-linear.md)。F32/W16/A16 的实际选核身份见 [CPU 覆盖台账 §3.1/§4](../cpu-coverage-baseline.md#31-矩阵计算)。

实现见 [Linear reference](../../../src/backend/cpu/kernels/linear/linear_a32_w16_reference.cpp)，测试见 [Linear regression](../../../tests/unit/backend/cpu/kernels/test_cpu_linear_kernel.cpp)，模型使用见 [W16 指南](../../guides/w16-fp32-inference.md)。遵循 [算子工作流](../../guides/operator-development-workflow.md)。

F32 Linear 在 binding 冷路径保存原始 Linear 参数和具体 driver；没有 SIMD 主体的小形状、空输出及不适合 optimized 的合法布局直接选择兼容 reference。执行时直接调用已冻结的 Linear driver；F32 标量/AVX2 路径内部的 GEMM 分块选择继续遵守既有合同。F32 B-panel 保持独立 candidate，物理布局与 identity 不混读。
