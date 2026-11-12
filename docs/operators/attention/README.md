# CPU Attention 与 P3 KV16 合同

当前 CPU production kernel 含 F32（Q/K/V/output 均 F32）与 KV16（Q=F32、K/V cache=同一 F16/BF16、output=F32）reference；operator inference 已接受同型与 A32→KV16 mixed 签名，`state_dtype` 参与 selector 身份。**P3 mixed-KV reference 已实现，并在 reduced fixture 上通过四组合模型验收（O1/O2，KV16 vs KV32 实测容差 + 固定排名规则）；§2.8 校准数值 fixture（2048）的正式生产验收已通过，SIMD/优化路径仍待推进。** 状态以 [11 号提案 §2.7–2.8](../../improvement-plan/11-fp16-bf16-execution-support.md) 的实施/验收门禁为准，遵循 [算子工作流](../../guides/operator-development-workflow.md)。

## 端口与布局（P3 O0）

| 端口 | dtype | 逻辑 shape / 语义 |
|---|---|---|
| `q` | scalar F32 | `[seq_len, num_q_heads * head_dim]` |
| `k_cache` / `v_cache` | 同一 scalar F16 或 BF16 | `[num_kv_heads, cache_len, head_dim]`，读取实际 state |
| `output` | scalar F32 | 跟随 q 的逻辑 shape |

保留已有同型 operator inference 合同；mixed tuple 只新增上述两种，既有 A16 语义不代表 CPU A16 kernel 可执行。head 数、head_dim 和 seq_len 均为正，`num_q_heads % num_kv_heads == 0`，K/V 逻辑 geometry 相等。query/output 支持既有合法正元素 stride，output 行不得重叠，实际地址须满足 float 自然对齐。

state specs、selector.state_dtype、KV view/layout 和 invocation binding 必须同型，不能仅放宽 entry 的 F32 检查。state ports 虽不参与 compact shape inference，仍参与独立 state dtype 推导。Attention 没有 weight port，selector 的 weight fallback 继续由 activation 决定，KV dtype 不写入 weight_dtype。精度来自端口 TensorSpec，现有 AttentionParams/serde 不新增平行的 KV 精度字段。

KV storage 采用 byte pointer 与 byte stride；head 内 logical elements 连续，token/head stride 必须覆盖实际几何、是两字节元素大小的整数倍，全部地址/字节运算 checked。只读取 logical head_dim，不读取 padding。生产 storage 遵守 KVCacheLayout 的 alignment；byte-safe 读取不另要求 float 对齐。

output 与 q、可见 K cache、可见 V cache 均不得重叠。只读输入间不额外限制。alias 证明沿用现有 strided footprint 的保守规则；地址包围区间无法证明不相交时可以拒绝，不宣称支持所有交错布局。

## 可见性与转换边界（P3 O0）

K 由 F32 RoPE 输出窄化后进入 cache，V 由 F32 projection 输出窄化后进入 cache。Attention 必须读取这份 cache，再按 F16/BF16 widening 合同转换，不能在 Prefill 旁路使用未窄化 activation。当前 query 对应 token 的 K/V 同样窄化；GQA/MQA 仍按连续 query-head group 映射到 KV head。

query row `i` 读取绝对位置 `[0, query_begin + i + 1)`，同时满足 invocation read binding 的 visible/query bounds。跨 plan 只允许读取 committed history；同一同步 plan 的 append window 只有在本层 KVUpdate 完成后可读。

KV 指针、read frontiers 和 query positions 只通过本次 `kv_read` binding 提供，不能保存在 prepared args。不能保留平行 F32 cache；若未来 tile widening 需要 scratch，由 workspace planner 提供。

## F32 reference 数值合同（P3 O1 已实现）

对每个 query row/head：

1. widening 每个 cache 元素，按 head_dim 升序进行 F32 product，再执行独立 F32 add，形成 dot。
2. scale 为 `1.0F / sqrt(float(head_dim))`；dot 与 scale 相乘并写为 F32 score。
3. 按可见 token 升序扫描 score 求 max；不物化完整 score matrix。
4. 第二遍按同序重算 score，使用 F32 `exp(score - max)`，按序累加 F32 denominator 和独立 product/add 的 weighted V。
5. 每个 output component 最后以 F32 denominator 相除。

新 P3 reference TU 禁用 FMA contraction 和 fast-math。既有 [F32 reference](../../../src/backend/cpu/kernels/attention/attention_f32_reference.cpp) 不因 KV16 合同自动取得该新增的局部编译要求，也不要求在本步修改其数值行为。优化路径在固定误差门禁下独立验收，不能隐式将 Q 窄化。

质量/误差门禁采用 FE_TONEAREST、F32 gradual underflow（x86 FTZ/DAZ 关闭）。Attention 不设置或恢复调用方浮点控制；`*`、`+`、`exp`、`/` 可以产生浮点异常 flags。cache widening 自身仍须满足 [FP16](../fp16_conversion/README.md) / [BF16](../bf16_conversion/README.md) 的转换环境合同。

不新增 Q/K/V payload 预扫描，不 clamp overflow 或 NaN/Inf。特殊值按固定 F32 算术和 `std::exp` 传播；含 +Inf 或全 -Inf score 可能产生 NaN softmax，不保证有限 output，不以零向量代替。分类回归覆盖 NaN/Inf、算术溢出和 `0 * Inf`，但不要求 Attention 算术 NaN payload 一致。有限值误差预算要求相应输入、中间 score/归约和输出有限。

## 校验与验收

每次调用仍检查动态 read binding 的 dtype、geometry、range、alias、generation/visibility 前提；这些借用不能跨 Execute 保存。参数/绑定错误在本次 Attention output 写入前失败；算术非有限结果本身不转换为 metadata 错误。Attention 成功不负责 commit，整个 plan 的发布权归 execution transaction。

- independent reference 必须显式量化 cache；不能用未量化 K/V 或同一生产 helper 生成 expected。
- 覆盖 F16/BF16、MHA/GQA/MQA、Prefill/Decode、history/current token、stride/tail/padding、alias/overflow 和特殊值分类。
- 模型实现正确性与同 W16 的 KV32 质量对照分开验收；固定预算、校准资产与四种 W×KV 组合以 [11 号提案 §2.8](../../improvement-plan/11-fp16-bf16-execution-support.md) 为唯一准入来源。
- O1/plain 与 O2/identity-packed 均已闭环；SIMD 与任何性能收益须另有 prepared、production-path 和端到端证据。

当前源码入口见 [attention_entry.cpp](../../../src/backend/cpu/kernels/attention/attention_entry.cpp)，已有 F32 回归见 [test_cpu_attention_kernel.cpp](../../../tests/unit/backend/cpu/kernels/test_cpu_attention_kernel.cpp)，事务回归见 [test_kvcache_update_kernel.cpp](../../../tests/unit/execution/test_kvcache_update_kernel.cpp)。
