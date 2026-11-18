# CPU Attention 优化方案

本方案实现 [CPU-04](../cpu-operator-evolution.md#41-工作包与退出条件) 的 SIMD、Decode GQA 与首个 Prefill 分块。端口、数值与 KV 可见性合同见 [Attention](README.md)，机器实测归 [benchmarks](benchmarks/)。

## 首段：保持归约顺序的 AVX2

| 路径 | AVX2 工作 | 兼容回退 |
|---|---|---|
| F32 Q / KV32 / F32 output | 8 个元素并行计算 QK 乘积、累加 weighted-V、输出除法 | 非连续 Q/output 列、head_dim < 8 或构建/feature 不可用 |
| F32 Q / KV FP16 或 BF16 / F32 output | 批量 widening 后复用 FP32 循环；FP16 需要 F16C，BF16 用整数扩展 | 同上；只有 FP16 cache 路径受 F16C 限制 |
| BF16 Q / KV / output | 每个 query head widening 一次；FP32 局部累加，最后逐元素窄化一次 | 同上；head_dim > 256 用独立 A16 reference；FP16 A16 使用既有 reference |

- 只替换 compute body。各候选复用原有 metadata、张量绑定及 invocation KV 检查；行 stride 和 KV padding 可以保留，SIMD 只读完整的 8 元素组，剩余用 scalar tail。
- 仍按原顺序扫描两遍可见 K：第一遍求最大值，第二遍重算 QK、求标准 `std::exp` 并累加 V。QK 乘积向量化后按原维度顺序做 FP32 加法；denominator、weighted-V 保留 token 顺序。禁用 FMA contraction 与 fast-math，保留 overflow/NaN 分类。F16/BF16 widening 在 FP 计算前 quiet NaN。
- A32 直接把已验证的输出作为 FP32 累加空间。BF16 A16 使用两个 256 元素 FP32 栈数组，每个 query head 复用；无需额外 workspace，不缓存 KV 地址、不改变 KV 事务或 commit。相比 A16 reference 单独求 denominator、按 64 维分块重扫 V，本方案在第二遍同时求 denominator 与完整 head 的加权和。
- ISA 代码独立编译、禁用 LTO 与 Unity 合并；descriptor 声明现有 AVX2/FMA 构建配置要求，FP16 cache 额外声明 F16C。选核通过已有 CPU feature policy；不采用可重排归约、向量近似 exp 或 score scratch，以便先隔离 widening 与 SIMD 的收益。

## 第二段：Decode 在一对 query heads 内复用 K/V

进入条件：SIMD 已可用、Q/output 列连续、head_dim 至少 8（BF16 A16 至多 256）；单行 Q、`query_begin == committed_end`、`query_end >= 16`，且至少两个 Q heads 对应一个 KV head。16 是本次实验覆盖到的最短生产启用点，不是跨硬件调优常量。单 token 首次 Prefill、普通 Prefill、MHA、非连续列与其余不适配组合继续走既有 per-head SIMD/reference。复用现有 descriptor 和 ISA feature policy；不增加公开选项。

- 按 KV head 分组，每次处理同组内相邻的两个 query heads。两遍扫描都把每个 K 向量组加载/widen 一次，供两条独立 QK 链使用；第二遍的 V 向量也共享。奇数组尾留给原 per-head driver，不跨 KV head 配对。
- 两个 heads 分别维护 FP32 max、denominator、weighted-V 和最终除法。每个 head 的维度累加、token 顺序与 narrowing 都保持原合同。单个 query head 的 NaN 不影响同组其他 head 的结果；共享的 K/V 值仍按原语义参与两者。
- 每次只读取 `[0, query_end)`。即使 invocation 的 `visible_end` 更大，也不读未来 token。所有 stride、alias 及状态检查继续在入口完成；不跨调用保存 KV 指针或 score。A32 写入既有输出，BF16 A16 需要固定的两个 widened Q 和两个 FP32 accum 数组，共 4 KiB（不计编译器栈帧开销）。不新增 workspace/allocator 调用，不改变 plan 的 KV commit。

| 方案 | 决策 |
|---|---|
| **配对、两遍扫描** | 本次生产路径。两条独立链可交错执行，K/V loads 与 widening 在一对 heads 内复用；context 任意增长均不增加 scratch。 |
| 配对 + 短 context 的 score tile | 固定 256-token / 2 KiB 原型可避免短 context 的第二次 QK，但超出容量仍重算；本机 prepared 算子有局部收益，模型 Decode 区间重叠，暂不增加生产分支。更长 context 的完整 score 缓存须单独核算 workspace。 |
| Online softmax | 可缩减 K 的扫描次数，但会改变归约顺序，尤其需处理非有限 score、denominator 重缩放和 A16 舍入。与 Prefill 的归约方案一起独立验证后再决策。 |

配对复用减少的是 kernel 发出的 load/widening 工作，不能由此计算 DRAM traffic。工作集、缓存层级与 register pressure 是否支持 4-head tile，需有单独 profiling 证据。

## 第三段：Prefill 双 query 行与 causal key 边界

进入条件：沿用四条 AVX2 精度路径及其连续列、head_dim 和 ISA 约束，至少 32 个 query rows。门槛来自本机 A/B，按 query 行数而非 KV cache 容量选路；短 Prefill、单 token Decode 和 reference fallback 继续使用原实现。

- **职责边界**：只在 Attention compute body 分块，保持当前单次同步 Prefill、KV layout、Graph IR、KernelSelector 和事务边界。每个 tile 由同一 query head 的相邻两行组成，按现有 GQA/MQA 映射读取同一个 KV head，MHA 同样适用。
- **数据流**：若第一行的 exclusive key end 为 `end`，两遍扫描均共享 `[0, end)` 的 K load/widening，第二遍共享 V。第一遍额外用位置 `end` 的 score 更新第二行的 max；第二遍结束后，仅为第二行累加该位置的 V。第一行从不读取对角尾部的 V，避免被 mask 的 NaN/Inf 参与运算。奇数末行使用原 per-head SIMD。
- **资源与数值**：沿用 Decode 的配对 AVX2 算术，每行保留 FP32 维度/位置归约顺序、独立 max/denominator，且 BF16 A16 只窄化一次。A32 在绑定的输出上累加，BF16 A16 使用固定的两份 widened Q 和两份 FP32 accumulator（合计 4 KiB 数组）。不新增 planner workspace、全局 score strip、payload 预扫或 hot-path allocation。KV 始终来自本次验证过的物理读绑定。
- **后续触发条件**：先用本实现做长 context profile，确认 QK 重算的实际占比。保存双行的全可见 FP32 scores 可省第二次 QK，但在 4096 keys 下需要 32 KiB strip，必须核算每 step 的 workspace 与整请求预算。只保存固定宽度的 score tile 无法在保持全局 max 与原归约顺序的同时消除重算。Online softmax 需独立冻结数值/非有限合同并跑模型校准；本阶段无此变更。

## 验证与后续

比较独立 reference 与实际 resolved/prepared kernel，覆盖 GQA、causal history、tail、padded stride、fallback、非有限分类及写出前拒绝非法 binding。Production A/B 使用 SIMD-eligible Llama geometry，通过 Compile/Prepare/ExecutionPlan 和公开 Generate 比较同精度轨迹；分别记录短、长 context 和可用的缓存轮转对照。每组仅在实际机器、构建、精度和 workload 范围内判读。

Decode 的生产决策与比较见 [本机证据](benchmarks/desktop-qhihogq-attention.md#cpu-04-第二段decode-gqakv-复用)。Prefill 双行验收见同一机器记录。更宽 query/key tiles、score strip 与 online softmax 由长 context profile 触发。
