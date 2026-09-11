# CPU FP32 RoPE 执行契约

本文描述当前 reference kernel 的执行约束。算子端口、shape、dtype 与 split-half
公式以 [ModelGraph 设计 §12.2](../model_graph_design.md#122-rope-语义与执行边界) 为准。
当前实现 FP32、kPlain、kBoth reference，支持 split-half/interleaved、partial rotary，
以及 Standard、Linear、Dynamic NTK、YaRN、Llama3 和 LongRoPE。SIMD 路径仍未实现。

## 参数与生命周期

- `RoPEParams` 以 typed `RoPEAlgorithmParams` 保存唯一算法 payload，并将 pairing 与
  `rotary_dim` 正交表达；metadata builder 将固定参数冻结到 `ResolvedKernel::attrs`。
- params builder 在 binding 阶段验证具体 shape、正 strides、偏移及 alias，并构造 POD args。
- args 借用 tensor data 指针；shape/address/stride 改变后必须重建 bindings。
- `position_ids` 的内容是运行时数据，每次执行重新读取。调用方不能在执行期间并发修改
  输入、位置或输出存储；kernel 不含共享可变 cache，也不为这些外部访问加锁。

## 算法解析

backend-independent frequency resolver 负责算法公式，CPU kernel 只处理 tensor layout、
pairing 和旋转。Standard、Linear、YaRN、Llama3 在 kernel preparation 阶段生成一张
immutable inverse-frequency table；LongRoPE 生成 short/long 两张表。这些 table 只存于
`ResolvedKernel::attrs`，不会复制进每次执行的 prepared args。

resolver 返回 `RoPERotationCoefficients`：`inv_freqs`、`position_divisor` 与
`rotary_output_scale`。所有 kernel 统一计算
`angle = (position_id / position_divisor) * inv_freq`；因此 Linear 保留未缩放的 base
frequency table，并将其 factor 冻结为 `position_divisor`。其余当前算法的 divisor 为 1。

Dynamic NTK 不把首次 position 固化进 `PreparedExecutionBindings`。每次执行先取得
`effective_sequence_length = max(position_ids) + 1`，据此计算 dynamic base，再按 pair
即时计算频率。LongRoPE 同样以该长度选择 short 或 long table。kernel invocation 路径
不分配 heap，也不维护跨 session 的可变频率 cache。

`rotary_dim` 是每个 head 被旋转的前缀，必须为正偶数且不大于 `head_dim`；剩余尾部
在 out-of-place 路径原样复制。split-half 配对 `i` 与 `rotary_dim/2+i`，interleaved
配对 `2*i` 与 `2*i+1`。算法与 pairing 互不推断。

YaRN/LongRoPE 的 `rotary_output_scale` 对应 HF `attention_factor`：它乘到 cos/sin，
因而同时缩放旋转后的 Q 和 K。full rotary 时可由经验证的 lowering/fusion 等价改写为
`attention_softmax_scale *= rotary_output_scale²`；不能直接乘一次原值。partial rotary 时
logit 等于 `scale² * dot(rotary_prefix) + dot(unrotated_tail)`，不存在单个全局 logit
系数，因此 reference 保留 RoPE 端缩放。该字段名称刻意避免把输出振幅与 logits scale
混为一谈。

## Alias 与地址范围

| 关系 | 允许条件 |
| --- | --- |
| q 与 q_output、k 与 k_output | 无重叠，或 dtype/shape/stride/data 完全相同的 exact in-place |
| 输出与其他输入（包括 position_ids） | 无重叠 |
| 两个输出 | 无重叠 |
| 只读输入之间 | 不额外限制 |

校验使用元素 stride 换算出的半开字节区间。元素偏移、字节跨度和地址加法均检查溢出，
不使用不同数组指针之间的减法。先用整个视图的包围区间快速排除；相交时以双指针扫描
单调排列的行区间。因此同一 QKV allocation 内各行分离的 Q/K 视图可以合法原地执行。

按行检查使用每行的包围区间，包含 column stride 产生的空隙；只因空隙交叠而实际
元素不相交的布局也可能被保守拒绝。该限制明确属于此 reference kernel，不能解释为
TensorView 的通用 alias 定义。输出各行也必须满足既有的非重叠行约束。

不支持的 overlap 和不可表示的地址范围返回 `InvalidArgument`，发生在 binding 阶段，
不修改任何 tensor 内容。验证地址范围不能证明 allocation 的真实容量；足够大的有效
backing storage 以及元素自然对齐仍由调用方保证。

## 派生数值范围与失败原子性

静态算法在 metadata build 阶段检查全部 inverse frequencies；Dynamic NTK 在执行前检查
本次 dynamic base 及每个 pair 的频率。所有算法都在写输出前检查最大角度可表示。

每次执行先扫描全部 position IDs；负值返回 `InvalidArgument`。之后检查：

```text
effective_sequence_length = max_position_id + 1
max_angle = (double(max_position_id) / position_divisor) * max_inverse_frequency
```

非有限的 inverse frequency、effective position 或 angle 返回 `Overflow`。
由于位置非负、频率为正，上述最大值检查覆盖所有实际角度。位置/角度预检查全部通过后
才开始写 Q/K，因而这些失败保持两个输出原值，包含 exact in-place 的输入。

该保证不承诺扫描或拒绝 Q/K 数据中的 NaN/Inf，也不承诺任意幅值输入的旋转结果都可用
Float32 表示。Linear 的 `0 < factor < 1` 仍合法；不 clamp position，也不新增
`position_ids < max_pos_embeddings` 限制。有限巨大角度只保证可执行，不代表已完成
对应上下文范围的 HF 兼容性验收。

成功执行不分配 heap 或 workspace。静态频率解析属于 cold path；每次执行的位置预检查
为 O(seq_len)，Dynamic NTK 另有 O(rotary_dim) 的无分配频率计算。按行 alias 检查只
发生在 binding 阶段。

## 精度与独立验收

当前 reference 使用 double 计算频率、角度、sin/cos 和旋转乘加，最终转换为 Float32。
它是数学 reference，不要求复现 HF FP32 运算的每次舍入。采用两类互补证据：

1. 手算 split-half golden、position 0 恒等性、pair 平方和近似不变等数学测试。
2. 固定 Transformers **4.57.1**、PyTorch **2.9.0+cpu**，直接调用
   `LlamaRotaryEmbedding` 与 `apply_rotary_pos_emb` 导出的独立 CPU Float32 golden。

既有 HF fixture 覆盖 head_dim 4/64/128、MHA/GQA/MQA、theta 4/10000/500000、Standard/Linear
scaling（factor 1/2.5/0.75），非连续 position 含 0、1、8192 和 32768，输入幅值不超过 1
并包含零值及较小幅值。同一逻辑 fixture 在 contiguous、strided/padded 和 exact in-place
三种布局上运行，检查 Q/K 输出、padding 及 position 数据。

整个固定 fixture 集统一使用以下兼容性门禁，不逐 case 调整：

```text
abs(actual - HF) <= 1e-3 + 1e-5 * abs(HF)
```

该门禁容纳长位置下 double reference 与 HF FP32 phase 的舍入差异；首次本机对照的
最大绝对误差约为 **6.27e-4**，出现在 head_dim 128 的长位置 MQA case。
短位置手算 golden 继续使用原有 1e-6 级别检查。该证据不等同于任意 theta、factor、
position 的兼容性证明或完整 Llama 数值验收；扩大验收域时应添加 fixture 并重新评估。

HF 的 config validator 会对 factor 0.75 发出警告，但其 linear 数值实现仍执行该公式。
该 case 用来验证 AetherMind 已声明的小于 1 的 factor 语义，不表示 HF 推荐这种模型配置。

### 重新生成

生成工具仅供开发者使用，不加入 CMake/runtime 依赖；无需模型下载：

```bash
uv venv /tmp/aethermind-rope-hf-venv --python python3.12
uv pip install --python /tmp/aethermind-rope-hf-venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu 'torch==2.9.0+cpu'
uv pip install --python /tmp/aethermind-rope-hf-venv/bin/python 'transformers==4.57.1'
HF_HUB_OFFLINE=1 /tmp/aethermind-rope-hf-venv/bin/python tools/generate_rope_hf_fixture.py
HF_HUB_OFFLINE=1 /tmp/aethermind-rope-hf-venv/bin/python tools/generate_rope_hf_fixture.py --check
```

数据位于 `tests/unit/backend/cpu/kernels/fixtures/rope_hf_v4_57_1.h`；相邻 JSON 记录
版本、CPU capability、种子、源文件与数据 SHA-256。`--check` 要求同一生成环境的数据
逐字节一致，不承诺不同 CPU/libm 环境重新生成得到相同位模式。C++ 测试只消费已提交数据。

```bash
cmake --build build --target aethermind_unit_tests -j 4
TMPDIR=/tmp ./build/tests/unit/aethermind_unit_tests \
  --gtest_filter='RoPEInference.*:CPUKernelRoPE.*:CPUKernelRoPEEntry.*:*CPUKernelRoPEHf*'
```

上游源码：
[HF Llama 实现](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/modeling_llama.py)、
[HF RoPE 参数实现](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/modeling_rope_utils.py)。

### 本次验证记录（2026-09-07）

- 2026-09-07 的基础实现验证为 87 个测试，包含 21 个 HF fixture/layout 对照。本次
  多算法扩展另覆盖 frequency resolver、HF 前端规范化、serde 迁移、dynamic binding
  reuse、pairing 与 partial rotary；以本次最终测试输出为准。
- 同一 filter 在 ASan/UBSan 构建下通过（`ASAN_OPTIONS=detect_leaks=0:halt_on_error=1`、
  `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`）。
- 单独开启 LSan 时退出检查未通过：`src/function.cpp` 的 `GlobalFunctionTable::Register`
  静态初始化路径报告 1384 字节、17 次分配；同一测试程序使用 `--gtest_filter=-*`
  运行零测试时也复现该报告。因此不能声称完整泄漏检查已通过，该问题不在 RoPE 修复范围内。
- HF fixture `--check` 与格式检查通过。未据默认配置微基准作 Release 性能结论。
