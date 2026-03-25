# AetherMind Phase 1 执行器 (Executor) 设计文档

**版本**: v1.1  
**日期**: 2026-03-25  
**作者**: AetherMind Team

---

## 目录

1. [概述](#1-概述)
2. [职责与范围](#2-职责与范围)
3. [设计目标与核心原则](#3-设计目标与核心原则)
4. [核心概念、对象与状态机](#4-核心概念对象与状态机)
5. [核心接口与模块拆分](#5-核心接口与模块拆分)
6. [推理执行流](#6-推理执行流)
7. [LayerRunner、OpContext 与 DispatchTable](#7-layerrunneropcontext-与-dispatchtable)
8. [KV Cache、Workspace 与内存约束](#8-kv-cacheworkspace-与内存约束)
9. [错误处理、验证与测试](#9-错误处理验证与测试)
10. [线程模型与实现路径约束](#10-线程模型与实现路径约束)
11. [Phase 2 演进兼容性](#11-phase-2-演进兼容性)

---

## 1. 概述

### 1.1 文档目的

本文档定义 AetherMind Phase 1 中执行器（Executor）的详细设计。执行器是推理执行阶段的组织核心，负责驱动从 Prompt Token 输入到输出 Token 序列生成的完整状态机转换。

本文档的目标不是描述模型结构本身，而是冻结以下内容：

- Executor 的职责边界
- Prefill / Decode 两条执行路径
- 执行阶段的核心对象与状态转移
- KV Cache / Workspace / Operator Layer 的交互契约
- Phase 1 的性能、并行与错误处理约束

### 1.2 Phase 1 定位

在 Phase 1 架构中，Executor 被定义为一个**单请求、同步阻塞、面向 CPU 的执行控制器**。它不承担请求调度、批处理、异步 API 或 GPU 后端适配职责，而是聚焦于 decoder-only Transformer 的标准 Prefill 与 Decode 流程。

### 1.3 一句话定义

**Executor 是一个面向单次生成会话的同步执行控制器，负责组织 Prefill 与 Decode 两条执行路径，通过统一的 LayerRunner 和 DispatchTable 驱动模型前向，在 Session Preparation 阶段完成 KV 与 Workspace 绑定，并保证 Decode 稳态路径零额外堆分配。**

---

## 2. 职责与范围

### 2.1 核心职责

- **同步流控**：管理单次生成任务的生命周期，确保 Session Preparation、Prefill 与 Decode 阶段有序转换。
- **执行组织**：按照 decoder-only Transformer 的层级结构驱动 Embedding、Norm、Linear、Attention、LM Head 与 Argmax。
- **状态维护**：在生成过程中，维护 `SessionState` 与 `SessionExecutionState` 中的当前位置、已生成 Token 与停止状态。
- **KV Cache 驱动**：通过 `KVCacheView` 协调 Prefill / Decode 的 KV 写入和历史 KV 读取。
- **Workspace 复用**：确保 Decode 稳态路径复用已绑定的工作区，不发生新的堆分配。
- **执行结果提交**：将生成 Token、停止原因和运行时状态回写到 Session。

### 2.2 非目标 (Non-Goals)

- **请求调度 (Request Scheduler)**：不处理多请求排队、优先级或 admission control。
- **连续批处理 (Continuous Batching)**：仅支持单请求（`batch_size = 1`）。
- **异步 API (Async API)**：执行器接口为同步阻塞设计。
- **分页 KV (Paged KV)**：Phase 1 仅支持静态线性预分配的 KV Cache。
- **GPU 支持**：执行逻辑完全基于 CPU 后端。
- **复杂采样**：Phase 1 仅支持贪婪采样（Argmax），不引入 Top-K / Top-P / Temperature。
- **图执行引擎**：Phase 1 不引入通用计算图调度器，优先实现针对 Llama-family dense decoder 的直接执行路径。

---

## 3. 设计目标与核心原则

### 3.1 设计目标

#### 3.1.1 正确性优先

Phase 1 的第一目标是建立清晰、可验证、可回归的执行路径。所有 Reference / Optimized 实现差异都必须建立在一致的执行语义之上。

#### 3.1.2 Decode 稳态零分配

一旦进入 Decode 循环，Executor 不得再触发新的堆分配。所有 hidden buffer、logits buffer、Q/K/V scratch 和 attention scratch 都必须在 Session Preparation 阶段完成绑定。

#### 3.1.3 Prefill / Decode 共用基础设施

Phase 1 不应实现两套互相独立的执行器。推荐设计为：

- 一个 `Executor`
- 两条内部路径：`PrefillPath` 与 `DecodePath`
- 一套共享的 `LayerRunner`
- 一套统一的 `DispatchTable`

#### 3.1.4 热路径轻量化

热路径必须避免：

- 动态查表
- 高层 owning tensor 传递
- 运行期字符串路由
- 运行期虚函数层层分发

Reference / Optimized 的差异应通过 init-time resolve 的函数指针表体现。

#### 3.1.5 为后续扩展保留边界

当前设计应尽量使以下演进在上层执行语义不变的前提下完成：

- Paged KV
- Continuous Batching
- GPU 后端
- 更复杂的采样策略

### 3.2 核心原则

- **Reference first, optimized second**
- **Session Preparation before execution**
- **Decode steady-state zero allocation**
- **Init-time resolve, hot-path direct call**
- **KV 通过 View 访问，不暴露底层物理布局**
- **Prefill 负责建立上下文，Decode 负责稳态单步推进**

---

## 4. 核心概念、对象与状态机

### 4.1 关键对象

| 对象 | 职责 | 备注 |
|------|------|------|
| **Executor** | 顶层执行控制器 | 对外优先暴露 `Generate`；细粒度步骤接口更适合作为内部或测试入口 |
| **ExecutorContext** | 执行期只读上下文 | 包含模型句柄、dispatch、线程策略等 |
| **SessionState** | 用户可见的生成状态 | 包括输入 Token、输出 Token、停止标志 |
| **SessionExecutionState** | 执行期附加状态 | 包括 KV 绑定、workspace 绑定、current_pos |
| **PrefillPath** | Prefill 执行路径 | 处理多 token Prompt 前向 |
| **DecodePath** | Decode 单步路径 | 处理 steady-state 单 token 前向 |
| **LayerRunner** | 单层执行骨架 | 复用 Transformer Block 固定流程 |
| **OpContext** | 算子执行上下文 | 线程策略、workspace、profiling、dispatch |
| **KVCacheView** | KV 的逻辑访问视图 | 供 Executor 与 Attention 访问 |
| **WorkspaceBinding** | Session 级工作区绑定 | 提供复用的临时 buffer |
| **DecodeStepOutput** | 单步输出结果 | 包括 next token 与 stop 信息 |

### 4.2 ExecutorContext

`ExecutorContext` 表示执行期不变的只读上下文，典型内容包括：

- `RuntimeContext* runtime`
- `LoadedModel* model`
- `DispatchTable* dispatch`
- `ThreadingPolicy* threading`
- `WorkspaceBinding` 或等价的 workspace 布局信息
- stop logic helper（如 EOS / max token 判定）

它的职责是减少函数参数膨胀，并将全局只读依赖集中管理。Phase 1 不要求这些依赖都演化为独立策略对象，只需保持职责边界清晰即可。

### 4.3 SessionState

`SessionState` 表示单次生成的用户态与运行态可见状态，至少应包含：

- `input_tokens`
- `output_tokens`
- `generation_config`
- `finished`
- `finish_reason`
- `current_pos`
- `prompt_len`
- `generated_len`

Phase 1 中建议在 Session 创建时预留 `output_tokens` 容量，以避免 Decode 中发生扩容。

### 4.4 SessionExecutionState

`SessionExecutionState` 是 `SessionState` 的执行侧补充，表示仅在 Executor 内部使用的绑定信息，至少包括：

- `KVCacheView kv`
- `WorkspaceBinding workspace`
- `bool prepared`
- `bool prefill_done`
- `std::size_t prompt_len`
- `std::size_t generated_len`
- `std::size_t current_pos`

该对象用于隔离用户态状态与执行期资源绑定信息。若 Phase 1 初版实现希望降低复杂度，也可先以内聚方式收敛在 Session 内部结构中，但应保留后续拆分空间。

### 4.5 DecodeStepOutput

`DecodeStepOutput` 表示单步 Decode 的产物，至少包含：

- `next_token`
- `finished`
- `finish_reason`
- 可选的 `logits_view`

它是瞬时对象，不拥有底层 buffer 生命周期。

### 4.6 执行状态机

Executor 应显式维护执行状态机，而不是依赖隐式 if 分支。

推荐状态：

- `kUnprepared`
- `kPrepared`
- `kPrefillDone`
- `kDecoding`
- `kFinished`
- `kError`

推荐状态流：

```text
Create Session
   -> PrepareSession
   -> Prefill
   -> DecodeStep ...
   -> Finished
```

异常情况下允许从任意状态转入 `kError`。

---

## 5. 核心接口与模块拆分

### 5.1 顶层接口建议

Phase 1 推荐将 `Generate()` 作为默认公开入口；`PrepareSession()`、`Prefill()`、`DecodeStep()` 更适合作为内部实现接口或测试辅助入口，以避免非法状态调用。

```cpp
namespace aethermind::execution {

class Executor {
public:
    Status Generate(SessionState& session);

private:
    Status PrepareSession(SessionState& session);
    Status Prefill(SessionState& session);
    Status DecodeStep(SessionState& session);
};

} // namespace aethermind::execution
```

### 5.2 推荐实现骨架

```cpp
class Executor final {
public:
    Executor(ExecutorContext ctx,
             PrefillPath* prefill,
             DecodePath* decode);

    Status Generate(SessionState& session);

private:
    Status PrepareSession(SessionState& session);
    Status Prefill(SessionState& session);
    Status DecodeStep(SessionState& session);
    Status ValidateSessionForPrefill(const SessionState& session) const;
    Status ValidateSessionForDecode(const SessionState& session) const;
    Status FinalizeStep(SessionState& session, const DecodeStepOutput& out);

private:
    ExecutorContext ctx_;
    PrefillPath* prefill_;
    DecodePath* decode_;
    SessionExecutionState exec_state_;
};
```

### 5.3 PrefillPath

`PrefillPath` 负责整段 Prompt 的前向计算。它必须处理多 token 输入，不允许通过重复调用单步 Decode 伪造 Prefill。

建议接口：

```cpp
class PrefillPath {
public:
    Status Run(ExecutorContext& ctx,
               SessionState& session,
               SessionExecutionState& exec_state,
               DecodeStepOutput& out);
};
```

### 5.4 DecodePath

`DecodePath` 负责 steady-state 的单步解码路径。

建议接口：

```cpp
class DecodePath {
public:
    Status RunStep(ExecutorContext& ctx,
                   SessionState& session,
                   SessionExecutionState& exec_state,
                   DecodeStepOutput& out);
};
```

### 5.5 LayerRunner

`LayerRunner` 负责执行 **Llama-family dense decoder layer** 的固定骨架，避免 Prefill 和 Decode 各自维护一套层循环。

建议接口：

```cpp
class LayerRunner {
public:
    Status Run(std::size_t layer_idx,
               Phase phase,
               const LayerWeightsView& weights,
               HiddenStateView hidden_in,
               HiddenStateView hidden_out,
               KVCacheView& kv,
               OpContext& op_ctx,
               std::size_t position_begin,
               std::size_t token_count);
};
```

---

## 6. 推理执行流

### 6.1 Session Preparation

在进入 Prefill 之前，Executor 必须完成 Session Preparation。该阶段是单次生成前的准备阶段，不等同于全局 Engine Warmup。

Preparation 至少应包括：

- 校验 Prompt 长度与 `max_new_tokens`
- 预留或验证输出 token 容量
- 绑定 `KVCacheView`
- 绑定 `WorkspaceBinding`
- 初始化 `current_pos / prompt_len / generated_len`
- 将执行状态切换为 `kPrepared`

### 6.2 Prefill 流程 (Prompt Processing)

Prefill 负责将 Prompt 编码进 KV Cache，并生成第一个输出 token。

推荐流程：

1. **输入准备**：将 Prompt Token 序列映射为 token view。
2. **Embedding**：将 Prompt Token 序列转换为隐向量。
3. **逐层前向**：逐层执行 Transformer Block。
   - 在 Attention 算子中一次性计算 Prompt 区间的 K / V。
   - 将 KV 写入 `KVCacheView` 的起始位置（`0 .. prompt_len - 1`）。
4. **Final Norm + LM Head**：对最后位置的 hidden state 计算 logits。
5. **Argmax**：从最后位置 logits 中选出第一个输出 token。
6. **Stop Check**：检查是否命中 EOS，或 `max_new_tokens == 0` 等停止条件。
7. **状态更新**：
   - 更新 `SessionState.output_tokens`
   - 更新 `generated_len`
   - 更新 `current_pos = prompt_len`
   - 设置 `prefill_done = true`

特别说明：

- Prefill 的主要价值是建立 Prompt 对应的上下文 KV，而不是仅仅“生成首 token”。
- Prefill 完成后，Decode 路径仅处理后续稳态单 token 推进。

### 6.3 Decode 流程 (Token Generation)

Decode 是 steady-state 热路径，目标是在最小控制开销下逐步生成 token。

推荐流程：

1. **取输入 token**：使用上一步生成的 token 作为当前输入。
2. **Embedding**：将单个 token 转换为隐向量。
3. **逐层执行**：
   - 对每层执行 norm / linear / rope / attention / mlp / residual
   - Attention 通过 `KVCacheView` 读取历史 KV，并在当前位置追加当前 token 的 K / V
4. **Final Norm + LM Head**：得到当前步 logits
5. **Argmax**：选择 `next_token`
6. **Stop Check**：判断是否命中 EOS 或达到 `max_new_tokens`
7. **步进更新**：
   - 追加输出 token
   - `current_pos++`
   - `generated_len++`

### 6.4 顶层 Generate 语义

推荐由 `Generate()` 统一组织完整执行流程：

```cpp
Status ExecutorImpl::Generate(SessionState& session) {
    RETURN_IF_ERROR(PrepareSession(session));
    RETURN_IF_ERROR(Prefill(session));

    while (!session.finished) {
        RETURN_IF_ERROR(DecodeStep(session));
    }
    return Status::Ok();
}
```

### 6.5 特殊情况约定

- **空 Prompt**：允许存在，但实现必须明确 BOS / 起始 token 策略。
- **`max_new_tokens == 0`**：不进入 Decode；是否在 Prefill 后直接结束应由产品语义明确。Phase 1 推荐在 Preparation 或 Prefill 后直接返回 finished。
- **首 token 即 EOS**：视为正常停止，不应返回错误。

---

## 7. LayerRunner、OpContext 与 DispatchTable

### 7.1 LayerRunner 的职责

`LayerRunner` 用于标准化单层 decoder layer 的执行骨架，推荐固定为：

1. RMSNorm / InputNorm
2. QKV Linear
3. RoPE
4. Attention（含 KV 读写）
5. Output Projection
6. Residual Add
7. Post-Attention Norm
8. Gate / Up Projection
9. Activation
10. Down Projection
11. Residual Add

其目标是：

- 避免 Prefill / Decode 分别维护各自的层逻辑
- 将“执行路径组织”与“单层骨架”解耦
- 为 Reference / Optimized 的一致签名提供锚点

### 7.2 OpContext

算子层不得直接依赖全局单例。Executor 应向算子传入明确的 `OpContext`，典型内容包括：

- `ThreadingPolicy* threading`
- `WorkspaceArena* workspace`
- `ProfilingSink* profiling`
- `DispatchTable* dispatch`

它的职责是为 Operator Layer 提供一致的执行期辅助信息。

### 7.3 DispatchTable

Reference / Optimized 路径不应各自维护一套执行器。推荐通过 `DispatchTable` 将差异收敛到算子分发层：

```cpp
struct DispatchTable {
    EmbeddingFn embedding{nullptr};
    RmsNormFn rmsnorm{nullptr};
    RopeFn rope{nullptr};
    LinearFn linear{nullptr};
    AttentionFn attention{nullptr};
    ArgmaxFn argmax{nullptr};
};
```

原则：

- init-time resolve
- hot-path direct call
- 不在热路径上使用 map / string / registry 动态查找

### 7.4 Reference / Optimized 共存原则

- Executor 只维护一套控制流
- LayerRunner 只维护一套逻辑骨架
- Reference / Optimized 的差异只通过 `DispatchTable` 体现
- 对同一逻辑输入，Reference 与 Optimized 应满足预期一致性或容差约束

---

## 8. KV Cache、Workspace 与内存约束

### 8.1 与 KVCacheView 的交互

- Executor 不直接操作底层 KV 物理内存，而是通过 `KVCacheView` 访问。
- `KVCacheView` 负责提供按 `layer / kv_head / seq_pos` 的逻辑访问能力。
- Attention 通过 `KVCacheView` 读取历史 KV，并追加写入当前位置 KV。
- Executor 不得把底层静态线性布局常量硬编码进控制流。

### 8.2 KV 写入与读取语义

#### Prefill

- 一次性处理 Prompt 区间
- 将 Prompt 对应的 KV 写入位置 `[0, prompt_len)`

#### Decode

- 每步只写入一个新的逻辑位置
- 每步读取 `[0, current_pos)` 的历史 KV

### 8.3 WorkspaceBinding

Executor 应在 Session Preparation 阶段为当前 Session 绑定工作区，典型内容包括：

- `hidden_a`
- `hidden_b`
- `logits`
- `q_buf`
- `k_buf`
- `v_buf`
- `attn_scratch`

建议通过简单的 `WorkspaceArena` 或等价布局描述在一块预留内存中进行切分；Phase 1 不要求先引入复杂的 `WorkspacePlanner` 体系。

### 8.4 内存约束

#### Decode 稳态零分配

在 Decode 循环中，禁止：

- `malloc / free`
- `new / delete`
- `std::vector` 扩容
- 隐式构造 owning tensor
- 按步动态申请 workspace

#### Buffer 复用

建议使用层间 ping-pong hidden buffer：

- `hidden_a`
- `hidden_b`

每层交替作为输入输出，以避免为每层分配新的 hidden state buffer。

### 8.5 生命周期

- `KVCacheView` 与 `WorkspaceBinding` 的逻辑绑定由 Session / Executor 管理
- 物理内存由各自的 Manager 或 Runtime 持有
- Session 结束后，逻辑绑定解除；是否立即归还底层物理资源由上层策略决定

---

## 9. 错误处理、验证与测试

### 9.1 错误分类

#### Preparation 阶段错误

- Prompt 长度非法
- `max_new_tokens` 非法
- KV 预留失败
- Workspace 绑定失败

#### 执行阶段错误

- 某层算子执行失败
- Dispatch 函数指针为空
- Logits / hidden / KV view 非法

#### 正常停止

以下情况属于正常结束，不是错误：

- 命中 EOS
- 达到 `max_new_tokens`

### 9.2 FinishReason

建议至少定义：

- `kNone`
- `kEos`
- `kMaxNewTokens`
- `kError`

### 9.3 错误传播策略

- Phase 1 推荐使用 `Status` 或 `Expected` 风格返回错误
- 热路径不抛出异常
- 错误一旦产生，应使 Session 进入 finished 或 error 终态，避免继续在不一致状态下运行

### 9.4 测试与验证

#### Operator Correctness

- 针对 Embedding / Norm / Linear / Attention / Argmax 做单元测试
- Reference kernel 作为主要正确性基线

#### Reference vs Optimized Consistency

- 同一输入下比较 Reference 与 Optimized 路径
- 误差标准由算子级 contract 定义，而不是统一追求 bit-identical

#### End-to-End Generation Regression

- 比较 Executor 输出 Token 序列与 HuggingFace 参考实现
- 验证 greedy generation、EOS、`max_new_tokens` 等停止逻辑

#### Zero-Allocation Check

- 在 Decode 稳态路径中通过分配钩子或计数器验证无新增堆分配

#### Determinism Regression

- 在固定平台、固定线程数和固定 dispatch 下，重复执行结果应一致

---

## 10. 线程模型与实现路径约束

### 10.1 线程模型

- **单控制流**：Executor 顶层逻辑在调用者线程内同步运行，无内部后台线程。
- **算子级并行 (Intra-op)**：允许算子内部使用 OpenMP 或等价机制进行并行加速。
- **禁止请求级并行**：Phase 1 不引入多会话共享的执行线程模型。

### 10.2 线程策略约束

Phase 1 应优先保持单一主导的 intra-op 线程策略，并避免 oversubscription：

- 若选择 OpenMP，则外部 BLAS / 数学库的线程策略必须受统一配置约束。
- 应避免多个线程运行时在热路径中重复叠加。
- 不建议在 Phase 1 中同时引入 OpenMP、自建线程池和多线程 BLAS。

### 10.3 热路径约束

- 热路径算子调用通过函数指针静态分发，不使用虚函数。
- 不在 Decode 每步动态解析 dispatch key。
- 不在热路径构造高层 owning tensor 或复杂容器。
- 不在热路径使用字符串或 registry 动态查找算子。

### 10.4 Phase 1 Practical

- **硬编码 Llama 结构**：Phase 1 优先满足 Llama-family dense decoder 路径。
- **Argmax 唯一**：不引入概率分布处理或采样策略层。
- **CPU first**：所有执行语义以 CPU 后端实现为基线。
- **Reference first**：先跑通 Reference 路径，再补充 Optimized Dispatch。

---

## 11. Phase 2 演进兼容性

当前 Executor 设计应支持后续平滑演进，而不破坏顶层执行语义。

### 11.1 向 Paged KV 演进

- 保持 `KVCacheView` 作为上层访问边界
- Phase 2 替换底层 `KVCacheStorage` 即可
- Executor / LayerRunner 尽量不感知底层物理布局变化

### 11.2 向 Batching 演进

- 将 `SessionState` / `SessionExecutionState` 从单 session 扩展为 batch view
- 保留 `LayerRunner`、`OpContext`、`DispatchTable` 的设计思路

### 11.3 向 GPU 后端演进

- 保持 `Executor` / `PrefillPath` / `DecodePath` 的高层语义不变
- 替换 `DispatchTable`、`WorkspaceBinding`、`OpContext` 和底层 view 实现

### 11.4 向复杂采样演进

- 当前 Argmax 路径可在 Decode 末端替换为独立的 Sampling 模块
- Executor 仅负责组织 logits 到 next token 的控制流

---

**文档所有者**: AetherMind 架构团队  
**下次更新**: 当 Executor 头文件骨架冻结后，补充接口与测试基线
