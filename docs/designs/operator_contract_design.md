# AetherMind Phase 1 Operator Contract 设计文档

**版本**: v1.0  
**日期**: 2026-03-25  
**作者**: AetherMind Team

---

## 目录

1. [概述](#1-概述)
2. [目标与非目标](#2-目标与非目标)
3. [最小 Tensor / Buffer / View 契约](#3-最小-tensor--buffer--view-契约)
4. [OpContext 与执行期依赖](#4-opcontext-与执行期依赖)
5. [算子签名原则](#5-算子签名原则)
6. [Reference / Optimized 一致性规则](#6-reference--optimized-一致性规则)
7. [Workspace、In-place 与布局约束](#7-workspacein-place-与布局约束)
8. [KV 访问边界与 Dispatch 契约](#8-kv-访问边界与-dispatch-契约)
9. [错误处理与验证](#9-错误处理与验证)

---

## 1. 概述

### 1.1 文档目的

本文档定义 AetherMind Phase 1 的算子层执行契约（Operator Contract）。其目标是冻结 Executor、Loaded Model 与各类 kernel 之间的最小接口语义，使 Reference 与 Optimized 实现可以共享一致的逻辑签名并进行稳定对照验证。

### 1.2 Phase 1 定位

Phase 1 的 Operator Contract 面向：

- 桌面/服务器 CPU 本地推理
- decoder-only Transformer
- 单请求同步执行
- 静态 KV Cache
- Reference-first, Optimized-second

它不是一个通用深度学习张量框架，也不承担图执行引擎职责。

---

## 2. 目标与非目标

### 2.1 设计目标

- **统一逻辑签名**：Reference / Optimized 路径共享一致的算子语义
- **轻量化数据视图**：热路径以 borrowed view 为主，不传 owning tensor
- **执行期依赖显式传递**：通过 `OpContext` 提供线程、workspace、profiling、dispatch 信息
- **KV 边界清晰**：Attention 通过 `KVCacheView` 访问历史 KV
- **零分配友好**：算子不在 Decode 热路径申请堆内存

### 2.2 非目标

- 通用自动求导系统
- 通用动态图/静态图 IR
- 调度器感知算子接口
- 多后端统一算子注册框架
- 全功能 Tensor 库

---

## 3. 最小 Tensor / Buffer / View 契约

### 3.1 设计原则

Phase 1 不引入完整深度学习框架式 Tensor 系统，但必须冻结最小可执行的数据视图契约。

### 3.2 最小元信息

每个 `TensorView` / `MutableTensorView` 至少应包含：

- `data pointer`
- `dtype`
- `rank`
- `shape`
- `stride`
- `alignment`
- `mutability`

### 3.3 所有权约束

- Execution / Operator 热路径主要处理 **borrowed view**
- 持久态数据由 Loaded Model / KV Cache / Workspace 持有
- 算子本身不拥有输入输出 buffer 的生命周期

### 3.4 设计准则

- 热路径优先使用轻量 view，而不是 owning tensor
- `shape / stride` 必须显式可见
- 不在算子接口中隐式推断布局
- 不在热路径构造高层容器包装数据

---

## 4. OpContext 与执行期依赖

### 4.1 目的

算子层不得直接依赖全局单例或隐藏状态。所有执行期辅助依赖应通过 `OpContext` 显式传入。

### 4.2 建议内容

`OpContext` 典型包含：

- `ThreadingPolicy* threading`
- `WorkspaceArena* workspace`
- `ProfilingSink* profiling`
- `DispatchTable* dispatch`

### 4.3 约束

- 算子不持有 `OpContext` 生命周期
- `OpContext` 只描述执行期环境，不持有模型权重
- 算子必须允许 `profiling` 等扩展项为空

---

## 5. 算子签名原则

### 5.1 基本原则

- 算子接口描述逻辑输入输出，而不是绑定具体物理实现
- Reference / Optimized 路径在同一签名语义下工作
- 是否 in-place、是否需要 workspace、误差容忍范围必须可定义

### 5.2 Phase 1 优先冻结的算子

- Embedding
- RMSNorm
- RoPE
- Linear
- Attention
- Argmax

### 5.3 每个算子至少应明确

- 输入 view
- 输出 view
- 所需 workspace
- 是否允许 in-place
- 对齐要求
- 数值语义要求

### 5.4 示例形态（非实现承诺）

```cpp
Status RmsNorm(
    const TensorView& input,
    const TensorView& weight,
    MutableTensorView& output,
    float eps,
    OpContext& op_ctx);
```

> 注：以上只是契约形态示意，不代表当前仓库已经存在该接口。

---

## 6. Reference / Optimized 一致性规则

### 6.1 总体原则

- Reference kernel 作为主要正确性基线
- Optimized kernel 必须与 Reference kernel 保持同一逻辑输入输出语义
- 差异仅体现在实现方式和允许的数值容差上

### 6.2 Reference Kernel 要求

- 优先保证正确性和数值稳定性
- 可使用更直接、可验证的实现方式
- 对量化路径可采用更高精度累积作为基线

### 6.3 Optimized Kernel 要求

- 可以使用向量化、blocking、量化优化
- 不改变逻辑签名
- 不引入未声明的额外 side effect
- 必须满足与 Reference 路径的预期一致性或容差约束

---

## 7. Workspace、In-place 与布局约束

### 7.1 Workspace 契约

- 算子所需临时 scratch 必须通过 `OpContext.workspace` 获取
- Decode 热路径中不允许按步动态堆分配 workspace
- Workspace 的切分由 Executor/Workspace 层负责，不由算子自行决定物理分配策略

### 7.2 In-place 规则

- 算子是否允许 in-place 必须显式声明
- 默认不假设 in-place 安全
- 如果允许 in-place，必须说明输入输出别名规则

### 7.3 对齐与步长要求

- 需要特定 alignment 的算子必须显式声明
- `stride` 与 layout 相关要求必须通过 view 或 contract 传递
- 不允许在算子内部偷偷假设连续布局而不在契约中表达

### 7.4 数值语义

- 必须明确累积精度、归一化/激活语义和误差容忍范围
- 量化算子必须与 PRD 定义的 INT8 / INT4 方案对齐
- 不以跨平台 bit-identical 作为通用承诺

---

## 8. KV 访问边界与 Dispatch 契约

### 8.1 KV 访问边界

Attention 算子必须通过 `KVCacheView` 访问 KV：

- 不得直接耦合到底层线性静态缓冲区实现
- 不得在算子签名中暴露物理布局常量
- `seq_pos` 必须按逻辑位置理解，而非物理地址

### 8.2 Dispatch 契约

DispatchTable 负责将算子逻辑签名绑定到具体实现：

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
- unsupported combination 在加载阶段显式失败
- 不在热路径使用 map / string / registry 动态查找

---

## 9. 错误处理与验证

### 9.1 错误处理原则

- 推荐使用 `Status` / `StatusOr` / `Expected`
- 热路径不抛出异常
- 参数非法、view 非法、workspace 不足、dispatch 缺失等都应显式失败

### 9.2 验证与测试

建议至少覆盖以下测试：

#### 1. Operator Correctness

- 对 Embedding / RMSNorm / RoPE / Linear / Attention / Argmax 做单元测试
- Reference kernel 作为主要正确性基线

#### 2. Reference vs Optimized Consistency

- 同一输入下比较 Reference 与 Optimized 路径
- 误差标准按算子 contract 定义，而不是统一追求 bit-identical

#### 3. Workspace / In-place Contract

- 验证需要 workspace 的算子不会隐式申请堆内存
- 验证允许 in-place 的算子遵守别名规则

#### 4. KV Access Contract

- 验证 Attention 只能通过 `KVCacheView` 访问 KV
- 验证算子不依赖底层物理布局常量

---

**文档所有者**: AetherMind 架构团队  
**下次更新**: 当核心算子签名冻结后补充更具体的接口与测试基线
