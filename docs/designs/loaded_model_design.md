# AetherMind Phase 1 Loaded Model 设计文档

**版本**: v1.0  
**日期**: 2026-03-25  
**作者**: AetherMind Team

---

## 目录

1. [概述](#1-概述)
2. [职责与范围](#2-职责与范围)
3. [对象边界与核心概念](#3-对象边界与核心概念)
4. [加载链路与构建流程](#4-加载链路与构建流程)
5. [Loaded Model 的内部组成](#5-loaded-model-的内部组成)
6. [量化适配与分发绑定](#6-量化适配与分发绑定)
7. [生命周期与只读保证](#7-生命周期与只读保证)
8. [错误处理与验证](#8-错误处理与验证)
9. [非目标与后续演进](#9-非目标与后续演进)

---

## 1. 概述

### 1.1 文档目的

本文档定义 AetherMind Phase 1 中 `Loaded Model` 的详细设计。`Loaded Model` 是模型加载链路的最终产物，也是 Executor 热路径依赖的唯一只读可执行模型对象。

### 1.2 Phase 1 定位

在 Phase 1 中，模型对象体系收敛为两层：

- **Model Assets**：来自磁盘与 Safetensors 的原始模型资产视图
- **Loaded Model**：经过验证、重排、量化适配和 dispatch 绑定后的只读内部可执行表示

本文档的重点不是描述模型推理过程本身，而是冻结 `Loaded Model` 的边界、组成、所有权与生命周期契约。

---

## 2. 职责与范围

### 2.1 核心职责

`Loaded Model` 在 Phase 1 中承担以下职责：

- **持有模型配置**：持有经校验后的 Llama-family 配置元数据
- **持有重排后权重**：持有适合 CPU 执行路径的内部权重表示
- **量化适配**：承载 INT8 / INT4 权重适配结果
- **分发绑定**：持有 init-time resolve 后的算子分发表
- **只读执行边界**：向 Executor 提供只读执行期访问能力
- **元数据暴露**：暴露层数、头数、hidden size、KV 相关维度等只读元信息

### 2.2 非目标

Phase 1 的 `Loaded Model` 不承担：

- 多模型调度
- serving 级模型热切换
- 动态权重修改
- LoRA 动态挂载
- MoE 路由
- 通用图执行框架的 graph ownership

---

## 3. 对象边界与核心概念

### 3.1 两层对象模型

#### Model Assets

表示来自模型目录的原始资产视图，典型包括：

- `config.json`
- Safetensors metadata
- tensor shard 索引信息
- 原始 tensor 的 mmap / 加载视图

职责：

- 输入校验
- 元数据提取
- 原始 tensor 定位

不承担热路径职责。

#### Loaded Model

表示经过构建后的内部只读执行对象。

职责：

- 持有解析后的模型配置
- 持有重排后的权重
- 持有量化适配信息
- 持有 dispatch binding
- 暴露 Executor 所需的只读访问边界

### 3.2 为什么不引入第三层对象

Phase 1 已有意避免将模型对象强行拆成更多层级（如 ExecutableModel / Handle / Graph）。原因是：

- 当前目标是尽快建立稳定可执行的 CPU runtime
- 热路径只需要一个只读执行对象
- 过早引入更多中间层会增加实现与命名复杂度

如果未来需要支持多后端、多模型共享或更复杂 ABI，再考虑进一步拆分。

---

## 4. 加载链路与构建流程

### 4.1 加载输入

Phase 1 模型加载输入固定为：

- HuggingFace 风格模型目录
- `config.json`
- `*.safetensors`

### 4.2 构建流程

推荐流程：

```text
Model Directory
   │
   ├─ config.json
   └─ *.safetensors
   │
   ▼
Config Parser
   │
   ▼
Model Assets
   │
   ▼
Safetensors Loader
   │
   ▼
Repack Store
   │
   ▼
Quantization Adaptation
   │
   ▼
Dispatch Binding
   │
   ▼
Loaded Model
```

### 4.3 各阶段职责

#### Config Parser

- 解析 `config.json`
- 验证是否为支持的 Llama-family dense 模型
- 显式拒绝不支持结构（如 sliding window / encoder-decoder / MoE）

#### Safetensors Loader

- 解析 Safetensors header
- 建立 tensor name 到底层存储的映射
- 校验 shard 与 dtype 元数据

#### Repack Store

- 将外部权重重排为 CPU 友好的内部布局
- 为 Linear / Attention 等算子提供稳定输入格式
- 避免 decode 时临时做布局转换

#### Quantization Adaptation

- 将 PRD 支持的量化权重映射到内部格式
- 支持 `INT8 per-channel`
- 支持 `INT4 group-wise (group_size=128)`

#### Dispatch Binding

- 根据 `op_id + cpu_feature + quant_scheme` 绑定函数指针
- 在加载阶段完成 resolve
- 不把动态分发留到 Decode 热路径

---

## 5. Loaded Model 的内部组成

### 5.1 建议组成

`Loaded Model` 至少应包含：

- **ModelConfig**
  - hidden size
  - num layers
  - num attention heads
  - num kv heads
  - head dim
  - vocab size
  - rope 参数
- **Repacked Weights**
  - embedding
  - layer-wise attention / MLP weights
  - final norm / lm head
- **Quantization Metadata**
  - quant scheme
  - scales / zero points / group metadata
- **DispatchTable**
  - 已绑定的算子函数指针
- **Read-only Runtime Metadata**
  - dtype
  - alignment
  - memory footprint
  - supported execution constraints

### 5.2 所有权边界

- `Loaded Model` 拥有其内部重排后权重与量化元数据的生命周期
- Executor 仅借用其只读视图
- Session 不拥有模型权重
- Runtime / ModelManager 负责 Loaded Model 的创建与销毁

### 5.3 只读保证

Phase 1 明确要求：

- Loaded Model 在成功构建后进入只读状态
- Executor / Operator 不允许修改模型权重
- 多 Session 场景下允许共享同一个 Loaded Model 的只读访问

---

## 6. 量化适配与分发绑定

### 6.1 量化边界

量化支持范围以 PRD 为准：

- `INT8 per-channel`
- `INT4 group-wise (group_size=128)`

其他量化方案应显式拒绝。

### 6.2 Repack 与 Quant 的关系

Repack 与量化适配需要协同设计：

- Repack 后的权重布局必须满足目标 kernel 的访问模式
- Quant metadata 必须和重排后张量绑定，而不是悬空存在
- 不允许在 Decode 热路径动态推导量化布局

### 6.3 Dispatch Binding

DispatchTable 绑定原则：

- init-time resolve
- hot-path direct call
- unsupported combination 在加载阶段失败

### 6.4 为什么要在 Loaded Model 中持有 Dispatch

因为执行器依赖的是“可执行模型”，而不是“原始权重”。因此：

- Dispatch binding 是 Loaded Model 的一部分
- 它和重排权重、量化适配结果一起构成内部可执行表示

---

## 7. 生命周期与只读保证

### 7.1 生命周期阶段

推荐状态流：

```text
Unloaded
   -> Load Assets
   -> Validate Config
   -> Repack Weights
   -> Bind Dispatch
   -> Loaded
   -> Unload
```

### 7.2 构建完成后的不变量

一旦进入 `Loaded` 状态，以下不变量应成立：

- 模型配置已校验完成
- 必需权重已加载并重排完成
- 量化元数据完整
- DispatchTable 已绑定完成
- Executor 所需元数据可稳定读取

### 7.3 卸载语义

卸载时：

- 释放 Loaded Model 自身持有的重排权重与量化元数据
- 释放与其绑定的辅助元数据
- 不再允许 Session / Executor 继续访问

如果存在活跃 Session，卸载必须由上层生命周期策略决定是否拒绝或延迟。

---

## 8. 错误处理与验证

### 8.1 错误来源

#### 配置错误

- 非 llama-family
- 不支持的 attention / activation / norm
- 缺少必需字段

#### 资产错误

- 缺失 `config.json` 或 `*.safetensors`
- tensor 名称不匹配
- shard 不完整
- dtype 非法

#### 构建错误

- repack 失败
- 量化适配失败
- dispatch 绑定失败
- 内存分配失败

### 8.2 错误处理策略

- 推荐使用 `Status` / `StatusOr` / `Expected`
- 加载阶段允许详细错误消息
- 热路径不负责恢复模型构建错误

### 8.3 验证与测试

建议至少覆盖以下验证：

#### 1. 配置契约验证

- 支持/拒绝策略与 PRD 一致

#### 2. Safetensors 完整性验证

- tensor 名称、shape、dtype 与期望一致

#### 3. Repack 正确性

- 重排前后张量数值语义一致

#### 4. Quantization Consistency

- INT8 / INT4 路径与 FP32 参考结果满足容差约束

#### 5. Dispatch Binding 正确性

- 所有必需算子函数指针非空
- 不支持组合在加载阶段显式失败

#### 6. 只读保证

- Executor / Session 不修改 Loaded Model 内容

---

## 9. 非目标与后续演进

### 9.1 Phase 1 非目标

- 多模型共享调度
- GPU 权重副本管理
- 在线量化
- 动态 LoRA 合并
- graph-level execution plan cache

### 9.2 后续演进方向

未来若需要支持：

- GPU backend
- 多后端 dispatch
- 多模型驻留
- LoRA / adapter
- 更复杂 ABI 句柄层

可在不破坏当前 `Loaded Model` 只读执行语义的前提下，逐步扩展对象结构。

---

**文档所有者**: AetherMind 架构团队  
**下次更新**: 当 ModelManager / Loader / Repack 接口冻结后补充实现细节
