# AetherMind Phase 1 实施分解文档 (Implementation Breakdown)

**版本**: v1.0  
**日期**: 2026-03-25  
**作者**: AetherMind Team

---

## 目录

1. [概述](#1-概述)
2. [实施阶段与里程碑](#2-实施阶段与里程碑)
3. [模块依赖图](#3-模块依赖图)
4. [具体工作包](#4-具体工作包)
5. [最小可行切片](#5-最小可行切片)
6. [验证网关](#6-验证网关)
7. [实施顺序推荐](#7-实施顺序推荐)
8. [风险点](#8-风险点)

---

## 1. 概述

本文档将当前已经冻结的架构设计映射为可立即执行的 Phase 1 工程落地拆解，目标是指导 AetherMind 从 0 到 1 完成桌面/服务器 CPU 本地推理运行时的实现。

本文档基于以下设计基线：

- `docs/designs/aethermind_arch_design.md`
- `docs/designs/executor_design.md`
- `docs/designs/kv_cache_design.md`
- `docs/designs/loaded_model_design.md`
- `docs/designs/operator_contract_design.md`

Phase 1 的边界保持不变：

- token IDs in / token IDs out
- 单请求同步执行
- 静态 KV Cache
- Reference-first, Optimized-second
- 桌面/服务器 CPU 本地推理

---

## 2. 实施阶段与里程碑

Phase 1 建议划分为四个阶段：

| 里程碑 | 名称 | 核心目标 | 关键交付物 |
|--------|------|----------|------------|
| **M1** | 契约与骨架 | 冻结核心接口与数据契约 | 目录结构、核心头文件、Status 基础设施 |
| **M2** | 加载与正确性 | 打通模型加载链，跑通 Reference 路径 | Loader、Reference 算子、最小执行器 |
| **M3** | 架构收敛 | 完善 KV Cache 与零分配 Decode 路径 | 静态 KV Manager、Workspace 绑定、Prefill/Decode 状态机 |
| **M4** | 交付准备 | 落地优化路径、C ABI 与集成验证 | INT8/INT4 优化算子、C ABI、基准与回归验证 |

---

## 3. 模块依赖图

```text
[HAL / Platform]
      │
      ▼
[Operator Layer (Ref/Opt)] ◄──── [Repack Store]
      │                                │
      ▼                                ▼
[Dispatch Table] ◄────────────── [Loaded Model]
      │                                ▲
      ▼                                │
[Executor (Prefill/Decode)] ◄─── [Model Manager / Loader]
      │                                │
      ▼                                ▼
[KV Cache Manager] ◄──────────── [Model Config]
      │
      ▼
[API Layer (C++ / C ABI)]
```

依赖顺序的关键原则：

- 先冻结 contract，再写实现
- 先跑通 Reference，再加 Optimized
- 先打通单条正确性链路，再补完整优化路径

---

## 4. 具体工作包

## WP 1: 基础设施与骨架

### 任务 1.1: 目录与构建系统收敛

- 按 `include/aethermind/` 与 `src/` 的分层布局建立目录
- 确保 CMake 目标结构能承载 Runtime / Model / Execution / Ops / Dispatch / Platform

### 任务 1.2: 核心数据契约

- 实现轻量 `TensorView` / `MutableTensorView`
- 定义 `Status` / `StatusOr` / 错误码基础设施
- 冻结 `OpContext`、`DispatchTable`、`GenerationConfig` 等基础头文件

### 任务 1.3: Runtime 基础对象骨架

- `RuntimeContext`
- `LoadedModel`
- `SessionState`
- `SessionExecutionState`

---

## WP 2: 模型加载链路

### 任务 2.1: Config Parser

- 解析 `config.json`
- 实现 Llama-family dense 支持/拒绝策略
- 校验 attention / activation / norm / quantization 配置

### 任务 2.2: Safetensors Loader

- 解析 header
- 建立 tensor name → 存储映射
- 处理 shard 与 dtype 校验

### 任务 2.3: Repack Store

- 将权重转为 CPU 友好布局
- 为 Linear / Attention 等算子提供稳定输入格式
- 避免 Decode 热路径临时布局转换

### 任务 2.4: Loaded Model 构建

- 组装 `Model Assets` → `Loaded Model`
- 绑定 `DispatchTable`
- 填充只读元数据与量化信息

---

## WP 3: 算子实现

### 任务 3.1: Reference Kernels

优先实现：

- Embedding
- RMSNorm
- RoPE
- Attention
- Linear
- SwiGLU
- Argmax

要求：

- 正确性优先
- 数值稳定
- 作为 Reference 基线

### 任务 3.2: Dispatch Table

- 实现 `(op_id, dtype, quant_scheme, cpu_feature) -> fn_ptr`
- 在加载阶段完成 resolve
- 不把查找留在热路径

### 任务 3.3: 第一条 Optimized 路径

- 优先落 `Linear INT8`
- 再逐步补 `INT4`
- 其他优化算子在 Reference 路径稳定后追加

---

## WP 4: 执行层

### 任务 4.1: Executor 状态机

- `PrepareSession()`
- `Prefill()`
- `DecodeStep()`
- `Generate()`

### 任务 4.2: LayerRunner

- 复用 decoder layer 固定执行骨架
- 统一 Prefill / Decode 的单层执行路径

### 任务 4.3: KV Cache Manager

- 静态预分配 KV
- 通过 `KVCacheView` 提供逻辑索引
- 支持 append / read / reset / release

### 任务 4.4: Workspace 绑定

- 在 Session Preparation 阶段完成绑定
- 保证 Decode 热路径稳态零分配

---

## WP 5: API 与交付接口

### 任务 5.1: C++ API

- Runtime / Model / Session 生命周期
- token IDs in / token IDs out

### 任务 5.2: C ABI

- `am_runtime_*`
- `am_model_*`
- `am_session_*`
- `am_error_*`

### 任务 5.3: 错误桥接与资源销毁

- C++ 错误到 C ABI 错误对象的桥接
- 生命周期安全销毁

---

## 5. 最小可行切片

建议按以下顺序推进，以便最快得到可验证反馈：

### Slice 1: Loader Smoke Test

目标：

- 输入一个 HuggingFace 风格 Llama 模型目录
- 正确解析 config 与 safetensors metadata
- 打印或验证关键模型结构

### Slice 2: Single Operator Correctness

目标：

- 用固定输入分别验证 Embedding / RMSNorm / RoPE / Linear / Attention
- 确立 Reference 基线

### Slice 3: Single-Step Decode

目标：

- 构造 `Loaded Model + KVCacheView + Workspace`
- 跑通单步 Decode

### Slice 4: End-to-End Greedy Generation

目标：

- 从 Prompt token 输入到输出 token 序列
- 跑通 Prefill + Decode 全链路

---

## 6. 验证网关

每个阶段进入下一阶段前建议通过如下网关：

### 6.1 Correctness Gate

- Reference kernel 与 Python / HuggingFace 参考结果一致

### 6.2 Consistency Gate

- Optimized kernel 与 Reference kernel 在容差范围内一致

### 6.3 Zero-Allocation Gate

- Decode 稳态路径中堆分配调用次数为 0

### 6.4 Ownership Gate

- ASAN：无内存泄漏
- TSAN：无数据竞争
- Session / Loaded Model 生命周期关系正确

### 6.5 End-to-End Gate

- 生成 token 序列满足 greedy contract
- EOS / max token 停止逻辑正确

---

## 7. 实施顺序推荐

推荐顺序：

1. **先冻结头文件与 contract**
2. **先打通 Loader 与 Loaded Model**
3. **先实现 Reference Kernels**
4. **再实现 Executor + KV Cache + Workspace**
5. **最后补 C ABI 与第一条 Optimized Path**

换句话说：

- 不要先写优化，再补正确性
- 不要先写 C ABI，再补内部执行链路
- 不要先扩展功能边界，再打通单请求主链路

---

## 8. 风险点

### 8.1 KV 布局泄漏

如果 Attention 或 Executor 直接依赖物理 offset 公式而不是 `KVCacheView`，Phase 2 切到 Paged KV 时会导致执行层返工。

### 8.2 过早优化

在 Reference 路径未跑通前投入大量精力做 SIMD / 量化优化，会显著增加调试复杂度。

### 8.3 量化误差堆叠

INT4 在 CPU 上更容易出现误差放大，需要尽早定义算子级容差标准。

### 8.4 状态对象膨胀

如果 `RuntimeContext / SessionState / SessionExecutionState / OpContext` 的职责边界不清晰，很容易导致函数参数膨胀与状态重复。

---

**文档所有者**: AetherMind 架构团队  
**下次更新**: 当 M1/M2 的头文件骨架和首条端到端链路打通后补充实现映射
