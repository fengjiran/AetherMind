---
title: AetherMind Phase 1 开发计划草稿
status: draft
version: v0.2
date: 2026-03-26
author: OpenCode / Sisyphus
source_documents:
  - docs/designs/phase1_implementation_breakdown.md
  - docs/designs/aethermind_arch_design.md
  - docs/designs/executor_design.md
  - docs/designs/kv_cache_design.md
  - docs/designs/loaded_model_design.md
  - docs/designs/operator_contract_design.md
---

# AetherMind Phase 1 开发计划草稿

**状态**: Draft  
**版本**: v0.2  
**日期**: 2026-03-26  
**作者**: OpenCode / Sisyphus

---

## 目录

1. [概述](#1-概述)
2. [当前仓库基线评估](#2-当前仓库基线评估)
3. [需先冻结的架构决策](#3-需先冻结的架构决策)
4. [里程碑执行计划](#4-里程碑执行计划)
5. [推荐开发切片](#5-推荐开发切片)
6. [验证计划](#6-验证计划)
7. [风险控制](#7-风险控制)
8. [建议开工顺序](#8-建议开工顺序)
9. [结论](#9-结论)

---

## 1. 概述

### 1.1 文档目的

本文档基于 `docs/designs/phase1_implementation_breakdown.md` 与当前仓库实现现状，整理出一份可执行的 Phase 1 开发计划草稿，用于指导 AetherMind 从现有基础设施推进到可运行的 CPU 本地推理主链路。

本文档的目标不是替代详细设计文档，而是回答以下工程落地问题：

- 当前仓库已有哪部分可直接复用
- Phase 1 主链路还缺哪些关键模块
- 里程碑应以什么顺序推进
- 每个阶段应如何验证，才能避免返工

### 1.2 设计基线

本计划建立在以下设计文档之上：

- `docs/designs/phase1_implementation_breakdown.md`
- `docs/designs/aethermind_arch_design.md`
- `docs/designs/executor_design.md`
- `docs/designs/kv_cache_design.md`
- `docs/designs/loaded_model_design.md`
- `docs/designs/operator_contract_design.md`

### 1.3 Phase 1 边界

Phase 1 的目标边界保持不变：

- token IDs in / token IDs out
- 单请求同步执行
- 静态 KV Cache
- Reference-first, Optimized-second
- 面向桌面 / 服务器 CPU 本地推理

---

## 2. 当前仓库基线评估

### 2.1 已有可复用底座

当前仓库已经具备以下基础设施，可作为 Phase 1 的直接起点：

- `ammalloc`：完整内存池实现
- Tensor / Storage / DataType / Device：基础张量与内存设施已存在
- `dispatcher.h` / `function.h` / `operator_name.h`：已有通用调度雏形
- 单元测试与 benchmark 基础设施已存在

这些基础设施说明仓库并非空白工程，尤其是在底层内存与基础类型方面已经有较好的工程支撑。

### 2.2 当前主要缺口

当前缺失的 Phase 1 核心能力包括：

- `TensorView` / `MutableTensorView`
- `Status` / `StatusOr`
- `RuntimeContext` / `LoadedModel` / `SessionState` / `SessionExecutionState`
- 模型加载链路：`config.json`、safetensors、repack
- Reference kernels：Embedding / RMSNorm / RoPE / Attention / Linear / SwiGLU / Argmax
- Executor / KV Cache / Workspace
- 面向推理 runtime 的 C++ API / C ABI
- 对应的 Phase 1 测试用例与验证脚手架

### 2.3 基线判断

结论上，当前仓库属于“基础设施较完整，但推理主链路尚未开始系统实现”的状态。

这意味着开发策略应当是：

- 复用已有底座
- 避免大范围重写现有基础库
- 从 contract、loader、reference kernels、executor、API 逐层推进

---

## 3. 需先冻结的架构决策

在正式开工前，建议先冻结两项关键决策，否则中后期会出现明显返工。

### 3.1 错误模型

建议新增 `Status` / `StatusOr`，不要直接将现有 `Error` 扩展为主执行路径错误模型。

原因：

- 设计文档已按 contract/result 风格建模
- 推理链路、loader 与 C ABI 更适合显式错误返回
- 现有 `Error` 可继续保留在断言、异常、非预期失败场景中使用

建议策略：

- `Status` / `StatusOr` 作为 Phase 1 主执行路径结果类型
- `Error` 继续作为底层异常或调试辅助设施存在
- 两者边界清晰，不互相替代

### 3.2 目录结构

建议新增 `include/aethermind/...` 与相应 `src/...` 分层目录，保留现有基础设施，不做大规模迁移。

建议分层如下：

- `include/aethermind/base/`
- `include/aethermind/model/`
- `include/aethermind/ops/`
- `include/aethermind/execution/`
- `include/aethermind/api/`

这样做的原因：

- 最小化对当前平铺 `include/` 结构的扰动
- 为 Phase 1 推理模块提供清晰边界
- 与设计文档中的模块划分保持一致

---

## 4. 里程碑执行计划

## M1 — 契约与骨架

目标：先冻结接口与基础对象，不进入复杂算法实现。

### M1.1 基础 contract

实现内容：

- `StatusCode`, `Status`, `StatusOr<T>`
- `TensorView`, `MutableTensorView`
- `GenerationConfig`
- `OpContext`
- 推理侧需要的基础 dtype / shape / quantization contract

交付物：

- 以头文件为主的基础契约层
- 少量必要 `.cpp` 实现

验证：

- 新增 `Status.*` 单测
- 新增 `TensorView.*` 单测
- 新增 `GenerationConfig.*` 单测

### M1.2 Runtime 基础骨架

实现内容：

- `RuntimeContext`
- `LoadedModel`
- `SessionState`
- `SessionExecutionState`
- `DispatchTable` 骨架
- `ModelConfig` 只读结构

验证：

- 新增 `RuntimeContract.*`
- 新增 `DispatchTable.*`

### M1.3 构建系统收敛

实现内容：

- 将新目录纳入 `src/CMakeLists.txt`
- 明确 Phase 1 对外头文件出口
- 初期继续挂在 `AetherMind` 主库下，不急于拆分子目标

验证：

- `cmake --build build --target AetherMind -j`

## M2 — Loader 与正确性主链路

目标：打通“读取模型元数据并构建 `LoadedModel`”主链路。

### M2.1 Config Parser

实现内容：

- 解析 HuggingFace 风格 `config.json`
- 仅支持 Llama-family dense
- 对 unsupported attention / norm / activation / quantization 配置执行 reject

验证：

- 新增 `ModelConfigContract.*`

### M2.2 Safetensors Loader

实现内容：

- 解析 safetensors header
- 构建 tensor name -> metadata 映射
- 处理 shard 与 dtype 校验
- 首阶段优先完成 metadata 与映射，不急于复杂预处理

验证：

- 新增 `SafetensorsLoader.*`
- 先打通 Loader Smoke Test

### M2.3 LoadedModel 构建

实现内容：

- 将 config 与 safetensors metadata 组装成 `LoadedModel`
- 绑定层数、head 数、hidden size、vocab size、rope 参数等只读元信息
- 首阶段允许权重以原始存储配合轻量视图存在

验证：

- 新增 `LoadedModel.*`

## M3 — Reference Kernels

目标：建立 correctness baseline，严格遵循 Reference-first。

建议实现顺序：

1. `Embedding`
2. `RMSNorm`
3. `Linear`
4. `SwiGLU`
5. `RoPE`
6. `Attention`
7. `Argmax`

实现顺序这样安排的原因是：

- 先完成无状态或低状态复杂度算子
- 再进入需要序列与 KV 语义的 Attention

验证：

- 新增 `ReferenceCpuKernels.Embedding*`
- 新增 `ReferenceCpuKernels.RMSNorm*`
- 新增 `ReferenceCpuKernels.Linear*`
- 其余算子按相同模式补齐
- 尽量与 Python / HuggingFace reference 固定输入输出比对

## M4 — 执行层

目标：打通 Prefill + Decode 最小主链路。

### M4.1 KV Cache

实现内容：

- 静态预分配
- `KVCacheManager`
- `KVCacheView`
- `append / read / reset / release`

关键约束：

- Executor 不直接依赖物理布局公式
- 统一通过 `KVCacheView` 间接访问，避免未来切换到 paged KV 时返工

验证：

- 新增 `KVCache.*`

### M4.2 Workspace Binding

实现内容：

- 在 session preparation 阶段完成 workspace 绑定
- Decode 稳态路径不再触发新的堆分配

验证：

- 增加 instrumentation 或分配计数验证
- 对应 Zero-allocation gate

### M4.3 Executor State Machine

实现内容：

- `PrepareSession()`
- `Prefill()`
- `DecodeStep()`
- `Generate()`
- `LayerRunner`

验证：

- 新增 `LlamaDecode.SingleStep*`
- 新增 `LlamaDecode.GreedyGeneration*`

## M5 — API 与交付接口

目标：形成对外可用 runtime 接口。

### M5.1 C++ API

实现内容：

- Runtime / Model / Session 生命周期管理
- token IDs in / token IDs out

### M5.2 C ABI

实现内容：

- `am_runtime_*`
- `am_model_*`
- `am_session_*`
- `am_error_*`

### M5.3 错误桥接

实现内容：

- `Status` / C++ 内部错误到 C ABI error object 的桥接
- 明确资源所有权与销毁边界

验证：

- 新增 `CAbiRuntime.*`

---

## 5. 推荐开发切片

建议按以下顺序推进，以尽快获得可验证反馈。

### Slice A：M1 最小骨架

完成 M1.1 + M1.2 最小骨架：

- contract
- runtime 对象骨架
- 不引入真实执行逻辑

### Slice B：Loader Smoke Test

- `config.json`
- safetensors metadata 解析

### Slice C：Single Operator Correctness

- 优先 Embedding / RMSNorm / Linear

### Slice D：Single-Step Decode

- `LoadedModel + KVCacheView + Workspace + DecodeStep`

### Slice E：End-to-End Greedy Generation

- Prompt token IDs 输入
- 输出 token IDs 序列

### Slice F：交付准备

- C ABI
- benchmark
- 第一条 INT8 optimized path

---

## 6. 验证计划

当前仓库已有测试框架，但缺少绝大多数 Phase 1 目标测试。建议按阶段同步补齐，而不是在主链路打通后再集中补测。

### 6.1 M1 对应测试

- `test_status.cpp`
- `test_tensor_view.cpp`
- `test_runtime_contract.cpp`
- `test_dispatch_table.cpp`

### 6.2 M2 对应测试

- `test_model_config.cpp`
- `test_safetensors_loader.cpp`
- `test_loaded_model.cpp`

### 6.3 M3 对应测试

- `test_reference_kernels.cpp`

### 6.4 M4 对应测试

- `test_kv_cache.cpp`
- `test_llama_decode.cpp`

### 6.5 M5 对应测试

- `test_c_abi_runtime.cpp`

### 6.6 推荐验证节奏

- 先跑最窄相关单测
- 再扩到受影响模块测试
- 最后再跑 benchmark / TSAN / ASAN 等较重验证

---

## 7. 风险控制

最需要优先规避的风险如下：

1. 不要将异常式 `Error` 直接当作 `Status` 替代品
2. 不要让 Executor 感知 KV 物理布局公式
3. 不要在 Reference 路径未稳定前过早投入 SIMD / INT4 优化
4. 不要让 `RuntimeContext / SessionState / SessionExecutionState` 职责混杂

换言之，Phase 1 成功的关键不是“尽快加优化”，而是先建立边界清晰、测试可回归、可逐层扩展的主链路。

---

## 8. 建议开工顺序

### 第 1 周期

- 冻结错误模型与目录结构决策
- 建立 Phase 1 目录
- 落 `Status` / `StatusOr`
- 落 `TensorView`
- 落 runtime skeleton
- 补对应单测

### 第 2 周期

- Config parser
- Safetensors metadata loader
- `LoadedModel`
- Loader smoke test

### 第 3 周期

- Embedding / RMSNorm / Linear / SwiGLU
- Reference kernel tests

### 第 4 周期

- RoPE / Attention
- KV cache
- Single-step decode

### 第 5 周期

- Prefill + Generate
- Greedy end-to-end

### 第 6 周期

- C ABI
- 第一条 INT8 optimized path
- benchmark / TSAN / ASAN 验证

---

## 9. 结论

当前仓库具备良好的基础设施底座，但 Phase 1 推理主链路仍需从 contract、loader、reference kernels、executor 到 API 逐层落地。

建议严格遵循以下顺序：

1. 先冻结 contract
2. 先打通 loader
3. 先完成 Reference 正确性
4. 再做 executor 与 KV
5. 最后补 C ABI 与优化路径

该顺序能最大限度降低返工风险，并尽早获得可验证的工程反馈。
