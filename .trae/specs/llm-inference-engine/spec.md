# AetherMind 大模型推理引擎 - 产品需求文档

## Overview
- **Summary**: 基于现有 AetherMind 基础设施构建高性能、可扩展的大语言模型 (LLM) 推理引擎，支持主流大模型架构（如 Transformer、GPT、BERT 等）的高效推理。
- **Purpose**: 提供一套完整的深度学习推理框架，具备张量计算、算子调度、设备抽象、内存管理等核心能力，支持多种硬件设备（CPU、CUDA、CANN）的高效推理。
- **Target Users**: 深度学习研究人员、AI 应用开发者、需要在生产环境部署大模型的工程师。

## Goals
- 基于现有 AetherMind 张量系统，构建完整的计算图执行引擎
- 实现 Transformer 系列模型的核心算子（如 Attention、FFN、LayerNorm 等）
- 支持多设备（CPU、CUDA、CANN）的算子调度和执行
- 提供 KV Cache 管理能力，优化生成式模型的推理性能
- 设计灵活的算子注册和分发机制
- 支持模型权重的加载和管理

## Non-Goals (Out of Scope)
- 不实现自动微分和训练功能（仅推理）
- 不实现完整的 Python 前端绑定（第一阶段）
- 不实现分布式训练/推理（第一阶段）
- 不实现动态图（第一阶段优先静态图或 eager 模式）

## 整体架构设计

### 架构概览
AetherMind 大模型推理引擎采用分层架构设计，从下到上分为以下层次：

```
┌─────────────────────────────────────────────────────────────┐
│                   高层 API 层                               │
│  (Model API, Generation API, KV Cache Management)         │
├─────────────────────────────────────────────────────────────┤
│                   计算图执行层                              │
│  (Graph Builder, Graph Executor, Memory Planner)          │
├─────────────────────────────────────────────────────────────┤
│                   算子层                                    │
│  (Transformer Ops: Attention, FFN, Norm, RoPE)           │
│  (Tensor Ops: MatMul, Softmax, Element-wise, Reduction)  │
├─────────────────────────────────────────────────────────────┤
│                   算子分发层                                │
│  (Dispatcher, Operator Registry, Dispatch Key)            │
├─────────────────────────────────────────────────────────────┤
│              AetherMind 现有基础设施                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  Tensor  │ │  Type    │ │  Device  │ │ Function │  │
│  │  System  │ │  System  │ │  Abstr.  │ │  System  │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│  ┌──────────────────────────────────────────────────┐    │
│  │          Memory Allocator (ammalloc)             │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件说明

#### 1. 算子分发层 (Dispatcher Layer)
- **Dispatcher**: 全局单例算子调度器
- **Operator Registry**: 算子注册表，存储所有已注册的算子
- **Dispatch Key**: 分发键，包括设备类型（CPU/CUDA/CANN）、数据类型等
- **职责**: 根据输入张量的属性自动选择并调用合适的算子实现

#### 2. 算子层 (Operator Layer)
- **核心张量算子**:
  - Element-wise: Add, Sub, Mul, Div, Exp, Log, Relu, Gelu, Silu
  - Reduction: Sum, Mean, Max, Min
  - MatMul: 矩阵乘法
  - Softmax: Softmax 计算
  - Shape Ops: Reshape, Transpose, Permute
- **Transformer 专用算子**:
  - LayerNorm / RMSNorm
  - Feed-Forward Network (FFN)
  - Multi-Head Attention (MHA)
  - Rotary Positional Embedding (RoPE)

#### 3. 计算图执行层 (Graph Execution Layer)
- **Graph Builder**: 计算图构建 API
- **Graph IR**: 简单的中间表示
- **Graph Executor**: 计算图执行引擎
- **Memory Planner**: 内存规划和复用优化器（可选）

#### 4. 高层 API 层 (High-level API Layer)
- **Model API**: 模型构建和加载 API
- **Generation API**: 文本生成 API
- **KV Cache Manager**: KV Cache 管理器

### 数据流

#### 推理数据流（Eager 模式）
```
用户输入 Tensor
      ↓
Operator 调用
      ↓
Dispatcher 根据设备/数据类型分发
      ↓
选择对应后端的算子实现
      ↓
执行计算
      ↓
返回结果 Tensor
```

#### 增量生成数据流
```
第 1 步：Prefill
   输入: 完整 prompt
   输出: 第 1 个 token + KV Cache

第 2..N 步：Decode
   输入: 上一个 token + KV Cache
   输出: 下一个 token + 更新后的 KV Cache
```

### 扩展点设计

#### 新增设备后端
1. 定义新的 DispatchKey
2. 为现有算子注册该设备的实现
3. 实现该设备的 Allocator
4. 无需修改高层代码

#### 新增算子
1. 定义 OperatorSchema
2. 实现各设备后端的算子逻辑
3. 注册到 Dispatcher
4. 编写单元测试

## Background & Context
AetherMind 项目已经具备以下基础设施：
- **张量系统**：完整的 Tensor、TensorImpl、Storage 抽象
- **类型系统**：支持多种数据类型（float32、float16、bfloat16、int8、float8 等）
- **设备抽象**：Device 类支持 CPU、CUDA、CANN
- **内存管理**：高效的自定义内存分配器（ammalloc）
- **函数系统**：Function 和 FunctionImpl 提供动态函数调用能力
- **调度器框架**：Dispatcher 和 OperatorSchema 提供算子分发基础设施

在此基础上，我们需要构建完整的 LLM 推理引擎。

## Functional Requirements
- **FR-1**: 实现核心张量算子库（Element-wise、Reduction、MatMul、Softmax 等）
- **FR-2**: 实现 Transformer 专用算子（Multi-Head Attention、Feed-Forward Network、LayerNorm、RMSNorm 等）
- **FR-3**: 实现 KV Cache 管理机制，支持增量生成
- **FR-4**: 实现算子注册和分发机制，支持多设备后端
- **FR-5**: 提供计算图构建和执行能力
- **FR-6**: 实现模型权重加载和管理接口
- **FR-7**: 提供简单的模型构建 DSL 或 API

## Non-Functional Requirements
- **NFR-1**: 性能：在 CUDA 设备上，单个 A100 GPU 上支持至少 7B 模型的实时推理
- **NFR-2**: 可扩展性：易于添加新的算子和新的设备后端
- **NFR-3**: 内存效率：支持内存复用和 KV Cache 优化，减少内存占用
- **NFR-4**: 正确性：所有算子实现必须通过数值正确性测试

## Constraints
- **Technical**: 基于现有 AetherMind C++20 代码库，保持 API 一致性
- **Business**: 优先支持主流硬件平台（x86 CPU、NVIDIA CUDA、华为 CANN）
- **Dependencies**: 可以选择性依赖 cuBLAS、oneDNN、CANN 等高性能库作为后端加速

## Assumptions
- 用户对 C++ 有一定了解
- 目标硬件平台支持相应的加速库
- 模型权重以标准格式（如 Safetensors、GGUF 或自定义格式）提供

## Acceptance Criteria

### AC-1: 核心张量算子实现
- **Given**: AetherMind 张量基础设施已就绪
- **When**: 实现 MatMul、Softmax、Element-wise Add/Mul 等核心算子
- **Then**: 所有算子通过数值正确性测试（与 PyTorch 参考实现对比，误差在可接受范围内）
- **Verification**: `programmatic`
- **Notes**: 使用随机张量进行测试，相对误差 < 1e-5

### AC-2: Transformer 算子实现
- **Given**: 核心张量算子已就绪
- **When**: 实现 Multi-Head Attention、FFN、LayerNorm、RMSNorm
- **Then**: 能够构建完整的 Transformer 层并通过端到端测试
- **Verification**: `programmatic`

### AC-3: KV Cache 管理
- **Given**: Transformer 算子已实现
- **When**: 实现 KV Cache 数据结构和管理逻辑
- **Then**: 支持增量生成，第二次及之后的推理时延显著降低
- **Verification**: `programmatic`
- **Notes**: 第二次推理时延应比第一次降低 > 50%

### AC-4: 算子分发机制
- **Given**: Dispatcher 框架已就绪
- **When**: 实现算子注册机制，支持为同一算子注册不同设备的实现
- **Then**: 能根据输入张量的设备类型自动调度到对应的后端实现
- **Verification**: `programmatic`

### AC-5: 设备抽象验证
- **Given**: 算子分发机制已就绪
- **When**: 在 CPU、CUDA（如可用）、CANN（如可用）设备上运行相同的计算图
- **Then**: 所有设备上的计算结果一致
- **Verification**: `programmatic`

### AC-6: API 设计质量
- **Given**: 所有核心功能已实现
- **When**: 评审 API 设计
- **Then**: API 简洁、一致、易用，符合 C++ 最佳实践
- **Verification**: `human-judgment`

## Open Questions
- [ ] 是否需要支持动态形状？
- [ ] 第一阶段是否需要 Python 绑定？
- [ ] 模型权重格式支持哪些？（Safetensors、GGUF、PyTorch .pt/.pth）
- [ ] 是否需要支持量化（INT8、INT4）推理？
