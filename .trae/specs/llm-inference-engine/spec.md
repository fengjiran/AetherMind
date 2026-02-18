# AetherMind 大模型推理引擎 - 产品需求文档

## Overview
- **Summary**: 基于现有 AetherMind 基础设施构建高性能、可扩展的大语言模型 (LLM) 推理引擎，支持主流大模型架构（如 Transformer、GPT、BERT 等）的高效推理。
- **Purpose**: 提供一套完整的深度学习推理框架，具备张量计算、算子调度、设备抽象、内存管理等核心能力，支持多种硬件设备（CPU、CUDA、CANN）的高效推理。
- **Target Users**: 深度学习研究人员、AI 应用开发者、需要在生产环境部署大模型的工程师。

## Goals
- 完善硬件抽象层（HAL），统一设备、内存、流接口
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
│              硬件抽象层 (HAL)                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Device   │ │ Memory   │ │  Stream  │ │  Event   │  │
│  │ Manager  │ │ Allocator│ │  Manager │ │ Manager  │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│  ┌──────────────────────────────────────────────────┐    │
│  │              Data Transfer (H2D, D2H, D2D)       │    │
│  └──────────────────────────────────────────────────┘    │
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

#### 1. 硬件抽象层 (Hardware Abstraction Layer, HAL)

HAL 层是整个推理引擎的基础，用于屏蔽不同硬件设备的差异，提供统一的设备、内存、流和事件管理接口。

##### 1.1 Device Manager
- **职责**: 设备管理和查询
- **功能**:
  - 枚举系统可用设备
  - 获取设备属性（内存大小、计算能力等）
  - 设置当前设备
  - 设备同步
- **API 设计**:
  ```cpp
  class DeviceManager {
  public:
      static DeviceManager& Instance();
      size_t GetDeviceCount(DeviceType type) const;
      Device GetDevice(DeviceType type, int8_t index) const;
      void SetCurrentDevice(const Device& device);
      Device GetCurrentDevice(DeviceType type) const;
      void SynchronizeDevice(const Device& device);
      DeviceProperties GetDeviceProperties(const Device& device) const;
  };
  ```

##### 1.2 Memory Allocator
- **职责**: 统一内存分配/释放接口
- **功能**:
  - 设备内存分配/释放
  - 主机内存分配/释放
  - 固定主机内存分配（用于高性能 H2D/D2H 传输）
  - 内存使用统计
- **API 设计**:
  ```cpp
  class Allocator {
  public:
      virtual ~Allocator() = default;
      virtual DataPtr Allocate(size_t nbytes) = 0;
      virtual void Deallocate(void* ptr) = 0;
      virtual size_t GetAllocatedSize() const;
      virtual size_t GetReservedSize() const;
  };
  ```

##### 1.3 Stream Manager
- **职责**: 异步执行流管理
- **功能**:
  - 创建/销毁计算流
  - 流同步
  - 流之间的依赖管理
- **API 设计**:
  ```cpp
  class Stream {
  public:
      virtual ~Stream() = default;
      virtual void Synchronize() = 0;
      virtual bool Query() const;
      virtual void* GetNativeHandle() const;
  };

  class StreamManager {
  public:
      static StreamManager& Instance();
      std::shared_ptr<Stream> CreateStream(const Device& device, int priority = 0);
      std::shared_ptr<Stream> GetDefaultStream(const Device& device);
      void SetCurrentStream(const Device& device, const std::shared_ptr<Stream>& stream);
      std::shared_ptr<Stream> GetCurrentStream(const Device& device);
  };
  ```

##### 1.4 Event Manager
- **职责**: 事件管理和同步
- **功能**:
  - 创建/销毁事件
  - 事件记录（记录流上的某个时间点）
  - 事件同步（等待事件完成）
  - 事件查询
  - 事件之间的依赖
- **API 设计**:
  ```cpp
  class Event {
  public:
      virtual ~Event() = default;
      virtual void Record(const std::shared_ptr<Stream>& stream) = 0;
      virtual void Synchronize() = 0;
      virtual bool Query() const;
      virtual float ElapsedTime(const Event& other) const;
      virtual void* GetNativeHandle() const;
  };

  class EventManager {
  public:
      static EventManager& Instance();
      std::shared_ptr<Event> CreateEvent(const Device& device, bool enable_timing = false);
      void WaitEvent(const std::shared_ptr<Event>& event, const std::shared_ptr<Stream>& stream);
  };
  ```

##### 1.5 Data Transfer
- **职责**: 统一数据传输接口
- **功能**:
  - Host 到 Device (H2D)
  - Device 到 Host (D2H)
  - Device 到 Device (D2D)
  - 同步/异步传输
- **API 设计**:
  ```cpp
  class DataTransfer {
  public:
      static DataTransfer& Instance();
      void Copy(void* dst, const void* src, size_t nbytes,
                const Device& dst_device, const Device& src_device);
      void CopyAsync(void* dst, const void* src, size_t nbytes,
                     const Device& dst_device, const Device& src_device,
                     const std::shared_ptr<Stream>& stream);
      void Fill(void* dst, int value, size_t nbytes, const Device& device);
      void FillAsync(void* dst, int value, size_t nbytes, const Device& device,
                     const std::shared_ptr<Stream>& stream);
  };
  ```

#### 2. 算子分发层 (Dispatcher Layer)
- **Dispatcher**: 全局单例算子调度器
- **Operator Registry**: 算子注册表，存储所有已注册的算子
- **Dispatch Key**: 分发键，包括设备类型（CPU/CUDA/CANN）、数据类型等
- **职责**: 根据输入张量的属性自动选择并调用合适的算子实现

#### 3. 算子层 (Operator Layer)
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

#### 4. 计算图执行层 (Graph Execution Layer)
- **Graph Builder**: 计算图构建 API
- **Graph IR**: 简单的中间表示
- **Graph Executor**: 计算图执行引擎
- **Memory Planner**: 内存规划和复用优化器（可选）

#### 5. 高层 API 层 (High-level API Layer)
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
1. 实现 HAL 层接口：
   - 实现该设备的 `DeviceManager` 后端
   - 实现该设备的 `Allocator`
   - 实现该设备的 `Stream` 和 `StreamManager`
   - 实现该设备的 `Event` 和 `EventManager`
   - 实现该设备的 `DataTransfer` 后端
2. 定义新的 DispatchKey
3. 为现有算子注册该设备的实现
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
- **FR-0**: 完善硬件抽象层（HAL），统一设备、内存、流、事件和数据传输接口
  - 实现 DeviceManager，支持设备枚举、属性查询和同步
  - 完善 Allocator，支持设备内存、主机内存和固定内存分配
  - 实现 Stream 和 StreamManager，支持异步执行流管理
  - 实现 Event 和 EventManager，支持事件记录和同步
  - 实现 DataTransfer，支持 H2D/D2H/D2D 同步/异步传输
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

### AC-0: HAL 层实现
- **Given**: 现有 Device 和 Allocator 框架已就绪
- **When**: 实现 DeviceManager、Stream/StreamManager、Event/EventManager、DataTransfer
- **Then**: HAL 层各组件功能完整，单元测试通过
- **Verification**: `programmatic`
- **Sub-ACs**:
  - **AC-0.1**: DeviceManager 可以正确枚举和查询设备
  - **AC-0.2**: Allocator 可以在各设备上正确分配/释放内存
  - **AC-0.3**: Stream 可以正确创建、同步和查询
  - **AC-0.4**: Event 可以正确记录、同步和测量时间
  - **AC-0.5**: DataTransfer 可以正确执行 H2D/D2H/D2D 传输

### AC-1: 核心张量算子实现
- **Given**: AetherMind 张量基础设施和 HAL 层已就绪
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
- **Given**: Dispatcher 框架和 HAL 层已就绪
- **When**: 实现算子注册机制，支持为同一算子注册不同设备的实现
- **Then**: 能根据输入张量的设备类型自动调度到对应的后端实现
- **Verification**: `programmatic`

### AC-5: 设备抽象验证
- **Given**: 算子分发机制和 HAL 层已就绪
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
