# AetherMind 大模型推理引擎 - 实现计划

## [ ] Task 0: 实现硬件抽象层 (HAL)
- **Priority**: P0
- **Depends On**: None
- **Description**: 
  - 实现 DeviceManager：设备枚举、属性查询、设备同步
  - 完善 Allocator：支持设备内存、主机内存、固定内存分配
  - 实现 Stream 和 StreamManager：异步执行流管理
  - 实现 Event 和 EventManager：事件记录、同步、时间测量
  - 实现 DataTransfer：H2D/D2H/D2D 同步/异步传输
  - 为 CPU 设备实现完整的 HAL 后端
- **Acceptance Criteria Addressed**: [AC-0]
- **Test Requirements**:
  - `programmatic` TR-0.1: DeviceManager 可以正确枚举和查询 CPU 设备
  - `programmatic` TR-0.2: Allocator 可以在 CPU 上正确分配/释放内存
  - `programmatic` TR-0.3: Stream 可以正确创建、同步和查询
  - `programmatic` TR-0.4: Event 可以正确记录、同步和测量时间
  - `programmatic` TR-0.5: DataTransfer 可以正确执行内存拷贝
- **Notes**: 先实现 CPU 后端，CUDA/CANN 后端后续扩展

## [ ] Task 1: 完善 Dispatcher 和算子注册机制
- **Priority**: P0
- **Depends On**: [Task 0]
- **Description**: 
  - 完善现有 Dispatcher 框架
  - 实现 OperatorSchema 的完整功能
  - 实现算子注册宏（类似 PyTorch 的 TORCH_LIBRARY）
  - 实现基于 DispatchKey 的算子分发逻辑
- **Acceptance Criteria Addressed**: [AC-4]
- **Test Requirements**:
  - `programmatic` TR-1.1: 可以成功注册一个算子
  - `programmatic` TR-1.2: 可以为同一算子注册不同设备的实现
  - `programmatic` TR-1.3: Dispatcher 能根据输入张量设备类型正确分发
- **Notes**: 参考 PyTorch 的 Dispatcher 设计，但保持简洁

## [ ] Task 2: 实现核心张量算子库 (CPU 后端)
- **Priority**: P0
- **Depends On**: [Task 1]
- **Description**: 
  - 实现 Element-wise 算子：Add, Sub, Mul, Div, Pow, Exp, Log, Relu, Gelu, Silu 等
  - 实现 Reduction 算子：Sum, Mean, Max, Min
  - 实现矩阵乘法 MatMul
  - 实现 Softmax
  - 实现索引和切片操作
  - 实现形状变换：Reshape, Transpose, Permute
- **Acceptance Criteria Addressed**: [AC-1]
- **Test Requirements**:
  - `programmatic` TR-2.1: 每个算子都有单元测试
  - `programmatic` TR-2.2: 数值测试与 PyTorch 参考实现对比，相对误差 < 1e-5
  - `programmatic` TR-2.3: 支持 float32、float16、bfloat16 数据类型

## [ ] Task 3: 实现 Transformer 专用算子
- **Priority**: P0
- **Depends On**: [Task 2]
- **Description**: 
  - 实现 LayerNorm
  - 实现 RMSNorm
  - 实现 Feed-Forward Network (FFN)
  - 实现 Multi-Head Attention (MHA)
  - 实现 Rotary Positional Embedding (RoPE)
- **Acceptance Criteria Addressed**: [AC-2]
- **Test Requirements**:
  - `programmatic` TR-3.1: 每个 Transformer 算子都有单元测试
  - `programmatic` TR-3.2: 可以构建并运行完整的 Transformer 层
  - `programmatic` TR-3.3: 端到端测试通过

## [ ] Task 4: 实现 KV Cache 管理
- **Priority**: P0
- **Depends On**: [Task 3]
- **Description**: 
  - 设计 KVCache 数据结构
  - 实现 KV Cache 的追加和更新逻辑
  - 实现带 KV Cache 的 Attention 算子
  - 支持滑动窗口 KV Cache（可选）
- **Acceptance Criteria Addressed**: [AC-3]
- **Test Requirements**:
  - `programmatic` TR-4.1: KV Cache 单元测试通过
  - `programmatic` TR-4.2: 增量生成测试，第二次及之后的推理时延降低 > 50%
  - `programmatic` TR-4.3: 带 KV Cache 的生成与不带 Cache 的生成结果一致

## [ ] Task 5: 计算图和执行引擎
- **Priority**: P1
- **Depends On**: [Task 2, Task 3]
- **Description**: 
  - 设计简单的计算图 IR
  - 实现计算图的构建 API
  - 实现计算图的执行引擎
  - 实现内存规划和复用（可选优化）
- **Acceptance Criteria Addressed**: [AC-5]
- **Test Requirements**:
  - `programmatic` TR-5.1: 可以构建简单的计算图
  - `programmatic` TR-5.2: 计算图执行结果正确
  - `programmatic` TR-5.3: 支持 eager 模式（即时执行）作为 fallback

## [ ] Task 6: 模型权重加载
- **Priority**: P1
- **Depends On**: [Task 2]
- **Description**: 
  - 实现 Safetensors 格式的加载
  - 实现权重映射机制（名称匹配）
  - 实现模型参数的初始化和加载
- **Acceptance Criteria Addressed**: [AC-6]
- **Test Requirements**:
  - `programmatic` TR-6.1: 可以加载小型测试权重文件
  - `programmatic` TR-6.2: 加载的权重数值正确
  - `programmatic` TR-6.3: 加载后的模型可以进行推理

## [ ] Task 7: 高层模型 API 和示例
- **Priority**: P2
- **Depends On**: [Task 3, Task 4, Task 6]
- **Description**: 
  - 提供简单的模型构建 API
  - 实现预定义的模型架构（如 Llama 风格、GPT 风格）
  - 编写端到端示例代码
  - 编写文档
- **Acceptance Criteria Addressed**: [AC-6]
- **Test Requirements**:
  - `programmatic` TR-7.1: 示例代码可以编译和运行
  - `human-judgement` TR-7.2: API 设计简洁易用

## [ ] Task 8: CUDA 后端支持 (可选，第一阶段之后)
- **Priority**: P2
- **Depends On**: [Task 0, Task 1, Task 2]
- **Description**: 
  - 为 HAL 层实现 CUDA 后端
  - 为核心算子实现 CUDA 后端
  - 集成 cuBLAS 用于矩阵乘法加速
  - 实现 CUDA KV Cache
- **Acceptance Criteria Addressed**: [AC-0, AC-4, AC-5]
- **Test Requirements**:
  - `programmatic` TR-8.1: CUDA HAL 后端测试通过
  - `programmatic` TR-8.2: CUDA 算子测试通过
  - `programmatic` TR-8.3: CUDA 后端性能显著优于 CPU

## [ ] Task 9: 性能优化和基准测试
- **Priority**: P2
- **Depends On**: [Task 3, Task 4]
- **Description**: 
  - 实现性能基准测试框架
  - 优化热点算子
  - 内存使用优化
- **Acceptance Criteria Addressed**: [AC-1, AC-2]
- **Test Requirements**:
  - `programmatic` TR-9.1: 基准测试可以运行
  - `human-judgement` TR-9.2: 性能满足预期目标
