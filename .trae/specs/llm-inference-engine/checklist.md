# AetherMind 大模型推理引擎 - 验证检查清单

## 设计评审
- [ ] PRD 中的所有目标都清晰明确
- [ ] 非目标（Out of Scope）已明确界定，避免范围蔓延
- [ ] 所有验收标准（Acceptance Criteria）都有明确的验证方式
- [ ] 开放问题已列出，以便后续决策

## 实现计划评审
- [ ] 所有任务都有明确的优先级（P0/P1/P2）
- [ ] 任务依赖关系正确，没有循环依赖
- [ ] 每个任务都有可验证的测试要求
- [ ] 任务粒度适中，没有过大或过小的任务
- [ ] 所有验收标准都被至少一个任务覆盖

## Task 1 验证：完善 Dispatcher 和算子注册机制
- [ ] OperatorSchema 类功能完整
- [ ] 算子注册宏可用，API 简洁
- [ ] 可以为同一算子注册不同设备的实现
- [ ] Dispatcher 能根据输入张量设备类型正确分发
- [ ] 相关单元测试全部通过

## Task 2 验证：核心张量算子库
- [ ] Element-wise 算子（Add, Sub, Mul, Div, Pow, Exp, Log, Relu, Gelu, Silu）全部实现
- [ ] Reduction 算子（Sum, Mean, Max, Min）全部实现
- [ ] MatMul 矩阵乘法实现正确
- [ ] Softmax 实现正确
- [ ] 形状变换（Reshape, Transpose, Permute）实现正确
- [ ] 每个算子都有单元测试
- [ ] 数值测试与 PyTorch 参考实现对比，相对误差 < 1e-5
- [ ] 支持 float32、float16、bfloat16 数据类型

## Task 3 验证：Transformer 专用算子
- [ ] LayerNorm 实现正确
- [ ] RMSNorm 实现正确
- [ ] Feed-Forward Network (FFN) 实现正确
- [ ] Multi-Head Attention (MHA) 实现正确
- [ ] Rotary Positional Embedding (RoPE) 实现正确
- [ ] 可以构建并运行完整的 Transformer 层
- [ ] 端到端测试通过

## Task 4 验证：KV Cache 管理
- [ ] KVCache 数据结构设计合理
- [ ] KV Cache 追加和更新逻辑正确
- [ ] 带 KV Cache 的 Attention 算子实现正确
- [ ] 增量生成功能正常
- [ ] 第二次及之后的推理时延比第一次降低 > 50%
- [ ] 带 KV Cache 的生成与不带 Cache 的生成结果一致

## Task 5 验证：计算图和执行引擎
- [ ] 计算图 IR 设计简洁合理
- [ ] 可以通过 API 构建计算图
- [ ] 计算图执行引擎工作正常
- [ ] 计算图执行结果正确
- [ ] 支持 eager 模式（即时执行）

## Task 6 验证：模型权重加载
- [ ] Safetensors 格式加载功能正常
- [ ] 权重映射机制（名称匹配）工作正常
- [ ] 可以加载小型测试权重文件
- [ ] 加载的权重数值正确
- [ ] 加载后的模型可以进行推理

## Task 7 验证：高层模型 API 和示例
- [ ] 模型构建 API 简洁易用
- [ ] 预定义模型架构（Llama 风格、GPT 风格）可用
- [ ] 端到端示例代码可以编译和运行
- [ ] API 设计符合 C++ 最佳实践

## Task 8 验证：CUDA 后端支持（可选）
- [ ] 核心算子的 CUDA 后端实现
- [ ] cuBLAS 集成正常，矩阵乘法性能提升
- [ ] CUDA KV Cache 实现正确
- [ ] CUDA 算子测试通过
- [ ] CUDA 后端性能显著优于 CPU

## Task 9 验证：性能优化和基准测试
- [ ] 性能基准测试框架可用
- [ ] 热点算子已优化
- [ ] 内存使用优化到位
- [ ] 基准测试可以正常运行
- [ ] 性能满足预期目标

## 整体验证
- [ ] 所有 P0 任务已完成
- [ ] 所有 P0 和 P1 任务的验收标准已满足
- [ ] 代码符合项目现有风格和约定
- [ ] 没有引入明显的性能回退
- [ ] 文档更新到位
