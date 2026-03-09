# CPU 优先 Llama 运行时 V1

## 概述
> **摘要**: 在现有 AetherMind 代码库中构建一个精简的 V1 嵌入式推理运行时，用于 Llama 家族的仅解码器模型，采用 CPU 优先执行、HuggingFace Safetensors 导入、混合算子分发和稳定的 C++/C ABI。该计划刻意推迟服务化、多请求调度、GPU 后端和通用图运行时，直到单请求端到端 CPU 生成稳定为止。
> **交付物**:
> - 用于单进程、单模型、单请求解码的嵌入式运行时模块
> - HuggingFace `config.json` + Safetensors 加载器和内部权重重排路径
> - CPU 内核注册/分发表，支持参考实现 + 量化路径
> - KV 缓存、预填充/解码循环、贪婪采样和确定性黄金测试
> - 稳定的 C++ API 和基于不透明句柄的薄 C ABI
> - 通过单元测试、基准测试、消毒器变体和证据工件进行加固
> **工作量**: 超大 (XL)
> **并行性**: 是 - 3 波次
> **关键路径**: 1 -> 3 -> 7 -> 10 -> 11 -> 12 -> 15

## 背景
### 原始请求
为大型模型推理引擎设计详细的架构和实施计划。

### 访谈摘要
- 产品形态: 嵌入式运行时优先，而非服务化系统
- 模型范围: 首先支持 Llama 家族仅解码器模型
- 部署目标: CPU 优先
- 容量目标: 最高支持 7B 参数，在共享运行时架构下支持 INT4/INT8
- 集成接口: C++ API 加 C ABI
- 依赖策略: 允许少量成熟的依赖项；保持运行时核心自主实现
- 模型资源格式: HuggingFace `config.json` + Safetensors 作为标准输入
- 分发风格: 混合分发 - 运行时算子/后端注册与静态优化的热路径内核
- 测试策略: 测试后置，但在编码前明确定义验证契约
- V1 成功标准: 在 CPU 上实现稳定的 prompt -> 预填充 -> 解码 -> 采样 -> 输出

### Metis 审查 (已解决的缺口)
- 将 V1 视为 HAL、内核、加载器、运行时和 KV 缓存的全新领域；不要假设已存在隐藏的推理基础设施
- 明确取代 `.trae/specs/llm-inference-engine/tasks.md:1` 和 `.trae/specs/llm-inference-engine/spec.md:274` 中的更广泛范围
- 将 token ID 冻结为核心运行时边界；将文本分词器作为薄适配器层，而非运行时重心
- 将混合分发限制在 CPU 密集型和 V1 所需的选定量化内核；不要首先构建通用框架项目
- 在实施开始前预定义黄金标准、KV 缓存一致性检查、ABI 冒烟测试和无稳态分配断言

## 工作目标
### 核心目标
在 AetherMind 内部交付一个 CPU 优先的嵌入式运行时，能够加载一个受支持的 HuggingFace 风格的 Llama 家族检查点，将权重重排到内部 CPU 布局，使用 KV 缓存执行确定性单请求生成，并通过 C++ 和 C ABI 入口点暴露该流程。

### 交付物
- 在新的推理/运行时导向的头文件和源文件下的运行时模块布局
- 受支持 Llama 配置字段、张量命名和拒绝行为的标准 V1 模型契约
- Safetensors 清单读取器、分片解析器、mmap/重排管道和内部权重存储
- 具有特性检测、线程池策略、临时内存和分发表的 CPU 执行上下文
- V1 所需的参考 CPU 内核加量化线性内核
- KV 缓存布局、解码层执行器、生成会话和贪婪采样器
- 用于创建/加载/预填充/解码/销毁/错误检索的 C++ 和 C ABI 句柄
- 单元测试、夹具资源、基准测试覆盖率和消毒器就绪的构建/测试路径

### 完成定义 (可验证条件及命令)
- `cmake -S . -B build-v1 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON && cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntime.*`
- `./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=SafetensorsLoader.*`
- `./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=LlamaDecode.*`
- `./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=KVCache.*`
- `./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=CAbiRuntime.*`
- `cmake -S . -B build-v1-tsan -DENABLE_TSAN=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF && cmake --build build-v1-tsan --target aethermind_unit_tests -j && ./build-v1-tsan/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntimeThreading.*`
- `cmake --build build-v1 --target aethermind_benchmark -j && ./build-v1/tests/benchmark/aethermind_benchmark --benchmark_filter=BM_LlamaDecode|BM_KVCache|BM_QuantLinear`

### 必须包含
- V1 的单进程、单模型、单请求运行时
- 核心 API 边界是 token ID 输入/输出；文本分词器支持是可选的适配器层
- 标准输入是 HuggingFace `config.json` + Safetensors；不受支持的资源会明确报错
- 确定性贪婪解码作为必需的基线采样器
- V1 CPU 的一个精确加载器契约、一个精确的 KV 缓存布局、一个精确的分发矩阵
- 在支持的预热单请求路径后，解码期间无稳态分配

### 必须不包含 (护栏、AI 垃圾模式、范围边界)
- V1 中不包含 HTTP/gRPC 服务器、OpenAI 兼容 API 或 SSE 流式传输
- V1 中不包含多请求调度器、连续批处理、分块预填充、前缀缓存、LoRA、推测解码或分布式执行
- 在直接解码器执行工作之前，不包含通用图 IR 或通用图执行器
- V1 中不包含 GPU/CUDA/CANN 后端实现
- 不包含"支持所有 Llama 变体"的模糊性；仅支持命名字段和拒绝规则
- 不包含模糊的量化支持；V1 必须交付精确的内核/布局支持并清晰地拒绝其他一切

## 验证策略
> 零人工干预 - 所有验证由代理执行。
- 测试决策: 测试后置，使用现有的 GoogleTest + Google Benchmark 基础设施
- QA 策略: 每个任务都包含代理执行的快乐路径和失败路径场景
- 证据: `.sisyphus/evidence/task-{N}-{slug}.{ext}`
- 所需夹具策略: 为配置解析、Safetensors 头解析、权重名称映射、单层解码、KV 缓存一致性、无分配探针和 ABI 冒烟测试创建微小的确定性夹具
- 加固策略: 即使仓库目前仅在顶层暴露 TSAN，也在计划中期间添加 ASAN/UBSAN CMake 支持

## 执行策略
### 并行执行波次
> 目标: 每波 5-8 个任务。在第 1 波中提取共享依赖以最大化并行性。

第 1 波: V1 范围契约、资源摄取契约、CPU 运行时上下文、分发骨架、验证夹具
第 2 波: 参考内核、加载器/重排、KV 缓存、解码器执行、生成会话
第 3 波: C ABI、INT4 扩展、分词器适配器、CI/消毒器/基准加固、文档/示例

### 依赖矩阵 (完整，所有任务)
| 任务 | 阻塞 | 被阻塞于 |
|------|--------|------------|
| 1 | 2, 3, 4, 5 | - |
| 2 | 8, 14 | 1 |
| 3 | 6, 7, 8, 9, 11, 12, 13 | 1 |
| 4 | 6, 7, 10, 13 | 1 |
| 5 | 6, 8, 9, 10, 11, 12, 13, 14, 15 | 1 |
| 6 | 10 | 3, 4, 5 |
| 7 | 10, 13 | 3, 4, 5 |
| 8 | 10, 11, 14 | 2, 3, 5 |
| 9 | 10, 11 | 3, 5 |
| 10 | 11 | 4, 5, 6, 7, 8, 9 |
| 11 | 12, 13, 14, 15 | 3, 5, 8, 9, 10 |
| 12 | 15 | 3, 5, 11 |
| 13 | 15 | 3, 4, 5, 7, 11 |
| 14 | 15 | 2, 5, 11 |
| 15 | F1, F2, F3, F4 | 5, 12, 13, 14 |

### 代理分派摘要 (波次 -> 任务数 -> 类别)
- 第 1 波 -> 5 个任务 -> `unspecified-high`, `deep`, `writing`
- 第 2 波 -> 5 个任务 -> `deep`, `ultrabrain`, `unspecified-high`
- 第 3 波 -> 5 个任务 -> `deep`, `quick`, `writing`, `unspecified-high`

## 待办事项
> 实现 + 测试 = 一个任务。永不分离。
> 每个任务必须有: 代理配置 + 并行化 + QA 场景。

- [ ] 1. 冻结 V1 运行时契约和模块布局

  **做什么**: 在 `include/` 和 `src/` 下为配置、加载器、运行时上下文、内核、kv 缓存、模型和 C ABI 包装器创建 V1 运行时模块边界。在面向代码的头文件中定义标准 V1 范围: 单进程、单模型、单请求、token ID 输入/输出、仅 CPU、同步生成。仅在执行器指导需要时，在 `.sisyphus/` 下添加一个小的 `README` 风格的架构说明，但通过头文件契约保持源代码树的权威性。
  **禁止做**: 不要在此任务中添加服务 API、调度器抽象、图 IR、CUDA/CANN 占位符或通用多后端脚手架。

  **推荐代理配置**:
  - 类别: `unspecified-high` - 原因: 设置架构边界和文件布局，解锁所有后续任务
  - 技能: `[]` - 不需要外部技能，只需仓库感知实现
  - 省略: `[playwright]` - 此仓库中没有浏览器或 UI 工作

  **并行化**: 能否并行: 否 | 第 1 波 | 阻塞: 2, 3, 4, 5 | 被阻塞于: -

  **参考资料** (执行者没有访谈上下文 - 要全面):
  - 模式: `include/tensor.h:12` - 使用现有的薄外观风格处理公共运行时句柄和值类型
  - 模式: `include/tensor_impl.h:88` - 遵循仓库在外观对象和底层实现/状态对象之间的分离
  - 模式: `include/function.h:94` - 为可调用/运行时组件镜像基于 Object/ObjectRef 的实现所有权
  - 模式: `include/c_api.h:16` - 重用不透明句柄思维进行 C ABI 表面设计
  - 模式: `src/CMakeLists.txt:1` - 源文件发现是 `src/` 下的递归，因此子目录是可接受的
  - 测试: `tests/unit/test_tensor.cpp:22` - 遵循当前简单的 GoogleTest 命名风格

  **验收标准** (仅代理可执行):
  - [ ] 新的运行时头文件/源文件在现有的根 `src/*.cpp` 递归构建下编译，而不添加第二个库目标
  - [ ] 存在一个新的面向契约的单元测试目标路径，并通过 `--gtest_filter=InferenceRuntimeContract.*`
  - [ ] 公共头文件明确编码 V1 范围边界，并省略服务/分布式/GPU API

  **QA 场景** (强制 - 没有这些任务不完整):
  ```text
  场景: 契约头文件编译并仅暴露 V1 范围
    工具: Bash
    步骤: cmake -S . -B build-v1 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF && cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntimeContract.*
    预期: 构建成功；契约测试通过；链接不需要服务器/调度器符号
    证据: .sisyphus/evidence/task-1-runtime-contract.txt

  场景: 禁止范围保持不存在
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntimeContract.RejectsOutOfScopeFeatures
    预期: 测试通过，断言分布式、GPU 和流式 API 未在 V1 契约中暴露
    证据: .sisyphus/evidence/task-1-runtime-contract-error.txt
  ```

  **提交**: 是 | 消息: `feat(runtime): define v1 runtime contracts and layout` | 文件: `include/`, `src/`, `tests/unit/`

- [ ] 2. 实现标准 HuggingFace 模型契约和拒绝策略

  **做什么**: 为 V1 支持的模型元数据实现严格的解析器/验证器。仅接受文档记录的 HuggingFace 风格 `config.json` 字段子集，这些字段是仅解码器 Llama 执行所需的: `model_type`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `rms_norm_eps`, `rope_theta`, `max_position_embeddings`, `vocab_size`, `bos_token_id`, `eos_token_id`, 和 `tie_word_embeddings`。冻结精确支持的变体: 仅解码器、RMSNorm、RoPE、允许 GQA/MQA、无 MoE、无编码器-解码器、无滑动窗口注意力。对不受支持的资源返回显式的结构化错误。
  **禁止做**: 不要解析"尽力而为"的配置，静默默认缺少的必填字段，或将不受支持的检查点行为留空。

  **推荐代理配置**:
  - 类别: `deep` - 原因: 这锁定了加载器、执行器和测试使用的资产契约和拒绝语义
  - 技能: `[]` - 仓库原语足够
  - 省略: `[playwright]` - 与后端解析器工作无关

  **并行化**: 能否并行: 否 | 第 1 波 | 阻塞: 8, 14 | 被阻塞于: 1

  **参考资料** (执行者没有访谈上下文 - 要全面):
  - 模式: `include/device.h:15` - 当前仓库已经建模显式枚举加验证；遵循该风格处理支持/不支持的变体
  - 模式: `include/function_schema.h:18` - 使用显式模式式验证而非宽松映射
  - 模式: `include/error.h` - 使用仓库错误报告风格处理结构化拒绝
  - 外部: `https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py` - 标准 Llama 配置字段名称和 GQA 语义的来源
  - 外部: `https://huggingface.co/docs/transformers/main/en/model_doc/llama` - 字段级行为和模型家族期望
  - 测试: `tests/unit/test_device.cpp` - 遵循当前"接受/拒绝显式不变量"的单元测试模式

  **验收标准** (仅代理可执行):
  - [ ] `config.json` 解析器接受一个受支持的微小 Llama 夹具并填充类型化的运行时配置对象
  - [ ] 解析器以确定性错误消息和测试覆盖率拒绝不受支持的字段/组合
  - [ ] 存在绑定嵌入、GQA/MQA、无效 `model_type` 和缺少必填字段的单元测试

  **QA 场景** (强制 - 没有这些任务不完整):
  ```text
  场景: 支持的 Llama 配置解析成功
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=ModelConfigContract.AcceptsSupportedLlama
    预期: 测试通过；解析的配置暴露预期的隐藏大小、层数、头数和 rope 参数
    证据: .sisyphus/evidence/task-2-model-contract.txt

  场景: 不受支持的配置被干净拒绝
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=ModelConfigContract.RejectsUnsupportedVariant
    预期: 测试通过，确认对 MoE、编码器-解码器或缺少必填字段的显式拒绝
    证据: .sisyphus/evidence/task-2-model-contract-error.txt
  ```

  **提交**: 是 | 消息: `feat(runtime): add llama config contract parser` | 文件: `include/`, `src/`, `tests/unit/`

- [ ] 3. 构建 CPU 运行时上下文和执行资源

  **做什么**: 实现用于特性检测、算子内线程配置、每线程临时分配、请求本地解码工作区和预热时资源保留的 CPU 运行时上下文对象。为 V1 定义一个精确的策略: 同步 API、仅算子内并行性、无请求间调度器、预热后无稳态解码分配。在需要所有权的地方使用现有的存储/数据指针基础设施；仅对临时/工作区缓冲区使用 `ammalloc`，不用于内存映射模型权重。
  **禁止做**: 不要在 V1 中引入流/事件抽象、异步协程执行或通用设备管理器。

  **推荐代理配置**:
  - 类别: `ultrabrain` - 原因: 执行资源、分配策略和线程不变量影响每个热路径
  - 技能: `[]` - 重推理的仓库工作，不需要特殊外部工具
  - 省略: `[playwright]` - 与系统/运行时工作无关

  **并行化**: 能否并行: 是 | 第 1 波 | 阻塞: 6, 7, 8, 9, 11, 12, 13 | 被阻塞于: 1

  **参考资料** (执行者没有访谈上下文 - 要全面):
  - 模式: `include/memory/storage.h:13` - 将所有权包装在 `ObjectRef` 风格的对象中
  - 模式: `include/memory/data_ptr.h:77` - 重用设备标记的指针所有权语义处理运行时拥有的缓冲区
  - 模式: `ammalloc/include/ammalloc/ammalloc.h` - 临时/工作区分配的分配器入口点
  - 模式: `ammalloc/include/ammalloc/thread_cache.h` - 当前仓库已包含 CPU 分配器热路径思维；将工作区策略与其对齐
  - 测试: `tests/unit/test_thread_cache.cpp` - 遵循线程敏感逻辑的并发导向测试风格
  - 测试: `tests/benchmark/benchmark_memory_pool.cpp:23` - 遵循基准测试风格进行分配/吞吐量测量

  **验收标准** (仅代理可执行):
  - [ ] 运行时上下文暴露单请求 CPU 执行的确定性线程计数、临时大小和预热配置
  - [ ] 解码预热保留所有必需的稳态工作区，后续测试可以断言零解码时分配
  - [ ] 线程安全测试验证独立的执行上下文可以在 TSAN 构建下安全共享一个加载的模型

  **QA 场景** (强制 - 没有这些任务不完整):
  ```text
  场景: 运行时上下文预热稳定资源使用
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntimeThreading.WarmupPreallocatesDecodeResources
    预期: 测试通过，并记录预热后支持的解码循环无分配增长
    证据: .sisyphus/evidence/task-3-runtime-context.txt

  场景: 共享模型与独立上下文保持无竞争
    工具: Bash
    步骤: cmake -S . -B build-v1-tsan -DENABLE_TSAN=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF && cmake --build build-v1-tsan --target aethermind_unit_tests -j && ./build-v1-tsan/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntimeThreading.SharedModelIndependentContexts
    预期: TSAN 构建通过，无报告的竞争或崩溃
    证据: .sisyphus/evidence/task-3-runtime-context-error.txt
  ```

  **提交**: 是 | 消息: `feat(runtime): add cpu execution context and scratch policy` | 文件: `include/`, `src/`, `tests/unit/`, `tests/benchmark/`

- [ ] 4. 实现 V1 内核分发表和注册骨架

  **做什么**: 构建一个最小的 V1 分发表面，根据精确的 `op_id + cpu_feature + quant_scheme` 元组在运行时选择 CPU 内核实现，并缓存直接函数指针供热使用。将算子表面限制为 V1 解码所需的内核: 嵌入收集、RMSNorm、RoPE、矩阵乘法/线性、注意力分数、注意力值累积、softmax、SwiGLU、残差加、logits 投影和贪婪采样助手。仅在简化初始化时注册的情况下重用现有的 `Registry`/`Function` 思想；不要强制热路径调用通过装箱的 `Any` 分发。
  **禁止做**: 不要在此任务中尝试完成通用 `Dispatcher` 框架、动态装箱路径或多后端分发网格。

  **推荐代理配置**:
  - 类别: `deep` - 原因: 必须将当前仓库注册思想连接到窄快路径分发表而不至于过度工程化
  - 技能: `[]` - 内部仓库引用足够
  - 省略: `[playwright]` - 无 UI/浏览器范围

  **并行化**: 能否并行: 是 | 第 1 波 | 阻塞: 6, 7, 10, 13 | 被阻塞于: 1

  **参考资料** (执行者没有访谈上下文 - 要全面):
  - 模式: `include/dispatcher.h:31` - 现有分发器是骨架；用它作为证据说明什么不应该首先过度构建
  - 模式: `include/registry.h:12` - 现有注册 API 可以激发初始化时注册形态
  - 模式: `include/function.h:146` - 当前可调用抽象对非热路径注册有用，但对紧凑解码循环来说太重
  - 模式: `include/dispatch_key.h` - 如果重用，将分发键窄范围限制在仅 CPU 密集型和选定的量化方案
  - 测试: `tests/unit/test_function.cpp` - 在适用的情况下遵循当前可调用/注册测试模式

  **验收标准** (仅代理可执行):
  - [ ] 分发表初始化为选定的 CPU 特性集解析每个必需的 V1 算子的一个具体函数指针
  - [ ] 热路径执行代码可以调用缓存的函数指针而不通过 `Any` 装箱
  - [ ] 不受支持的算子/量化/cpu 特性组合在初始化期间失败，而非解码中途

  **QA 场景** (强制 - 没有这些任务不完整):
  ```text
  场景: 分发表解析所有必需的 V1 算子
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=KernelDispatchTable.ResolvesRequiredCpuOps
    预期: 测试通过；为选定的 V1 CPU 后端填充所有必需的算子槽位
    证据: .sisyphus/evidence/task-4-dispatch-table.txt

  场景: 不受支持的分发组合在初始化时失败
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=KernelDispatchTable.RejectsUnsupportedQuantScheme
    预期: 测试通过，确认对未知量化/算子组合的确定性初始化失败
    证据: .sisyphus/evidence/task-4-dispatch-table-error.txt
  ```

  **提交**: 是 | 消息: `feat(runtime): add narrow cpu dispatch table` | 文件: `include/`, `src/`, `tests/unit/`

- [ ] 5. 添加 V1 的确定性夹具和验证工具

  **做什么**: 创建计划的其余部分所依赖的测试/夹具脚手架: 微小的支持/无效配置夹具、最小的 safetensors 头夹具、层权重映射夹具、确定性 token 黄金标准、KV 缓存一致性夹具、无分配探针和 ABI 冒烟测试工具。现在定义精确的测试套件和基准名称，以便后续任务只实现它们，而不是发明它们。
  **禁止做**: 不要在单元测试中使用巨大的外部检查点或脆弱的文本生成期望。

  **推荐代理配置**:
  - 类别: `unspecified-high` - 原因: 这是验证架构，而不仅仅是测试编写
  - 技能: `[]` - 仅使用当前的测试/基准模式
  - 省略: `[playwright]` - 仅后端验证

  **并行化**: 能否并行: 是 | 第 1 波 | 阻塞: 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 | 被阻塞于: 1

  **参考资料** (执行者没有访谈上下文 - 要全面):
  - 模式: `tests/unit/CMakeLists.txt:1` - 测试已经编译成一个 `aethermind_unit_tests` 二进制文件；保持相同的约定
  - 模式: `tests/benchmark/CMakeLists.txt:1` - 基准测试存在于一个 `aethermind_benchmark` 目标中；扩展它而不是发明新的工具
  - 模式: `tests/unit/test_tensor.cpp:97` - 确定性命名测试用例是仓库规范
  - 模式: `tests/unit/test_storage.cpp` - 对所有权和缓冲区不变量使用聚焦的低级测试
  - 模式: `tests/benchmark/benchmark_memory_pool.cpp:23` - 遵循当前基准命名风格 `BM_*`
  - 外部: `https://github.com/huggingface/safetensors` - 用于夹具生成的 safetensors 头/体不变量

  **验收标准** (仅代理可执行):
  - [ ] 存在支持和拒绝配置/资源路径的夹具文件，并被命名测试使用
  - [ ] 在解码器实现开始之前定义黄金 token 序列、KV 奇偶校验、ABI 冒烟和无分配断言
  - [ ] 存在解码、KV 缓存和量化线性路径的基准占位符，带有确定性的 CLI 过滤器

  **QA 场景** (强制 - 没有这些任务不完整):
  ```text
  场景: 夹具工具加载并验证所有计划的测试资源
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=InferenceFixtures.*
    预期: 所有夹具加载测试通过并生成确定性的黄金元数据
    证据: .sisyphus/evidence/task-5-fixtures.txt

  场景: 显式执行无效夹具路径
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=InferenceFixtures.RejectsMalformedAssets
    预期: 测试通过，确认格式错误的配置/safetensors 头以精确的预期错误失败
    证据: .sisyphus/evidence/task-5-fixtures-error.txt
  ```

  **提交**: 是 | 消息: `test(runtime): add deterministic inference fixtures and harness` | 文件: `tests/unit/`, `tests/benchmark/`, `tests/fixtures/`

- [ ] 6. 实现参考 CPU 解码器原语

  **做什么**: 实现证明解码器正确性所需的非量化参考内核表面: 嵌入收集、RMSNorm、RoPE 应用、残差加、softmax、注意力分数/值累积、SwiGLU、logits 投影和贪婪 argmax 助手。使用正确性优先路径，使用 float32 累积和确定性数学；这是针对后续量化内核进行比较的预言机路径。
  **禁止做**: 不要首先优化、积极融合，或在同一内核入口点混合正确性和量化逻辑。

  **推荐代理配置**:
  - 类别: `deep` - 原因: 核心数学原语必须在模型循环组装之前正确
  - 技能: `[]` - 仅使用当前的 tensor/storage 基础
  - 省略: `[playwright]` - 仅后端数学

  **并行化**: 能否并行: 是 | 第 2 波 | 阻塞: 10 | 被阻塞于: 3, 4, 5

  **参考资料** (执行者没有访谈上下文 - 要全面):
  - 模式: `include/tensor.h:12` - 重用现有的 tensor 外观，而不是发明第二个 tensor 抽象
  - 模式: `include/tensor_impl.h:69` - tensor/存储别名规则对视图安全内核实现很重要
  - 模式: `include/data_type.h` - 现有的 dtype 支持应该通知显式运行时 dtype 检查
  - 模式: `tests/unit/test_tensor.cpp` - 在仓库现有风格中添加数学聚焦测试
  - 外部: `https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py` - RMSNorm、RoPE、注意力和 SwiGLU 顺序的参考语义

  **验收标准** (仅代理可执行):
  - [ ] 参考内核为微小的单层夹具路径生成确定性输出
  - [ ] 单元测试将输出与夹具黄金标准在显式容差内进行比较
  - [ ] 内核入口点可以通过 V1 分发表调用而不装箱

  **QA 场景** (强制 - 没有这些任务不完整):
  ```text
  场景: 参考解码器原语匹配黄金输出
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=ReferenceCpuKernels.*
    预期: 所有原语内核测试在声明的容差内通过，输出确定性
    证据: .sisyphus/evidence/task-6-reference-kernels.txt

  场景: 无效的 tensor 形状被显式拒绝
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=ReferenceCpuKernels.RejectsInvalidShapes
    预期: 测试通过，确认形状/步幅不匹配以显式错误失败
    证据: .sisyphus/evidence/task-6-reference-kernels-error.txt
  ```

  **提交**: 是 | 消息: `feat(runtime): add reference cpu decoder kernels` | 文件: `src/`, `include/`, `tests/unit/`

- [ ] 7. 实现 INT8 仅权重量化内核和分发条目

  **做什么**: 为 V1 实现第一个必需的量化路径: INT8 仅权重量化线性内核，用于 q/k/v、o_proj、gate/up/down 投影和 lm_head（如适用）。冻结 INT8 权重、缩放和偏置处理的一个精确内部打包/布局；在 V1 分发表中注册这些内核。保持激活和累积行为显式和确定性。
  **禁止做**: 不要在此任务中添加 INT4，不要支持多种 INT8 打包格式，也不要让缩放粒度模糊。

  **推荐代理配置**:
  - 类别: `ultrabrain` - 原因: 量化 CPU 内核和布局选择是计划中第一个主要的性能/正确性分支
  - 技能: `[]` - 实现是仓库本地的但重推理
  - 省略: `[playwright]` - 与计算内核无关

  **并行化**: 能否并行: 是 | 第 2 波 | 阻塞: 10, 13 | 被阻塞于: 3, 4, 5

  **参考资料** (执行者没有访谈上下文 - 要全面):
  - 模式: `include/device.h:37` - V1 保持仅 CPU；此处不要添加其他后端分支
  - 模式: `include/function.h:167` - 当前可调用系统不是热路径；在初始化时分发后使用直接内核指针
  - 模式: `tests/benchmark/benchmark_memory_pool.cpp` - 遵循基准命名/调用风格进行吞吐量基准测试
  - 外部: `https://oneapi-src.github.io/oneDNN/dev_guide_matmul.html` - 如果 oneDNN 被选为基线验证，仅将其用作正确性/性能参考，而非运行时架构
  - 外部: `https://arxiv.org/abs/2210.17323` - LLM.int8 上下文的仅权重量化矩阵乘法期望；实现仍然必须冻结一个精确布局

  **验收标准** (仅代理可执行):
  - [ ] INT8 内核在记录的容差内通过数值奇偶校验测试，与参考线性路径对比
  - [ ] 分发初始化仅为支持的量化方案选择 INT8 内核，否则失败
  - [ ] 至少存在一个夹具形状族的 INT8 线性吞吐量基准条目

  **QA 场景** (强制 - 没有这些任务不完整):
  ```text
  场景: INT8 线性内核匹配参考路径
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=QuantLinearInt8.*
    预期: 所有 INT8 奇偶校验测试在支持的形状声明的容差内通过
    证据: .sisyphus/evidence/task-7-int8-kernels.txt

  场景: 未知的 INT8 布局在注册/初始化时被拒绝
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=QuantLinearInt8.RejectsUnknownPacking
    预期: 测试通过，确认不支持的打包/布局值确定性失败
    证据: .sisyphus/evidence/task-7-int8-kernels-error.txt
  ```

  **提交**: 是 | 消息: `feat(runtime): add int8 weight-only cpu linear kernels` | 文件: `src/`, `include/`, `tests/unit/`, `tests/benchmark/`

- [ ] 8. 实现 Safetensors 资源读取器、分片解析器和权重重排存储

  **做什么**: 为 V1 实现标准资源摄取路径。读取 HuggingFace 模型目录，解析一个或多个 Safetensors 分片，验证头和张量元数据，mmap 或缓冲加载原始权重存储，并将张量重排到任务 7、10 和 11 期望的精确内部 CPU 布局。支持选定的 Llama 家族契约的显式张量名称映射，并确定性地处理绑定嵌入。
  **禁止做**: 不要支持 PyTorch pickle 权重、GGUF 或模糊的张量名称重映射启发式。

  **推荐代理配置**:
  - 类别: `deep` - 原因: 资产加载正确性和内部布局选择控制整个运行时
  - 技能: `[]` - 不需要特殊外部技能
  - 省略: `[playwright]` - 不适用

  **并行化**: 能否并行: 是 | 第 2 波 | 阻塞: 10, 11, 14 | 被阻塞于: 2, 3, 5

  **参考资料** (执行者没有访谈上下文 - 要全面):
  - 模式: `include/memory/data_ptr.h:77` - 遵循映射/重排缓冲区的显式所有权/生命周期规则
  - 模式: `include/memory/storage.h:17` - 存储包装器已支持预分配的后备内存
  - 模式: `include/object.h` - 重用现有的 Object/ObjectRef 所有权模型处理加载的资产/清单
  - 外部: `https://github.com/huggingface/safetensors/blob/main/README.md#format` - 头长度、JSON 头、偏移和填充行为的标准文件格式
  - 外部: `https://huggingface.co/docs/safetensors/index` - 实用的 safetensors 约束和元数据期望
  - 测试: `tests/unit/test_storage.cpp` - 遵循所有权/生命周期验证模式

  **验收标准** (仅代理可执行):
  - [ ] 加载器接受一个受支持的微小 Safetensors 夹具，验证偏移/形状/dtype，并构建类型化清单
  - [ ] 加载器以显式错误拒绝格式错误的头、不支持的 dtype、缺少的张量和冲突的分片清单
  - [ ] 重排存储为解码路径暴露确定性内部缓冲区，而不依赖按需解码时格式转换

  **QA 场景** (强制 - 没有这些任务不完整):
  ```text
  场景: 支持的 safetensors 资源集正确加载和重排
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=SafetensorsLoader.*
    预期: 加载器测试通过；清单、分片解析和重排布局与夹具黄金标准匹配
    证据: .sisyphus/evidence/task-8-safetensors-loader.txt

  场景: 格式错误的 safetensors 元数据干净失败
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=SafetensorsLoader.RejectsMalformedHeader
    预期: 测试通过，确认对无效头大小、偏移或 dtype 元数据的确定性拒绝
    证据: .sisyphus/evidence/task-8-safetensors-loader-error.txt
  ```

  **提交**: 是 | 消息: `feat(runtime): add safetensors asset loader and repack store` | 文件: `src/`, `include/`, `tests/unit/`, `tests/fixtures/`

- [ ] 9. 实现 KV 缓存布局和请求状态生命周期

  **做什么**: 为 V1 CPU 解码实现一个精确的 KV 缓存设计: 按层、K/V 头分组、序列位置和头维度分区的连续预分配存储。添加拥有当前位置、解码 token 计数、临时引用和 KV 追加/更新助手的请求状态对象。提供参考无缓存路径，以便奇偶校验测试可以将缓存解码与重新计算进行比较。
  **禁止做**: 不要在 V1 中添加分页注意力、滑动窗口、驱逐、前缀共享或多请求缓存管理。

  **推荐代理配置**:
  - 类别: `deep` - 原因: 缓存布局正确性和生命周期语义对解码稳定性至关重要
  - 技能: `[]` - 仓库本地系统工作
  - 省略: `[playwright]` - 不相关

  **并行化**: 能否并行: 是 | 第 2 波 | 阻塞: 10, 11 | 被阻塞于: 3, 5

  **参考资料** (执行者没有访谈上下文 - 要全面):
  - 模式: `include/memory/storage.h:13` - KV 缓冲区应使用现有的存储所有权模式
  - 模式: `include/tensor_impl.h:76` - 张量视图可能别名后备存储；小心利用这一点进行缓存切片
  - 模式: `ammalloc/include/ammalloc/ammalloc.h` - 临时临时存储与持久 KV 后备分开
  - 外部: `https://huggingface.co/docs/transformers/main/en/cache_explanation` - 概念性 KV 缓存语义；实现仍然需要固定的 V1 布局
  - 测试: `tests/unit/test_storage.cpp` - 重用后备缓冲区的低级存储正确性风格检查

  **验收标准** (仅代理可执行):
  - [ ] KV 缓存追加/更新 API 为支持的形状确定性地存储和检索每层 K/V 切片
  - [ ] 缓存解码在微小夹具上与无缓存参考路径产生相同的 token 输出
  - [ ] 运行时断言在配置的最大序列长度之外零缓存增长，超出预分配容量

  **QA 场景** (强制 - 没有这些任务不完整):
  ```text
  场景: KV 缓存追加和奇偶校验测试通过
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=KVCache.*
    预期: KV 缓存测试通过，包括与无缓存解码参考的奇偶校验
    证据: .sisyphus/evidence/task-9-kv-cache.txt

  场景: KV 缓存容量溢出被显式拒绝
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=KVCache.RejectsCapacityOverflow
    预期: 测试通过，确认最大上下文溢出以确定性错误失败，而非损坏内存
    证据: .sisyphus/evidence/task-9-kv-cache-error.txt
  ```

  **提交**: 是 | 消息: `feat(runtime): add kv cache layout and request state` | 文件: `src/`, `include/`, `tests/unit/`

- [ ] 10. 组装 Llama 解码器块执行器

  **做什么**: 为一个受支持的 Llama 家族块和完整的块堆栈实现手写解码器执行路径: 输入嵌入、每层 RMSNorm -> qkv 投影 -> RoPE -> 注意力 -> 输出投影 -> 注意力后残差 -> FFN/SwiGLU -> 残差 -> 最终范数 -> lm_head。使用早期任务的精确内核和 KV 缓存契约支持预填充和单 token 解码。从解析的配置中明确 GQA/MQA 行为。
  **禁止做**: 不要在此任务中添加通用图运行时、动态图构建器或非 Llama 层抽象。

  **推荐代理配置**:
  - 类别: `ultrabrain` - 原因: 这是第一个完整的模型执行组装点和最高风险的正确性任务
  - 技能: `[]` - 仅内部运行时/内核
  - 省略: `[playwright]` - 仅后端工作

  **并行化**: 能否并行: 否 | 第 2 波 | 阻塞: 11 | 被阻塞于: 4, 5, 6, 7, 8, 9

  **参考资料** (执行者没有访谈上下文 - 要全面):
  - 模式: `include/tensor.h:12` - 使用现有的张量对象作为执行值
  - 模式: `include/function_schema.h:130` - 如果模式用于内部验证，保持它们狭窄和显式
  - 模式: `include/dispatcher.h:31` - 不要通过未完成的通用分发器路由
  - 外部: `https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py` - Llama 块的规范前向顺序参考
  - 外部: `https://arxiv.org/abs/2302.13971` - RMSNorm/RoPE/GQA 假设的 LLaMA 架构参考

  **验收标准** (仅代理可执行):
  - [ ] 单层和完整堆栈解码器执行测试通过确定性夹具黄金标准
  - [ ] 预填充和单 token 解码都通过相同的运行时契约执行，显式路径覆盖
  - [ ] 执行器需要的不受支持配置分支在模型初始化期间失败，而非运行中途

  **QA 场景** (强制 - 没有这些任务不完整):
  ```text
  场景: Llama 解码器块和堆栈确定性执行
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=LlamaDecode.*
    预期: 预填充和解码测试通过精确或容差边界内的夹具黄金标准
    证据: .sisyphus/evidence/task-10-llama-decode.txt

  场景: 不受支持的执行器配置在解码前失败
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=LlamaDecode.RejectsUnsupportedExecutorConfig
    预期: 测试通过，确认不支持的 rope/头/布局组合在初始化时失败
    证据: .sisyphus/evidence/task-10-llama-decode-error.txt
  ```

  **提交**: 是 | 消息: `feat(runtime): assemble llama decoder executor` | 文件: `src/`, `include/`, `tests/unit/`

- [ ] 11. 实现模型组装、同步生成会话和贪婪解码循环

  **做什么**: 构建将解析的配置、重排的权重、分发表和 KV 缓存策略绑定到一个可执行工件中的运行时模型对象。实现具有 token ID 输入/token ID 输出语义、最大长度/eos 停止规则、确定性贪婪采样和显式预热钩子的同步预填充/解码会话对象。这是 V1 成功路径任务: CPU 上的稳定单请求生成。
  **禁止做**: 不要添加束搜索、top-k/top-p、推测解码、流式回调或并发请求调度。

  **推荐代理配置**:
  - 类别: `ultrabrain` - 原因: 这是最高级别的集成点和官方 V1 成功标准
  - 技能: `[]` - 不需要额外的外部工具
  - 省略: `[playwright]` - 仅后端系统工作

  **并行化**: 能否并行: 否 | 第 3 波 | 阻塞: 12, 14, 15 | 被阻塞于: 3, 5, 8, 9, 10

  **参考资料** (执行者没有访谈上下文 - 要全面):
  - 模式: `include/object.h` - 运行时模型/会话所有权应在适当时遵循仓库侵入式对象约定
  - 模式: `include/function.h:189` - 如果任何内部全局可调用查找保留，将其排除在热解码路径之外
  - 模式: `include/c_api.h:40` - 生命周期敏感操作应保持显式和基于句柄
  - 测试: `tests/unit/test_function.cpp` - 遵循运行时入口点的直接调用/调用验证风格
  - 外部: `https://huggingface.co/docs/transformers/main/en/generation_strategies` - 只有贪婪基线行为对 V1 重要；用作语义参考，而非范围扩展

  **验收标准** (仅代理可执行):
  - [ ] 受支持的微小夹具模型通过 C++ 运行时 API 执行确定性的 prompt -> 预填充 -> 解码 -> eos 停止
  - [ ] 预热后的解码断言在支持的路径中无稳态分配
  - [ ] 至少存在一个确定性 prompt 夹具的黄金 token 测试

  **QA 场景** (强制 - 没有这些任务不完整):
  ```text
  场景: 端到端贪婪生成在 CPU 上成功
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntime.EndToEndGreedyGeneration
    预期: 测试通过，并为受支持的微小夹具模型发出精确预期的 token 序列
    证据: .sisyphus/evidence/task-11-generation-session.txt

  场景: 解码循环拒绝无效的停止/配置设置
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntime.RejectsInvalidGenerationConfig
    预期: 测试通过，确认无效的最大长度或 eos 配置在生成前干净失败
    证据: .sisyphus/evidence/task-11-generation-session-error.txt
  ```

  **提交**: 是 | 消息: `feat(runtime): add synchronous llama generation session` | 文件: `src/`, `include/`, `tests/unit/`

- [ ] 12. 暴露稳定的 C++ API 和薄 C ABI

  **做什么**: 围绕不透明运行时/模型/会话句柄发布 V1 嵌入表面。C++ API 可以在内部使用丰富的类型，但 C ABI 必须暴露创建/加载/预填充/解码/销毁/错误检索操作，具有显式的所有权和缓冲区契约。保持 ABI 薄于 C++ 运行时，而不是发明第二个实现。使 token ID 成为 ABI 有效负载；分词器/文本处理保持在 ABI 核心之外。
  **禁止做**: 不要通过 C ABI 直接暴露模板、STL 容器或分词器/文本语义。

  **推荐代理配置**:
  - 类别: `unspecified-high` - 原因: 这是产品边界，必须在人体工程学、生命周期安全和未来可扩展性之间取得平衡
  - 技能: `[]` - 仓库本地 API 设计
  - 省略: `[playwright]` - 不相关

  **并行化**: 能否并行: 是 | 第 3 波 | 阻塞: 15 | 被阻塞于: 3, 5, 11

  **参考资料** (执行者没有访谈上下文 - 要全面):
  - 模式: `include/c_api.h:16` - 现有仓库已经使用不透明句柄和显式引用计数/生命周期钩子
  - 模式: `include/object.h` - 尽可能重用所有权语义，而不是发明特别的生命周期规则
  - 模式: `include/container/array.h` - 如果需要任何内部动态数组，将它们保留在 C++ 内部
  - 测试: `tests/unit/test_object.cpp` - 遵循生命周期/引用管理验证模式

  **验收标准** (仅代理可执行):
  - [ ] C++ API 和 C ABI 都支持受支持的 V1 路径的创建/加载/预填充/解码/销毁
  - [ ] ABI 测试验证确定性 token 输出、显式错误检索和干净拆卸
  - [ ] 没有 ABI 符号要求调用者理解内部 C++ 对象布局

  **QA 场景** (强制 - 没有这些任务不完整):
  ```text
  场景: C ABI 冒烟路径端到端工作
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=CAbiRuntime.*
    预期: ABI 冒烟测试通过创建/加载/预填充/解码/销毁并与 C++ 运行时 token 输出匹配
    证据: .sisyphus/evidence/task-12-c-abi.txt

  场景: ABI 报告错误而不泄漏资源
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=CAbiRuntime.ReportsLoadFailureCleanly
    预期: 测试通过，确认无效模型路径/配置呈现确定性错误并释放所有临时资源
    证据: .sisyphus/evidence/task-12-c-abi-error.txt
  ```

  **提交**: 是 | 消息: `feat(runtime): expose stable cpp and c runtime apis` | 文件: `include/`, `src/`, `tests/unit/`

- [ ] 13. 在相同的运行时契约下添加 INT4 仅权重量化扩展

  **做什么**: 将 V1 运行时扩展为支持 INT8 已覆盖的相同线性表面的一个精确 INT4 仅权重量化打包/布局。重用相同的分发表、加载器清单和模型组装流程。将 INT4 严格限制在选定的方案后面；不支持的 INT4 变体必须在加载/注册期间失败。
  **禁止做**: 不要添加多种 INT4 布局、混合激活量化或 INT4 的单独模型/运行时路径。

  **推荐代理配置**:
  - 类别: `deep` - 原因: 必须在不破坏现在工作的 V1 运行时契约的情况下扩展量化覆盖
  - 技能: `[]` - 不需要特殊外部技能
  - 省略: `[playwright]` - 仅计算/运行时

  **并行化**: 能否并行: 是 | 第 3 波 | 阻塞: 15 | 被阻塞于: 3, 4, 5, 7, 11

  **参考资料** (执行者没有访谈上下文 - 要全面):
  - 模式: `include/device.h:37` - 保留仅 CPU V1 后端假设
  - 模式: `tests/benchmark/benchmark_memory_pool.cpp` - 使用与仓库一致的基准风格进行量化内核性能检查
  - 外部: `https://huggingface.co/docs/transformers/main/en/quantization/concept_guide` - 一般量化语义参考；实现仍然必须冻结一个精确方案
  - 外部: `https://github.com/huggingface/safetensors` - 量化权重仍然通过相同的资产路径到达，必须保留元数据完整性

  **验收标准** (仅代理可执行):
  - [ ] INT4 路径通过相同的模型/会话 API 加载、注册和执行，与 INT8 相同
  - [ ] INT4 奇偶校验测试在受支持的夹具的显式容差内通过参考路径
  - [ ] 不支持的 INT4 元数据/布局组合在解码开始前失败

  **QA 场景** (强制 - 没有这些任务不完整):
  ```text
  场景: INT4 路径通过相同的运行时契约执行
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=QuantLinearInt4.*
    预期: INT4 测试通过，在受支持的路径上进行加载、分发和解码奇偶校验
    证据: .sisyphus/evidence/task-13-int4-extension.txt

  场景: 不支持的 INT4 元数据在执行前被拒绝
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=QuantLinearInt4.RejectsUnsupportedPacking
    预期: 测试通过，确认无效 INT4 打包/布局在加载/注册时失败
    证据: .sisyphus/evidence/task-13-int4-extension-error.txt
  ```

  **提交**: 是 | 消息: `feat(runtime): add int4 weight-only extension` | 文件: `src/`, `include/`, `tests/unit/`, `tests/benchmark/`

- [ ] 14. 添加可选的分词器适配器和文本便利示例

  **做什么**: 实现核心运行时之外的薄文本适配器层，为受支持的 V1 演示路径将文本转换为 token ID 并返回。保持核心运行时基于 token。为 V1 文本演示支持一个精确的分词器资源策略: 首先支持 SentencePiece 兼容的 `tokenizer.model`；用显式错误拒绝不支持的分词器资源类型。提供一个端到端示例，加载模型 + 分词器资源，对 prompt 进行分词，运行生成，并通过仅 C++ API 反分词输出。
  **禁止做**: 不要将分词器支持作为核心 C ABI 的先决条件，或将 V1 扩展为通用分词器框架。

  **推荐代理配置**:
  - 类别: `deep` - 原因: 除非保持薄和显式，否则分词器是隐藏的范围炸弹
  - 技能: `[]` - 仓库代码加官方分词器引用足够
  - 省略: `[playwright]` - 不适用

  **并行化**: 能否并行: 是 | 第 3 波 | 阻塞: 15 | 被阻塞于: 2, 5, 8, 11

  **参考资料** (执行者没有访谈上下文 - 要全面):
  - 模式: `include/c_api.h:16` - 分词器保持在 C ABI 核心之外；保持 ABI 基于 token
  - 模式: `include/container/string.h:45` - C++ 中的字符串所有权/传递应遵循现有仓库约定
  - 外部: `https://github.com/google/sentencepiece` - `tokenizer.model` 的官方 SentencePiece 参考
  - 外部: `https://huggingface.co/docs/tokenizers/index` - 分词器资源语义和陷阱
  - 外部: `https://github.com/huggingface/tokenizers` - 分词器 JSON/模型元数据参考；V1 中必须清楚拒绝不支持的表单

  **验收标准** (仅代理可执行):
  - [ ] 示例代码演示通过 C++ API 的 prompt 文本 -> token ID -> 运行时生成 -> 反分词输出
  - [ ] 分词器适配器以显式错误拒绝不支持的分词器资源
  - [ ] 核心运行时测试保持基于 token，不依赖分词器可用性

  **QA 场景** (强制 - 没有这些任务不完整):
  ```text
  场景: 文本便利示例与支持的分词器资源一起工作
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=TokenizerAdapter.*
    预期: 分词器适配器测试通过，微小夹具上的示例 prompt 往返是确定性的
    证据: .sisyphus/evidence/task-14-tokenizer-adapter.txt

  场景: 不支持的分词器资源干净失败
    工具: Bash
    步骤: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=TokenizerAdapter.RejectsUnsupportedAsset
    预期: 测试通过，确认在运行时执行开始之前拒绝不支持的分词器文件
    证据: .sisyphus/evidence/task-14-tokenizer-adapter-error.txt
  ```

  **提交**: 是 | 消息: `feat(runtime): add tokenizer adapter and text example` | 文件: `src/`, `include/`, `tests/unit/`, `examples/`

- [ ] 15. 使用 CI、消毒器变体、基准和端到端回归门加固 V1

  **做什么**: 添加集成后保持 V1 稳定所需的工程加固: GitHub Actions CI、CTest 集成或等效的显式测试运行器契约、ASAN/UBSAN CMake 选项以及现有的 TSAN 支持、解码/KV/int8/int4 路径的基准注册，以及在黄金 token、ABI 冒烟或无分配回归上失败的回归门。仅在需要时更新项目文档以指向新的运行时构建/测试命令。
  **禁止做**: 不要在没有基准基线的情况下添加性能承诺，也不要将失败测试隐藏在可选/仅手动脚本后面。

  **推荐代理配置**:
  - 类别: `unspecified-high` - 原因: 最终稳定跨越构建系统、测试、性能和贡献者工作流
  - 技能: `[]` - 仓库原生构建/测试工作
  - 省略: `[playwright]` - 不相关

  **并行化**: 能否并行: 否 | 第 3 波 | 阻塞: F1, F2, F3, F4 | 被阻塞于: 5, 11, 12, 13, 14

  **参考资料** (执行者没有访谈上下文 - 要全面):
  - 模式: `CMakeLists.txt:15` - 测试/基准/消毒器切换的现有选项风格
  - 模式: `tests/unit/CMakeLists.txt:1` - 现有的一二进制测试组织，有意保留或扩展
  - 模式: `tests/benchmark/CMakeLists.txt:1` - 当前基准目标组织
  - 模式: `AGENTS.md:29` - 仅当实现更改它们时才应更新当前记录的构建/测试命令
  - 测试: `tests/benchmark/benchmark_memory_pool.cpp:23` - 保持与当前仓库一致的基准注册风格

  **验收标准** (仅代理可执行):
  - [ ] CI 为 V1 运行时路径运行构建 + 单元测试 + 选定的基准健全性检查
  - [ ] ASAN/UBSAN 和 TSAN 变体作为记录的构建选项存在，至少一个运行时聚焦的测试套件通过它们
  - [ ] 回归门在黄金 token 漂移、ABI 冒烟失败和解码分配回归上失败

  **QA 场景** (强制 - 没有这些任务不完整):
  ```text
  场景: V1 运行时的强化构建/测试矩阵通过
    工具: Bash
    步骤: cmake -S . -B build-v1 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON && cmake --build build-v1 -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntime.*:CAbiRuntime.*:TokenizerAdapter.* && ./build-v1/tests/benchmark/aethermind_benchmark --benchmark_filter=BM_LlamaDecode|BM_KVCache|BM_QuantLinear
    预期: 构建、单元测试和基准健全性检查都通过受支持的 V1 运行时路径
    证据: .sisyphus/evidence/task-15-hardening.txt

  场景: 消毒器构建捕获运行时回归
    工具: Bash
    步骤: cmake -S . -B build-v1-asan -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF -DENABLE_ASAN=ON -DENABLE_UBSAN=ON && cmake --build build-v1-asan --target aethermind_unit_tests -j && ./build-v1-asan/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntime.* && cmake -S . -B build-v1-tsan -DENABLE_TSAN=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF && cmake --build build-v1-tsan --target aethermind_unit_tests -j && ./build-v1-tsan/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntimeThreading.*
    预期: 消毒器目标测试套件通过，无内存或竞争诊断
    证据: .sisyphus/evidence/task-15-hardening-error.txt
  ```

  **提交**: 是 | 消息: `chore(runtime): harden v1 with ci sanitizers and regressions` | 文件: `.github/workflows/`, `CMakeLists.txt`, `tests/unit/`, `tests/benchmark/`, `docs/`, `AGENTS.md`

## 最终验证波次 (4 个并行代理，全部必须批准)
- [ ] F1. 计划合规审计 - oracle
- [ ] F2. 代码质量审查 - unspecified-high
- [ ] F3. 真实手动 QA - unspecified-high
- [ ] F4. 范围保真度检查 - deep

## 提交策略
- 提交 1: V1 契约、构建目标、夹具和验证管道
- 提交 2: CPU 运行时上下文、分发表和参考内核表面
- 提交 3: 加载器/重排、KV 缓存、解码器执行和贪婪生成
- 提交 4: C ABI、分词器适配器和示例
- 提交 5: INT4 扩展、CI/消毒器、基准和加固

## 成功标准
- 受支持的微小夹具模型和一个受支持的真实 Llama 家族量化检查点都通过标准 HF 资源路径加载
- 运行时在启用 KV 缓存的情况下在 CPU 上执行确定性贪婪生成，并产生预期的黄金 token 序列
- 解码循环在支持的预热单请求流程后无稳态分配
- C++ API 和 C ABI 都通过创建/加载/预填充/解码/销毁冒烟测试
- 存在解码、KV 缓存访问和量化线性内核的基准，并记录回归阈值
