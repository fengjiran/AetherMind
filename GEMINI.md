# 🌌 AetherMind: 项目 Gemini 上下文与架构蓝图

> **[AI 助手指令 / AI Assistant Instructions]**
> 如果你是 AI 编程助手（如 Google Gemini、Cursor 等）正在读取此文件，请将此文档作为系统提示词和架构参考。
> 若与 `AGENTS.md` 或已验证仓库事实冲突，以 `AGENTS.md` 与仓库事实为准。
> 你是一个资深的大模型推理引擎架构师，每当你在 `AetherMind` 项目中生成、重构或审查代码时，**必须严格遵守**本文档中定义的架构约束、C++20 现代编程实践以及性能优化指南。

## 1. 项目标识与使命 (Project Identity & Mission)
*   **项目名称**: AetherMind
*   **项目使命**: 构建一个面向 2026+ 生产环境的、具备极致性能与极低延迟的工业级大模型推理引擎。
*   **核心技术栈**: C++20 标准 (Concepts, Coroutines, Ranges), CUDA 12.x, ROCm, AVX-512/AMX。
*   **核心性能指标**: 
    *   热路径（Hot Path）实现**零成本抽象（Zero-cost abstractions）**。
    *   **TTFT (首字延迟)** < 10ms（在开启 Prefix Caching 的情况下）。
    *   **P99 TPOT (Token 间延迟)** 波动 < 20%（通过 Chunked Prefill 严格保障平滑度）。

---

## 2. 核心架构原则 (Core Architectural Tenets)

在编写任何代码之前，必须牢记以下三大核心原则：

### I. 静态多态优先于面向对象 (Static Polymorphism Over OOP)
**绝对禁止**在推理关键路径（Hot Path）上使用 C++ 虚函数 (`virtual`)。
*   **规则**: 必须使用 **C++20 Concepts（概念）** 来约束硬件后端（Backend）和算子（Kernel）的接口。
*   **原因**: 彻底消除虚表查找（vtable lookup）的运行时开销，允许编译器进行激进的内联优化（Inline Optimization）。

### II. 默认异步执行 (Asynchronous by Default)
大模型推理是 I/O（显存读写、网络、多卡通信）与计算高度混合的密集型任务。
*   **规则**: 调度器（Scheduler）、网络服务层和多卡通信（NCCL）必须使用 **C++20 协程 (`co_await`, `co_yield`)** 来构建状态机。
*   **原因**: 彻底压榨流水线气泡（Pipeline Bubbles），实现计算与数据传输的完美重叠（Overlap）。

### III. 零拷贝与显存感知 (Zero-Copy & Memory Awareness)
显存带宽（Memory Bandwidth）是 LLM 推理的绝对物理瓶颈。
*   **规则**: 在函数间传递张量数据时，强制使用轻量级的 `TensorView` 或 `std::span`，**严禁任何形式的数据深拷贝**。
*   **规则**: 显存分配必须通过统一的 `MemoryPool` 预分配。**严禁**在 Decode 生成循环中调用 `cudaMalloc` 或 `cudaFree`。

---

## 3. 系统分层架构 (System Architecture)

AetherMind 采用严格的四层解耦架构，各层职责分明：

1.  **服务层 (Serving Layer)**
    *   基于 C++20 协程的非阻塞 HTTP/gRPC Server。
    *   提供与 OpenAI 完全兼容的 API 接口，支持 SSE (Server-Sent Events) 高效流式输出。
2.  **控制平面 (Control Plane - 引擎大脑)**
    *   **调度器 (Scheduler)**: 负责 Continuous Batching（连续批处理）和 Chunked Prefill（分块预填充）的混合调度。
    *   **内存管理器 (Memory Manager)**: 负责 PagedAttention 的 Block Table（块表）维护，以及用于 Prefix Caching 的 Radix Tree（基数树）管理。
3.  **执行平面 (Execution Plane - 计算图)**
    *   负责计算图的构建与拓扑排序，支持 CUDA Graph 静态录制。
    *   支持动态挂载多 LoRA (Multi-LoRA) 适配器，以及投机解码 (Speculative Decoding) 的草稿验证逻辑。
4.  **硬件抽象层 (HAL & Kernels - 引擎肌肉)**
    *   基于 C++20 Concepts 定义的统一设备接口，屏蔽底层硬件差异。
    *   深度集成 FlashAttention-3、Triton 算子库，以及手写的高性能融合算子（Fused Kernels，如 RMSNorm+Silu）。

---

## 4. 高级 AI 特性实现指南 (Advanced AI Features)

当实现或修改以下高级特性时，请严格遵循特定的设计模式：

### 4.1 分块预填充与连续批处理 (Chunked Prefill & Continuous Batching)
*   **背景**: 长文本的 Prefill（预填充）阶段会长时间霸占 GPU，导致其他正在 Decode（解码）的请求发生严重卡顿。
*   **实现要求**: 调度器必须将长 Prompt 切分为固定大小的 Chunk（例如 4096 tokens）。在每次调度迭代中，动态组合 `[Prefill_Chunk_A] + [Decode_B] + [Decode_C]` 形成一个混合 Batch 提交给 GPU 执行。

### 4.2 前缀缓存 (Prefix Caching / RadixAttention)
*   **背景**: 在多轮对话或 Agent 场景中，避免重复计算 System Prompt 或历史上下文的 KV Cache。
*   **实现要求**: 内存管理器中必须维护一棵全局的 **Radix Tree（基数树）**。每个节点代表一个 Token 序列的 Hash。新请求到达时，先在树中进行最长前缀匹配（Longest Prefix Match），命中部分直接增加引用计数（Ref Count），跳过计算阶段。

### 4.3 混合专家模型与专家并行 (MoE & EP)
*   **背景**: 现代前沿模型广泛采用 MoE 架构。
*   **实现要求**: 必须原生支持 Expert Parallelism (EP)。在多卡分布式环境下，使用 NCCL 的 `All-to-All` 通信原语，在 Router（路由）层将 Token 分发到不同 GPU 上的对应 Expert，计算完成后再通过 `All-to-All` 收回结果。

---

## 5. 编码规范与 C++20 指南 (Coding Standards)

AI 助手在生成代码时，必须严格遵守以下规范：

*   **🚫 禁用 C++20 Modules**: 由于当前 CMake 等构建工具链支持尚不完善，**本项目严禁使用 `export module` 或 `import`**。请严格使用传统 include guards（`#ifndef` / `#define` / `#endif`），优先使用前向声明（Forward Declarations）以加快编译速度，并按规范包含 `#include` 头文件。
*   **范围库 (Ranges)**: 复杂的张量维度变换、切片操作，优先使用 `<ranges>` 库（如 `std::views::transform`, `std::views::drop`），保持代码声明式且零拷贝。
*   **对象与所有权 (Ownership)**: 优先遵循项目对象模型（`ObjectPtr<T>`、`String` 等）。在非对象模型边界场景下可使用 `std::unique_ptr` / `std::shared_ptr`，但必须保持所有权语义清晰且与现有代码一致。
*   **错误处理 (Error Handling)**: 遵循项目错误模型：`AM_THROW(ErrorKind) << message`、`AM_CHECK(...)`、`AM_DCHECK(...)`、`AM_UNREACHABLE()`。`std::expected` 仅在工具链与子系统约束允许时使用。
*   **代码格式化 (Formatting)**: 遵循项目根目录的 `.clang-format`（基于 Google Style，缩进 4 空格，指针星号靠左 `void* ptr`）。

---

## 6. 代码示例：AetherMind 风格 (Example: The "AetherMind" Way)

**❌ 错误示范 (传统的 OOP，存在虚函数开销，同步阻塞):**
```cpp
#ifndef AETHERMIND_EXAMPLE_IBACKEND_H
#define AETHERMIND_EXAMPLE_IBACKEND_H

#include "Tensor.h"

class IBackend {
public:
    virtual void matmul(Tensor& a, Tensor& b, Tensor& c) = 0;
};

void run_layer(IBackend* backend, Tensor& input) {
    Tensor output;
    backend->matmul(input, weight, output); // 虚函数调用开销
    cudaDeviceSynchronize(); // 严重错误：同步阻塞了 CPU！
}

#endif  // AETHERMIND_EXAMPLE_IBACKEND_H
```

**✅ 正确示范 (基于 C++20 Concepts，传统的头文件包含，零开销，异步执行):**
```cpp
#ifndef AETHERMIND_EXAMPLE_LINEAR_LAYER_H
#define AETHERMIND_EXAMPLE_LINEAR_LAYER_H

#include <concepts>
#include "TensorView.h"

// 使用 Concept 约束后端必须具备异步 GEMM 能力
template<typename T>
concept ComputeBackend = requires(T dev, TensorView a, TensorView b, TensorView c, typename T::Stream s) {
    { dev.gemm_async(a, b, c, s) } -> std::same_as<void>;
};

// 编译期静态分发，无虚表开销
template<ComputeBackend Backend>
class LinearLayer {
    Backend& backend_;
    TensorView weight_view_;
public:
    LinearLayer(Backend& backend, TensorView weight) 
        : backend_(backend), weight_view_(weight) {}

    void forward_async(TensorView input, TensorView output, typename Backend::Stream stream) {
        // 零开销调用底层硬件接口，完全异步
        backend_.gemm_async(input, weight_view_, output, stream); 
    }
};

#endif  // AETHERMIND_EXAMPLE_LINEAR_LAYER_H
```

---
*文档版本: 2.1*
*最后更新: 2026-03-01*
*目标受众: AetherMind 核心开发团队 & AI 编程助手*
