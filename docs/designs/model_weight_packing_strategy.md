# 模型权重打包策略设计文档

## 1. 概述

本文档描述 AetherMind 推理引擎的模型权重打包策略，包括打包时机、多后端支持和内存优化方案。

## 2. 核心原则

### 2.1 打包是一次性工作

权重打包（layout 转换、padding、重排）是**静态转换**：
- **输入**：原始模型权重（logical weights）
- **输出**：backend 优化格式（packed weights）
- **特性**：只依赖权重本身和目标 backend/selector，不依赖输入数据

这类工作应该**预计算并缓存**，而不是每次推理重复做。

### 2.2 打包后的权重属于模型状态

```cpp
class ModelInstance {
    BackendSidecar backend_sidecar_;  // 持有 PackedWeights
};
```

打包后的权重随模型加载而生成，随模型销毁而释放，是**模型状态的一部分**。

## 3. 打包时机

### 3.1 正确时序

```
[模型加载阶段]
    │
    ├── 从磁盘读取原始权重（logical weights）
    │
    ├── 对每个需要打包的权重：
    │     └── CpuWeightPrepacker::Pack(logical_weight, selector)
    │           └── 生成 PackedWeights
    │
    ├── ModelInstance::StorePackedWeights(...)
    │     └── 存入 BackendSidecar
    │
    └── [ModelInstance 持有 packed weights，长期存活]

[执行计划构建阶段]
    │
    └── ExecutionPlanBuilder 从 ModelInstance 查找 packed weights
          └── 绑定到 ExecutionStep.packed_params
          └── （只取地址，不重新打包）

[推理执行阶段]
    │
    └── Kernel 直接使用 step.packed_params
          └── （已经是打包格式，无需转换）
```

### 3.2 各阶段职责

| 阶段 | 打包相关职责 |
|------|-------------|
| **模型加载** | **核心职责**：读取权重、打包转换、存入 ModelInstance |
| **执行计划构建** | 查找 packed weights，绑定地址到 plan |
| **推理执行** | 无打包职责，直接使用 packed_params |

### 3.3 反模式

#### ❌ 在 ExecutionPlanBuilder 里打包

```cpp
// 错误：每次构建 plan 都重新打包
ExecutionPlanBuilder::Build(...) {
    for (each node) {
        if (needs_packed_weights) {
            auto packed = CpuWeightPrepacker::Pack(...);  // ❌ 重复工作
            step.packed_params = packed->storage().data();
        }
    }
}
```

**问题**：
- 每次构建 plan 都重新打包（可能多次）
- 没有缓存，性能浪费
- packed weights 生命周期管理混乱

#### ❌ 在 Kernel 执行时打包

```cpp
// 错误：每次 kernel 调用都打包
Kernel::Execute(...) {
    auto packed = PackWeightsOnTheFly(...);  // ❌ 极端浪费
}
```

**问题**：
- 每次推理都重新打包
- 完全违背了打包优化的初衷

## 4. 打包策略

### 4.1 Eager Packing（预打包）

```cpp
class ModelInstance {
public:
    Status Load(const std::string& path) {
        // 1. 读取原始权重
        auto logical_weights = ReadFromDisk(path);
        
        // 2. 对所有可能的 (op_type, selector) 组合预打包
        for (const auto& weight : logical_weights) {
            for (const auto& selector : GetSupportedSelectors()) {
                auto packed = PrepackerFor(selector.device_type)
                    ->Pack(weight, selector);
                StorePackedWeights(std::move(packed));
            }
        }
        return Status::Ok();
    }
};
```

**优点**：
- 推理时零延迟，所有权重已就绪
- 执行计划构建简单，直接查找即可
- 内存布局稳定，便于调试

**缺点**：
- 加载时间长（需要打包所有组合）
- 内存占用大（原始 + 所有 packed 格式）
- 可能打包了永远用不到的格式

### 4.2 Lazy Packing（按需打包）

```cpp
class ModelInstance {
public:
    const PackedWeights* FindOrPack(
            OpType op_type,
            const KernelSelector& selector,
            const Tensor& logical_weight) {
        // 1. 先查找
        if (auto* packed = sidecar_.Find(op_type, selector)) {
            return packed;
        }
        
        // 2. 未找到则即时打包
        auto packed = PrepackerFor(selector.device_type)
            ->Pack(logical_weight, selector);
        
        // 3. 存储并返回
        StorePackedWeights(std::move(packed));
        return sidecar_.Find(op_type, selector);
    }
};
```

**优点**：
- 加载快，只打包实际用到的格式
- 内存占用小（只存必要的 packed 格式）
- 支持动态发现新 selector

**缺点**：
- 第一次推理有打包延迟（cold start）
- 需要线程安全（并发访问可能同时触发打包）
- 执行计划构建可能触发打包，增加复杂度

### 4.3 混合策略（推荐）

```cpp
class ModelInstance {
public:
    // 核心 selector（已知会用的）eager 打包
    Status Load(const std::string& path) {
        auto logical_weights = ReadFromDisk(path);
        
        // Eager: 默认 selector（CPU scalar）
        for (const auto& weight : logical_weights) {
            auto packed = cpu_prepacker_.Pack(weight, kDefaultSelector);
            StorePackedWeights(std::move(packed));
        }
        
        // 保存 logical weights 供 lazy 打包
        logical_weights_ = std::move(logical_weights);
        return Status::Ok();
    }
    
    // 非核心 selector lazy 打包
    const PackedWeights* FindOrPack(
            OpType op_type,
            const KernelSelector& selector) {
        if (auto* packed = sidecar_.Find(op_type, selector)) {
            return packed;
        }
        
        // Lazy: 从 logical weights 重新打包
        auto& logical = FindLogicalWeight(op_type);
        auto packed = PrepackerFor(selector.device_type)
            ->Pack(logical, selector);
        StorePackedWeights(std::move(packed));
        
        // 可选：释放已打包的 logical weight
        MaybeReleaseLogicalWeight(op_type);
        
        return sidecar_.Find(op_type, selector);
    }

private:
    std::unordered_map<OpType, Tensor> logical_weights_;
};
```

**策略**：
- 80% 场景用 eager（默认 selector）
- 20% 场景用 lazy（特殊 selector）
- 平衡加载速度和内存占用

## 5. 多后端支持

### 5.1 同一权重，不同格式

| Backend | 打包格式 | 示例 |
|---------|---------|------|
| CPU Scalar | 原始 layout | 直接 memcpy |
| CPU AVX2 | 8x32 重排 | 便于 SIMD 访问 |
| CPU AMX | tile 格式 | Intel AMX 指令 |
| CUDA | NCuDNN 格式 | NHWC 或自定义 |
| CANN | Ascend 格式 | 达芬奇架构优化 |

### 5.2 BackendSidecar 抽象

```cpp
class BackendSidecar {
    // 不区分 backend 类型，只按 selector 存储
    std::vector<std::unique_ptr<PackedWeights>> packed_weights_;
    
public:
    const PackedWeights* Find(OpType op_type, 
                              const KernelSelector& selector);
};
```

**优点**：
- ModelInstance 不感知具体 backend
- selector 包含 device_type，自然分区
- 新增 backend 只需新增 prepacker，无需改 ModelInstance

**关键**：`KernelSelector` 作为 key 包含 device_type

```cpp
struct KernelSelector {
    DeviceType device_type;      // CPU/CUDA/CANN
    DataType activation_dtype;   // Float32/BFloat16
    IsaLevel isa;                // Scalar/AVX2/AMX
    WeightFormat weight_format;  // Plain/Packed
    // ...
};
```

### 5.3 分层打包管理（推荐）

```cpp
// 1. 基础打包：跨 backend 通用
class WeightPrepacker {
public:
    virtual StatusOr<std::unique_ptr<PackedWeights>> Pack(
        const Tensor& logical,
        const KernelSelector& selector) = 0;
};

// 2. 每个 backend 注册自己的 prepacker
class CpuWeightPrepacker : public WeightPrepacker { ... };
class CudaWeightPrepacker : public WeightPrepacker { ... };

// 3. ModelInstance 通过工厂获取 prepacker
class ModelInstance {
    std::unordered_map<DeviceType, std::unique_ptr<WeightPrepacker>> prepackers_;
    
    const PackedWeights* FindOrPack(OpType op_type,
                                    const KernelSelector& selector) {
        auto* prepacker = prepackers_[selector.device_type].get();
        // ...
    }
};
```

**扩展性**：新增 backend 只需注册 prepacker，ModelInstance 自动支持

## 6. 内存优化

### 6.1 问题：打包后，原始权重还留着吗？

```
Memory layout:
┌─────────────────┐
│ Logical Weight  │ 100MB (原始格式)
│ (Tensor)        │
├─────────────────┤
│ Packed Weight   │ 80MB (CPU packed)
│ (PackedWeights) │
├─────────────────┤
│ Packed Weight   │ 75MB (CUDA packed)
│ (PackedWeights) │
└─────────────────┘
Total: 255MB
```

### 6.2 方案 A：保留原始权重（保守）

```cpp
class ModelInstance {
    std::unordered_map<OpType, Tensor> logical_weights_;  // 保留
    BackendSidecar sidecar_;
};
```

**场景**：
- 需要支持动态重新打包（lazy 策略）
- 多 backend 切换，可能需要不同格式
- 调试需要对比原始值

### 6.3 方案 B：释放原始权重（激进）

```cpp
Status ModelInstance::FinalizePacking() {
    // 标记所有已打包的权重
    for (const auto& [op_type, packed] : sidecar_) {
        packed_logical_weights_.insert(op_type);
    }
    
    // 释放原始 Tensor 的底层 Buffer
    for (auto& [op_type, tensor] : logical_weights_) {
        if (packed_logical_weights_.contains(op_type)) {
            tensor.Reset();  // 释放底层内存，保留元数据壳
        }
    }
    
    // 或者完全删除
    logical_weights_.clear();
    logical_weights_.shrink_to_fit();
}
```

**触发时机**：
- Eager packing 完成后立即释放
- Lazy packing 达到覆盖率阈值后（如 95% 权重已打包）

**风险**：
- 无法再重新打包（除非从磁盘重新加载）
- 需要确保所有可能的 selector 都已打包

### 6.4 方案 C：引用计数共享（高级）

```cpp
class PackedWeights {
    // 共享底层 storage，但不同 view
    std::shared_ptr<Buffer> storage_;
    Layout layout_;  // 描述如何解释 storage
};

// 某些打包格式可以与原始权重共享内存
class CpuPlainPackedWeights : public PackedWeights {
    // 如果 selector 要求 Plain format
    // 直接引用 logical weight 的 Buffer，不复制
};
```

**优点**：零拷贝，内存最优  
**缺点**：复杂度极高，需要 format 支持

## 7. 推荐架构

```cpp
/// ModelInstance 负责权重的生命周期和打包管理
class ModelInstance {
public:
    /// 加载模型，eager 打包核心格式
    Status Load(const std::string& path, 
                const RuntimeConfig& config);
    
    /// 查找或按需打包（lazy 回退）
    const PackedWeights* FindOrPack(
        OpType op_type,
        const KernelSelector& selector);
    
    /// 完成打包后释放原始权重
    Status FinalizePacking();
    
    /// 序列化 packed weights（缓存到磁盘）
    Status SavePackedCache(const std::string& cache_path);
    
    /// 从缓存加载（跳过打包）
    Status LoadPackedCache(const std::string& cache_path);

private:
    // 原始权重（lazy 打包期间保留，finalize 后可释放）
    std::unordered_map<OpType, Tensor> logical_weights_;
    
    // 打包后的权重（长期持有）
    BackendSidecar sidecar_;
    
    // 各 backend 的 prepacker
    std::unordered_map<DeviceType, std::unique_ptr<WeightPrepacker>> prepackers_;
    
    // 加载配置（决定哪些 selector 需要 eager 打包）
    RuntimeConfig config_;
};
```

## 8. 总结

- **打包时机**：模型加载阶段，避免执行时重复打包
- **打包策略**：混合方案（eager 核心格式 + lazy 边缘格式）
- **多后端**：通过 BackendSidecar + KernelSelector 自然分区
- **内存优化**：finalize 后释放原始权重，或按需保留支持动态重新打包
