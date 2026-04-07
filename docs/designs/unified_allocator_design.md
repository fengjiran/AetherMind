---
scope: global
module: project
parent: none
depends_on: [ammalloc]
adr_refs: []
last_verified: 2026-04-08
owner: team
status: active
---

# AetherMind 统一设备分配器架构设计 (Unified Device Allocator)

## 1. 设计目标 (Goals)

- **运行时持有注册中心**：`AllocatorRegistry` 必须由显式的 `Runtime` 或 `InferenceContext` 实例持有，严禁使用全局静态单例。
- **Provider-Instance 分层模型**：
  - **Provider**：按 `DeviceType` (CPU, CUDA, CANN) 注册，定义特定后端分配策略。
  - **Instance**：按 `Device` (Type + Index) 解析，绑定具体的硬件设备实例。
- **解耦生命周期**：`Buffer` 与 `MemoryHandle` 是唯一持有的内存抽象。通过函数指针（Deleter）实现自包含释放，`Tensor` 或 `Storage` 严禁持有对 `IAllocator` 的引用。
- **收敛遗留接口**：明确 `Storage` / `DataPtr` / `AllocatorTable` 为仅限迁移使用的 **Legacy** 目标，新功能必须基于 `Buffer` 路径。

## 2. 核心架构 (Target Architecture)

1. **AllocatorProvider**: 每种 `DeviceType` 注册一个 Provider。
2. **AllocatorRegistry**: 由 `Runtime` 持有，负责按 `Device` 缓存并管理 `IAllocator` 实例。
3. **IAllocator**: 具体的分配执行器，绑定到特定 `Device`。
4. **Buffer & MemoryHandle**: `MemoryHandle` 采用函数指针 + 上下文（Context）模式，实现无 `IAllocator` 依赖的 RAII 释放。

## 3. 核心接口定义 (Core Interfaces)

```cpp
// 内存句柄：采用函数指针 + 上下文模式，不持有 Allocator 引用
struct MemoryHandle {
    void* ptr = nullptr;
    size_t size = 0;
    Device device;
    // 兼容当前风格：使用原始函数指针与上下文实现释放
    void (*deleter)(void* context, void* ptr) = nullptr;
    void* deleter_context = nullptr;
};

// 分配器实例接口
class IAllocator {
public:
    virtual ~IAllocator() = default;
    // 返回引用计数的 Buffer (ObjectRef)
    virtual Buffer Allocate(size_t nbytes) = 0;
    virtual Device GetDevice() const = 0;
};

// 运行时持有的注册中心 (Owned by Runtime/Context)
class AllocatorRegistry {
public:
    // 按 DeviceType 注册提供商
    void RegisterProvider(DeviceType type, std::unique_ptr<IAllocatorProvider> provider);
    
    // 按 Device (Type + Index) 获取或动态创建分配器实例
    IAllocator* GetAllocator(Device device);

private:
    std::unordered_map<DeviceType, std::unique_ptr<IAllocatorProvider>> providers_;
    std::map<Device, std::unique_ptr<IAllocator>> instances_;
};
```

## 4. 所有权与生命周期 (Ownership & Lifetime)

- **IAllocator**：由 `AllocatorRegistry` 拥有，随 `Runtime` 上下文销毁。
- **Buffer/MemoryHandle**：
  - `Buffer` 是基于 `ObjectRef` 的引用计数对象，作为 Tensor 的底层所有权抽象。
  - **核心约束**：`Buffer` 及其内部的 `MemoryHandle` **严禁**持有 `IAllocator` 的指针。
  - **释放机制**：在分配时，`IAllocator` 将释放逻辑填充到 `MemoryHandle.deleter` 和 `deleter_context` 中。当 `Buffer` 引用计数归零时，直接调用该函数指针，确保即使分配器实例已销毁，内存仍能安全回收。

## 5. 迁移与兼容性 (Legacy & Migration)

- **Legacy 状态**：
  - `AllocatorTable`：全局静态表，标记为 **Deprecated**。
  - `DataPtr` / `Storage`：过渡期容器，标记为 **Migration-Only**。
- **演进策略**：
  - 新算子路径（Operator Contract）必须通过 `Buffer` 获取数据指针。
  - `StorageImpl` 将作为 `Buffer` 的装饰器存在，直到 `Storage` 被完全移除。

## 6. 后端实现 (Backend Examples)

- **CPU (ammalloc)**：
  - `CPUAllocatorProvider` 创建绑定到 CPU 的 `IAllocator`。
  - Deleter 闭包上下文可存储页面分配器信息。
- **CUDA/CANN**：
  - 每个 `DeviceIndex` 对应独立的 `IAllocator` 实例。
  - Deleter 直接封装驱动级释放函数（如 `cudaFree`）。

## 7. 实施清单 (Implementation Checklist)

- [ ] 在 `include/aethermind/memory/` 中定义 `IAllocator` 与 `IAllocatorProvider`。
- [ ] 升级 `MemoryHandle` 以支持函数指针 + context 风格。
- [ ] 在 `Runtime` 类中集成 `AllocatorRegistry`，取代全局 `AllocatorTable`。
- [ ] 实现 `CPUAllocator` 并适配 `ammalloc` 的 PageAllocator。
- [ ] 验证测试：确保在 `Runtime` 销毁后，存量的 `Buffer` 仍能通过 `MemoryHandle.deleter` 正常释放。

## 8. 反模式 (Anti-Patterns to Avoid)

- ❌ **全局单例**：严禁在代码中直接使用全局分配器查找。
- ❌ **引用回溯**：Buffer 不得增加分配器的引用计数或持有其指针。
- ❌ **手动内存管理**：禁止在算子内部手动释放，必须由 `Buffer` 的引用计数触发。

