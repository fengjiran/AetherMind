---
title: Object 实现审查结论与最小侵入修复清单
status: draft
version: v0.1
date: 2026-03-26
author: OpenCode / Sisyphus
source_files:
  - include/object.h
  - src/object.cpp
  - include/object_allocator.h
  - include/c_api.h
  - tests/unit/test_object.cpp
---

# Object 实现审查结论与最小侵入修复清单

**状态**: Draft  
**版本**: v0.1  
**日期**: 2026-03-26

---

## 1. 审查范围

本次审查聚焦 `Object/ObjectPtr/WeakObjectPtr` 的生命周期管理、并发引用计数与 C API 边界一致性：

- `include/object.h`
- `src/object.cpp`
- `include/object_allocator.h`
- `include/c_api.h`
- `tests/unit/test_object.cpp`

---

## 2. 结论摘要（Review Conclusion）

当前 intrusive refcount 核心架构可行，`make_object()` 路径设计完整且正确（先设置 deleter，再建立 weak 基线，再创建首个 strong owner）。

但在“原始指针接管 API”和“null sentinel 对外语义”上存在高风险边界问题，应优先收敛。

结论分级：

- **P0（优先处理）**：接管路径不变量破坏风险、`reclaim(nullptr)` 状态异常、`nullptr` 比较语义冲突
- **P1（高优先）**：`release/get` 的边界语义统一、危险构造路径可见性收窄
- **P2（可选优化）**：内存序微调、文档契约强化

---

## 3. 主要问题清单（按风险）

## 3.1 P0 - Correctness 风险

1. `ObjectPtr(std::unique_ptr<T>)` 走 raw 接管路径，与 `make_object()` 初始化不变量不一致  
   - 相关位置：`include/object.h`（`271`, `504-509`）

2. `ObjectPtr::reclaim(nullptr)` 可制造“逻辑 defined 但内部空指针”状态  
   - 相关位置：`include/object.h`（`466-468`, `395-397`, `376-381`）

3. `ObjectPtr == nullptr` 与 sentinel 设计冲突（空对象并非真实 `nullptr`）  
   - 相关位置：`include/object.h`（`255-262`, `737-745`）

## 3.2 P1 - API 健壮性风险

1. `release()` 对外暴露 sentinel 指针，不利于跨边界安全使用  
   - 相关位置：`include/object.h`（`454-458`）

2. `ObjectPtr(T*, DoNotIncRefCountTag)` 暴露过宽，外部可绕过不变量  
   - 相关位置：`include/object.h`（`276`）

## 3.3 P2 - 维护与性能项

1. `TryPromoteWeakPtr()` 成功 CAS 使用 `ACQ_REL`，可评估收敛为 `ACQUIRE`（微优化）  
   - 相关位置：`src/object.cpp`（`54-67`）

2. `ObjectHandle` 与 `ObjectHeader` 的文档边界需更明确（opaque handle）  
   - 相关位置：`include/c_api.h`（`14-40`）

---

## 4. 最小侵入修复清单（按文件与函数粒度）

以下清单遵循“先保正确性、尽量不改调用方 API”的原则。

## A. `include/object.h`

### A1. `ObjectPtr(std::unique_ptr<T>)`

- **改动**：限制接管来源（仅允许内部已初始化对象），或改为内部专用路径
- **目的**：避免绕过 `make_object()` 的 weak/deleter 初始化不变量
- **侵入性**：中

### A2. `ObjectPtr::reclaim(T*)`

- **改动**：对 `nullptr` 显式归一化为空对象；补 debug invariant 检查
- **目的**：避免构造“defined + 空指针”的非法状态
- **侵入性**：低

### A3. `ObjectPtr::get()` / `ObjectPtr::release()`

- **改动**：对外语义统一为空返回 `nullptr`（内部若需 sentinel，新增内部接口）
- **目的**：消除 C++/C 边界语义冲突，降低误用概率
- **侵入性**：中（需评估内部依赖点）

### A4. `operator==(nullptr)` / `operator!=(nullptr)`

- **改动**：改为基于 `defined()` 的语义判断
- **目的**：使 null 比较符合直觉与智能指针约定
- **侵入性**：低

### A5. `ObjectPtr(T*, DoNotIncRefCountTag)`

- **改动**：收窄可见性（internal/friend-only）
- **目的**：防止外部越过所有权不变量
- **侵入性**：中

## B. `src/object.cpp`

### B1. `Object::DecRef()` / `Object::DecWeakRef()`

- **改动**：增加 debug 断言，防止 refcount 下溢
- **目的**：尽早暴露错误接管路径
- **侵入性**：低

### B2. `Object::TryPromoteWeakPtr()`

- **改动**：评估成功序由 `ACQ_REL` 收敛到 `ACQUIRE`（在不破坏语义前提下）
- **目的**：小幅优化原子开销
- **侵入性**：低（可选）

## C. `include/c_api.h`

### C1. `ObjectHandle` 与 `ObjectHeader` 注释契约

- **改动**：明确 `ObjectHandle` 是 opaque handle，不应依赖内部布局
- **目的**：避免外部错误 reinterpret 造成 ABI 风险
- **侵入性**：低（文档层）

---

## 5. 建议实施顺序（最小风险）

1. **先做 P0**：A2 + A4 + A1
2. **再做边界收敛**：A3 + A5
3. **最后做增强项**：B1 + C1 +（可选）B2

---

## 6. 验证建议（每步改动后执行）

建议最小验证回路：

1. `lsp_diagnostics` 覆盖修改文件
2. 构建：`cmake --build build --target aethermind_unit_tests -j`
3. 测试：`./build/tests/unit/aethermind_unit_tests --gtest_filter=object.*:weak_object_ptr.*`

建议新增回归测试点：

- `ObjectPtr::reclaim(nullptr)` 行为
- `ObjectPtr == nullptr` 与 `defined()` 一致性
- `release()` 空对象返回值语义
- 非 `make_object` 接管路径的 debug invariant 校验

---

## 7. 备注

本清单刻意避免替换 intrusive refcount 架构，优先修复 API 边界风险并保持兼容性。

在正确性收敛后，再依据性能数据决定是否进行更深层优化（例如原子序、header 布局微调）。
