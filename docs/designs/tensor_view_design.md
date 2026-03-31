---
title: TensorView / MutableTensorView 设计文档
status: draft
version: v0.1
date: 2026-03-31
author: Sisyphus
source_documents:
  - docs/designs/operator_contract_design.md
  - docs/designs/phase1_m1_execution_checklist.md
---

# TensorView / MutableTensorView 设计文档

**状态**: Draft  
**版本**: v0.1  
**日期**: 2026-03-31  
**作者**: Sisyphus

---

## 1. 设计依据

### 1.1 来源文档

本设计基于以下文档的最小契约要求：

- `docs/designs/operator_contract_design.md` §3.2 最小元信息
- `docs/designs/phase1_m1_execution_checklist.md` §3.3 Base Contract

### 1.2 Phase 1 目标定位

Phase 1 Operator Contract 面向：

- 桌面/服务器 CPU 本地推理
- decoder-only Transformer
- 单请求同步执行
- 静态 KV Cache
- Reference-first, Optimized-second

**非目标**：通用深度学习张量框架、图执行引擎。

---

## 2. 最小元信息要求

每个 `TensorView` / `MutableTensorView` 至少应包含（来自 operator_contract_design.md §3.2）：

| 元信息 | 类型 | 说明 |
|--------|------|------|
| `data pointer` | `void*` / `const void*` | 原始数据指针 |
| `dtype` | `DataType` | 元素类型 |
| `rank` | `int32_t` | 维度数（可从 shape 推导） |
| `shape` | `IntArrayView` | 对外暴露为 view；内部元数据应安全持有 |
| `stride` | `IntArrayView` | 对外暴露为 view；内部元数据应安全持有 |
| `alignment` | `std::optional<size_t>` | 已知的最小数据对齐（字节）；未知时为空 |
| `mutability` | 类型区分 | `TensorView` vs `MutableTensorView` |

---

## 3. Ownership 模式

### 3.1 设计原则

来自 operator_contract_design.md §3.3：

- Execution / Operator 热路径处理 **borrowed view**
- 持久态数据由 Loaded Model / KV Cache / Workspace 持有
- 算子本身不拥有输入输出 buffer 的生命周期

### 3.2 实现方式

- 直接持有 `data_ptr + dtype + shape/stride metadata + alignment hint` 元数据
- **不依赖** `ObjectPtr<TensorImpl>`（零分配友好）
- `data` 生命周期仍然是 borrowed；但 `shape/stride` 元数据不应长期借用外部临时对象
- 推荐内部使用 `ShapeAndStride` 或等价的小对象元数据容器保存 shape/stride，再通过 `IntArrayView` 对外暴露
- 在 Phase 1 常见 rank（2~4）下，shape/stride 元数据复制应优先走 inline storage，避免堆分配

### 3.3 与现有 Tensor 的关系

| 特性 | 现有 Tensor | 新增 TensorView |
|------|-------------|-----------------|
| **所有权** | owning (`ObjectPtr<TensorImpl>`) | non-owning borrowed view |
| **数据指针** | `void*` / typed | `const void*` / `void*` |
| **dtype** | `DataType` | `DataType` |
| **shape/stride** | `IntArrayView`（由 `TensorImpl` 内部元数据支撑） | 对外返回 `IntArrayView`，但 view 内部安全持有元数据 |
| **alignment** | 未显式定义 | 可选的已知最小对齐提示 |
| **mutability** | const handle 可写 data | 类型区分（TensorView vs MutableTensorView） |
| **热路径** | 构造有分配开销 | 零分配友好 |

---

## 4. 参考 ArrayView 模式

### 4.1 现有 ArrayView 设计

`include/container/array_view.h` 特点：

- 构造：`const T* + size_t`
- 接口：`data()`, `size()`, `begin()/end()`, `slice()`, `at()`
- 边界检查：`AM_CHECK`
- 无所有权，可从 `std::vector`, `std::array`, C 数组构造

### 4.2 TensorView 需要的扩展

- 多维：shape/stride 两个数组
- dtype：携带元素类型信息
- alignment：携带对齐约束
- typed access：模板方法访问 typed 数据

---

## 5. 设计决策

### 5.1 类结构：两个独立类

**决策**: 采用两个独立类（而非单一模板特化）

**理由**:
- 符合现有代码风格（ArrayView 只有 const 版本）
- 接口更清晰，避免模板复杂性
- MutableTensorView 的 `data()` 返回 `void*`，TensorView 返回 `const void*`

```cpp
class TensorView;       // const void* data
class MutableTensorView; // void* data
```

### 5.2 shape/stride 元数据：内部持有，不长期借用

**决策**: `TensorView` / `MutableTensorView` 对外仍暴露 `IntArrayView`，但内部持有 shape/stride 元数据

**理由**:
- `ArrayView` 明确不适合作为长期存储字段；直接保存借用 view 容易悬垂
- `slice()` / `KVCacheView` 这类按值返回新 view 的 API 需要派生后的 shape/stride 元数据
- 现有仓库已有 `ShapeAndStride`，且对小 rank 做了 inline 优化，适合作为内部元数据承载

```cpp
class TensorView {
private:
    const void* data_{nullptr};
    DataType dtype_{};
    ShapeAndStride shape_and_stride_{};
    std::optional<size_t> known_data_alignment_{};
};
```

### 5.3 alignment：可选提示，不设静默默认值

**决策**: alignment 不设静默默认值；仅在调用方或上层 contract 已知时显式提供

**理由**:
- 算子 contract 会声明对齐要求，但并不意味着每个 view 都天然满足该要求
- `TensorImpl::storage_offset` 会改变逻辑 `data()` 指针，导致实际可用对齐低于底层 storage 对齐
- `alignment` 更适合作为“已知的最小对齐提示”，未知时为空，而不是统一默认 16

```cpp
TensorView(data, dtype, shape, strides);                                 // alignment unknown
TensorView(data, dtype, shape, strides, /*known_alignment=*/32);         // 显式声明
std::optional<size_t> known_data_alignment() const noexcept;
```

### 5.4 rank 字段：不单独存储

**决策**: rank 不作为独立字段存储

**理由**:
- rank 可从 `shape.size()` 计算
- 避免冗余存储
- `rank()` 方法返回 `shape().size()`

```cpp
int32_t rank() const noexcept { return static_cast<int32_t>(shape().size()); }
```

### 5.5 Tensor → TensorView 转换：显式构造

**决策**: 提供显式构造函数，不支持隐式转换

**理由**:
- 避免隐式转换带来的生命周期风险
- TensorView 借用 Tensor 数据，需显式表达意图
- 用户明确知道正在创建视图

```cpp
Tensor t = ...;
TensorView view(t);   // ✅ 显式构造
TensorView view = t;  // ❌ 隐式转换（不允许）
```

### 5.6 状态语义：`defined()` 不等于 `data() != nullptr`

**决策**: `defined()` 表示 view 是否绑定了合法的逻辑对象；空 view 允许 `data() == nullptr` 但仍为已定义状态

**理由**:
- 现有 `TensorImpl::data()` 在空 tensor 上可能返回 `nullptr`
- 如果把 `defined()` 绑定到 `data() != nullptr`，零长度 slice 或空 tensor 派生 view 会被误判为 undefined

```cpp
TensorView undefined;                      // 默认构造：undefined
TensorView empty_view = MakeEmptyView(...);// 已定义，但 numel()==0 且 data()==nullptr 可接受

bool defined() const noexcept;
bool empty() const noexcept;
```

---

## 6. 接口规范

### 6.1 TensorView

```cpp
// include/aethermind/base/tensor_view.h

namespace aethermind {

class TensorView {
public:
    // === 构造 ===
    
    /// 默认构造：undefined view
    TensorView() noexcept;
    
    /// 从原始指针构造（完整参数）
    TensorView(const void* data,
               DataType dtype,
               IntArrayView shape,
               IntArrayView strides,
               std::optional<size_t> known_alignment = std::nullopt);
    
    /// 从 Tensor 显式构造（借用其数据）
    explicit TensorView(const Tensor& tensor);
    
    // === 元信息访问 ===
    
    /// 数据指针（const）
    const void* data() const noexcept;
    
    /// 元素类型
    DataType dtype() const noexcept;
    
    /// 形状（借用视图）
    IntArrayView shape() const noexcept;
    
    /// 步长（借用视图）
    IntArrayView strides() const noexcept;
    
    /// 已知的最小数据对齐（字节）；未知时为空
    std::optional<size_t> known_data_alignment() const noexcept;
    
    /// 维度数（= shape.size()）
    int32_t rank() const noexcept;
    
    /// 元素总数
    int64_t numel() const noexcept;
    
    /// 单元素大小（字节）
    size_t itemsize() const noexcept;
    
    /// 逻辑元素字节数 = numel * itemsize
    /// 注意：这不是 backing storage 的 addressable span，也不等于物理占用大小
    size_t logical_nbytes() const noexcept;
    
    // === 状态查询 ===
    
    /// 是否为空（numel == 0）
    bool empty() const noexcept;
    
    /// 是否连续布局
    bool is_contiguous() const noexcept;
    
    /// 是否已绑定逻辑 view；不等同于 data() 是否非空
    bool defined() const noexcept;
    
    // === Typed 访问 ===
    
    /// 获取 typed 数据指针
    /// 要求：dtype.Match<T>()
    template<typename T>
    const T* typed_data() const;
    
    /// 按 indices 访问元素
    /// 要求：indices.size() == rank()
    template<typename T>
    const T& at(std::initializer_list<int64_t> indices) const;
    
    /// 按 indices 数组访问元素
    template<typename T>
    const T& at(IntArrayView indices) const;
    
    // === Slice / View 操作 ===
    
    /// 在指定维度切片
    /// @param dim 维度索引
    /// @param start 起始位置
    /// @param end 结束位置（不含）
    /// 保持 rank 不变，仅调整目标维度长度；data 按 start * stride[dim] * itemsize 前移
    TensorView slice(int64_t dim, int64_t start, int64_t end) const;
    
private:
    const void* data_;
    DataType dtype_;
    ShapeAndStride shape_and_stride_;
    std::optional<size_t> known_data_alignment_;
    bool defined_{false};
};

}  // namespace aethermind
```

### 6.2 MutableTensorView

```cpp
class MutableTensorView {
public:
    // === 构造 ===
    
    /// 默认构造：undefined view
    MutableTensorView() noexcept;
    
    /// 从原始指针构造（完整参数）
    MutableTensorView(void* data,
                      DataType dtype,
                      IntArrayView shape,
                      IntArrayView strides,
                      std::optional<size_t> known_alignment = std::nullopt);
    
    /// 从 Tensor 显式构造（可写）
    explicit MutableTensorView(Tensor& tensor);
    
    // === 元信息访问（同 TensorView） ===
    
    void* data() const noexcept;  // 非 const
    DataType dtype() const noexcept;
    IntArrayView shape() const noexcept;
    IntArrayView strides() const noexcept;
    std::optional<size_t> known_data_alignment() const noexcept;
    int32_t rank() const noexcept;
    int64_t numel() const noexcept;
    size_t itemsize() const noexcept;
    size_t logical_nbytes() const noexcept;
    
    // === 状态查询（同 TensorView） ===
    
    bool empty() const noexcept;
    bool is_contiguous() const noexcept;
    bool defined() const noexcept;
    
    // === Typed 访问（可写） ===
    
    template<typename T>
    T* typed_data() const;
    
    template<typename T>
    T& at(std::initializer_list<int64_t> indices) const;
    
    template<typename T>
    T& at(IntArrayView indices) const;
    
    // === Slice / View 操作 ===
    
    MutableTensorView slice(int64_t dim, int64_t start, int64_t end) const;
    
private:
    void* data_;
    DataType dtype_;
    ShapeAndStride shape_and_stride_;
    std::optional<size_t> known_data_alignment_;
    bool defined_{false};
};
```

---

## 7. 验证约束

### 7.1 热路径零分配

- 构造 TensorView/MutableTensorView 不应因 data ownership 触发堆分配
- shape/stride 元数据可复制到内部小对象存储；Phase 1 常见小 rank 应优先走 inline 路径
- 不依赖 ObjectPtr 引用计数

### 7.2 边界检查

- `at()` 使用 `AM_CHECK` 检查 indices 范围
- `slice()` 检查 dim/start/end 合法性
- `shape.size() == strides.size()` 为强不变量
- `typed_data<T>()` 检查 dtype.Match<T>()

### 7.3 类型安全

- `typed_data<T>()` 在 dtype 不匹配时抛出/报错
- 编译期无警告（避免隐式类型转换）

---

## 8. 使用示例

### 8.1 从 Tensor 构造视图

```cpp
Tensor weight = model.get_weight("layer.0.weight");
TensorView weight_view(weight);  // 借用 weight 数据

// 传给算子
Status status = Linear(input_view, weight_view, output_view, op_ctx);
```

### 8.2 从原始指针构造

```cpp
// workspace 分配的临时 buffer
void* scratch = workspace->allocate(1024 * 1024, 32);
std::array<int64_t, 2> scratch_shape{1024, 256};
std::array<int64_t, 2> scratch_strides{256, 1};

MutableTensorView scratch_view(
    scratch,
    DataType::Float32(),
    IntArrayView(scratch_shape),
    IntArrayView(scratch_strides),
    32  // known alignment
);
```

### 8.3 Slice 操作

```cpp
TensorView input = ...;  // shape: [seq_len, hidden_dim]
TensorView token_view = input.slice(0, 5, 6);  // 第 5 个 token
// token_view.shape(): [1, hidden_dim]
// token_view.strides(): 与 input 相同
// token_view.data(): input.data() + 5 * input.strides()[0] * input.itemsize()
```

---

## 9. 风险与注意事项

### 9.1 生命周期风险

TensorView 借用外部数据，不管理生命周期：

```cpp
TensorView create_bad_view() {
    Tensor temp = Tensor::rand({100, 100});
    return TensorView(temp);  // ❌ temp 离开作用域后被销毁
}
```

**建议**: TensorView 应仅用于局部传递，不存储。

### 9.2 shape/stride 来源

对外接口虽然返回 `IntArrayView`，但 view 内部应拥有稳定的 shape/stride 元数据，避免以下悬垂风险：

```cpp
std::vector<int64_t> shape = {100, 100};
std::vector<int64_t> strides = {100, 1};
TensorView view(data, dtype, IntArrayView(shape), IntArrayView(strides));
// ✅ 如果 TensorView 在构造时复制/持有元数据，则调用后不再依赖外部容器生命周期
```

**禁止模式**：

```cpp
MutableTensorView bad(
    data,
    dtype,
    IntArrayView({1024, 256}),
    IntArrayView({256, 1})
);
// ❌ 若 TensorView 仅保存借用 IntArrayView，该构造会立即悬垂
```

### 9.3 alignment 语义

- `known_data_alignment()` 表示当前 `data()` 指针已知满足的最小对齐，而不是期望对齐
- 算子要求的 alignment 仍应在 operator/layout contract 中声明和验证
- 对从 `Tensor` 派生的 view，需要基于逻辑 `data()` 指针而不是底层 storage base 评估可用对齐

### 9.4 `logical_nbytes()` 语义

- `logical_nbytes() = numel() * itemsize()`
- 对 strided / padded / non-contiguous view，`logical_nbytes()` 不等于 backing storage 的物理跨度
- 如后续确有需要，可再补充 `addressable_span_bytes()` 一类更精确的 API，但不在 M1 冻结

---

## 10. 待办事项

M1 阶段需要完成：

- [ ] 实现 `include/aethermind/base/tensor_view.h`
- [ ] 实现 `src/aethermind/base/tensor_view.cpp`（如有需要）
- [ ] 创建 `tests/unit/test_tensor_view.cpp`
- [ ] 补充默认/空 view、non-contiguous、slice、dtype mismatch、mutable write-through 测试
- [ ] 验证小 rank 元数据 inline 路径不引入额外堆分配
- [ ] 与 Tensor 的转换接口测试

---

## 11. 相关文档

- `docs/designs/operator_contract_design.md` — Operator Contract 契约
- `docs/designs/phase1_m1_execution_checklist.md` — M1 执行清单
- `include/container/array_view.h` — ArrayView 参考实现
- `include/tensor.h` — Tensor owning handle
- `include/tensor_impl.h` — TensorImpl entity

---

**文档所有者**: AetherMind 架构团队  
**下次更新**: 实现 TensorView 后补充实际接口细节
