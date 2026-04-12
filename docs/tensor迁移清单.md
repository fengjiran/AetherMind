# Storage-Tensor → Buffer-Tensor 迁移清单
## 目标
- [ ] 统一内存拥有模型到 `Allocator + Buffer + MemoryHandle`
- [ ] 统一张量模型到 `Tensor`
- [ ] 渐进淘汰 `Storage / DataPtr / AllocatorBK / TensorImpl / Tensor_BK`
- [ ] 保证迁移期间始终可编译、可测试、可回滚
---
## Batch 1：建立兼容层
### 新增文件
- [x] `include/aethermind/migration/tensor_compat.h`
  - [x] 定义 `Tensor TensorFromLegacy(const Tensor_BK&)`
  - [x] 定义 `Tensor_BK LegacyTensorFromTensor(const Tensor&)`
  - [x] 定义 `Buffer BufferFromLegacyStorage(const Storage&)`
  - [x] 文档化 `storage_offset`（元素）→ `byte_offset`（字节）换算规则
  - [x] 文档化失败前提和兼容边界
- [x] `src/migration/tensor_compat.cpp`
  - [x] 实现新旧 Tensor 双向转换
  - [x] 实现 Storage → Buffer 转换
  - [x] 对 offset/itemsize 做显式校验
  - [x] 对未初始化对象做失败处理
  - [x] 确保 deleter/context 生命周期正确
### 既有文件
- [x] `include/tensor_bk.h`
  - [x] 仅补最小 compat 所需声明
  - [x] 标记 legacy 身份
  - [x] 不新增业务能力
- [x] `include/memory/storage_impl.h`
  - [x] 补最小只读桥接访问能力
  - [x] 标记 migration-only / legacy
  - [x] 不扩展旧 API 面
- [x] `include/aethermind/base/tensor.h`
  - [x] 如 compat 所需，补最小构造支持（已满足，无需额外改动）
  - [x] 保持不反向依赖 legacy 头
### 验收
- [ ] compat 层独立可编译
- [ ] 新旧 Tensor 能互转
- [ ] 无循环 include
- [ ] 生命周期语义清晰
---
## Batch 2：外围双接口
### Type System
- [x] `include/type_system/tensor_type.h`
  - [x] 增加 `Create(const Tensor&)`
  - [x] 增加 `MatchTensor(const Tensor&)`
  - [x] 保留 `Tensor_BK` 版本接口
  - [x] 注释说明新接口是主方向
- [x] `src/type_system/tensor_type.cpp`
  - [x] 实现 `Tensor` 版本 metadata 抽取
  - [x] 对齐新旧逻辑的 dtype/shape/device 语义
  - [x] 避免把 offset 迁移逻辑散落到这里
### Any
- [x] `include/any.h`
  - [x] 保留 `Tensor_BK ToTensor()`
  - [x] 增加 `Tensor ToNewTensor()`（或确定的新命名）
- [x] `src/any.cpp`
  - [x] 实现 `Tensor` 装箱/拆箱
  - [x] 统一通过 compat/type-system 层做必要分发
  - [x] 校验 Any round-trip 行为
### Format / Traits / Utils
- [x] `src/format.cpp`
  - [x] 增加 `Tensor` 格式化支持
  - [x] 保留 `Tensor_BK` 格式化
  - [x] 尽量抽公共逻辑
- [x] `include/type_traits.h`
  - [x] 增加/调整 `Tensor` 相关 trait（已核对，Batch 2 无需额外改动）
  - [x] 暂不删除 legacy trait
- [x] `include/any_utils.h`
  - [x] 增加 `Tensor` 类型名映射
  - [x] 保留 `Tensor_BK` 类型名映射
### 验收
- [x] type system 同时支持新旧 Tensor
- [x] Any 同时支持新旧 Tensor
- [x] format 支持新旧 Tensor
- [x] 无重载歧义
---
## Batch 3：测试基础设施迁移
### 工厂与随机张量
- [x] `tests/unit/test_utils/tensor_factory.h`
  - [x] 确认为新 Tensor 官方测试构造入口
  - [x] 检查默认构造全部走 `Buffer`
  - [x] 覆盖 shape/stride/dtype/device/byte_offset
  - [x] 新增 Tensor_BK 兼容层（MakeEmptyTensorBK / MakeContiguousTensorBK）
- [x] `tests/unit/test_utils/tensor_random.h`
  - [x] 新增 `RandomUniformTensor(...) -> Tensor`
  - [x] 新增 `RandomNormalTensor(...) -> Tensor`
  - [x] 新增 `RandomIntTensor(...) -> Tensor`
  - [x] 保留 legacy 版本一段时间
  - [x] 确保内部默认走新 Tensor 路径（使用 MakeContiguousTensor + mutable_data）
### 断言
- [x] `tests/unit/test_utils/tensor_assert.h`
  - [x] 增加 `ExpectTensorAllClose(const Tensor&, const Tensor&, ...)`
  - [x] 增加 `ExpectTensorEqual(const Tensor&, const Tensor&, ...)`
  - [x] Tensor_BK 版本完全保留
### 验收
- [x] 新测试工具默认服务 `Tensor`
- [x] 旧测试仍可继续运行
- [x] 新旧工具不冲突
---
## Batch 4：迁移测试用例
### 核心测试
- [ ] `tests/unit/test_tensor.cpp`
  - [ ] 主路径改用 `Tensor`
  - [ ] 覆盖 initialized
  - [ ] 覆盖 shape/stride
  - [ ] 覆盖 byte_offset
  - [ ] 覆盖 data/mutable_data
  - [ ] 覆盖 slice
  - [ ] 覆盖 contiguous
  - [ ] 覆盖 alignment
  - [ ] 覆盖 range validity
- [x] `tests/unit/test_tensor_random.cpp`
  - [x] 改用新 `tensor_random.h`
  - [x] 校验 shape/dtype/device/数值分布
  - [x] 新增 Tensor 版本的随机测试（TensorRandomNew.*）
  - [x] 新增 Tensor 版本的断言测试（TensorAllCloseNew.*）
- [x] `tests/unit/test_any.cpp`
  - [x] 增加 `Tensor` 装箱/拆箱测试
  - [x] 保留 `Tensor_BK` 兼容测试
  - [x] 增加 compat round-trip 测试
### 旧存储测试处理
- [x] `tests/unit/test_storage.cpp`
  - [x] 区分 legacy-only 测试与可迁移测试
  - [x] 可迁移内容迁到 Buffer 测试
  - [x] 保留内容显式标记 legacy（文件头注释说明）
### 建议新增
- [x] `tests/unit/test_buffer.cpp`
  - [x] 测试 Buffer 构造
  - [x] 测试 nbytes/device/alignment
  - [x] 测试 shared ownership
  - [x] 测试 deleter 调用
- [x] `tests/unit/test_tensor_compat.cpp`
  - [x] 测试 `Tensor_BK -> Tensor`
  - [x] 测试 `Tensor -> Tensor_BK`
  - [x] 测试 offset 换算
  - [x] 测试非法条件失败路径
### 验收
- [x] 新 Tensor 测试可独立运行
- [x] compat 测试覆盖关键桥接语义
- [x] legacy 测试仅保留过渡价值
---
## Batch 5：切换分配主路径
### Allocator
- [x] `include/aethermind/memory/allocator.h`
  - [x] 标记 `AllocatorBK` deprecated
  - [x] 标记 `AllocatorTable` deprecated
  - [x] 注释强调 `Allocator::Allocate()` 为主入口
- [x] `include/aethermind/memory/cpu_allocator.h`
- [x] `src/memory/cpu_allocator.cpp`
  - [x] 确认主路径返回 `Buffer`（CPUAllocator::Allocate 已返回 Buffer）
  - [x] 检查 `MemoryHandle` deleter/context/alignment 正确（cpu_buffer_deleter + alignment）
  - [x] legacy BK 路径仅保留兼容（CPUAllocatorBK marked deprecated）
### Runtime
- [x] `include/aethermind/runtime/runtime_context.h`
- [x] `src/runtime/runtime_context.cpp`
- [x] `include/aethermind/runtime/runtime_builder.h`
- [x] `src/runtime/runtime_builder.cpp`
  - [x] 确保 runtime 主路径围绕新 `Allocator`（RuntimeContext 只使用 AllocatorRegistry）
  - [x] 避免 runtime 扩散 legacy allocator 语义（grep 无 legacy allocator 使用）
  - [x] 如有 tensor 创建辅助，统一走 Buffer（runtime 无 tensor 创建辅助）
### 验收
- [x] 新建 Tensor/Buffer 默认走新分配路径（tensor_factory.h 使用 Buffer）
- [x] 无新增 `AllocatorBK/DataPtr/Storage` 用法（deprecated 警告会阻止新用法）
- [x] runtime 不再是 legacy 扩散源（runtime 仅使用 Allocator/AllocatorRegistry）
---
## Batch 6：旧 Tensor 降级为 shim
### Legacy Tensor
- [x] `include/tensor_impl.h`
- [x] `src/tensor_impl.cpp`
  - [x] 停止新增功能（标记 deprecated）
  - [x] 尽量收敛到 compat / 新 Tensor（工厂建议通过 compat）
  - [x] 标记 legacy（文件头 + [[deprecated]]）
- [x] `include/tensor_bk.h`
- [x] `src/tensor_bk.cpp`
  - [x] 工厂逻辑尽量走新 Tensor/Buffer 再桥接（建议通过 tensor_compat）
  - [x] 标记 deprecated（类 + 构造函数）
  - [x] 降级为兼容壳
### Legacy Storage
- [x] `include/memory/storage_impl.h`
- [x] `src/memory/storage_impl.cpp`
  - [x] 只保留最小兼容行为
  - [x] 不再增加新特性（标记 deprecated）
  - [x] 为删除做准备（[[deprecated]] 标记）
### 验收
- [x] 主流程不再依赖 `Tensor_BK`（新 Tensor 测试覆盖）
- [x] 旧类型仅承担兼容职责（deprecated 警告生效）
- [x] 大部分核心调用切到 `Tensor`（compat 层提供桥接）
---
## Batch 7：删除旧体系
### 删除文件
- [ ] 删除 `include/memory/storage_impl.h`
- [ ] 删除 `src/memory/storage_impl.cpp`
- [ ] 删除 `include/aethermind/memory/data_ptr.h`
- [ ] 删除 `include/tensor_impl.h`
- [ ] 删除 `src/tensor_impl.cpp`
- [ ] 删除 `include/tensor_bk.h`
- [ ] 删除 `src/tensor_bk.cpp`
### 清理调用点
- [ ] `include/aethermind/memory/allocator.h`
  - [ ] 删除 `AllocatorBK`
  - [ ] 删除 `AllocatorTable`
- [ ] `include/type_system/tensor_type.h`
- [ ] `src/type_system/tensor_type.cpp`
  - [ ] 删除 `Tensor_BK` 重载
- [ ] `include/any.h`
- [ ] `src/any.cpp`
  - [ ] 删除 legacy Tensor 接口
- [ ] `src/format.cpp`
  - [ ] 删除 legacy Tensor 格式化分支
- [ ] `tests/unit/*`
  - [ ] 删除 legacy-only 测试
  - [ ] 清理 compat 过渡逻辑
- [ ] `docs/designs/*`
  - [ ] 更新为 Buffer-Tensor 最终架构
  - [ ] 标记迁移完成
### 最终验收
- [ ] grep 无生产级 `Storage/DataPtr/Tensor_BK/TensorImpl/AllocatorBK` 引用
- [ ] 最小相关目标可编译
- [ ] 单元测试通过
- [ ] 关键 benchmark 无明显退化
- [ ] 文档已同步
---
## 迁移期间通用检查项
- [ ] 不在业务代码中手写 offset 单位换算，统一走 compat
- [ ] 不把 legacy include 反向引入新 Tensor 主头
- [ ] 不为旧体系继续扩展新能力
- [ ] 所有删除动作必须在 grep 和测试验证后进行
- [ ] 每个 Batch 都保持可编译、可回滚
---
## 推荐优先处理文件（Top 12）
- [ ] `include/aethermind/migration/tensor_compat.h`
- [ ] `src/migration/tensor_compat.cpp`
- [ ] `include/type_system/tensor_type.h`
- [ ] `src/type_system/tensor_type.cpp`
- [ ] `include/any.h`
- [ ] `src/any.cpp`
- [ ] `src/format.cpp`
- [ ] `tests/unit/test_utils/tensor_factory.h`
- [ ] `tests/unit/test_utils/tensor_random.h`
- [ ] `tests/unit/test_utils/tensor_assert.h`
- [ ] `tests/unit/test_tensor.cpp`
- [ ] `tests/unit/test_any.cpp`
