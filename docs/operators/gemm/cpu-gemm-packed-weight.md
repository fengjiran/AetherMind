# CPU GEMM Packed Weight 提案

- **状态**: Proposed
- **版本**: 1.2
- **日期**: 2026-09-22
- **文档定位**: `exact recipe` 与 `packed-B` 阶段（[CPU GEMM 优化方案](cpu-gemm-optimization.md) §7）的具体化设计；只含方案内容，状态与机器级证据不在此维护。
- **产品边界**: [AetherMind 当前产品 PRD](../../products/aethermind_prd.md)
- **工作流规范**: [算子开发与优化工作流](../../guides/operator-development-workflow.md)
- **架构基线**: [架构总览](../../designs/architecture/architecture_overview.md)
- **关联代码**: `src/backend/cpu/cpu_weight_prepacker.cpp`、`src/backend/cpu/kernels/gemm/`、`src/backend/cpu/kernels/common/packed_weight_utils.{h,cpp}`（packed recipe 校验闸口）、`src/compiler/packing_request_builder.{h,cpp}`（生产 packing request 来源）、`include/aethermind/backend/resolved_kernel.h`（`expected_packing_recipe` 已存在）、`include/aethermind/backend/packed_weights.h`、`include/aethermind/model/{weight_prepack_planner,packed_weight_store}.h`
- **关联测试**: `tests/unit/backend/cpu/kernels/`、`tests/unit/model/test_weight_prepack_planner.cpp`、`tests/benchmark/cpu_kernels/`
- **关联 ADR**: 无（exact recipe 合同落地时新建）
- **关联模块**: backend / execution / compiler / model / benchmark

## 1. 结论

把 GEMM 的 B 打包从**每次 Execute 前的运行时成本**迁移为 **model preparation 期一次性成本**（immutable packed weight）。落地后：

- 运行时 B 面板重复打包（每个 `mb` 块重打包一次 `(kb, nb)` 面板）消失，prefill 的 B 面板直接从 artifact 顺序流式消费；
- steady-state 热路径不再执行 B 侧转置打包；A 侧打包与 96KB 栈缓冲保留（packed-B 可省去 32KB `buf_b`）；
- 为后续量化 recipe（`量化与新 ISA` 阶段）铺好合同基座。

打包与消费两侧的合同骨架大多已存在（`PackingRecipe`、`PackedWeights`、`PackedWeightStore`、`PackedWeightView`、`ResolvedKernel::expected_packing_recipe`）。本方案补四个缺口：① recipe 表达与校验闸口；② recipe 传递链（descriptor → request → artifact）；③ 按 recipe 打包的服务体；④ packed-B 消费 driver。**生产编排点（谁在 model preparation 调用 packing request 生成与执行）当前不存在**，属前置缺失，见 §5 里程碑 M4。

## 2. 现状与缺口

| 环节 | 现状 | 位置 |
|---|---|---|
| recipe 合同 | `PackingRecipe{layout, alignment}`，无 tile 字段 | `include/aethermind/backend/packed_weights.h` |
| 打包服务 | `CpuWeightPrepacker::Pack/RecipeFor`，唯一 recipe 为 `cpu_identity` 对齐拷贝；`Pack` 内部自己调 `RecipeFor(selector)` | `src/backend/cpu/cpu_weight_prepacker.cpp` |
| 打包 request（生产） | `BuildWeightPackingRequests(lowered, resolved)`：纯数据映射，填 components/op_type/source_id，**never touches a backend**，不携带 recipe | `include/aethermind/compiler/packing_request_builder.h` |
| 打包 request（legacy） | `WeightPrepackPlanner::BuildRequests`：头注明 "ModelLoader does not call this legacy planner"；实现 `UNUSED(registry)`；把 q/k/v/gate/up 全发成 `OpType::kLinear` + per-role binding、从不填 components | `include/aethermind/model/weight_prepack_planner.h`、`src/model/weight_prepack_planner.cpp` |
| 打包执行 | `PrepackAndStore(store, requests)` 无 Backend 参数；key 的 recipe 由它自己调 `RecipeFor(req.selector)` 得出；model 层直接实例化 `CpuWeightPrepacker`（母提案 §4.5 第 4 点要消除的偏差） | `src/model/weight_prepack_planner.cpp` |
| 存储/定位 | `PackedWeightStore` + `WeightArtifactKey{source_id, value_index, binding, selector, recipe}`；plan 组装用 `Find(exact_key)`，冻结期校验 `artifact->recipe() != expected_packing_recipe` 即失败 | `include/aethermind/model/packed_weight_store.h`、`src/execution/execution_plan_builder.cpp` |
| resolve 期 recipe | `ResolvedKernel::expected_packing_recipe` 已在 resolve 期由 backend 填充；resolve 期**没有 shape**（`LinearParams {}` 为空，`QkvLinearParams` 只有 q/k/v out_features） | `include/aethermind/backend/resolved_kernel.h`、`include/aethermind/operators/op_params.h` |
| binding 期消费 | `KernelParamsBuildContext::packed_weight`（opaque `PackedWeightView`：data/nbytes/logical dtype+shape/recipe_layout/alignment）；**无 tile 字段** | `include/aethermind/backend/kernel_types.h` |
| packed 校验闸口 | `ValidateIdentityPackedWeight` 硬编码只接受 `cpu_identity`，是所有 packed 消费者的共同闸口 | `src/backend/cpu/kernels/common/packed_weight_utils.cpp` |
| execute 期消费 | `KernelContext::packed_weights` 指针 | `include/aethermind/backend/kernel_context.h` |
| packed 消费者样板 | QkvLinear/GateUpLinear 已走 packed（identity）链路；三段重量指针来自 identity 大 artifact 的行偏移切分 | `src/backend/cpu/kernels/qkv_linear/qkv_linear_entry.cpp` |
| 打包循环原型 | `PackBPanel`（当前在运行时执行，kc_len clamp 传入） | `src/backend/cpu/kernels/gemm/gemm_f32_avx2.cpp` |

缺口：

1. **tile 化 recipe 表达与消费侧闸口**：`PackingRecipe` 只有 layout 名与 alignment；`PackedWeightView` 无 tile 字段；`packed_weight_utils` 只认 identity。
2. **recipe 传递链**：生产 request（`BuildWeightPackingRequests`）不携带 recipe，且按模块边界（AGENTS.md：compiler 不得依赖 backend/runtime、不得查询 kernel registry）不可能在 compiler 内查 descriptor；descriptor 的 recipe 无法到达 request，`PrepackAndStore` 只能自己 `RecipeFor(selector)`。
3. **按 recipe 打包的服务体**：`CpuWeightPrepacker` 只有 identity copy。
4. **packed-B 消费 driver**：plain `GemmF32Args` 无法表达 packed 布局；现有 microkernel 的 M/N 尾处理依赖 plain B 的 row-pair 路径，packed-only 下不存在。

## 3. 详细设计

### 3.1 recipe 表达（缺口 1 之一）

`PackingRecipe` **不扩展、不增加 shape 派生字段**。理由（硬约束）：resolve 期无 shape，kernel 侧永远算不出 `k_panels/k_pad` 之类字段；artifact 侧一旦携带这些字段，`WeightArtifactKey::operator==` 必然不等 → 所有 packed step 在 plan 组装期 `Find` 失败。而这类字段正是 logical N/K 与 tile 常量的纯函数，与"recipe 不重复携带 logical 尺寸、避免双源漂移"的原则也冲突。

立约：

- recipe 保持 `{layout, alignment}`；layout 名为 `cpu_bpanel_f32_v1_avx2` 形式，版本编入布局名，布局变更必须改名。
- `nr`/`kc` 是**layout 常量**，由 layout 名唯一确定（常量头 `cpu_bpanel_packing.h`，仿 `identity_packing.h` 模式），不进 recipe struct、不随 shape 自适应。
- panel 数与 pad 由 packer 与 consumer **各自**从 `PackedWeights::logical_shape()` + layout 常量派生；交叉校验用不变式：

```text
packed_nbytes == k_panels * n_blocks * kc * nr * 4
k_panels = ceil(K / kc), n_blocks = ceil(N / nr)   （K/N 取自 logical_shape）
```

即 nbytes 目标是**校验项而非 recipe 字段**，两侧独立推导、以等式互证，recipe 相等性合同不受 shape 影响。

### 3.2 物理布局（v1, AVX2）

与现有 `PackBPanel` / `MicroKernel4x16` **块内字节布局**逐字节一致；块间遍历顺序不同——运行时 blocked 路径是 `(kb, mb, nb)`（B 每 `mb` 块重打包），artifact 是 `(k_panels, n_blocks)` 顺序流。仅把打包时机从运行时移到准备期：

```text
B 逻辑 [N, K]（权重 [out, in]）
  → 按 K 面板切 k_panels 段（每段 kc 行，尾段 pad 0）
     每个 K 面板内：n_blocks 个转置块依次排列
       块 (pk, nb) = buf[kk][col], kk∈[0,kc), col∈[0,nr)   // 即 buf_b[kk*nr+col]
```

- 每个面板块起始 64B 对齐，块大小 `kc*nr*4`。
- 放大率公式（pad 贷方）：`ceil(N/nr)*nr/N × ceil(K/kc)*kc/K`。逐项数值表见 §4 第 1 条；**K 侧对非整除 K 是固定成本**（kc 是 layout 常量）。
- 尾面板 pad 0 的作用只有两点：统一块 stride（消费侧 B 读永不越 artifact 边界）、消去消费端的 K 尾分支前提。它**不**消除输出侧 tail 处理（见 §3.3）。

### 3.3 tail 处理（修正稿）

pad-0 满 tile **不等于**"kernel 永不做 tail 分支"。三类 tail 分别立约：

- **K 尾（A 侧读）**：K 循环按 binding 期计算的 `kc_len = min(kc, K - pk*kc)` clamp——每面板一次常量、非 per-element 分支。不得读满 kc 而越过该面板的 K 边界（末面板满宽读取会越出该行 `lhs` 尾部，末行会越出 tensor）。
- **N 尾（输出列）**：`MicroKernel4x16` 无条件写 16 列，`output_m_stride == n` 时写满会污染下一行（末行越界写出）。处理三选一（driver 内部实现选择）：masked/partial column store 写末块；NR<16 列变体（窄块）；N 尾交由 compatible packed driver（见 §3.6）。
- **M 尾（输出行）**：`m % kBlockedMR != 0` 的底部行同样不能用满 4 行 microkernel，复用 §3.4 scan driver 的 MR∈{1..8} 广播变体（MR=余数行数）或交 compatible driver。

今天 blocked 路径的 M/N 尾复用 row-pair 消费 plain B（`args.rhs + n_full*rhs_n_stride`），packed-only artifact 下该回退路径不存在——**这是 packed-B 相对 plain 路径的合同收窄，必须在 §3.6 逐条列出**，不能依赖"顺手复用"。

### 3.4 打包与消费 primitive（缺口 3、4）

**打包侧**：`CpuWeightPrepacker` 演进为 recipe→packer 注册表。`Pack` 签名增加 `PackingRecipe` 参数（不再内部 `RecipeFor`）；`RecipeFor(selector)` 保持返回 identity（现有 packed 链路零扰动）。新增 `BpanelPackerV1`，复用 `PackBPanel` 逻辑 + pad 处理。

**消费侧**：新建 backend-private `PackedGemmF32Args`（POD：packed 指针 + 切片描述 + logical N/K + tile 常量与 k_panels/n_blocks，**不得整体嵌入 `PackingRecipe`**——params 受 arena 硬约束，见 §4 第 3 条）与三个 driver：

| Driver | 形状 | 行为 |
|---|---|---|
| `RunGemmF32PackedBScan` | M 小（decode/skinny） | `MicroKernel4x16` 的 MR∈{1..8} 广播变体：每个 `kk` 取该行 A 标量广播、加载 16 宽连续 B 向量、FMA 进 16 个累加器（沿 N 向量化、**无水平归约**）；M=1 即 MR=1。不做"沿 K 向量化 + 归纳"形态——bpanel 下同列相邻 K 元素间隔 `nr*4=64B`，K 向量化会丢掉 B 读顺序化的全部收益 |
| `RunGemmF32PackedBBlocked` | M 大（prefill） | 现有 `RunBlockedGemmAvx2` 去掉 `PackBPanel`，按 `(k_panels, n_blocks)` 直接消费 |
| `RunGemmF32PackedBReference` | 兼容 fallback / 测试对照 | 标量消费 packed 布局；与 double oracle 对比时按 layout 常量 unpad 视角求和 |

数值合同：packed 布局只是数据搬移 + 零填充，乘加次序与累加路数不变。验收容差沿用现有测试口径：绝对误差 `2.0e-4 + 2.0e-5*sqrt(k)` **或**相对误差 `2.0e-4`（见 `tests/unit/backend/cpu/kernels/test_cpu_gemm_avx2.cpp`）。pad 只在末面板尾部追加 `+0.0` 项，累加次序不变，结果与未 pad 逐 bit 相同——唯一例外是 `-0.0 + 0.0 → +0.0` 的符号位变化（写 bit-exact 断言时注意）。

### 3.5 recipe 传递链（缺口 2，修正稿）

生产链路（`BuildWeightPackingRequests` 是唯一生产 request 来源）：

```text
KernelDescriptor  ── 新增 .packing_recipe 字段（packed descriptor 声明其精确 recipe）
      │ backend 提供不依赖 OpParams 的 recipe 查询入口
      │   （不能复用 PrepareKernel：QkvLinear 的 metadata_builder 消费 params，
      │    packing request 不带 params；且 compiler 不得依赖 backend）
      │ 由持有 Backend 的编排层调用，注入
WeightPrepackPlanner::Request  ── 新增 .recipe 字段
      │ PrepackAndStore(store, requests) 按 req.recipe 调用
CpuWeightPrepacker::Pack(op, weight, selector, recipe)  ── 不再内部 RecipeFor
      │ 三处联动（Request.recipe == artifact.recipe == key.recipe）
PackedWeightStore.Store(key{..., recipe})
      │ execution plan 组装：Find({binding, selector, expected_packing_recipe})
      │   （expected_packing_recipe 已存在于 ResolvedKernel，非本提案新增）
PrepareExecutionBindings ── params builder 固化 packed 指针 + POD 切片
      │
Execute ── packed driver，零分配
```

要点：

- 真缺口是 **recipe 无法到达 request 且到达路径受模块边界约束**。指定归属：`KernelDescriptor::packing_recipe` 新字段 + backend 查询入口（纯数据、不需要 OpParams），由持有 Backend 的编排层（非 compiler、非 model loader）注入 `Request::recipe`。
- **recipe 查询入口复用 resolve 逻辑（硬约束）**：查询入口为 `CpuBackend` 成员（使用 `capabilities_` 快照），内部即 `ResolveEligibleDescriptor(...)` → `descriptor->packing_recipe`；`PrepareKernel` 同时改为从 `descriptor->packing_recipe` 取值，替换现在的 `CpuWeightPrepacker::RecipeFor(selector)` 赋值（`cpu_backend.cpp:87-91`）。pack 侧与 resolve 侧共用同一条 eligibility 路径，否则"另写一套选择逻辑"会让 pack 期与 resolve 期选到不同 descriptor → recipe 不等 → `Find(exact_key)` 返回 nullptr（§6 第 1 行风险的来源之一）。`RecipeFor(selector)` 退化为 identity descriptor 的默认 recipe 来源（现有链路零扰动）。
- `Request 加字段 / Pack 加参数 / key 用同一份 recipe` 三处必须一起改，否则 artifact recipe 与 key recipe 分叉（今天 `Pack` 内部自行 `RecipeFor(selector)`，不加参数产不出 bpanel artifact）。
- **共存期策略（显式立约）**：identity 与 bpanel 并存期间，二者**按 op_type/selector 不相交**——bpanel 只上新增 descriptor（首落 Linear），现有 QKV/GateUp descriptor 保持 identity 不动；任何时刻 `{binding, selector}` 只映射一种 recipe，同键多 recipe 不出现。若未来需要同键多 recipe（如多 ISA 并存），须先改 `PackedWeightStore::FindByBindingSelector` 的 `kFailedPrecondition` 语义（当前无生产调用者，仅测试使用），不在本方案范围内。
- **packed 使能是图级全局开关（M5 前置，硬阻塞）**：`GraphLoweringConfig::enable_packed_weights` 是 `bool`，一旦开启，所有带 kWeight 端口的 step 全部翻成 `kPacked`（`graph_lowering.cpp:110-118`）。带 kWeight 端口的算子共 6 个（Embedding/RmsNorm/Linear/QkvLinear/GateUpLinear/AddRmsNorm，`operator_schema.cpp`），当前注册 packed descriptor 的只有 QkvLinear/GateUpLinear/AddRmsNorm。真实 Llama 图开 packed 时 Embedding/RmsNorm/Linear 在 resolve 期 NotFound——"bpanel 首落 Linear"只在单算子测试里成立。M5 端到端前置：补齐 Linear + RmsNorm + Embedding 的 packed descriptor（identity recipe 即可），或引入 per-op packed 使能（属 lowering 语义变更，按 AGENTS.md compiler 边界另立 workstream，不混入本方案）。
- **生产编排点不存在**：全仓只有测试调用 `BuildWeightPackingRequests + PrepackAndStore`，无任何 `src/` 编排点；PRD 将 `PackedWeightStore/WeightPrepackPlanner` 定位为"仅为现有 ExecutionPlan packed-weight API 的兼容设施"。M4 的端到端验证是其前置依赖（improvement-plan 01 InferenceSession/Generate readiness），须显式标注。

### 3.6 合同收窄清单（descriptor 声明范围）

按母提案 §3.3/§4.4：packed descriptor 必须覆盖其声明的完整合法 layout 合同，不适合 SIMD 的合法输入在 descriptor 内走 `RunGemmF32PackedBReference`。逐条列出：

1. `lhs_k_stride != 1`、`output_n_stride != 1` → packed reference；
2. `n % nr != 0` 的 N 尾（§3.3 未选 masked-store 实现时）→ packed reference 或 masked 路径；
3. `m % mr != 0` 的 M 尾（同上）；
4. `k == 0` → 直接写 `+0.0`（与 plain 一致），不读 artifact；
5. `m == 0 || n == 0` → no-op；
6. rhs 不再以裸指针出现：任何需要 plain B 的回退（如 row-pair 尾路径）在 packed 路径下都必须以 packed reference 完成，**不得**把 packed 字节流当作 plain B 解释。

### 3.7 端到端时序

```text
[model preparation]                          [steady-state]
Load HF → BuildLlamaDense → Lower            Execute (zero alloc):
  → resolve（无 shape，填 expected_recipe）      M=1  : PackedBScan
  → BuildWeightPackingRequests（compiler）      M 中 : PackedBScan / Blocked
  → 编排层注入 Request.recipe（backend 查询）   无 registry、无 heap、无 B 转置打包
  → PrepackAndStore → Store（模型准备期一次）
```

编排点缺失警示：图中"编排层注入"一步今天没有任何生产代码处，属前置依赖（见 §5 M4）。

## 4. 关键决策与权衡

1. **pad 放大量化（修正）**：只论证 N 侧会低估成本。K 侧以 kc=512 计：

   | logical K | k_panels | K 侧放大 | 注释 |
   |---|---|---|---|
   | 4096 | 8 | 0% | 整除 |
   | 11008 | 22 | +2.3% | 11264/11008（down_proj） |
   | 14336 | 28 | 0% | 整除 |
   | 1280 | 3 | +20% | 3×512/1280（小模型宽） |
   | 768 | 2 | +33% | 2×512/768（GQA 的 k/v 常见量级） |

   备选：kc=256 使 4096/11008/14336 全整除（11008=43×256），K=768 放大降为 0，块降至 16KB、A 面板 buffer 降至 48KB。但 kc 减半 → k_panels 翻倍 → 输出 C 的 read-modify-write 趟数与 microkernel 调用次数翻倍（`MicroKernel4x16` 每面板都 load-add-store 输出，`gemm_f32_avx2.cpp:169-188`）；A 打包总量不变（恒 M×K×4）。大 M 下这是二阶成本，方向与 pad 收益相反——kc 定值归 M3 消费侧 benchmark。kc 是 layout 常量，无论取哪个值都编入 layout 名。

2. **独立 `PackedGemmF32Args` vs 扩展 `GemmF32Args`** → 独立结构。plain 合同是普通指针 + 步长，packed 需要切片描述，混合只会互相拖累。

3. **params arena 硬约束**：`KernelParamsBuilder` 产物必须 trivially destructible 且 `params_size <= kMaxKernelParamsSize = 512`（`kernel_types.h`）。`PackingRecipe` 含 `std::string`，不能整体入 params——`PackedGemmF32Args` 只存 POD 切片（指针 + int64 切片尺寸）；layout 常量在每个 driver TU 内编译期绑定（§3.1 已立约 layout 名唯一确定 nr/kc，不引入 op/selector→layout 的二级查表，避免再添一层可漂移映射）。

4. **artifact 生命周期维持内存态**：`PackedWeightStore` 不持久化；磁盘缓存/序列化不属本方案。

5. **QKV/GateUp 大 artifact 与 nr 对齐（修正）**：复合绑定下"三段 N 范围"只有在**每个 component 行数都是 nr 整数倍**时才成立（4096/1024/1024 恰好对齐是巧合不是合同）。否则一个 16 宽输出 tile 跨 Q/K 边界，无法写进单一输出 tensor。三选一：

   - (i) 要求 component 行数 nr 对齐，不对齐走 packed reference fallback；
   - (ii) 按 component 分别 pad/分别成块，artifact 记录 component 块偏移表（当前 `logical_shape` 只有 `[total_rows, in_features]`，无 component 表，需扩展 contract）；
   - (iii) 支持跨段 tile 拆分写出（复杂度最高）。

   默认选 (i)，M5 落地时再按 benchmark 决定是否升级。另：现有 alias/injectivity 校验基于"连续字节范围 + 三个连续行区间"（`qkv_linear_entry.cpp`），bpanel 布局下这套安全证明对 packed payload 仍成立（payload 是只读 opaque 缓冲），但"weight 与 input/output 不重叠"的证明对象从三段裸指针变成单个 artifact 缓冲，adapter 校验逻辑需重写而非复用。

6. **workspace 边界**：首版保持 zero-workspace（B 已 pack 免去 32KB `buf_b`，A 侧保留 96KB `buf_a` 栈缓冲）。若未来 A 也要 pad-to-kc / 面板驻留，须按母提案 §4.6 走 binding-dependent workspace 合同，不得在 kernel 内 `malloc`。

## 5. 里程碑与验证设计

| 里程碑 | 内容 | 退出证据 |
|---|---|---|
| M1 合同 | layout 命名/版本规则 + `cpu_bpanel_packing.h` 常量头；nbytes 不变式两侧独立推导互证单测；**不扩展 PackingRecipe** | 合同单测通过 |
| M2 打包器 | `BpanelPackerV1`（identity 链路保持不动）；pad/对齐/放大公式单测（K 侧数值表入 params）；unpack 后与逻辑权重逐元素一致、pad 区为 0；打包吞吐与放大实测（kc=256 vs 512，作为 M3 决策输入） | 工作流 O0/O1 证据 |
| M3 packed primitive | 三个 driver；与 `RunGemmF32Reference` 对照覆盖母提案 §6.2 boundary shapes（含 M/N/K=0、非整除、pad 尾、masked/partial store 路径）；**测"移除 pack 的净收益"**并记录 B 流量模型（去掉 PackBPanel 不改变 B 总流量：无 NC 层级时每 `mb` 仍重扫 K 面板全部 n_blocks，只是省掉 transpose-gather 与 32KB 写） | O1 correctness + O2 prepared micro/operator benchmark |
| M4 传递链 | `KernelDescriptor::packing_recipe` + backend recipe 查询入口 + 编排层注入 `Request::recipe` + `Pack(..., recipe)` 三处联动；mismatch（layout/alignment/nbytes）明确失败；**前置：生产编排点存在**（依赖 improvement-plan 01） | 母提案 §7「exact recipe 与 packed B」退出条件 |
| M5 消费者 | **前置：补齐 Linear/RmsNorm/Embedding 的 packed descriptor 或引入 per-op packed 使能（§3.5 全局开关）**；component nr 对齐立约（§4 第 5 条方案 i）+ adapter 校验重写；Qkv/GateUp 切 bpanel（跨步受 §4 决策约束）；Linear 新增 packed descriptor | fused 数值全链路 + packing break-even + hot/streaming 双模式 |

依赖关系与母提案依赖图一致：M1-M3 可与「Decode direct-weight AVX2」并行；M4/M5 前置合同与证据基线重采 + 生产编排点落地。

验证方法补充：

- 全部 correctness 与 contract 单测走 `tests/unit/`；packed driver 在 ASAN 变体下跑，验证 pad 读取永不越 artifact 边界、masked store 不越输出边界。
- 消费侧校验复用/扩展 `packed_weight_utils.cpp` 闸口：新增 bpanel 校验项（recipe 名、alignment、nbytes 不变式、logical rank/尺寸），identity 校验保持原样。
- packing 性能单独测（GB/s、size amplification、break-even invocation count），不混入 steady-state kernel loop。
- 性能结论遵守逐机噪声 floor 协议（母提案 §6.8.1），不外推跨机数值。

## 6. 风险

| 风险 | 缓解 |
|---|---|
| 打包期与 resolve 期 recipe 不一致 → plan 组装 `Find(exact_key)` 返回 nullptr → "Packed weights not found"、模型无法组装（这是生产路径真实风险；`FindByBindingSelector` 无生产调用者，仅测试使用） | M4 端到端单测覆盖不一致时的明确失败与诊断信息；三处联动（§3.5） |
| ISA 特有 recipe 撞既有不变式（`cpu_capability_design.md`：recipe 只由 selector 导出、一份 packed 权重服务所有特征等级、key 机器无关）；layout 名含 `avx2` 使 key 变成 machine/policy 相关：feature policy 关 AVX2 或换机 → resolve 到别的 descriptor → recipe 变 → 已有 artifact 查不到 | 立约：packing 必须由将来 resolve kernel 的**同一 backend 实例/同一 feature policy** 驱动；mismatch → NotFound 作为预期行为 + 诊断要求写进 M4 测试 |
| 模型加载变慢（prepack 一次性成本） | M5 单独测 packing GB/s + break-even invocation count，不达标不提升 priority |
| **PRD 预算对齐**：冷启动 ≤2s、内存 ≤4GB、稳态零分配 | (a) bpanel 是 transpose-gather，单字节成本远高于 identity memcpy，2s 预算需按实测 packing GB/s 倒算可接受模型规模；(b) 打包期 raw + packed 双份驻留（fused 还会临时再复制 Q+K+V），需决策"打包后是否释放 raw weight backing storage"；(c) FP32 packed 7B 约 26GB 量级，与 4GB/INT4 目标不在一个数量级——**明确 FP32 packed-B 是合同/stepping-stone 里程碑，达标路径是量化 recipe** |
| 体积放大 | nbytes 不变式在 M1 单测保单；按 §4 第 1 条数值表逐 shape 记录 |
| packed driver 读超 logical 尾部的 pad 数据 | pad 写入 artifact 内（分配按 padded 尺寸），读取永不越 Buffer 边界；M3 ASAN 验证 |
| 未来 AVX-512/量化 layout 分叉 | layout 名内嵌 ISA 家族 + 版本；量化 metadata 按量化合同演进（另立提案，不混入本方案） |

## 7. 变更记录

| 日期 | 版本 | 变更 | 原因 |
|---|---|---|---|
| 2026-09-22 | 1.2 | 第二轮审核修订：scan driver 形态改为 MR∈{1..8} 广播变体（沿 N 向量化、无水平归约）；recipe 查询入口硬约束为复用 ResolveEligibleDescriptor、PrepareKernel 改读 descriptor->packing_recipe；新增 M5 前置（packed 使能是图级全局开关，须补齐 6 个 kWeight 算子的 packed descriptor 或 per-op 使能）；修正 N 尾三选一（NR<16 列变体/partial column store）、kc 取舍维度（C read-modify-write 趟数翻倍，定值归 M3）、layout 常量编译期绑定（去二级查表）、块内/块间布局表述、K 尾边界表述 | 块内布局与遍历序分开声明；scan driver 原"沿 K 向量化"形态在 bpanel 布局下丢失全部收益；图级开关使"首落 Linear"在端到端不成立 |
| 2026-09-22 | 1.1 | 按审核意见修订：recipe 不扩展（shape 派生字段不可实现）；pad-0 改为只统一块 stride、输出侧须 masked/partial store；传递链改指生产 `BuildWeightPackingRequests` 并指定 recipe 注入 API 与三处联动；补 K 侧放大量化表、PRD 预算、component nr 对齐、params arena 约束、合同收窄清单、真实风险（exact key NotFound）、编排点缺失标注；修容差公式与关联代码清单 | 初稿按字面实施会因 recipe 相等性合同失败、tail 处理越界、现状描述失真带偏实施顺序 |
| 2026-09-22 | 1.0 | 初稿：从 B panel 重复打包审核结论展开，具体化 exact recipe 与 packed-B 设计 | GEMM AVX2 kernel 审核提出长期方向 |