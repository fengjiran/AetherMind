# CPU Capability 能力模型

## 1. 目标与范围

CPU capability 模型回答两个问题：

1. **这台机器能安全执行哪些指令集？** —— 硬件位（cpuid/hwcaps）、OS 状态（XCR0、AMX 权限、SVE VL）、运行时策略（`CpuFeaturePolicy`）三者共同决定。
2. **某个 kernel descriptor 是否可在此机器运行？** —— 注册时声明指令集要求，resolve 时按能力快照过滤。

范围：x86-64 与 AArch64 的指令集能力检测、数据契约、kernel 选择集成。属 `backend/cpu` 模块：类型声明在 `include/aethermind/backend/cpu/cpu_capabilities.h`，检测在 `src/backend/cpu/cpu_info.cpp`，字符串化在 `src/backend/cpu/cpu_capabilities.cpp`。

## 2. 设计原则

CPU 能力模型基于**特征集合（feature set）**建模：

- **原子性**：每个 `CpuFeature` 是单一、不可再分的指令集能力（如 `kAvx2`、`kFma`、`kAvx512Vnni`），组合要求以特征集的并集表达。

- **互不隐含**：特征之间不存在蕴含关系，`KernelDescriptor::cpu_requirements` 必须显式声明一条路径所需的全部特征。

- **严格子集判定**：kernel 可运行当且仅当机器 `effective_features` 包含其全部要求（即 `ContainsAll` 语义）。

- **与执行行为正交**：device/DType/weight 布局/phase 属 `KernelSelector` 的结构维度；指令集要求属 `cpu_requirements` 能力维度，二者在 resolve 中按序组合、互不重叠。

若未来出现现有特征无法表达的指令集要求，按 §3.2 的 append-only 规则扩展枚举。

## 3. 数据模型

类型均位于 `include/aethermind/backend/cpu/cpu_capabilities.h`（base 层不可依赖，仅 backend/cpu 及其上层消费方依赖）。

### 3.1 `CpuArchitecture`

```cpp
enum class CpuArchitecture : uint8_t {
    kUnknown = 0,
    kX86_64,
    kAArch64
};
```

### 3.2 `CpuFeature`

扁平、append-only 特征枚举，x86 与 AArch64 共用一个命名空间，`kCount` 为哨兵（bitset 尺寸）。

```cpp
enum class CpuFeature : uint8_t {
    // x86-64
    kSse41, kAvx, kAvx2, kFma, kF16c,
    kAvx512F, kAvx512Bw, kAvx512Vnni, kAvxVnni, kAvx512Bf16,
    kAmxTile, kAmxInt8, kAmxBf16,
    // AArch64
    kNeon, kFp16, kDotProd, kI8mm, kBf16, kSve, kSve2,
    kCount
};
```

追加规则：

1. 只能 append（不得改既有值，防破坏哈希与序列化）；位置随意但需保持分组注释。
2. 必须同步扩充 `src/backend/cpu/cpu_capabilities.cpp` 的 `kCpuFeatureNames` 名字表。该不变量由测试 `CpuCapabilities.ToStringFeatureTableCoversEveryEnumValue` 护栏（漏改 → 输出 "Unknown" 即失败）。

### 3.3 `CpuFeatureSet`

128 位定长 bitmask（`kWordCount = 2`），无堆分配，全部成员 `constexpr`/`noexcept`，值类型可自由拷贝，`operator==` 与 `Hash()` 一致（`hash_combine` 逐字混合，供 `RegistrationKeyHash` 复用）。

```cpp
class CpuFeatureSet {
    static constexpr size_t kWordCount = 2;
    static constexpr CpuFeatureSet From(std::initializer_list<CpuFeature>) noexcept;
    AM_NODISCARD constexpr bool Contains(CpuFeature) const noexcept;
    AM_NODISCARD constexpr bool ContainsAll(const CpuFeatureSet& required) const noexcept; // 子集
    AM_NODISCARD constexpr bool empty() const noexcept;
    constexpr void Enable(CpuFeature) noexcept;          // 越界(kCount)忽略
    AM_NODISCARD constexpr CpuFeatureSet Intersect(const CpuFeatureSet&) const noexcept;
    AM_NODISCARD constexpr CpuFeatureSet Difference(const CpuFeatureSet&) const noexcept;
    AM_NODISCARD constexpr size_t Hash() const noexcept;
    friend constexpr bool operator==(...) noexcept = default;
};
```

### 3.4 字符串化

三个 `ToString`（`CpuArchitecture` / `CpuFeature` / `CpuFeatureSet`，后者输出 `{AVX2, FMA}` 形式，枚举序升序）定义在 `cpu_capabilities.cpp`——非 constexpr 格式化逻辑，与"constexpr 位运算留头文件"的准则区分。仅供诊断/错误信息，非热路径。

## 4. 三层能力快照

```cpp
struct CpuCapabilities {
    CpuArchitecture architecture = CpuArchitecture::kUnknown;
    CpuFeatureSet hardware_features{};  // 硬件原始位，仅供诊断
    CpuFeatureSet usable_features{};    // 硬件 + OS 状态门控后可用
    CpuFeatureSet effective_features{}; // policy 应用后；kernel 选择唯一依据
    uint32_t sve_vector_bytes = 0;      // SVE VL（bytes），thread-local，不参与 kernel 选择
};
```

**不变量**：`hardware_features ⊇ usable_features ⊇ effective_features`（`ApplyCpuFeaturePolicy` 校验前两者）。

分层语义：

| 层         | 含义                                | 例子                                                                    |
| --------- | --------------------------------- | --------------------------------------------------------------------- |
| hardware  | cpuid/hwcaps 报告的硬件位，不关心 OS        | Intel Ice Lake 报告 AVX512F，即使 macOS 从不保存 zmm 状态                        |
| usable    | 硬件 + 当前 OS/进程上下文**真正可安全执行**       | x86：XCR0 状态位全置才进；AMX 还需 `arch_prctl` 授权；SVE 需 `PR_SVE_GET_VL` 可查，否则不进 |
| effective | usable 应用 `CpuFeaturePolicy` 后的结果 | 测试停用 `{kAvx2, kFma}` → scalar 回退                                      |

**为什么不用单层 bool**：单层无法区分"硬件不支持"与"硬件支持但 OS 没开"——后者在 macOS（AVX512）、虚拟机/容器（XSAVE 限制）上普遍存在，直接执行会触发 `#UD` 或 `SIGILL`。kernel 选择必须只看 effective。

## 5. 检测矩阵

入口：`StatusOr<CpuCapabilities> DetectCpuCapabilities(const CpuFeaturePolicy& = {})`（`cpu_info.h`），等价于 `ApplyCpuFeaturePolicy(DetectUsableCapabilities(), policy)`。未知平台返回空快照。

### 5.1 x86-64（`src/backend/cpu/cpu_info.cpp`）

| 特征                         | 检测位（cpuid）                                         | 进 usable 门控                                                                        |
| -------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------- |
| SSE4.1                     | leaf1.ECX bit19                                    | 无（简化：现代 OS XCR0.bit1 恒使能）                                                          |
| AVX                        | leaf1.ECX bit28                                    | OSXSAVE(bit27) ∧ XCR0 bits1-2 (`0x6`)                                              |
| FMA / F16C                 | leaf1.ECX bit12 / bit29                            | 随 AVX 状态                                                                           |
| AVX2                       | leaf7.0.EBX bit5                                   | 随 AVX 状态                                                                           |
| AVXVNNI                    | leaf7.1.EAX bit4                                   | 随 AVX 状态（VEX 编码，无需 AVX512）                                                         |
| AVX512F / BW / VNNI / BF16 | leaf7.0.EBX bit16/bit30、ECX bit11、leaf7.1.EAX bit5 | AVX ∧ XCR0 全状态 `0xE6`（opmask+ZMM\_Hi256+Hi16\_ZMM）；leaf7.1 有 `leaf7.eax≥1` 保护      |
| AMX TILE / INT8 / BF16     | leaf7.0.EDX bit24/25/22                            | `arch_prctl(ARCH_REQ_XCOMP_PERM, XTILEDATA)` 授予 ∧ XCR0 bit17∧bit18；非 Linux 恒 false |

注意：AVX512 的硬件位与 OS 状态**分立**。If XCR0 无 `0xE6`，即使 CPUID 报 AVX512F 也不进 usable（典型：Intel Mac）。

`leaf0.EAX < 1` 或 `< 7` 时提前返回（不越界读 leaf7）。

### 5.2 AArch64（`src/backend/cpu/cpu_info.cpp`）

| 特征      | Linux（auxv）                       | Apple（sysctl）                  | 其它（编译期宏）                               |
| ------- | --------------------------------- | ------------------------------ | -------------------------------------- |
| NEON    | `HWCAP_ASIMD`                     | 恒 true                         | 恒 true                                 |
| FP16    | `HWCAP_ASIMDHP`                   | `hw.optional.arm.FEAT_FP16`    | `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` |
| DotProd | `HWCAP_ASIMDDP`                   | `hw.optional.arm.FEAT_DotProd` | `__ARM_FEATURE_DOTPROD`                |
| I8MM    | `HWCAP_I8MM`（或 `HWCAP2_I8MM`）     | `hw.optional.arm.FEAT_I8MM`    | `__ARM_FEATURE_MATMUL_INT8`            |
| BF16    | `HWCAP2_BF16`                     | `hw.optional.arm.FEAT_BF16`    | `__ARM_FEATURE_BF16_VECTOR_ARITHMETIC` |
| SVE     | `HWCAP_SVE`                       | 未检测                            | `__ARM_FEATURE_SVE`                    |
| SVE2    | `HWCAP2_SVE2`（需 usable SVE 才有有用值） | 未检测                            | `__ARM_FEATURE_SVE2`                   |

SVE 特例：`HWCAP_SVE` 置位只说明硬件支持；usable 需 `prctl(PR_SVE_GET_VL)` 返回值 ≥0（进程上下文可用），此时 `sve_vector_bytes = vl & PR_SVE_VL_LEN_MASK`（**bytes**，内核约定 VL 单位，非 bits）。SVE2 只进入 usable 当 SVE 已 usable。

## 6. `CpuFeaturePolicy`

```cpp
struct CpuFeaturePolicy {
    CpuFeatureSet disabled_features{}; // 从 usable 移除
    CpuFeatureSet required_features{}; // 必须保留在 effective，否则失败
};
```

**语义约束：策略只能削减，永远不能启用**。应用规则（`ApplyCpuFeaturePolicy`，对任意快照可测，无需真实硬件）：

```text
校验 usable ⊆ hardware          → 违反返回 InvalidArgument
effective = usable − disabled
校验 required ⊆ effective       → 违反返回 FailedPrecondition（消息携带缺失差集）
```

注入路径：

- `CpuBackendFactory(CpuFeaturePolicy)` → `CpuBackend(policy)`：构造期一次性求快照并冻结

- `RuntimeOptions.backend.cpu_feature_policy`（运行时正式入口）

- 测试可用 policy 停用 AVX2/FMA 强制 scalar 回退，无需全局可变状态

## 7. Kernel 选择集成

### 7.1 注册期

```cpp
struct KernelDescriptor {
    ...
    CpuFeatureSet cpu_requirements{}; // CPU kernel 的完整指令集要求；空 = 任意机器
    ...
};
```

- `ValidateKernelDescriptor`：非 CPU device 携带非空 `cpu_requirements` → InvalidArgument。

- `RegistrationKey = {op_type, selector, cpu_requirements}`：重复注册判定含特征集（scalar 与 AVX2 变体同 selector 可共存）；`RegistrationKeyHash` 复用 `CpuFeatureSet::Hash()`。

- `KernelRegistry::FindCandidates` **只做结构匹配**（device/dtype/weight\_format/phase + kBoth），不按特征过滤——过滤属于 backend 职责。

### 7.2 resolve 期（`CpuBackend::PrepareKernel`）

```text
FindCandidates(op_type, selector)          // 结构匹配
  → effective_features.ContainsAll(cpu_requirements)   // 特征过滤
  → priority 取优（tie 先注册者胜）
  → 无命中：NotFound（诊断含 selector + effective_features）
```

`KernelDescriptor` 注册的 `cpu_requirements` 与 `RegistrationKey`/hash 保持一致，`effective_features` 来自构造时冻结的 `CpuCapabilities` 快照。

### 7.3 与打包的关系

`CpuWeightPrepacker::RecipeFor(selector)` 只由 selector 结构字段导出（**不含特征维度**）——一份 packed 权重服务所有特征等级，`WeightArtifactKey` 因此机器无关。若未来引入 ISA 特有排布（如 VNNI 专用 layout），需扩展 recipe 或 selector 维度并重新评估此权衡。

## 8. 平台差异与边界

| 场景           | 行为                                                                                                                                                                                                              |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| x86-64 macOS | AVX2/FMA 正常（XCR0 有 YMM 状态）；AVX512 硬件位存在但 `0xE6` 不满足 → 正确排除；AMX 非 Linux 恒 false                                                                                                                                  |
| 未知/未支持平台     | `DetectUsableCapabilities` 返回空快照 → `ApplyCpuFeaturePolicy` InvalidArgument → `DetectCapabilitiesOrDie` 在 `CpuBackend` 构造时 `AM_CHECK` 终止（`BackendFactory::Create()` 无 Status 通道，当前 API 下唯一选择；x86-64/AArch64 不触发） |
| 容器/虚拟机       | XSAVE 状态未经透传时 AVX512/AMX 会被 OS 门控正确排除                                                                                                                                                                           |
| SVE VL       | `sve_vector_bytes` 是创建快照线程的 VL（thread-local），故不参与 kernel 选择，只供诊断/内存规划                                                                                                                                           |
| 检测副作用        | x86-64 Linux 下 `DetectCpuCapabilities` 可能发起 `arch_prctl(ARCH_REQ_XCOMP_PERM)` 进程级权限申请                                                                                                                           |

## 9. 测试策略

| 文件                                                 | 覆盖                                                                                                                           |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `tests/unit/backend/cpu/test_cpu_capabilities.cpp` | 默认快照、`ToString` 三件套（含名字表全枚举护栏与越界回退）、`ToString(CpuFeatureSet)` 格式、`Intersect`/`Difference` 边界                                 |
| `tests/unit/backend/cpu/test_cpu_info.cpp`         | 真实检测不变量（hardware⊇usable⊇effective）、policy 减法/required 拒绝（FailedPrecondition）、usable⊄hardware 拒绝（InvalidArgument）、空 policy 幂等 |
| `tests/unit/backend/cpu/test_cpu_backend.cpp`      | policy 禁 FMA → scalar 回退；RuntimeBuilder 禁 AVX2 生效                                                                            |
| `tests/unit/backend/test_kernel_registry*.cpp`     | RegistrationKey 含特征集去重、非 CPU descriptor 拒绝、FindCandidates 不过滤特征                                                              |

结构性不可注入（不做）：cpuid/XCR0 分层、"硬件有 OS 无"场景依赖真实硬件，`DetectX86Capabilities` 无 provider 注入缝；如需多 ISA 交叉验证再引入注入重构。

## 10. 已知取舍

1. `CpuFeatureSet::Intersect` / `Difference` 中 Intersect 当前无生产调用者，保留为纯数据 API。
2. 特征枚举 append-only；删除特征会破坏既有 hash/去重语义，弃用不改值。
3. `ToString(CpuFeature)` 越界（`kCount`）返回 "Unknown"，由测试固定该行为。

## 11. 相关文档与代码

- 派发总纲：`docs/designs/dispatch_design.md`（§4 引用本模型）

- Backend 概览：`docs/designs/backend_design.md`

- 代码：`include/aethermind/backend/cpu/cpu_capabilities.h`、`src/backend/cpu/cpu_info.cpp`、`src/backend/cpu/cpu_capabilities.cpp`、`src/backend/cpu/cpu_backend.cpp`

