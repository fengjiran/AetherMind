# utils 清理与正确性修复验收

- **状态**: Current
- **日期**: 2026-10-23
- **接口合同**: [公共 API 工具说明](../api/public-api.md#3-通用-c-工具includeutils)
- **环境**: Linux/x86_64，GCC 14.2.0，CMake 3.28.3，C++20

## 1. 范围与结果

本次完成 utils 审核中的修复与清理。**714 项聚焦原生测试、38 项 Python CLI 黑盒测试、112 项普通 ASAN/UBSAN 测试通过**，所选用例无失败或跳过。未运行全仓测试或性能基准，不据此声明速度提升或新增 dtype/model/device 支持。

| 范围 | 实现与合同 |
|---|---|
| checked arithmetic | [overflow_check.h](../../include/utils/overflow_check.h) 保留 builtin 优先路径；portable helper 按目标范围检查有符号溢出，使用足够宽的无符号计算生成模位宽结果，消除窄整数提升 UB。两条路径均始终写入输出；CheckNarrow 仍仅成功时写入。 |
| hashing | [hash.h](../../include/utils/hash.h) 为自包含 C++20 头，删除标准复合类型的通用 std::hash 注入；定制顺序明确，pair 无元素复制。复合键使用显式 aethermind::hash。 |
| Map helper | [FibonacciHash](../../include/container/detail/fibonacci_hash.h) 归 container detail，使用 uint64_t 运算；shift >= 64 返回单 bucket 索引零，三个 Map 的消费者均迁移。 |
| 浮点解析 | [parse_number.cpp](../../src/utils/parse_number.cpp) 两分支完整成功后才发布输出；共享 POSIX fallback 检查 general 字面量格式、可表示 subnormal、OOM 与 locale 失败，局部 locale RAII 并恢复线程 rounding mode。合法 inf/nan 留给领域层准入。 |
| 断言 | [logging.h](../../include/utils/logging.h) 限定 helper 命名空间，保留 Check failed 前缀和 debug/release 行为；移除 spdlog 传递 include，实际消费者声明直接依赖，核心库链接依赖为 PRIVATE。构建暴露的 memory/functional 传递依赖已在使用头补齐。 |
| memory | [memory_junk_fill.cpp](../../src/memory/memory_junk_fill.cpp) 为 memory 私有实现，byte 指针和 memcpy 支持任意对齐，保留旧 pattern、null/zero no-op 和默认 junk_fill 策略。删除旧公共头、实现及 pattern getters。 |
| 删除与保留 | 删除零消费者的 lru.h、xxh3.h、bits.h，以及 DataType 对 bits.h 的无效 include；保留 DataType 枚举与行为。numeric_utils、variant_utils、qualified_name 保留现有接口和归属。无兼容转发。 |

上述 C++ 头与定制点变更需要调用方迁移并重新编译，C ABI 合同保持不变。并发协议与资源所有权未扩大；locale 和 rounding guard 均为调用局部状态，不引入共享可变缓存。

## 2. 验证矩阵

| 验证 | 结果与边界 |
|---|---|
| Release 构建 | aethermind_unit_tests 与独立 aethermind_infer 成功；独立 CLI 配置 tests/benchmark 均 OFF。 |
| Debug ASAN/UBSAN 构建 | build-asan 的 AetherMind 成功，使用 address/undefined 与 frame pointers。 |
| 工具单测 | 114 项通过：原有 CheckNarrow 15 项，本次新增算术 43、hash/Fibonacci 16、解析 28、断言 8、memory 4，共 99 项新增。 |
| 消费者回归 | 452 项通过：kernel registry/selector、allocator、HF JSON、serde、OperatorName/QualifiedName、Map/Array/Any/Complex/DataType、constant evaluator/folding。 |
| C ABI 与 CLI/Session | 148 项通过，包含 100 项 C ABI 和 48 项 CLI/直接 Session 对照。 |
| CLI 黑盒 | 38 项通过，包括真实 THP warning 的 stderr 与 JSON 输出隔离。 |
| Sanitizer | 独立 harness 的 112 项普通用例通过，排除两项 malloc 注入；两项故障用例在原生主目标中实际命中并验证重试、输出与 rounding mode 恢复。 |
| portable 算术 | 直接测试生产 helper；穷举 signed/unsigned 8-bit operand pairs，覆盖各位宽边界、alias 输出及固定种子 64-bit builtin 差分。 |
| 解析平台路径 | Linux 同时编译并调用共享 POSIX fallback，覆盖四种 rounding mode、已有 exception flag、range/subnormal、buffer OOM。另以 newlocale 链接 wrapper 验证 NULL 返回后的恢复与重试。 |
| 头独立性 | 7 个保留 utils 头与 3 个 Map 头各自单独包含均编译成功，无 spdlog include 路径。 |
| 断言配置 | 独立 debug 与 release 各 8 项通过；主目标为 release，sanitizer harness 为 debug。 |

未进行真实 macOS、MSVC 或 32 位平台验证；Linux 上的 fallback/helper 测试不等价于这些平台的完整构建。未运行 TSAN，未修改同步或并发协议。

Sanitizer harness 关闭既有 malloc interposer，并设置 detect_leaks=0；这不构成零泄漏证据。此前整体 sanitizer/interposer 与静态注册泄漏边界见 [M6 验收记录](upper_level_runtime_assembly_m6_validation_2026-10-22.md)，本轮未重新评定 LSAN 门禁。

## 3. 复跑命令

以下复用本地已配置的 Release build、tests/benchmark OFF 的 Release build-infer，以及 Debug address/undefined 的 build-asan；HF fixtures 沿根 README 准备。

```bash
cmake --build build --target aethermind_unit_tests -j 6
cmake --build build-infer --target aethermind_infer -j 6
cmake --build build-asan --target AetherMind -j 4

./build/tests/unit/aethermind_unit_tests --gtest_filter='CheckNarrow.*:OverflowPortable*.*:HashUtils.*:FibonacciHash.*:CheckMacro*.*:DebugCheckMacro*.*:ParseNumber/*.*:MemoryJunkFill.*'

./build/tests/unit/aethermind_unit_tests --gtest_filter='KernelRegistry.*:KernelSelector.*:CPUAllocator.*:AlignedCpuBuffer.*:*/AlignedCpuBuffer.*:AllocatorOwnership.*:ModelLoader_HfJsonReaderTest.*:OpParamsSerde.*:Operators_OperatorName.*:QualifiedName.*:Map*.*:Array.*:ComplexTest.*:ComplexHalfTest.*:DataType*.*:Any*.*:ConstEvaluator.*:ConstantFoldingPass.*'

TMPDIR=/tmp AETHERMIND_INFER_BINARY="$PWD/build-infer/apps/inference/aethermind_infer" \
  ./build/tests/unit/aethermind_unit_tests --gtest_filter='CAbiRuntime.*:CAbiModel.*:CAbiSession.*:CAbiGenerate.*:CAbiKVValidation.*:KvPlanes/CAbiRuntimeAllocationFailure.*:HfPresets/CAbiRuntimeIntegration.*:CliMatrix/InferenceCliIntegration.*'

TMPDIR=/tmp python3 tests/unit/apps/inference/test_inference_cli.py \
  --infer "$PWD/build-infer/apps/inference/aethermind_infer"
```

```bash
g++ -std=c++20 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
  -DAETHERMIND_TEST_MALLOC_INTERPOSER_DISABLED=1 -Iinclude -Isrc -Itests/unit \
  tests/unit/base/test_check.cpp tests/unit/base/test_hash.cpp \
  tests/unit/base/test_overflow_arithmetic.cpp tests/unit/base/test_overflow_check.cpp \
  tests/unit/base/test_parse_number.cpp tests/unit/memory/test_memory_junk_fill.cpp \
  tests/unit/inference/test_malloc_interposer.cpp \
  -Lbuild-asan/src -Wl,-rpath,"$PWD/build-asan/src" \
  -lAetherMind -lgtest_main -lgtest -pthread -ldl \
  -o /tmp/aethermind-utils-cleanup-asan-tests

ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  /tmp/aethermind-utils-cleanup-asan-tests --gtest_filter='*-*AllocationFailure*'
```

## 4. 证据身份

受测工作树以 HEAD `84a1bd43d8c8e41079b98b9e1fbda7a254c82ed7` 加本次未提交改动组成。关键文件 SHA-256：

```text
478d09a164557718c6292148a4417f8e0c4d5c16b6134bfd65161b7253ec1fb4  include/utils/overflow_check.h
94b9f9191139e16ab7f9e34ddc23cd79c92c6c31f2c9a057f53897f59fce48ed  include/utils/hash.h
ed2aa9c13de630909edbb1eb1d11a0b68952a805b1ab989b43b485147d0212d1  include/utils/logging.h
48ce899b6b974aba24ab11256c37aaa6c53087e5c8aa3032be090862feb9f4c0  src/utils/parse_number.cpp
8c963560aca94a7ffe1580bcb0ae41cea43cb1a4013c76f7dd2a1cf2fbe981f7  src/memory/memory_junk_fill.cpp
a7af02f865a0b07407dabe54a4baf373d992517d20d1e36bf87aac2debab64b4  include/container/detail/fibonacci_hash.h
b47fd0eeb4c77a2a3577274c1d76d17337856572c4e1308bf9cc5568d188de4f  src/CMakeLists.txt
```

本地原始日志在 `/tmp/aethermind-utils-cleanup-{build,cli-build,asan-build,unit-tests,consumer-tests,integration-tests,cli-tests,asan-tests}.log`；这些是会话临时文件，不随仓库保存。
