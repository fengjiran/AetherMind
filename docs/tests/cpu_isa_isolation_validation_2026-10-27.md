# CPU ISA 编译隔离第三批验收

- **状态**: Current
- **日期**: 2026-10-27
- **关联方案**: [13 号方案 Q2](../improvement-plan/13-deepinfer-quality-and-validation.md)、[算子工作流](../guides/operator-development-workflow.md)
- **执行基线**: `ff2f94797033938e8e5d7c2e39810904142f67fd` 加本批工作树补丁，执行时 dirty；未暂存、未提交。
- **环境**: Linux x86_64、GCC 14.2.0、CMake/CTest 3.28.3、GoogleTest 1.16.0；当前主机具有 OS-usable AVX2/FMA。

## 1. 实现与边界

[ISA 构建配置](../../cmake/AetherMindCpuIsa.cmake) 以 AETHERMIND_ENABLE_AVX2 控制可选实现；实际可用性还需目标架构和 compiler intrinsic compile probe 通过。该探测不运行 AVX 指令，不能冒充硬件执行证据。

移除整个 unit target 的 -mavx2/-mfma。四个库实现及 [学习 helper](../../tests/unit/backend/cpu/kernels/avx_learning/test_avx_learning_avx2.cpp) 独立接收 ISA flags，使用 -fno-lto 和 SKIP_UNITY_BUILD_INCLUSION 保持边界。[学习 wrapper](../../tests/unit/backend/cpu/kernels/test_avx_learning.cpp) 保留 17 个测试名，不含 intrinsic，SetUp 检查真实 usable/effective AVX2/FMA 后才调用无 SIMD 类型的私有接口。未编译 helper 的同名 stub 在 CTest 中明确 disabled/not_applicable。

baseline 构建直接排除四个可选库 TU 与学习 helper，不定义对应 availability macros。首次真实构建发现 QKV/GateUp 的 descriptor 虽有 guard，函数体仍引用被排除的 packed-B 实现；现已为专用声明、preparer、entry 和执行函数补齐相同 guard，Linear 的专用 preparer 也保持一致。reference/identity 路径保留，未修改 kernel 计算算法、layout 身份或 priority。

[ISA 编译检查器](../../tools/verify_cpu_isa_build.py) 检查真实 compile_commands：baseline 无可选源码或宏；AVX 配置恰有四个库实现与一个 helper 接收 ISA flags，其余 TU 禁止全局 ISA uplift。它读取 generated Unity TU 的 include 归属，防止 build-tree 命令漏检；可选实现进入 Unity 或后置 LTO 覆盖边界会失败。

CTest 使用精确 direct-candidate 名单，纯 metadata/policy 用例保持可运行。AVX 专用入口依赖 compile-boundary fixture 和 [baseline-safe host probe](../../tests/cmake/test_avx2_host_probe.cpp)，未编译、无硬件、缺二进制、空 discovery、零 ISA case 或 required skip 都不能算 AVX 验收通过。

## 2. 实际结果

| 配置与集合 | CTest 项数 | 结果 |
|---|---:|---|
| validation-baseline 的生产/lifecycle 集合 | 379 | 全部通过，0 failed/skipped |
| validation-avx2 构建的同一生产/lifecycle 集合 | 379 | 全部通过，0 failed/skipped |
| validation-avx2 专用集合 | 62 | 60 个 GTest candidate/learning case + 2 fixtures，全部通过 |
| ASAN/UBSAN/LSAN 的生产/lifecycle 集合 | 285 | 全部通过，0 failed/skipped，无 sanitizer 报告 |
| ASAN/UBSAN/LSAN 的 AVX 专用集合 | 62 | 全部通过，0 failed/skipped，无 sanitizer 报告 |

集合复用既有用例，不能相加为独立测试数。生产集合包含 C++、CLI/C API 对照、模型 fixture、Python 及 CMake 构建合同；不等同于全部都是 instrumented GTest。

新增 [10 个 policy/availability case](../../tests/unit/backend/cpu/test_cpu_isa_fallback.cpp) 在 baseline 和 AVX 构建均通过：分别禁用 AVX2/FMA，覆盖 Prefill/Decode 的 RMSNorm reference 与 Linear identity recipe；compiled macros 与真实 descriptor 注册一致；factory 无法要求不可用的另一架构 feature。backend policy 只证明 descriptor 过滤，不能据此声称独立 dot-product dispatch 已强制 scalar。

完整模型证据复用既有 HF→Compile→Prepare→Generate、tiny tied-GQA scalar oracle、Prefill/两步 Decode、logits/KV/watermark 与重复请求回归。没有新增平行 runner，也没有将 loader 或 descriptor 成功提升为模型支持声明。

## 3. 静态与负面证据

- baseline compile_commands 中没有五个可选 ISA TU，所有 project TU 无 ISA uplift flags；AVX 配置只有上述五个 TU 接收 -mavx2/-mfma。
- baseline 实际 libAetherMind.so 没有可选 AVX compute 符号；整库反汇编未发现 AVX/VEX 指令。
- baseline/AVX 两种真实 wrapper 对象及 dot-product dispatch 对象均无 AVX/VEX 指令；学习 helper 对象包含 168 条目标指令。该数量仅是本编译环境的对象检查，不是性能指标。
- 四个库 AVX 对象与 helper 没有动态启动初始化函数；GTest 注册和硬件检查保持 baseline 编译。
- [编译检查器合同](../../tests/unit/cmake/test_verify_cpu_isa_build.py) 9 项通过，覆盖误编可选源码/宏、缺实现、wrapper 全局 flags、LTO 顺序及 Unity 污染。
- 6 项真实 CTest 状态探测覆盖 baseline 不适用、host 失败、缺 binary、空 discovery、零 ISA case、required skip，均维持明确失败或 not-applicable 状态。

## 4. 复现

先按 [fixture 合同](../../tests/fixtures/inference_admission/README.md) 准备固定资产，再运行：

```bash
cmake --preset validation-baseline
cmake --build --preset validation-baseline
ctest --preset validation-baseline

cmake --preset validation-avx2
cmake --build --preset validation-avx2
ctest --preset validation-avx2
ctest --test-dir build/validation/validation-avx2 \
  -L '^validation$' -LE '^isa-avx2$' --no-tests=error --output-on-failure \
  --output-junit "$PWD/build/validation/validation-avx2/junit-production.xml"

cmake --preset validation-asan-ubsan
cmake --build --preset validation-asan-ubsan
ctest --preset validation-asan-ubsan
ASAN_OPTIONS=halt_on_error=1:detect_leaks=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
ctest --test-dir build/validation/validation-asan-ubsan \
  -L '^isa-avx2$' --no-tests=error --output-on-failure \
  --output-junit "$PWD/build/validation/validation-asan-ubsan/junit-isa.xml"
```

本次配置通过 -DFETCHCONTENT_SOURCE_DIR_SPDLOG 指向既有 build/_deps/spdlog-src，复用 1.17.0 源码。ASAN/UBSAN 使用 detect_leaks=1，没有关闭 LSAN 或增加 suppression。没有运行 TSAN 或 benchmark。

## 5. 证据与未验证范围

configure/build/CTest/JUnit/compile_commands 位于 gitignored build/validation 对应 preset 目录；反汇编、符号、配置结果和负面探测归档在 build/validation/isa-artifacts。首次 baseline 编译失败保留为 build-initial-failure.log，不将失败历史抹去。

当前证据来自具备 AVX2/FMA 的同一主机，尚无物理无 AVX CPU 的运行证据；baseline 构建证明可选实现被排除及完整生成可运行，不表示整个进程的 libc/libm 都被强制 scalar。没有验收全仓 Unity/LTO 构建、其他 compiler/architecture 或模型性能收益。warnings profile、物理平台与后续外部数值 golden 仍按各自门禁推进，Q2 未整体关闭。
