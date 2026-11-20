## 1. 使用的框架与方式

AetherMind 使用 **spdlog** 作为唯一日志后端，未引入自定义 logger 抽象、结构化字段或日志级别配置层。日志调用直接以 `spdlog::warn(...)` 等形式出现；进程入口通过 `spdlog::stderr_logger_mt("aethermind_infer")` 创建线程安全（`_mt`）的 stderr sink，并以 `spdlog::set_default_logger` 安装为进程默认 logger。

构建期通过顶层 `CMakeLists.txt` 集成 spdlog：
- 关闭共享库构建 (`SPDLOG_BUILD_SHARED OFF`)、启用 PIC (`SPDLOG_BUILD_PIC ON`)。
- 根据 C++20 能力默认设置 `SPDLOG_USE_STD_FORMAT=ON`，仅在外部提供 fmt 时才回退到 fmt 后端，并在配置时输出所选后端信息。
- README 明确说明 AetherMind 会向消费者传递 C++20 要求，并尊重父项目已存在的 `spdlog::spdlog` target。

## 2. 关键文件

| 文件 | 作用 |
|---|---|
| `apps/inference/inference_main.cpp` | CLI 进程唯一 logger 初始化点：创建 `stderr_logger_mt` 并设为默认 logger，捕获 `bad_alloc` 等异常，失败时用 `std::fputs` 写固定诊断后退出 1 |
| `src/memory/cpu_allocator.cpp` | 代码中目前唯一一处运行时日志调用：`spdlog::warn("thp madvise for HUGEPAGE failed")` |
| `tests/cmake/build_contract_project/logger.cpp` | 依赖契约测试用例，验证 `spdlog::info` 可通过链接 |
| `tests/benchmark/models/model_benchmark.cpp` | 基准套件在启动时调用 `spdlog::set_level(spdlog::level::off)` 关闭日志输出，避免干扰 benchmark JSON |
| `CMakeLists.txt` (根) | spdlog 编译选项与 `SPDLOG_USE_STD_FORMAT` 决策逻辑 |
| `README.md` | 文档化 spdlog 集成行为及旧目录缓存兼容说明 |
| `docs/roadmap/10-upper-level-runtime-assembly.md` | 设计记录中明确约定：仅独立工具 main 配置进程策略，库与 C ABI 不替换宿主 logger；logger 初始化必须是 main 首项可失败操作 |

## 3. 架构与约定

- **单默认 logger，stderr 输出**：CLI 在 `main` 开头构造 `spdlog::stderr_logger_mt("aethermind_infer")` 并覆盖默认 logger。当前仓库中没有自定义 sink、格式化器、日志轮转或分级路由。
- **无结构化字段**：所有日志均为扁平字符串，没有键值对、trace_id、session_id 等结构化字段。
- **日志级别极少使用**：实际源码中只出现 `spdlog::warn` 一个级别；未见 debug/info/error/critical 的使用，也未定义任何日志级别枚举。
- **库不持有 logger**：核心库 (`include/aethermind/*`, `src/*`) 除 `cpu_allocator.cpp` 外几乎不调用 spdlog；C ABI 适配层也不修改宿主 logger（见 `AGENTS.md` 与 `public-api.md` 的约束描述）。这使宿主可以保留自己的 logger 配置。
- **logger 初始化是容错第一关**：`inference_main.cpp` 将 logger 创建放在 `try` 中，捕获 `std::bad_alloc` 输出固定字符串 `"aethermind_infer: logger initialization: RESOURCE_EXHAUSTED\n"` 后返回 1；其他异常则输出 `"aethermind_infer: logger initialization failed\n"`。后续模型加载/Runtime 构建走另一层 try/catch，避免 OOM 导致诊断自身崩溃。
- **OOM 诊断不分配内存**：logger 初始化失败分支和后续异常分支统一使用 `std::fputs` 写入 stderr，这是刻意选择——在内存耗尽场景下避免二次分配。
- **基准套件主动关闭日志**：`model_benchmark.cpp` 在运行前调用 `spdlog::set_level(spdlog::level::off)` 禁用所有日志，防止日志混入 benchmark JSON 输出。

## 4. 约定与约束

- **仅 CLI 安装默认 logger**：`docs/roadmap/10-upper-level-runtime-assembly.md` 明确要求「仅独立工具 main 配置该进程策略，库与 C ABI 不擅自替换宿主 logger」；`AGENTS.md` 与 `docs/api/public-api.md` 同样声明 C adapter 不改宿主 logger。
- **logger 初始化必须先于模型加载/Runtime 构建**：同一份改进计划将其列为 M4 测试约束，并通过 `fail_logger_allocation.c` 静态注入 malloc 失败来断言「logger OOM 先于缺路径的 Load 失败」。集成测试同时捕获 stdout/stderr，验证正式工具无额外注入开关。
- **spdlog 格式后端由构建决定**：根 `CMakeLists.txt` 在无外部 fmt 时默认 `SPDLOG_USE_STD_FORMAT=ON`，否则回退 fmt；README 警告旧 build 目录可能缓存 `OFF`，需要显式 `-DSPDLOG_USE_STD_FORMAT=ON` 重建。
- **日志调用保持无分配语义**：logger 初始化失败分支以及 main 外层 catch 全部使用 `std::fputs`，避免在 bad_alloc 路径再次分配。
- **基准输出不污染日志**：`tests/benchmark/models/model_benchmark.cpp` 通过 `spdlog::set_level(spdlog::level::off)` 在基准开始前关闭日志，确保 JSON 结果纯净。

## 5. 适用性评估

该仓库确实存在 logging_system，但实现非常轻量：仅有 spdlog 集成 + CLI 默认 logger 安装 + 一处 `warn` 调用 + 基准端 level off，没有结构化字段、级别管理、sink 切换或日志 API 封装。因此本卡片标记为 **medium** 置信度——体系存在且有一致约定，但规模有限。
