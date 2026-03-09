# CPU-First Llama Runtime V1

## TL;DR
> **Summary**: Build a narrow V1 embedded inference runtime for Llama-family decoder-only models in the existing AetherMind codebase, using CPU-first execution, HuggingFace Safetensors import, hybrid operator dispatch, and a stable C++/C ABI. The plan deliberately defers serving, multi-request scheduling, GPU backends, and a generic graph runtime until single-request end-to-end CPU generation is stable.
> **Deliverables**:
> - Embedded runtime module for single-process, single-model, single-request decode
> - HuggingFace `config.json` + Safetensors loader and internal weight repack path
> - CPU kernel registry/dispatch table for reference + quantized paths
> - KV cache, prefill/decode loop, greedy sampling, and deterministic golden tests
> - Stable C++ API and thin C ABI over opaque handles
> - Hardening via unit tests, benchmarks, sanitizer variants, and evidence artifacts
> **Effort**: XL
> **Parallel**: YES - 3 waves
> **Critical Path**: 1 -> 3 -> 7 -> 10 -> 11 -> 12 -> 15

## Context
### Original Request
Design a detailed architecture and implementation plan for a large-model inference engine.

### Interview Summary
- Product shape: embedded runtime first, not a serving system
- Model scope: Llama-family decoder-only first
- Deployment target: CPU-first
- Capacity target: up to 7B, with INT4/INT8 support under a shared runtime architecture
- Integration surface: C++ API plus C ABI
- Dependency policy: allow a small number of mature dependencies; keep runtime core in-house
- Model asset format: HuggingFace `config.json` + Safetensors as canonical input
- Dispatch style: hybrid dispatch - runtime operator/backend registration with statically optimized hot-path kernels
- Test strategy: tests-after, but with explicit verification contracts defined before coding
- V1 success: stable prompt -> prefill -> decode -> sampling -> output on CPU

### Metis Review (gaps addressed)
- Treat V1 as greenfield for HAL, kernels, loader, runtime, and KV cache; do not assume hidden inference infrastructure already exists
- Explicitly supersede the broader scope in `.trae/specs/llm-inference-engine/tasks.md:1` and `.trae/specs/llm-inference-engine/spec.md:274` for this V1 plan
- Freeze token-IDs as the core runtime boundary; keep text tokenization as a thin adapter layer, not the runtime center of gravity
- Limit hybrid dispatch to CPU dense + chosen quantized kernels needed for V1; do not build a generic framework project first
- Predefine goldens, KV-cache parity checks, ABI smoke tests, and no-steady-state-allocation assertions before implementation starts

## Work Objectives
### Core Objective
Ship a CPU-first embedded runtime inside AetherMind that can load one supported HuggingFace-style Llama-family checkpoint, repack weights into an internal CPU layout, execute deterministic single-request generation with KV cache, and expose that flow through both C++ and C ABI entry points.

### Deliverables
- Runtime module layout under new inference/runtime-oriented headers and sources
- Canonical V1 model contract for supported Llama config fields, tensor naming, and rejection behavior
- Safetensors manifest reader, shard resolver, mmap/repack pipeline, and internal weight store
- CPU execution context with feature detection, threadpool policy, scratch memory, and dispatch table
- Reference CPU kernels plus quantized linear kernels needed for V1
- KV cache layout, decoder-layer executor, generation session, and greedy sampler
- C++ and C ABI handles for create/load/prefill/decode/destroy/error retrieval
- Unit tests, fixture assets, benchmark coverage, and sanitizer-ready build/test paths

### Definition of Done (verifiable conditions with commands)
- `cmake -S . -B build-v1 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON && cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntime.*`
- `./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=SafetensorsLoader.*`
- `./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=LlamaDecode.*`
- `./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=KVCache.*`
- `./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=CAbiRuntime.*`
- `cmake -S . -B build-v1-tsan -DENABLE_TSAN=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF && cmake --build build-v1-tsan --target aethermind_unit_tests -j && ./build-v1-tsan/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntimeThreading.*`
- `cmake --build build-v1 --target aethermind_benchmark -j && ./build-v1/tests/benchmark/aethermind_benchmark --benchmark_filter=BM_LlamaDecode|BM_KVCache|BM_QuantLinear`

### Must Have
- Single-process, single-model, single-request runtime for V1
- Core API boundary is token-IDs in / token-IDs out; text tokenizer support is optional adapter surface
- Canonical input is HuggingFace `config.json` + Safetensors; unsupported assets fail with explicit errors
- Deterministic greedy decoding as the required baseline sampler
- One exact loader contract, one exact KV-cache layout, one exact dispatch matrix for V1 CPU
- No steady-state allocations during decode after warmup for the supported single-request path

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- No HTTP/gRPC server, OpenAI-compatible API, or SSE streaming in V1
- No multi-request scheduler, continuous batching, chunked prefill, prefix cache, LoRA, speculative decoding, or distributed execution in V1
- No generic graph IR or general graph executor before direct decoder execution works
- No GPU/CUDA/CANN backend implementation in V1
- No "support all Llama variants" ambiguity; only named config fields and rejection rules are in scope
- No vague quantization support; V1 must ship exact kernel/layout support and reject everything else clearly

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: tests-after using existing GoogleTest + Google Benchmark infrastructure
- QA policy: Every task includes agent-executed happy-path and failure-path scenarios
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`
- Required fixture policy: create tiny deterministic fixtures for config parsing, safetensors header parsing, weight-name mapping, one-layer decode, KV-cache parity, and ABI smoke tests
- Hardening policy: add ASAN/UBSAN CMake support during the plan even though the repo currently only exposes TSAN at top level

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. Extract shared dependencies into Wave 1 to maximize parallelism.

Wave 1: V1 scope contract, asset ingestion contract, CPU runtime context, dispatch skeleton, verification fixtures
Wave 2: reference kernels, loader/repack, KV cache, decoder execution, generation session
Wave 3: C ABI, INT4 extension, tokenizer adapter, CI/sanitizer/benchmark hardening, documentation/examples

### Dependency Matrix (full, all tasks)
| Task | Blocks | Blocked By |
|------|--------|------------|
| 1 | 2, 3, 4, 5 | - |
| 2 | 8, 14 | 1 |
| 3 | 6, 7, 8, 9, 11, 12, 13 | 1 |
| 4 | 6, 7, 10, 13 | 1 |
| 5 | 6, 8, 9, 10, 11, 12, 13, 14, 15 | 1 |
| 6 | 10 | 3, 4, 5 |
| 7 | 10, 13 | 3, 4, 5 |
| 8 | 10, 11, 14 | 2, 3, 5 |
| 9 | 10, 11 | 3, 5 |
| 10 | 11 | 4, 5, 6, 7, 8, 9 |
| 11 | 12, 13, 14, 15 | 3, 5, 8, 9, 10 |
| 12 | 15 | 3, 5, 11 |
| 13 | 15 | 3, 4, 5, 7, 11 |
| 14 | 15 | 2, 5, 11 |
| 15 | F1, F2, F3, F4 | 5, 12, 13, 14 |

### Agent Dispatch Summary (wave -> task count -> categories)
- Wave 1 -> 5 tasks -> `unspecified-high`, `deep`, `writing`
- Wave 2 -> 5 tasks -> `deep`, `ultrabrain`, `unspecified-high`
- Wave 3 -> 5 tasks -> `deep`, `quick`, `writing`, `unspecified-high`

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Freeze V1 runtime contract and module layout

  **What to do**: Create the V1 runtime module boundaries under `include/` and `src/` for config, loader, runtime context, kernels, kv-cache, model, and C ABI wrappers. Define the canonical V1 scope in code-facing headers: single-process, single-model, single-request, token-IDs in/out, CPU-only, synchronous generation. Add a small `README`-style architecture note under `.sisyphus/` only if needed for executor guidance, but keep the source tree authoritative through header contracts.
  **Must NOT do**: Do not add serving APIs, scheduler abstractions, graph IR, CUDA/CANN placeholders, or generic multi-backend scaffolding in this task.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: sets architectural boundaries and file layout that unblock all following tasks
  - Skills: `[]` - no external skill is required beyond repo-aware implementation
  - Omitted: `[playwright]` - no browser or UI work exists in this repo

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 2, 3, 4, 5 | Blocked By: -

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `include/tensor.h:12` - use the existing thin facade style for public runtime handles and value types
  - Pattern: `include/tensor_impl.h:88` - follow the repo's split between facade objects and lower-level impl/state objects
  - Pattern: `include/function.h:94` - mirror Object/ObjectRef-based implementation ownership for callable/runtime components
  - Pattern: `include/c_api.h:16` - reuse opaque-handle thinking for C ABI surface design
  - Pattern: `src/CMakeLists.txt:1` - source discovery is recursive under `src/`, so subdirectories are acceptable
  - Test: `tests/unit/test_tensor.cpp:22` - follow current simple GoogleTest naming style

  **Acceptance Criteria** (agent-executable only):
  - [ ] New runtime headers/sources compile under the existing root `src/*.cpp` recursive build without adding a second library target
  - [ ] A new contract-focused unit test target path exists and passes `--gtest_filter=InferenceRuntimeContract.*`
  - [ ] Public headers explicitly encode V1 scope boundaries and omit serving/distributed/GPU APIs

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Contract headers compile and expose only V1 scope
    Tool: Bash
    Steps: cmake -S . -B build-v1 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF && cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntimeContract.*
    Expected: Build succeeds; contract tests pass; no server/scheduler symbols are required for linkage
    Evidence: .sisyphus/evidence/task-1-runtime-contract.txt

  Scenario: Forbidden scope stays absent
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntimeContract.RejectsOutOfScopeFeatures
    Expected: Test passes by asserting that distributed, GPU, and streaming APIs are not exposed in V1 contracts
    Evidence: .sisyphus/evidence/task-1-runtime-contract-error.txt
  ```

  **Commit**: YES | Message: `feat(runtime): define v1 runtime contracts and layout` | Files: `include/`, `src/`, `tests/unit/`

- [ ] 2. Implement canonical HuggingFace model contract and rejection policy

  **What to do**: Implement a strict parser/validator for V1-supported model metadata. Accept only a documented subset of HuggingFace-style `config.json` fields needed for decoder-only Llama execution: `model_type`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `rms_norm_eps`, `rope_theta`, `max_position_embeddings`, `vocab_size`, `bos_token_id`, `eos_token_id`, and `tie_word_embeddings`. Freeze exact supported variants: decoder-only, RMSNorm, RoPE, GQA/MQA allowed, no MoE, no encoder-decoder, no sliding-window attention. Return explicit structured errors for unsupported assets.
  **Must NOT do**: Do not parse "best effort" configs, silently default missing required fields, or leave unsupported checkpoint behavior undefined.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: this locks the asset contract and rejection semantics used by loader, executor, and tests
  - Skills: `[]` - repo primitives are sufficient
  - Omitted: `[playwright]` - not relevant to backend parser work

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 8, 14 | Blocked By: 1

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `include/device.h:15` - current repo already models explicit enums plus validation; follow that style for supported/unsupported variants
  - Pattern: `include/function_schema.h:18` - use explicit schema-like validation rather than permissive maps
  - Pattern: `include/error.h` - use repo error reporting style for structured rejections
  - External: `https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py` - source of canonical Llama config field names and GQA semantics
  - External: `https://huggingface.co/docs/transformers/main/en/model_doc/llama` - field-level behavior and model-family expectations
  - Test: `tests/unit/test_device.cpp` - follow current "accept/reject explicit invariants" unit-test pattern

  **Acceptance Criteria** (agent-executable only):
  - [ ] `config.json` parser accepts one supported tiny Llama fixture and populates a typed runtime config object
  - [ ] Parser rejects unsupported fields/combinations with deterministic error messages and test coverage
  - [ ] Unit tests exist for tied embeddings, GQA/MQA, invalid `model_type`, and missing required fields

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Supported Llama config parses successfully
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=ModelConfigContract.AcceptsSupportedLlama
    Expected: Test passes; parsed config exposes exact expected hidden size, layer count, head count, and rope parameters
    Evidence: .sisyphus/evidence/task-2-model-contract.txt

  Scenario: Unsupported config is rejected cleanly
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=ModelConfigContract.RejectsUnsupportedVariant
    Expected: Test passes by confirming explicit rejection for MoE, encoder-decoder, or missing required fields
    Evidence: .sisyphus/evidence/task-2-model-contract-error.txt
  ```

  **Commit**: YES | Message: `feat(runtime): add llama config contract parser` | Files: `include/`, `src/`, `tests/unit/`

- [ ] 3. Build CPU runtime context and execution resources

  **What to do**: Implement CPU runtime context objects for feature detection, intra-op thread configuration, per-thread scratch allocation, request-local decode workspace, and warmup-time resource reservation. Define one exact policy for V1: synchronous API, intra-op parallelism only, no inter-request scheduler, and no steady-state decode allocations after warmup. Use existing storage/data-pointer infrastructure where ownership is required; use `ammalloc` only for transient/workspace buffers, not for memory-mapped model weights.
  **Must NOT do**: Do not introduce stream/event abstractions, async coroutine execution, or a generic device manager in V1.

  **Recommended Agent Profile**:
  - Category: `ultrabrain` - Reason: execution resources, allocation policy, and threading invariants affect every hot path
  - Skills: `[]` - reasoning-heavy repo work, no special external tool needed
  - Omitted: `[playwright]` - not relevant to systems/runtime work

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 6, 7, 8, 9, 11, 12, 13 | Blocked By: 1

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `include/memory/storage.h:13` - keep ownership wrapped in `ObjectRef`-style objects
  - Pattern: `include/memory/data_ptr.h:77` - reuse device-tagged pointer ownership semantics for runtime-owned buffers
  - Pattern: `ammalloc/include/ammalloc/ammalloc.h` - allocator entry points for transient/workspace allocations
  - Pattern: `ammalloc/include/ammalloc/thread_cache.h` - current repo already contains CPU allocator hot-path thinking; align workspace strategy with it
  - Test: `tests/unit/test_thread_cache.cpp` - follow concurrency-oriented test style for thread-sensitive logic
  - Test: `tests/benchmark/benchmark_memory_pool.cpp:23` - follow benchmark style for allocation/throughput measurements

  **Acceptance Criteria** (agent-executable only):
  - [ ] Runtime context exposes deterministic thread-count, scratch-size, and warmup configuration for single-request CPU execution
  - [ ] Decode warmup reserves all required steady-state workspace and later tests can assert zero decode-time allocations
  - [ ] Thread-safety tests verify independent execution contexts can share one loaded model safely on TSAN build

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Runtime context warmup stabilizes resource usage
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntimeThreading.WarmupPreallocatesDecodeResources
    Expected: Test passes and records no allocation-growth after warmup for the supported decode loop
    Evidence: .sisyphus/evidence/task-3-runtime-context.txt

  Scenario: Shared model with independent contexts stays race-free
    Tool: Bash
    Steps: cmake -S . -B build-v1-tsan -DENABLE_TSAN=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF && cmake --build build-v1-tsan --target aethermind_unit_tests -j && ./build-v1-tsan/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntimeThreading.SharedModelIndependentContexts
    Expected: TSAN build passes without reported races or crashes
    Evidence: .sisyphus/evidence/task-3-runtime-context-error.txt
  ```

  **Commit**: YES | Message: `feat(runtime): add cpu execution context and scratch policy` | Files: `include/`, `src/`, `tests/unit/`, `tests/benchmark/`

- [ ] 4. Implement the V1 kernel dispatch table and registration skeleton

  **What to do**: Build a minimal V1 dispatch surface that selects CPU kernel implementations at runtime based on exact `op_id + cpu_feature + quant_scheme` tuples and caches direct function pointers for hot use. Limit the operator surface to the kernels required for V1 decode: embedding gather, RMSNorm, RoPE, matmul/linear, attention score, attention value accumulation, softmax, SwiGLU, residual add, logits projection, and greedy sampling helpers. Reuse existing `Registry`/`Function` ideas only where they simplify init-time registration; do not force hot-path calls through boxed `Any` dispatch.
  **Must NOT do**: Do not attempt to finish a generic `Dispatcher` framework, dynamic boxing path, or multi-backend dispatch lattice in this task.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: must connect current repo registration ideas to a narrow fast-path dispatch table without overengineering
  - Skills: `[]` - internal repo references are sufficient
  - Omitted: `[playwright]` - no UI/browser scope

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 6, 7, 10, 13 | Blocked By: 1

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `include/dispatcher.h:31` - existing dispatcher is skeletal; use it as evidence of what NOT to overbuild first
  - Pattern: `include/registry.h:12` - existing registration API can inspire init-time registration shape
  - Pattern: `include/function.h:146` - current callable abstraction is useful for non-hot-path registration but too heavy for tight decode loops
  - Pattern: `include/dispatch_key.h` - if reused, keep dispatch keys narrowly scoped to CPU dense and chosen quant schemes only
  - Test: `tests/unit/test_function.cpp` - follow current callable/registration test patterns where applicable

  **Acceptance Criteria** (agent-executable only):
  - [ ] Dispatch table initialization resolves one concrete function pointer per required V1 op for the selected CPU feature set
  - [ ] Hot-path execution code can call cached function pointers without boxing through `Any`
  - [ ] Unsupported op/quant/cpu-feature combinations fail with explicit errors during initialization, not midway through decode

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Dispatch table resolves all required V1 operators
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=KernelDispatchTable.ResolvesRequiredCpuOps
    Expected: Test passes; all mandatory op slots are populated for the chosen V1 CPU backend
    Evidence: .sisyphus/evidence/task-4-dispatch-table.txt

  Scenario: Unsupported dispatch combinations fail at init time
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=KernelDispatchTable.RejectsUnsupportedQuantScheme
    Expected: Test passes by confirming deterministic initialization failure for unknown quant/op combinations
    Evidence: .sisyphus/evidence/task-4-dispatch-table-error.txt
  ```

  **Commit**: YES | Message: `feat(runtime): add narrow cpu dispatch table` | Files: `include/`, `src/`, `tests/unit/`

- [ ] 5. Add deterministic fixtures and verification harness for V1

  **What to do**: Create the test/fixture scaffolding the rest of the plan depends on: tiny supported `config.json` fixtures, tiny invalid config fixtures, minimal safetensors header fixtures, layer-weight mapping fixtures, deterministic token goldens, KV-cache parity fixtures, no-allocation probes, and ABI smoke harnesses. Define exact test suites and benchmark names now so later tasks only implement them, not invent them.
  **Must NOT do**: Do not use giant external checkpoints or flaky text-generation expectations in unit tests.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: this is verification architecture, not just test writing
  - Skills: `[]` - use current test/benchmark patterns only
  - Omitted: `[playwright]` - backend-only verification

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 | Blocked By: 1

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `tests/unit/CMakeLists.txt:1` - tests already compile into one `aethermind_unit_tests` binary; keep the same convention
  - Pattern: `tests/benchmark/CMakeLists.txt:1` - benchmarks live in one `aethermind_benchmark` target; extend it rather than inventing a new harness
  - Pattern: `tests/unit/test_tensor.cpp:97` - deterministic named test cases are the repo norm
  - Pattern: `tests/unit/test_storage.cpp` - use focused low-level tests for ownership and buffer invariants
  - Pattern: `tests/benchmark/benchmark_memory_pool.cpp:23` - follow current benchmark naming style `BM_*`
  - External: `https://github.com/huggingface/safetensors` - safetensors header/body invariants for fixture generation

  **Acceptance Criteria** (agent-executable only):
  - [ ] Fixture files exist for supported and rejected config/asset paths and are consumed by named tests
  - [ ] Golden token sequence, KV parity, ABI smoke, and no-allocation assertions are defined before decoder implementation begins
  - [ ] Benchmark placeholders exist for decode, KV cache, and quantized linear paths with deterministic CLI filters

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Fixture harness loads and validates all planned test assets
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=InferenceFixtures.*
    Expected: All fixture-loading tests pass and produce deterministic golden metadata
    Evidence: .sisyphus/evidence/task-5-fixtures.txt

  Scenario: Invalid fixture path is exercised explicitly
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=InferenceFixtures.RejectsMalformedAssets
    Expected: Test passes by confirming malformed configs/safetensors headers fail with exact expected errors
    Evidence: .sisyphus/evidence/task-5-fixtures-error.txt
  ```

  **Commit**: YES | Message: `test(runtime): add deterministic inference fixtures and harness` | Files: `tests/unit/`, `tests/benchmark/`, `tests/fixtures/`

- [ ] 6. Implement reference CPU decoder primitives

  **What to do**: Implement the non-quantized reference kernel surface required to prove decoder correctness: embedding gather, RMSNorm, RoPE application, residual add, softmax, attention score/value accumulation, SwiGLU, logits projection, and greedy argmax helper. Use a correctness-first path with float32 accumulation and deterministic math; this is the oracle path against which later quantized kernels are compared.
  **Must NOT do**: Do not optimize first, fuse aggressively, or mix correctness and quantization logic in the same kernel entry points.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: core math primitives must be correct before the model loop is assembled
  - Skills: `[]` - use current tensor/storage foundations only
  - Omitted: `[playwright]` - backend math only

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 10 | Blocked By: 3, 4, 5

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `include/tensor.h:12` - reuse existing tensor facade instead of inventing a second tensor abstraction
  - Pattern: `include/tensor_impl.h:69` - tensor/storage aliasing rules matter for view-safe kernel implementations
  - Pattern: `include/data_type.h` - existing dtype support should inform explicit runtime dtype checks
  - Pattern: `tests/unit/test_tensor.cpp` - add math-focused tests in the repo's existing style
  - External: `https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py` - reference semantics for RMSNorm, RoPE, attention, and SwiGLU order

  **Acceptance Criteria** (agent-executable only):
  - [ ] Reference kernels produce deterministic outputs for the tiny one-layer fixture path
  - [ ] Unit tests compare outputs against fixture goldens within explicit tolerances
  - [ ] Kernel entry points are callable through the V1 dispatch table without boxing

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Reference decoder primitives match golden outputs
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=ReferenceCpuKernels.*
    Expected: All primitive-kernel tests pass with deterministic outputs inside stated tolerances
    Evidence: .sisyphus/evidence/task-6-reference-kernels.txt

  Scenario: Invalid tensor shapes are rejected explicitly
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=ReferenceCpuKernels.RejectsInvalidShapes
    Expected: Test passes by confirming shape/stride mismatches fail with explicit errors
    Evidence: .sisyphus/evidence/task-6-reference-kernels-error.txt
  ```

  **Commit**: YES | Message: `feat(runtime): add reference cpu decoder kernels` | Files: `src/`, `include/`, `tests/unit/`

- [ ] 7. Implement INT8 weight-only linear kernels and dispatch entries

  **What to do**: Implement the first required quantized path for V1: INT8 weight-only linear kernels used by q/k/v, o_proj, gate/up/down projections, and lm_head where applicable. Freeze one exact internal packing/layout for INT8 weights, scales, and bias handling; register these kernels in the V1 dispatch table. Keep activations and accumulation behavior explicit and deterministic.
  **Must NOT do**: Do not add INT4 in this task, do not support multiple INT8 packing formats, and do not leave scale granularity ambiguous.

  **Recommended Agent Profile**:
  - Category: `ultrabrain` - Reason: quantized CPU kernels and layout choices are the first major performance/correctness fork in the plan
  - Skills: `[]` - implementation is repo-local but reasoning-heavy
  - Omitted: `[playwright]` - not relevant to compute kernels

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 10, 13 | Blocked By: 3, 4, 5

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `include/device.h:37` - V1 remains CPU-only; do not add other backend branching here
  - Pattern: `include/function.h:167` - current callable system is not the hot path; use direct kernel pointers after init-time dispatch
  - Pattern: `tests/benchmark/benchmark_memory_pool.cpp` - follow benchmark naming/invocation style for throughput benchmarks
  - External: `https://oneapi-src.github.io/oneDNN/dev_guide_matmul.html` - if oneDNN is chosen for baseline validation, use it only as a correctness/perf reference, not as the runtime architecture
  - External: `https://arxiv.org/abs/2210.17323` - LLM.int8 context for weight-only matmul expectations; implementation still must freeze one exact layout

  **Acceptance Criteria** (agent-executable only):
  - [ ] INT8 kernels pass numerical parity tests against the reference linear path within documented tolerances
  - [ ] Dispatch initialization selects INT8 kernels only for the supported quant scheme and fails otherwise
  - [ ] Benchmark entries exist for INT8 linear throughput on at least one fixture shape family

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: INT8 linear kernels match reference path
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=QuantLinearInt8.*
    Expected: All INT8 parity tests pass within the declared tolerances for supported shapes
    Evidence: .sisyphus/evidence/task-7-int8-kernels.txt

  Scenario: Unknown INT8 layout is rejected at registration/init
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=QuantLinearInt8.RejectsUnknownPacking
    Expected: Test passes by confirming unsupported packing/layout values fail deterministically
    Evidence: .sisyphus/evidence/task-7-int8-kernels-error.txt
  ```

  **Commit**: YES | Message: `feat(runtime): add int8 weight-only cpu linear kernels` | Files: `src/`, `include/`, `tests/unit/`, `tests/benchmark/`

- [ ] 8. Implement Safetensors asset reader, shard resolver, and weight repack store

  **What to do**: Implement the canonical asset ingestion path for V1. Read HuggingFace model directories, resolve one or more Safetensors shards, validate headers and tensor metadata, mmap or buffered-load the raw weight storage, and repack tensors into the exact internal CPU layout expected by Tasks 7, 10, and 11. Support explicit tensor-name mapping for the chosen Llama-family contract and handle tied embeddings deterministically.
  **Must NOT do**: Do not support PyTorch pickle weights, GGUF, or ambiguous tensor-name remapping heuristics.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: asset loading correctness and internal layout choices gate the whole runtime
  - Skills: `[]` - no special external skill needed
  - Omitted: `[playwright]` - not applicable

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 10, 11, 14 | Blocked By: 2, 3, 5

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `include/memory/data_ptr.h:77` - follow explicit ownership/lifetime rules for mapped/repacked buffers
  - Pattern: `include/memory/storage.h:17` - storage wrappers already support pre-allocated backing memory
  - Pattern: `include/object.h` - reuse existing Object/ObjectRef ownership model for loaded assets/manifests
  - External: `https://github.com/huggingface/safetensors/blob/main/README.md#format` - canonical file format for header length, JSON header, offsets, and padding behavior
  - External: `https://huggingface.co/docs/safetensors/index` - practical safetensors constraints and metadata expectations
  - Test: `tests/unit/test_storage.cpp` - follow ownership/lifetime validation patterns

  **Acceptance Criteria** (agent-executable only):
  - [ ] Loader accepts a supported tiny Safetensors fixture, validates offsets/shapes/dtypes, and builds a typed manifest
  - [ ] Loader rejects malformed headers, unsupported dtypes, missing tensors, and conflicting shard manifests with explicit errors
  - [ ] Repack store exposes deterministic internal buffers for the decoder path without relying on on-demand decode-time format conversion

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Supported safetensors asset set loads and repacks correctly
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=SafetensorsLoader.*
    Expected: Loader tests pass; manifest, shard resolution, and repack layout match the fixture goldens
    Evidence: .sisyphus/evidence/task-8-safetensors-loader.txt

  Scenario: Malformed safetensors metadata fails cleanly
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=SafetensorsLoader.RejectsMalformedHeader
    Expected: Test passes by confirming deterministic rejection for invalid header size, offsets, or dtype metadata
    Evidence: .sisyphus/evidence/task-8-safetensors-loader-error.txt
  ```

  **Commit**: YES | Message: `feat(runtime): add safetensors asset loader and repack store` | Files: `src/`, `include/`, `tests/unit/`, `tests/fixtures/`

- [ ] 9. Implement KV cache layout and request state lifecycle

  **What to do**: Implement one exact KV-cache design for V1 CPU decode: contiguous preallocated storage partitioned by layer, K/V head grouping, sequence position, and head dimension. Add request-state objects that own current position, decoded token count, scratch references, and KV append/update helpers. Provide a reference no-cache path so parity tests can compare cached decode against recomputation.
  **Must NOT do**: Do not add paged attention, sliding window, eviction, prefix sharing, or multi-request cache management in V1.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: cache layout correctness and lifecycle semantics are central to decode stability
  - Skills: `[]` - repo-local systems work
  - Omitted: `[playwright]` - not relevant

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 10, 11 | Blocked By: 3, 5

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `include/memory/storage.h:13` - KV buffers should use existing storage ownership patterns
  - Pattern: `include/tensor_impl.h:76` - tensor views may alias backing storage; exploit that for cache slices carefully
  - Pattern: `ammalloc/include/ammalloc/ammalloc.h` - transient scratch is separate from persistent KV backing
  - External: `https://huggingface.co/docs/transformers/main/en/cache_explanation` - conceptual KV-cache semantics; implementation still needs a fixed V1 layout
  - Test: `tests/unit/test_storage.cpp` - reuse low-level storage correctness style for backing-buffer checks

  **Acceptance Criteria** (agent-executable only):
  - [ ] KV cache append/update API stores and retrieves per-layer K/V slices deterministically for supported shapes
  - [ ] Cached decode produces the same token outputs as a no-cache reference path on the tiny fixture
  - [ ] Runtime asserts zero cache growth beyond preallocated capacity for the configured max sequence length

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: KV cache append and parity tests pass
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=KVCache.*
    Expected: KV cache tests pass, including parity against a no-cache decode reference
    Evidence: .sisyphus/evidence/task-9-kv-cache.txt

  Scenario: KV cache capacity overflow is rejected explicitly
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=KVCache.RejectsCapacityOverflow
    Expected: Test passes by confirming max-context overflow fails with a deterministic error instead of corrupting memory
    Evidence: .sisyphus/evidence/task-9-kv-cache-error.txt
  ```

  **Commit**: YES | Message: `feat(runtime): add kv cache layout and request state` | Files: `src/`, `include/`, `tests/unit/`

- [ ] 10. Assemble the Llama decoder block executor

  **What to do**: Implement the handwritten decoder execution path for one supported Llama-family block and the full stack of blocks: input embedding, per-layer RMSNorm -> qkv projections -> RoPE -> attention -> output projection -> post-attention residual -> FFN/SwiGLU -> residual -> final norm -> lm_head. Support both prefill and single-token decode using the exact kernels and KV-cache contract from earlier tasks. Make GQA/MQA behavior explicit from the parsed config.
  **Must NOT do**: Do not add a general graph runtime, dynamic graph builder, or non-Llama layer abstractions in this task.

  **Recommended Agent Profile**:
  - Category: `ultrabrain` - Reason: this is the first full model-execution assembly point and the highest-risk correctness task
  - Skills: `[]` - internal runtime/kernels only
  - Omitted: `[playwright]` - backend-only work

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 11 | Blocked By: 4, 5, 6, 7, 8, 9

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `include/tensor.h:12` - use existing tensor objects as execution values
  - Pattern: `include/function_schema.h:130` - if schemas are used for internal validation, keep them narrow and explicit
  - Pattern: `include/dispatcher.h:31` - do not route this through the unfinished generic dispatcher
  - External: `https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py` - canonical forward-order reference for Llama blocks
  - External: `https://arxiv.org/abs/2302.13971` - LLaMA architecture reference for RMSNorm/RoPE/GQA assumptions

  **Acceptance Criteria** (agent-executable only):
  - [ ] One-layer and full-stack decoder execution tests pass against deterministic fixture goldens
  - [ ] Prefill and single-token decode both execute through the same runtime contract with explicit path coverage
  - [ ] Unsupported config branches required by the executor fail during model initialization, not mid-run

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Llama decoder block and stack execute deterministically
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=LlamaDecode.*
    Expected: Prefill and decode tests pass with exact or tolerance-bounded fixture goldens
    Evidence: .sisyphus/evidence/task-10-llama-decode.txt

  Scenario: Unsupported executor config fails before decode
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=LlamaDecode.RejectsUnsupportedExecutorConfig
    Expected: Test passes by confirming unsupported rope/head/layout combinations fail at initialization
    Evidence: .sisyphus/evidence/task-10-llama-decode-error.txt
  ```

  **Commit**: YES | Message: `feat(runtime): assemble llama decoder executor` | Files: `src/`, `include/`, `tests/unit/`

- [ ] 11. Implement model assembly, synchronous generation session, and greedy decode loop

  **What to do**: Build the runtime model object that binds parsed config, repacked weights, dispatch table, and KV-cache policy into one executable artifact. Implement synchronous prefill/decode session objects with token-IDs in/token-IDs out semantics, max-length/eos stop rules, deterministic greedy sampling, and explicit warmup hooks. This is the V1 success-path task: stable single-request generation on CPU.
  **Must NOT do**: Do not add beam search, top-k/top-p, speculative decode, streaming callbacks, or concurrent request scheduling.

  **Recommended Agent Profile**:
  - Category: `ultrabrain` - Reason: this is the highest-level integration point and the official V1 success criterion
  - Skills: `[]` - no additional external tool needed
  - Omitted: `[playwright]` - backend systems work only

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: 12, 14, 15 | Blocked By: 3, 5, 8, 9, 10

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `include/object.h` - runtime model/session ownership should follow repo intrusive-object conventions where appropriate
  - Pattern: `include/function.h:189` - if any internal global callable lookup remains, keep it out of the hot decode path
  - Pattern: `include/c_api.h:40` - lifetime-sensitive operations should remain explicit and handle-based
  - Test: `tests/unit/test_function.cpp` - follow direct call/invocation validation style for runtime entry points
  - External: `https://huggingface.co/docs/transformers/main/en/generation_strategies` - only greedy baseline behavior matters for V1; use this as semantics reference, not as scope expansion

  **Acceptance Criteria** (agent-executable only):
  - [ ] A supported tiny fixture model performs deterministic prompt -> prefill -> decode -> eos stop through the C++ runtime API
  - [ ] Warmup followed by decode asserts no steady-state allocations in the supported path
  - [ ] Golden-token tests exist for at least one deterministic prompt fixture

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: End-to-end greedy generation succeeds on CPU
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntime.EndToEndGreedyGeneration
    Expected: Test passes and emits the exact expected token sequence for the supported tiny fixture model
    Evidence: .sisyphus/evidence/task-11-generation-session.txt

  Scenario: Decode loop rejects invalid stop/config settings
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntime.RejectsInvalidGenerationConfig
    Expected: Test passes by confirming invalid max length or eos configuration fails cleanly before generation
    Evidence: .sisyphus/evidence/task-11-generation-session-error.txt
  ```

  **Commit**: YES | Message: `feat(runtime): add synchronous llama generation session` | Files: `src/`, `include/`, `tests/unit/`

- [ ] 12. Expose the stable C++ API and thin C ABI

  **What to do**: Publish the V1 embedding surface around opaque runtime/model/session handles. The C++ API may use rich types internally, but the C ABI must expose create/load/prefill/decode/destroy/error-retrieval operations with explicit ownership and buffer contracts. Keep the ABI thin over the C++ runtime rather than inventing a second implementation. Make token IDs the ABI payload; tokenizer/text handling remains outside the ABI core.
  **Must NOT do**: Do not expose templates, STL containers, or tokenizer/text semantics directly through the C ABI.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: this is a product boundary and must balance ergonomics, lifetime safety, and future extensibility
  - Skills: `[]` - repo-local API design
  - Omitted: `[playwright]` - not relevant

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 15 | Blocked By: 3, 5, 11

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `include/c_api.h:16` - existing repo already uses opaque handles and explicit refcount/lifetime hooks
  - Pattern: `include/object.h` - reuse ownership semantics where possible instead of inventing ad hoc lifetime rules
  - Pattern: `include/container/array.h` - if any internal dynamic arrays are needed, keep them inside C++ only
  - Test: `tests/unit/test_object.cpp` - follow lifetime/reference management validation patterns

  **Acceptance Criteria** (agent-executable only):
  - [ ] C++ API and C ABI both support create/load/prefill/decode/destroy for the supported V1 path
  - [ ] ABI tests verify deterministic token output, explicit error retrieval, and clean teardown
  - [ ] No ABI symbol requires callers to understand internal C++ object layouts

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: C ABI smoke path works end-to-end
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=CAbiRuntime.*
    Expected: ABI smoke tests pass for create/load/prefill/decode/destroy and match C++ runtime token output
    Evidence: .sisyphus/evidence/task-12-c-abi.txt

  Scenario: ABI reports errors without leaking resources
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=CAbiRuntime.ReportsLoadFailureCleanly
    Expected: Test passes by confirming invalid model path/config surfaces a deterministic error and frees all temporary resources
    Evidence: .sisyphus/evidence/task-12-c-abi-error.txt
  ```

  **Commit**: YES | Message: `feat(runtime): expose stable cpp and c runtime apis` | Files: `include/`, `src/`, `tests/unit/`

- [ ] 13. Add the INT4 weight-only extension under the same runtime contract

  **What to do**: Extend the V1 runtime to support one exact INT4 weight-only packing/layout for the same linear surfaces already covered by INT8. Reuse the same dispatch table, loader manifest, and model assembly flow. Keep INT4 gated strictly behind the chosen scheme; unsupported INT4 variants must fail during load/registration.
  **Must NOT do**: Do not add multiple INT4 layouts, mixed activation quantization, or separate model/runtime paths for INT4.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: must extend quantization coverage without destabilizing the now-working V1 runtime contract
  - Skills: `[]` - no special external skill needed
  - Omitted: `[playwright]` - compute/runtime only

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 15 | Blocked By: 3, 4, 5, 7, 11

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `include/device.h:37` - preserve CPU-only V1 backend assumptions
  - Pattern: `tests/benchmark/benchmark_memory_pool.cpp` - use benchmark style consistent with repo for quant-kernel perf checks
  - External: `https://huggingface.co/docs/transformers/main/en/quantization/concept_guide` - general quantization semantics reference; implementation still must freeze one exact scheme
  - External: `https://github.com/huggingface/safetensors` - quantized weights still arrive through the same asset path and must preserve metadata integrity

  **Acceptance Criteria** (agent-executable only):
  - [ ] INT4 path loads, registers, and executes through the same model/session API as INT8
  - [ ] INT4 parity tests pass against the reference path within explicit tolerances for supported fixtures
  - [ ] Unsupported INT4 metadata/layout combinations fail before decode begins

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: INT4 path executes through the same runtime contract
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=QuantLinearInt4.*
    Expected: INT4 tests pass for load, dispatch, and decode parity on the supported fixture path
    Evidence: .sisyphus/evidence/task-13-int4-extension.txt

  Scenario: Unsupported INT4 metadata is rejected before execution
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=QuantLinearInt4.RejectsUnsupportedPacking
    Expected: Test passes by confirming invalid INT4 packing/layout fails at load/registration time
    Evidence: .sisyphus/evidence/task-13-int4-extension-error.txt
  ```

  **Commit**: YES | Message: `feat(runtime): add int4 weight-only extension` | Files: `src/`, `include/`, `tests/unit/`, `tests/benchmark/`

- [ ] 14. Add the optional tokenizer adapter and text convenience example

  **What to do**: Implement a thin text adapter layer outside the core runtime that converts text to token IDs and back for the supported V1 demonstration path. Keep the core runtime token-based. Support one exact tokenizer asset strategy for V1 text demos: SentencePiece-compatible `tokenizer.model` first; reject unsupported tokenizer asset types with explicit errors. Provide one end-to-end example that loads model + tokenizer assets, tokenizes a prompt, runs generation, and detokenizes output through the C++ API only.
  **Must NOT do**: Do not make tokenizer support a prerequisite for the core C ABI or expand V1 into a general tokenizer framework.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: tokenization is a hidden scope bomb unless kept thin and explicit
  - Skills: `[]` - repo code plus official tokenizer references are enough
  - Omitted: `[playwright]` - not applicable

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: 15 | Blocked By: 2, 5, 8, 11

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `include/c_api.h:16` - tokenizer stays outside the C ABI core; keep the ABI token-based
  - Pattern: `include/container/string.h:45` - string ownership/passing in C++ should follow existing repo conventions
  - External: `https://github.com/google/sentencepiece` - official SentencePiece reference for `tokenizer.model`
  - External: `https://huggingface.co/docs/tokenizers/index` - tokenizer asset semantics and pitfalls
  - External: `https://github.com/huggingface/tokenizers` - tokenizer JSON/model metadata reference; unsupported forms must be rejected clearly in V1

  **Acceptance Criteria** (agent-executable only):
  - [ ] Example code demonstrates prompt text -> token IDs -> runtime generation -> detokenized output through the C++ API
  - [ ] Tokenizer adapter rejects unsupported tokenizer assets with explicit errors
  - [ ] Core runtime tests remain token-based and do not depend on tokenizer availability

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Text convenience example works with supported tokenizer asset
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=TokenizerAdapter.*
    Expected: Tokenizer adapter tests pass and example prompt round-trip is deterministic on the tiny fixture
    Evidence: .sisyphus/evidence/task-14-tokenizer-adapter.txt

  Scenario: Unsupported tokenizer asset fails cleanly
    Tool: Bash
    Steps: cmake --build build-v1 --target aethermind_unit_tests -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=TokenizerAdapter.RejectsUnsupportedAsset
    Expected: Test passes by confirming unsupported tokenizer files are rejected before runtime execution begins
    Evidence: .sisyphus/evidence/task-14-tokenizer-adapter-error.txt
  ```

  **Commit**: YES | Message: `feat(runtime): add tokenizer adapter and text example` | Files: `src/`, `include/`, `tests/unit/`, `examples/`

- [ ] 15. Harden V1 with CI, sanitizer variants, benchmarks, and end-to-end regression gates

  **What to do**: Add the engineering hardening required to keep V1 stable after integration: GitHub Actions CI, CTest integration or an equivalent explicit test runner contract, ASAN/UBSAN CMake options alongside existing TSAN support, benchmark registration for decode/KV/int8/int4 paths, and a regression gate that fails on golden-token, ABI-smoke, or no-allocation regressions. Update project docs only as needed to point contributors at the new runtime build/test commands.
  **Must NOT do**: Do not add performance promises without benchmark baselines, and do not hide failing tests behind optional/manual-only scripts.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: final stabilization spans build system, tests, perf, and contributor workflow
  - Skills: `[]` - repo-native build/test work
  - Omitted: `[playwright]` - not relevant

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: F1, F2, F3, F4 | Blocked By: 5, 11, 12, 13, 14

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `CMakeLists.txt:15` - existing options style for test/benchmark/sanitizer toggles
  - Pattern: `tests/unit/CMakeLists.txt:1` - existing one-binary test organization to preserve or extend deliberately
  - Pattern: `tests/benchmark/CMakeLists.txt:1` - current benchmark target organization
  - Pattern: `AGENTS.md:29` - current documented build/test commands should be updated only if implementation changes them
  - Test: `tests/benchmark/benchmark_memory_pool.cpp:23` - keep benchmark registration style consistent with current repo

  **Acceptance Criteria** (agent-executable only):
  - [ ] CI runs build + unit-test + selected benchmark sanity checks for the V1 runtime path
  - [ ] ASAN/UBSAN and TSAN variants exist as documented build options and at least one runtime-focused test suite passes under them
  - [ ] Regression gates fail on golden-token drift, ABI smoke failures, and decode allocation regressions

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Hardened build/test matrix passes for V1 runtime
    Tool: Bash
    Steps: cmake -S . -B build-v1 -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON && cmake --build build-v1 -j && ./build-v1/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntime.*:CAbiRuntime.*:TokenizerAdapter.* && ./build-v1/tests/benchmark/aethermind_benchmark --benchmark_filter=BM_LlamaDecode|BM_KVCache|BM_QuantLinear
    Expected: Build, unit tests, and benchmark sanity checks all pass for the supported V1 runtime path
    Evidence: .sisyphus/evidence/task-15-hardening.txt

  Scenario: Sanitizer builds catch runtime regressions
    Tool: Bash
    Steps: cmake -S . -B build-v1-asan -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF -DENABLE_ASAN=ON -DENABLE_UBSAN=ON && cmake --build build-v1-asan --target aethermind_unit_tests -j && ./build-v1-asan/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntime.* && cmake -S . -B build-v1-tsan -DENABLE_TSAN=ON -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF && cmake --build build-v1-tsan --target aethermind_unit_tests -j && ./build-v1-tsan/tests/unit/aethermind_unit_tests --gtest_filter=InferenceRuntimeThreading.*
    Expected: Sanitizer-targeted test suites pass without memory or race diagnostics
    Evidence: .sisyphus/evidence/task-15-hardening-error.txt
  ```

  **Commit**: YES | Message: `chore(runtime): harden v1 with ci sanitizers and regressions` | Files: `.github/workflows/`, `CMakeLists.txt`, `tests/unit/`, `tests/benchmark/`, `docs/`, `AGENTS.md`

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [ ] F1. Plan Compliance Audit - oracle
- [ ] F2. Code Quality Review - unspecified-high
- [ ] F3. Real Manual QA - unspecified-high
- [ ] F4. Scope Fidelity Check - deep

## Commit Strategy
- Commit 1: V1 contracts, build targets, fixtures, and verification plumbing
- Commit 2: CPU runtime context, dispatch table, and reference kernel surface
- Commit 3: loader/repack, KV cache, decoder execution, and greedy generation
- Commit 4: C ABI, tokenizer adapter, and examples
- Commit 5: INT4 extension, CI/sanitizer, benchmark, and hardening

## Success Criteria
- A supported tiny fixture model and one supported real Llama-family quantized checkpoint both load through the canonical HF asset path
- The runtime performs deterministic greedy generation on CPU with KV cache enabled and produces the expected golden token sequence
- The decode loop performs no steady-state allocations after warmup in the supported single-request flow
- The C++ API and C ABI both pass create/load/prefill/decode/destroy smoke tests
- Benchmarks exist for decode, KV-cache access, and quantized linear kernels, and regression thresholds are recorded
