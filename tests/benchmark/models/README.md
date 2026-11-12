# Model benchmark measurements

The Linux-only `aethermind_model_benchmark` developer executable measures the CPU Llama path with FP32 activations and selectable F32/F16/BF16 KV, including the retained W16 weight policies. It separates internal prepared execution diagnostics from the public `InferenceSession::Generate` interval. It does not add a product runner or change inference APIs.

The measurement rules follow the [operator workflow](../../../docs/guides/operator-development-workflow.md) and [quality plan Q4](../../../docs/improvement-plan/13-deepinfer-quality-and-validation.md). Results describe the specified model, build, machine, workload, and measurement boundary; they do not establish model quality or an optimization benefit by themselves.

## Build and prepare assets

Run commands from the repository root. Prepare the checked [numerical fixtures](../../fixtures/inference_numerical/README.md) before the examples below.

```bash
cmake --preset benchmark-model
cmake --build --preset benchmark-model

cmake --preset benchmark-model-baseline
cmake --build --preset benchmark-model-baseline
```

Both presets use Release, no sanitizer, `BUILD_BENCHMARKS=ON`, `BUILD_TESTS=OFF`, and build only `aethermind_model_benchmark`. The model executable links AetherMind and the existing spdlog dependency; it needs neither GTest nor Google Benchmark at link time. The current `BUILD_BENCHMARKS` option also configures the existing Google Benchmark dependency and target. Dependency acquisition therefore remains part of configuration, even when building only the model executable.

The baseline preset excludes optional AVX2/FMA implementations. An O2/packed workload can still use applicable fallback/identity paths; the preset name does not guarantee that a particular optimized kernel is selected.

The current probe uses Linux process RSS accounting, and the artifact driver resolves `libAetherMind.so` with `ldd`. Other platforms do not create the model target.

## Capture a timing run

Use the artifact driver to collect model/executable/library identities and launch one isolated probe process:

```bash
python3 tools/run_model_benchmark.py \
  --binary build/validation/benchmark-model/tests/benchmark/aethermind_model_benchmark \
  --model .models/inference-numerical/mha-untied/model \
  --prompt 1,7,3 --max-new-tokens 3 --preset o1-plain \
  --warmup 1 --repetitions 20 --mode timing --variant o1-baseline
```

The driver prints the accepted `report.json` path. The default output is:

```text
benchmark-results/models/YYYYMMDDTHHMMSSZ_<git-sha>_<host-id>_<variant>-<nonce>/
```

Timing runs preserve `invocation.json`, `source.patch`, `machine.json`, `dynamic-libraries.txt`, `engine.stdout`, `engine.stderr`, `engine.json`, `report.json`, `summary.json`, and `status.json`. Diagnostic runs omit `summary.json`. Failed launches/validation preserve the available diagnostic artifacts and a failed status; they do not create an accepted report. `--output-root` can select another artifact root. Keep large raw outputs in gitignored storage or a retained CI artifact.

The driver reads model files to compute their identities before launching the probe. The recorded cache policy is `pre_read_by_identity_collection`: loading measurements are **not cold filesystem-cache startup measurements**. It checks model, binary, linked library, and relevant source identities again after execution. Avoid rebuilding or editing those files during capture.

`--weights` selects the source weight policy: `require-fp32` (default), `convert-fp32`, `keep-fp16`, or `keep-bf16`. `convert-fp32` widens all required weights before compilation; the retained W16 policies keep F16/BF16 matrix weights, widen only Norm weights to FP32, and execute with FP32 activations and the selected KV dtype (F32 by default), mirroring the production `ExecutableModel::Load` orchestration. The retained-W16 path supports CPU O1/plain and O2/identity-packed; O2 additionally holds identity-packed two-byte artifacts. W16 measurement fixtures can come from `.models/inference-admission` (for example `f16-auto` or `bf16-auto`):

```bash
python3 tools/run_model_benchmark.py \
  --binary build/validation/benchmark-model/tests/benchmark/aethermind_model_benchmark \
  --model .models/inference-admission/f16-auto \
  --prompt 1,7,3 --max-new-tokens 3 --preset o1-plain --weights keep-fp16 \
  --warmup 1 --repetitions 20 --mode timing --variant w16-f16
```

The policy is recorded in the workload and participates in comparison identity; runs with different weight policies are not comparable.

`--kv-dtype f32|f16|bf16` independently selects KV storage and defaults to `f32`.
F16/BF16 KV requires `--weights keep-fp16` or `keep-bf16`; all four W16/KV16
combinations support the verified O1/plain and O2/identity-packed paths. The same
KV dtype drives physical Runtime allocation and graph state specs. For example:

```bash
python3 tools/run_model_benchmark.py \
  --binary build/tests/benchmark/aethermind_model_benchmark \
  --model .models/kv16-acceptance/mha-near-limit-wf16 \
  --prompt 1,7,3 --max-new-tokens 3 --preset o2-packed \
  --weights keep-fp16 --kv-dtype bf16 \
  --warmup 1 --repetitions 3 --mode timing --variant p3-wf16-kvbf16-o2
```

New engine documents use `schema_version: 2` and require explicit
`workload.kv_dtype` plus `memory.prefill_activation_bytes` and
`memory.decode_activation_bytes`. The enclosing artifact report stays at schema version 1.
The checker continues accepting historical engine schema 1 only with its original
fields and interprets its KV as F32. Comparison identity normalizes that legacy
representation to explicit `f32`; schema 2 cannot omit the field, and an engine
schema 1 report cannot relabel itself as KV16. Different KV dtypes are different
workloads even if row padding makes their physical byte counts equal.

Identity collection includes all index-referenced shards, including nested paths, and rejects symlink/path escapes. The environment contract records allocator settings such as `THP_MEM_ALLOC_ENABLE` as well as loader/thread settings. Empty recorded values mean the input variable was unset; library defaults remain tied to the recorded binary/source identity. Machine metadata records visible CPU, OS/kernel, affinity, memory, NUMA and available governor/turbo information; unavailable virtualization details do not become hardware claims.

Each warmup and measured sample uses the same fixed order:

```text
Load → Runtime build → (policy conversion when --weights != require-fp32)
     → Compile → Prepare → internal prepared Prefill/Decode
     → request resource query (untimed) → public Generate
```

Every sample owns a fresh Runtime/model. Runtime stays at its final address while prepared kernels borrow it. The public Generate call starts a fresh logical request, but follows the internal diagnostic in that sample and therefore sees already-used model/cache pages. This order must remain the same when comparing runs. Warmups use the complete pipeline and are excluded from the numbered measured samples; every requested measured repetition is retained.

## Timing boundaries and counts

| Field | Included interval |
|---|---|
| `preparation.load_ns` | `ModelLoader::Load`; later benchmark admission checks and raw-byte inspection are outside it |
| `preparation.compile_ns` | `ModelCompiler::Compile` |
| `preparation.runtime_ns` | `RuntimeBuilder::TryBuild`, including the configured KV storage |
| `preparation.conversion_ns` | `ConvertRawModelWeightsToFp32` with the policy scope and backend host conversion callbacks, including its validation and owned payload allocation; null for `require-fp32`. Post-conversion admission is enforced by Compile/Prepare, which are measured separately |
| `preparation.prepare_ns` | `PrepareExecutableModel`, including packing and plan/binding preparation |
| `internal.request_setup_ns` | Reservation, request inputs/positions, output capacity, workspace, and Prefill bindings/context |
| `internal.first_token_request_ns` | From reservation setup through Prefill execution and reading the first token; excludes prior plan/allocator lookup and diagnostic-array allocation |
| `internal.prefill_execute_ns` | Workspace reset and `Executor::Execute`, including the model's Argmax kernel; excludes reading the token |
| `internal.decode_setup_ns` | Clearing Prefill context, creating stable one-token inputs, and binding the Decode context once |
| `internal.decode_steps[].execute_ns` | One workspace reset and prepared Decode execution; excludes token reading/feedback, loop bookkeeping, and report storage |
| `generate.duration_ns` | The public `InferenceSession::Generate` call, including its request preparation, execution, owning output, and cleanup; Session creation is outside it |

The internal diagnostic is not the public Generate implementation. It prepares the Decode context once, then updates stable token/position storage. Input binding, snapshot creation, JSON serialization, and logging are outside Execute-only intervals. Summing Decode Execute intervals gives executed model time, not the complete continuous Decode-loop latency. The public Generate interval supplies the complete request measurement.

For an actual output length `G > 0`, the first token comes from Prefill, Decode executes `G - 1` times, and final committed KV is `prompt_length + G - 1`. The last returned token has not yet been appended to KV. Every step records its input/output token and `kv_before → kv_after`; statistics keep different KV lengths separate.

Optional `--eos-token ID` stops at the first generated EOS and includes it in the output. Rates use actual returned tokens, not `--max-new-tokens`. A zero limit returns no tokens, performs no internal Prefill/Decode or KV commit, and still measures the public zero-generation call. An output limit of one, or EOS predicted by Prefill, has no Decode sample.

## Profiling and allocation diagnostics

Use separate processes for diagnostic modes:

```bash
python3 tools/run_model_benchmark.py \
  --binary build/validation/benchmark-model/tests/benchmark/aethermind_model_benchmark \
  --model .models/inference-numerical/mha-untied/model \
  --prompt 1,7,3 --max-new-tokens 3 --preset o1-plain \
  --warmup 1 --repetitions 5 --mode profile

python3 tools/run_model_benchmark.py \
  --binary build/validation/benchmark-model/tests/benchmark/aethermind_model_benchmark \
  --model .models/inference-numerical/mha-untied/model \
  --prompt 1,7,3 --max-new-tokens 3 --preset o1-plain \
  --warmup 1 --repetitions 5 --mode allocations
```

`timing` uses the normal allocator and requires a Release, non-sanitized build. The driver rejects loader injection through `LD_PRELOAD`/`LD_AUDIT` for timing. `profile` uses the normal allocator but makes no probe clock queries; an external profiler can observe this separate execution. It does not create timing summaries.

`allocations` also makes no clock queries. Its separate Runtime allocator wrapper counts allocation requests during the specified Execute intervals. These are **Runtime allocator requests**, not all process heap operations, malloc-family calls, or a global zero-allocation proof. Policy conversion runs before those intervals, so its allocations are not counted. The wrapper is absent from normal timing/profile execution. Diagnostic reports use null timing fields and cannot enter timing comparisons/statistics.

Resource fields have separate meanings:

- `raw_unique_visible_bytes`: deduplicated final raw weight views by data/size, after any policy conversion (retained W16 matrices plus widened FP32 Norm weights); logical visible bytes, not mapped file bytes or RSS.
- `packed_unique_storage_bytes`: packed storage deduplicated by Buffer implementation, including sharing across phase plans.
- `kv_capacity_bytes` and `workspace_bytes`: configured KV planes and maximum phase workspace capacity. KV bytes use the selected dtype and the actual 64-byte row padding, not a fixed fraction of KV32 bytes. With small head dimensions, KV16 and KV32 can occupy equal physical bytes despite different logical payload sizes.
- `prefill_activation_bytes` and `decode_activation_bytes`: the public Session resource planner's concrete request activation arena sizes, queried outside timing. Decode reports zero when the requested output limit requires no Decode; a zero output limit reports zero for both. These are planned phase payloads, not simultaneous resident allocations or process heap counts.
- `process_peak_rss_bytes`: Linux process cumulative high-water RSS, including earlier warmups/samples; not a per-phase peak or current request RSS.

Do not sum these fields as independent resident-memory categories.

## Verify, summarize, and compare

The standard-library checker validates the complete report before summarizing. Missing/duplicate samples, non-finite or invalid timing, inconsistent tokens/EOS/counts, wrong KV intervals, and mismatched internal/public trajectories fail.

```bash
python3 tools/verify_model_benchmark.py --report <run-dir>/report.json

python3 tools/verify_model_benchmark.py \
  --report <baseline-run-dir>/report.json \
  --compare <candidate-run-dir>/report.json
```

Comparison requires the same machine/environment/affinity, build configuration, model identity, non-preset workload (including the weight policy and KV dtype), and observed token/context trajectory. Warmup and repetition counts must match. It may compare O1/plain and O2/packed, but does not automatically call a difference an improvement or regression. All metric groups must be present; it does not compare only an overlapping subset.

Statistics use all measured repetitions and report `n/min/median/max`. Observed p90 appears only with at least 10 samples in the group; observed p99 requires at least 100. These nearest-rank observations are not promises of stable tail latency.

For same-implementation process dispersion, capture at least four separate driver invocations with the same binary, library, source, machine, model, preset, and workload. Preserve all runs and their order, then pass the first report plus at least three additional reports:

```bash
python3 tools/verify_model_benchmark.py \
  --report <aa-run-1>/report.json \
  --noise-floor <aa-run-2>/report.json <aa-run-3>/report.json <aa-run-4>/report.json
```

The result preserves all run medians and pairwise differences. Its observed noise floor is a local dispersion observation, not a global percentage or a production regression threshold. The report keeps `performance_claim: none` / Needs More Data. Candidate evaluation still needs controlled independent-process interleaved A/B, the corresponding numerical gates, and adequate machine/workload evidence.

## Correctness checks

The real CLI contract script consumes the already verified numerical bundle and checks probe/driver behavior. It is a correctness test, not a performance regression gate:

```bash
python3 tests/unit/inference/test_model_benchmark_cli.py \
  --binary build/validation/benchmark-model/tests/benchmark/aethermind_model_benchmark \
  --fixtures .models/inference-numerical -v

python3 tests/unit/inference/test_model_benchmark_cli.py \
  --binary <instrumented-build>/tests/benchmark/aethermind_model_benchmark \
  --fixtures .models/inference-numerical --instrumented -v

python3 tests/unit/inference/test_verify_model_benchmark.py -v
python3 tests/unit/inference/test_run_model_benchmark.py -v
```

`--instrumented` selects untimed profile execution for instrumented probe checks and verifies that such a build cannot enter timing statistics. The standard-library contracts are registered as `python.model_benchmark_report_contract` and `python.model_benchmark_runner_contract` in ordinary CTest correctness slices. Actual performance capture is explicit and is not run by the default correctness presets.
