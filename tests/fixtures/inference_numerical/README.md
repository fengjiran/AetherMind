# Independent HF numerical references

This directory versions the recipe, reference-environment requirements, and reviewed `checksums.json`. Generated models and reference tensors belong under the gitignored `.models/`; they are not included in the repository. These fixtures extend the [FP32 admission fixtures](../inference_admission/README.md) with an independent numerical oracle.

The exporter uses the pinned Hugging Face Llama implementation and PyTorch CPU operations. It does not call AetherMind graph, RoPE, attention, or kernel code. The C++ tests use the serialized tensors through the actual Load → Compile → Runtime → Prepare → Execute/Generate paths. PyTorch, transformers, numpy, and the Python safetensors package are needed only to generate references; normal C++ tests and the asset checker do not import them.

## Reference contract

The frozen reference environment is Linux little-endian, CPython 3.12.3, `torch==2.9.0+cpu`, `transformers==4.57.1`, `numpy==2.3.4`, and `safetensors==0.6.2`. The generator checks these package versions and records Python identity, the relevant HF source hashes, and the installed `libtorch_cpu.so` identity. The manifest also pins the generator, recipe, requirements, source model, and every output file and tensor payload.

Reference execution uses CPU FP32, eager attention, `eval()`, `inference_mode()`, deterministic algorithms, and one intra-op/inter-op thread. Before importing torch, the exporter sets `ATEN_CPU_CAPABILITY=default`, `MKL_CBWR=COMPATIBLE`, and the OMP/MKL/OpenBLAS thread limits. Package pins alone do not promise byte-identical results on every OS or wheel build: provenance changes require explicit review.

| Case | Source and geometry | Trajectory |
|---|---|---|
| `mha-untied` | Reviewed `f32-explicit` admission checkpoint; 2 layers, 4 query/KV heads, head dim 4 | Short prompt |
| `mha-tied` | Reviewed `f32-tied` admission checkpoint; tied embedding/head | Short prompt |
| `gqa-tail` | Seed 20261027; 2 layers, hidden 24, intermediate 37, vocab 31, 4 query / 2 KV heads, head dim 6 | Short prompt |
| `mqa-tail` | Seed 20261028; 1 layer, hidden 18, intermediate 22, vocab 29, 3 query / 1 KV heads, head dim 6 | Short prompt |
| `mha-near-limit` | Same untied checkpoint; context capacity 2048 | Prefill 2046 positions, then two Decode steps to 2048 |

The real checkpoint source and revision are pinned by the admission recipe: [peft-internal-testing/tiny-random-LlamaForCausalLM at d0b81c5](https://huggingface.co/peft-internal-testing/tiny-random-LlamaForCausalLM/tree/d0b81c5e23f2d61bb7a5f430a8cd6f3e711c0cb3). Synthetic weights are serialized and then loaded by HF before reference execution.

Teacher forcing and greedy generation are independent runs with fresh caches:

- Teacher forcing uses short prompt `[1, 7, 3]` and fixed Decode inputs `[17, 23]`. The near-limit prompt repeats that pattern to length 2046. Prefill and both Decode stages store logits, row identity, per-row argmax Token IDs, all committed layers of K/V, and the committed-position watermark.
- Short references store every Prefill logit row and the full vocabulary. The near-limit reference stores the last Prefill logit row only; its argmax Token IDs and committed K/V cover all Prefill positions. Each Decode stores its one full logit row.
- A separate closed-loop greedy run feeds each selected Token ID into the next step and records three new tokens and the top-two logit margins. The public `InferenceSession::Generate` test checks the four short cases for O1Plain and O2Packed, twice through the same Session as fresh requests. The near-limit C++ case checks the fixed trajectory only.

There are 24 tensors per case, 120 total. K/V layout is `[layer, committed_position, kv_head, head_dim]`; K is post-RoPE and V is unrotated. Tolerance is `abs(actual - expected) <= atol + rtol * abs(expected)`: logits use `atol=2e-5, rtol=2e-4`; K/V use `atol=2e-6, rtol=2e-4`. Shapes, selected rows, Token IDs, and watermarks must match exactly; non-finite values fail.

These cases cover standard RoPE and the listed FP32 geometries. They do not qualify other model families, quantized/low-precision execution, extended-context RoPE variants, model quality, or performance.

## Prepare a new candidate

Run from the repository root. First prepare the checked admission variants using the [admission instructions](../inference_admission/README.md). Create an isolated Python environment with pip; the normal build environment does not need the reference packages.

```bash
REF_ENV=/tmp/aethermind-numerical-reference
python3 -m venv "$REF_ENV"
"$REF_ENV/bin/python3" -m pip install \
  -r tests/fixtures/inference_numerical/requirements-reference.txt

"$REF_ENV/bin/python3" tools/generate_llama_numerical_reference.py \
  --admission-root .models/inference-admission \
  --output-dir .models/inference-numerical-candidate
```

The output directory must not already exist and must not overlap the admission source. The exporter writes a staged bundle and renames it only after generation completes. It does not update `checksums.json` or replace an accepted bundle.

Each case has a `model/` containing only `config.json` and `model.safetensors`. Its `reference.safetensors` and `reference.json` are outside that model directory, so the HF loader cannot mistake the oracle tensors for model weights. The bundle root contains `manifest.json`.

Check a candidate against the reviewed identity using standard-library Python:

```bash
python3 tools/verify_llama_numerical_reference.py \
  --root .models/inference-numerical-candidate \
  --expected tests/fixtures/inference_numerical/checksums.json
```

This read-only command checks strict JSON/Safetensors structure, provenance, model identity, file/payload hashes, tensor sets and shapes, finite values, and trajectory consistency. Expected values must be a separately reviewed file, not the current bundle's own manifest. Exit 0 means verified; exit 2 means invalid.

## Compare and review an intentional change

Keep the accepted bundle and generate into another fresh directory, for example `.models/inference-numerical-next`. Compare it without changing either side:

```bash
python3 tools/verify_llama_numerical_reference.py \
  --root .models/inference-numerical \
  --expected tests/fixtures/inference_numerical/checksums.json \
  --compare .models/inference-numerical-next
```

The comparison reports provenance/model/file changes and per-tensor differences, including maximum errors and the first failing coordinate. Exit 0 means identical, 1 means review required, and 2 means an invalid bundle. Even numerical differences within tolerance require review; matching output tokens alone cannot accept a candidate.

When changing the source, recipe, generator, reference libraries, or serialized results, inspect that report and run the C++ tests against the candidate. Update the authoritative `checksums.json` and accepted asset location only after an explicit manual review of the intended change. Neither exporter, checker, CTest, nor CI automatically accepts a new baseline.

## Run validation

The default asset path is `.models/inference-numerical`. To test a verified candidate directly, configure its absolute path:

```bash
cmake --preset validation-avx2 \
  -DAETHERMIND_TEST_NUMERICAL_DIR="$PWD/.models/inference-numerical-candidate"
cmake --build --preset validation-avx2
ctest --preset validation-numerical

python3 tests/unit/inference/test_verify_llama_numerical_reference.py -v
```

The `validation-numerical` test preset uses the `validation-avx2` build and selects `numerical-model`, including the separate near-context-limit cases. It writes `numerical-junit.xml` without replacing the AVX2 candidate-test report. Both O1Plain and O2Packed execution are checked. To run the same model slice with optional AVX2/FMA implementations excluded:

```bash
cmake --preset validation-baseline \
  -DAETHERMIND_TEST_NUMERICAL_DIR="$PWD/.models/inference-numerical-candidate"
cmake --build --preset validation-baseline
ctest --test-dir build/validation/validation-baseline \
  -L '^numerical-model$' --output-on-failure --no-tests=error
```

CTest runs the independent `fixture.numerical_assets` checker before model tests. Required-asset presets fail on missing/changed assets, absent/empty C++ test discovery, and attempted skips. Ordinary optional-asset configurations mark unavailable model cases disabled with `missing_numerical_assets`. Configure/build may precede asset preparation; the required runtime fixture still fails until valid assets are available.

Short model cases, comparison-contract tests, and the Python checker tests are in the bounded `validation` slice. The near-limit cases carry `long-context` and require the numerical-model slice above. The weight-perturbation test requires a structurally valid cloned checkpoint to Load, Compile, and Execute successfully before the numerical comparator detects changed logits.
