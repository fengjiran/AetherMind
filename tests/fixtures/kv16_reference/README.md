# Independent KV16 production references

This bundle closes the P3 implementation comparison: actual CPU execution is
compared with the independently generated, quantization-aware FP64-reduction
oracle. The KV32 quality comparison remains a separate test. Frozen budgets
and ranking semantics are in proposal 11 §2.8; this fixture does not change them.

The five source cases are the reviewed `inference_numerical` cases: untied/tied
MHA, GQA/MQA tails, and 2046-token Prefill plus two Decode positions to 2048.
Each has F16/BF16 matrix checkpoints (Norm stays F32), and independent
F16/BF16 KV oracle tensors. Both O1/plain and O2/identity-packed execute all four
weight/cache combinations. Every stage compares logits, full committed K/V,
watermark, selected-row identities, and reference-derived clear-margin rankings.

`checksums.json` is the reviewed SHA-256 identity of the imported models and
oracles. The immutable calibration manifest is pinned by SHA-256 in the import
tool; it identifies the original FP32 sources, calibrator, pinned HF/PyTorch
environment, and every independent tensor payload. Import rechecks all of those
payloads. Matrix conversion also cross-checks the serialized 16-bit bits against
the independent integer RNE oracle. Accepted FP32 assets are never modified.

Generate into a fresh gitignored directory using the pinned reference Python
environment (the same requirements as `inference_numerical`):

```bash
/tmp/aethermind-numerical-reference/bin/python3 tools/prepare_kv16_reference.py prepare \
  --source .models/inference-numerical \
  --calibration benchmark-results/models/kv16-calibration/20261112T053100Z_592df0e4-dirty_desktop-54h5mmi_kv16-step1/calibration-A \
  --output .models/kv16-reference
python3 tools/prepare_kv16_reference.py verify --output .models/kv16-reference
```

Preparation requires NumPy/PyTorch/Safetensors. Verification requires only the
Python standard library, checks the exact file set and SHA-256 identities, and
rejects a manifest that differs from the reviewed `checksums.json`. It never
updates expected identities. `--record-identities` was used once for the initial
reviewed bundle and refuses to overwrite the committed pins.

The C++ direct-test path additionally checks each file size and its compiled
FNV-1a fingerprint from `test_kv16_reference_identities.h`; this catches absent
or altered files even if the separate SHA-256 check was skipped. This secondary
noncryptographic fingerprint is an accidental-corruption guard, not a substitute
for the SHA-256 asset verifier. Missing fixtures fail the production gate.

Run the shortest implementation comparison before the near-limit cases:

```bash
TMPDIR=/tmp ./build/tests/unit/aethermind_unit_tests \
  --gtest_filter=KV16CalibratedAcceptance.ShortFixturesMatchIndependentQuantizedOracle
TMPDIR=/tmp ./build/tests/unit/aethermind_unit_tests \
  --gtest_filter=KV16CalibratedAcceptance.NearLimitMatchesIndependentQuantizedOracle
```

GoogleTest XML properties record per-case/preset/weight/cache/stage absolute,
relative, and normalized tensor errors. The independently generated oracle uses
FP64 reductions but writes the cache boundary and result tensors back to F32;
it is a numerical oracle, not a claim that production performs FP64 reductions.
The scalar-conversion bit contract remains independent of model tolerances.
