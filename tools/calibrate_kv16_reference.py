#!/usr/bin/env python3
"""P3 step-1 independent KV16 numerical calibration.

Independent of AetherMind: the pinned Hugging Face Llama implementation supplies
the model algorithm and an integer round-to-nearest-even oracle validates the
16-bit conversion. Mathematical W16 is derived from the fixed F32 source (Norm
stays F32); K (post-RoPE) and V (raw) are narrowed once at the cache write
boundary and widened on read, including the Prefill current token.

This tool does not implement or replace production kernels, allocate two-byte
KV storage, exercise O1/O2 execution, or measure performance. It only produces
the independent references and measurements that freeze the §2.8 numerical
budgets before any KV16 kernel is written. See
docs/improvement-plan/11-fp16-bf16-execution-support.md §2.7-2.8.

Modes:
  selftest    converter bit-oracle checks + comparator negative checks
  crosscheck  reproduce the accepted FP32 HF golden (calibrator harness check)
  calibrate   four W16 x KV16 combinations against independent references
"""

import argparse
import json
import os
from pathlib import Path
import platform
import struct
import tempfile
import time

import generate_llama_numerical_reference as golden


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
CALIBRATOR_VERSION = 1

# §2.8 fixed finite model budgets (draft until calibration freezes them).
# (atol, rtol) per comparison layer and KV dtype.
BUDGETS = {
    ("impl", "f16"): {"logits": (1e-6, 2e-5), "kv": (2e-4, 2e-3)},
    ("impl", "bf16"): {"logits": (1e-6, 2e-5), "kv": (1.5e-3, 2e-3)},
    ("quality", "f16"): {"logits": (1e-3, 2e-3), "kv": (4e-4, 4e-3)},
    ("quality", "bf16"): {"logits": (5e-3, 1e-2), "kv": (2e-3, 2e-2)},
}
GOLDEN_TOLERANCE = {"logits": (2e-5, 2e-4), "kv": (2e-6, 2e-4)}
STAGES = ("prefill", "decode_0", "decode_1")


class CalibrationError(ValueError):
    """The calibration source, environment, or measurement violates its contract."""


# ---------------------------------------------------------------------------
# Independent integer RNE oracle (expected values; never uses torch conversion)
# ---------------------------------------------------------------------------

def _rne_shift(sig, n):
    if n <= 0:
        return sig << (-n)
    q = sig >> n
    rem = sig & ((1 << n) - 1)
    half = 1 << (n - 1)
    if rem > half or (rem == half and (q & 1)):
        q += 1
    return q


def oracle_f32_to_f16(b):
    sign = (b >> 16) & 0x8000
    mag = b & 0x7FFFFFFF
    exp = (mag >> 23) & 0xFF
    frac = mag & 0x7FFFFF
    if exp == 0xFF:
        if frac == 0:
            return sign | 0x7C00
        return sign | 0x7E00 | ((frac >> 13) & 0x1FF)
    if exp == 0:
        return sign
    e = exp - 127 + 15
    sig = 0x800000 | frac
    if e >= 31:
        return sign | 0x7C00
    if e <= 0:
        m = _rne_shift(sig, 14 - e)
        if m >= 0x400:
            return sign | 0x0400
        return sign | m
    m = _rne_shift(sig, 13)
    if m & 0x800:
        e += 1
        m >>= 1
        if e >= 31:
            return sign | 0x7C00
    return sign | (e << 10) | (m & 0x3FF)


def oracle_f32_to_bf16(b):
    sign = (b >> 16) & 0x8000
    mag = b & 0x7FFFFFFF
    exp = (mag >> 23) & 0xFF
    frac = mag & 0x7FFFFF
    if exp == 0xFF:
        if frac == 0:
            return sign | 0x7F80
        return sign | 0x7FC0 | ((frac >> 16) & 0x3F)
    hi = mag >> 16
    lo = mag & 0xFFFF
    if lo > 0x8000 or (lo == 0x8000 and (hi & 1)):
        hi += 1
    return sign | hi


def _f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _f32_from_bits(b):
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


# ---------------------------------------------------------------------------
# Converter selftest: validate the pinned torch quantizer against the oracle
# ---------------------------------------------------------------------------

def converter_selftest(torch, numpy):
    report = {}
    for name, narrow_dtype, oracle, widen_shift in (
        ("f16", torch.float16, oracle_f32_to_f16, None),
        ("bf16", torch.bfloat16, oracle_f32_to_bf16, 16),
    ):
        patterns = numpy.arange(65536, dtype=numpy.uint16)
        if widen_shift is None:
            widened = patterns.view(numpy.float16).astype(numpy.float32)
        else:
            widened = (patterns.astype(numpy.uint32) << widen_shift).view(numpy.float32)
        f32_bits = widened.view(numpy.uint32)
        finite = numpy.isfinite(widened)

        narrowed = torch.from_numpy(widened.copy()).to(narrow_dtype)
        narrowed_bits = narrowed.view(torch.int16).numpy().astype(numpy.uint16)

        # Finite round-trip must be bit-exact for every representable pattern.
        rt_ok = narrowed_bits[finite] == patterns[finite]
        if not rt_ok.all():
            bad = int(numpy.argmax(~rt_ok))
            raise CalibrationError(f"{name} finite round-trip mismatch at pattern {bad}")

        # NaN inputs are outside the finite calibration domain (§2.8 requires
        # finite fixtures). The production converter's NaN sign/payload bit
        # contract is validated separately against AetherMind's own converters
        # (steps 5/6). Here we only sanity-check that torch keeps NaN as NaN
        # (never Inf/finite) and record its canonicalization for transparency.
        if name == "f16":
            nan_mask = ((patterns & 0x7C00) == 0x7C00) & ((patterns & 0x03FF) != 0)
            exp_mask = 0x7C00
        else:
            nan_mask = ((patterns & 0x7F80) == 0x7F80) & ((patterns & 0x007F) != 0)
            exp_mask = 0x7F80
        nan_out = narrowed_bits[nan_mask]
        if not ((nan_out & exp_mask) == exp_mask).all():
            raise CalibrationError(f"{name} NaN narrowing produced non-NaN")
        nan_sign_preserved = bool(((nan_out & 0x8000) ==
                                   (patterns[nan_mask] & 0x8000)).all())

        # Midpoints and F32 neighbours between adjacent finite patterns.
        mid_checked = 0
        max_finite = 0x7BFF if name == "f16" else 0x7F7F
        for p in range(max_finite):
            a = float(widened[p])
            b = float(widened[p + 1])
            mid = (a + b) / 2.0
            mid_bits = _f32_bits(mid)
            for cand_bits in (mid_bits - 1, mid_bits, mid_bits + 1):
                cand = _f32_from_bits(cand_bits)
                if not numpy.isfinite(cand):
                    continue
                expected = oracle(cand_bits)
                got = int(torch.tensor([cand], dtype=torch.float32).to(narrow_dtype)
                          .view(torch.int16).item()) & 0xFFFF
                if got != expected:
                    raise CalibrationError(
                        f"{name} midpoint RNE mismatch bits={cand_bits:#010x} "
                        f"got={got:#06x} expected={expected:#06x}")
                mid_checked += 1

        # Deterministic random F32 bit patterns (finite) against the oracle.
        rng = numpy.random.default_rng(20261112)
        random_bits = rng.integers(0, 0x7F800000, size=200000, dtype=numpy.uint32)
        random_vals = random_bits.view(numpy.float32)
        got = (torch.from_numpy(random_vals.copy()).to(narrow_dtype)
               .view(torch.int16).numpy().astype(numpy.uint16))
        expected = numpy.array([oracle(int(b)) for b in random_bits], dtype=numpy.uint16)
        if not (got == expected).all():
            bad = int(numpy.argmax(got != expected))
            raise CalibrationError(f"{name} random RNE mismatch at index {bad}")

        # Signed zero and overflow thresholds.
        for value, expected in ((0.0, 0x0000), (-0.0, 0x8000)):
            got = int(torch.tensor([value], dtype=torch.float32).to(narrow_dtype)
                      .view(torch.int16).item()) & 0xFFFF
            if got != expected:
                raise CalibrationError(f"{name} signed zero mismatch")
        report[name] = {"finite_patterns": int(finite.sum()),
                        "nan_patterns": int(nan_mask.sum()),
                        "nan_sign_preserved_by_torch": nan_sign_preserved,
                        "nan_domain": "outside_finite_calibration; see production converter tests",
                        "midpoint_checks": mid_checked, "random_checks": 200000}
    return report


# ---------------------------------------------------------------------------
# Quantizer and quantizing cache (the single KV16 conversion boundary)
# ---------------------------------------------------------------------------

def make_narrow_widen(torch, kv_dtype):
    narrow_dtype = torch.float16 if kv_dtype == "f16" else torch.bfloat16

    def narrow_widen(tensor):
        return tensor.to(narrow_dtype).to(tensor.dtype)

    return narrow_widen


def make_quantized_cache_class(torch, narrow_widen):
    from transformers.cache_utils import DynamicCache

    class QuantizedKVCache(DynamicCache):
        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            return super().update(narrow_widen(key_states), narrow_widen(value_states),
                                  layer_idx, cache_kwargs)

    return QuantizedKVCache


def derive_math_w16(torch, model, weight_dtype):
    """Round every matrix (2D) weight to W16 and widen back; Norm (1D) stays F32.

    Tied lm_head shares embed_tokens storage, so converting the deduplicated
    parameter set preserves the tie. RoPE inv_freq is a buffer and is untouched.
    """
    narrow_dtype = torch.float16 if weight_dtype == "f16" else torch.bfloat16
    converted = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim == 2:
                parameter.copy_(parameter.to(narrow_dtype).to(parameter.dtype))
                converted.append(name)
            elif parameter.ndim != 1:
                raise CalibrationError(f"Unexpected parameter rank: {name}")
    if not converted:
        raise CalibrationError("No matrix weights converted")
    return converted


# ---------------------------------------------------------------------------
# Trajectories (mirror the golden harness; injectable cache + compute dtype)
# ---------------------------------------------------------------------------

def run_trajectories(torch, model, geometry, prompt, decode_inputs, near_limit,
                     cache_factory):
    tensors = {"prompt": torch.tensor(prompt, dtype=torch.int64),
               "decode_inputs": torch.tensor(decode_inputs, dtype=torch.int64)}
    positions = {}
    selected_rows = {}
    committed_positions = []
    cache = cache_factory() if cache_factory is not None else None
    stages = [("prefill", prompt, 0), ("decode_0", [decode_inputs[0]], len(prompt)),
              ("decode_1", [decode_inputs[1]], len(prompt) + 1)]
    with torch.inference_mode():
        for stage, inputs, position in stages:
            output = golden.forward(torch, model, inputs, position, cache)
            cache = output.past_key_values
            committed = position + len(inputs)
            logits = output.logits[0]
            rows = [len(inputs) - 1] if stage == "prefill" and near_limit else list(range(len(inputs)))
            key, value = golden.cache_planes(torch, cache, geometry, committed)
            tensors[stage + ".logit_rows"] = torch.tensor(rows, dtype=torch.int64)
            tensors[stage + ".logits"] = logits[rows].contiguous().clone()
            tensors[stage + ".tokens"] = logits.argmax(dim=-1).to(torch.int64).clone()
            tensors[stage + ".keys"] = key
            tensors[stage + ".values"] = value
            tensors[stage + ".position"] = torch.tensor([committed], dtype=torch.int64)
            positions[stage] = list(range(position, committed))
            selected_rows[stage] = rows
            committed_positions.append(committed)

        cache = cache_factory() if cache_factory is not None else None
        inputs = prompt
        position = 0
        greedy = []
        margins = []
        for _ in range(3):
            output = golden.forward(torch, model, inputs, position, cache)
            cache = output.past_key_values
            last = output.logits[0, -1]
            token = int(last.argmax().item())
            top_two = torch.topk(last, 2).values
            greedy.append(token)
            margins.append(float((top_two[0] - top_two[1]).item()))
            position += len(inputs)
            inputs = [token]
    tensors["greedy_tokens"] = torch.tensor(greedy, dtype=torch.int64)
    tensors["greedy_margins"] = torch.tensor(margins, dtype=torch.float32)
    trajectory = {"prompt": prompt, "decode_inputs": decode_inputs,
                  "positions": positions, "logit_rows": selected_rows,
                  "commit_positions": committed_positions,
                  "greedy_tokens": greedy, "greedy_margins": margins}
    return tensors, trajectory


# ---------------------------------------------------------------------------
# Comparator (§2.8 fixed budgets, ranking rule, identity)
# ---------------------------------------------------------------------------

def scalar_compare(actual, ref, atol, rtol):
    import numpy
    actual = numpy.asarray(actual, dtype=numpy.float64)
    ref = numpy.asarray(ref, dtype=numpy.float64)
    if actual.shape != ref.shape:
        return {"pass": False, "reason": "shape_mismatch",
                "actual_shape": list(actual.shape), "ref_shape": list(ref.shape)}
    if not numpy.isfinite(actual).all() or not numpy.isfinite(ref).all():
        return {"pass": False, "reason": "non_finite"}
    abs_err = numpy.abs(actual - ref)
    bound = atol + rtol * numpy.abs(ref)
    ok = abs_err <= bound
    denom = numpy.where(numpy.abs(ref) > 0, numpy.abs(ref), 1.0)
    normalized = abs_err / bound
    result = {"pass": bool(ok.all()),
                "max_abs": float(abs_err.max()) if abs_err.size else 0.0,
                "max_rel": float((abs_err / denom).max()) if abs_err.size else 0.0,
                "max_normalized": float(normalized.max()) if normalized.size else 0.0,
                "count": int(abs_err.size)}
    if not result["pass"]:
        idx = numpy.unravel_index(int(numpy.argmax(~ok)), ok.shape)
        result["reason"] = "budget_exceeded"
        result["first_failure"] = [int(i) for i in idx]
    return result


def classify_rows(ref_logits, atol, rtol):
    """Ranking classification from the reference and budget only (no candidate)."""
    import numpy
    ref = numpy.asarray(ref_logits, dtype=numpy.float64)
    if ref.ndim != 2 or ref.shape[1] < 5:
        raise CalibrationError("ranking classification needs [rows, vocab>=5]")
    epsilon = (atol + rtol * numpy.abs(ref)).max(axis=1)
    order = numpy.argsort(-ref, axis=1, kind="stable")
    rows = []
    for r in range(ref.shape[0]):
        idx = order[r]
        top1_margin = float(ref[r, idx[0]] - ref[r, idx[1]])
        top4_margin = float(ref[r, idx[3]] - ref[r, idx[4]])
        rows.append({
            "row": r,
            "epsilon": float(epsilon[r]),
            "top1_margin": top1_margin,
            "top4_margin": top4_margin,
            "top1_required": bool(top1_margin > 2 * epsilon[r]),
            "top4_required": bool(top4_margin > 2 * epsilon[r]),
            "ref_top1": int(idx[0]),
            "ref_top4": sorted(int(i) for i in idx[:4]),
        })
    return rows


def check_ranking(candidate_logits, classification):
    import numpy
    cand = numpy.asarray(candidate_logits, dtype=numpy.float64)
    failures = []
    near_ties = []
    order = numpy.argsort(-cand, axis=1, kind="stable")
    for entry in classification:
        r = entry["row"]
        idx = order[r]
        cand_top1 = int(idx[0])
        cand_top4 = sorted(int(i) for i in idx[:4])
        if not entry["top1_required"] or not entry["top4_required"]:
            near_ties.append(entry["row"])
        if entry["top1_required"] and cand_top1 != entry["ref_top1"]:
            failures.append({"row": r, "kind": "top1", "ref": entry["ref_top1"],
                             "candidate": cand_top1})
        if entry["top4_required"] and cand_top4 != entry["ref_top4"]:
            failures.append({"row": r, "kind": "top4", "ref": entry["ref_top4"],
                             "candidate": cand_top4})
    return {"pass": not failures, "failures": failures, "near_tie_rows": near_ties}


def compare_trajectory(actual_tensors, actual_traj, ref_tensors, ref_traj,
                       budgets, require_exact_tokens):
    """Compare one run against a reference trajectory under fixed budgets."""
    import numpy
    report = {"stages": {}, "pass": True, "identity_pass": True}
    if actual_traj["commit_positions"] != ref_traj["commit_positions"]:
        report["identity_pass"] = False
        report["pass"] = False
        report["identity_reason"] = "watermark_mismatch"
    for stage in STAGES:
        stage_report = {}
        if actual_traj["logit_rows"][stage] != ref_traj["logit_rows"][stage]:
            report["identity_pass"] = False
            report["pass"] = False
            stage_report["identity"] = "logit_row_mismatch"
        if actual_traj["positions"][stage] != ref_traj["positions"][stage]:
            report["identity_pass"] = False
            report["pass"] = False
            stage_report["identity"] = "position_mismatch"
        atol, rtol = budgets["logits"]
        logits = scalar_compare(actual_tensors[stage + ".logits"].numpy(),
                                ref_tensors[stage + ".logits"].numpy(), atol, rtol)
        stage_report["logits"] = logits
        kv_atol, kv_rtol = budgets["kv"]
        keys = scalar_compare(actual_tensors[stage + ".keys"].numpy(),
                              ref_tensors[stage + ".keys"].numpy(), kv_atol, kv_rtol)
        values = scalar_compare(actual_tensors[stage + ".values"].numpy(),
                                ref_tensors[stage + ".values"].numpy(), kv_atol, kv_rtol)
        stage_report["keys"] = keys
        stage_report["values"] = values
        classification = classify_rows(ref_tensors[stage + ".logits"].numpy(), atol, rtol)
        ranking = check_ranking(actual_tensors[stage + ".logits"].numpy(), classification)
        stage_report["ranking"] = ranking
        stage_report["near_tie_count"] = len(ranking["near_tie_rows"])
        if require_exact_tokens:
            same_tokens = bool((actual_tensors[stage + ".tokens"] ==
                                ref_tensors[stage + ".tokens"]).all().item())
            stage_report["tokens_exact"] = same_tokens
            if not same_tokens:
                report["pass"] = False
        for key in ("logits", "keys", "values"):
            if not stage_report[key]["pass"]:
                report["pass"] = False
        if not ranking["pass"]:
            report["pass"] = False
        report["stages"][stage] = stage_report
    if require_exact_tokens:
        greedy_same = bool((actual_tensors["greedy_tokens"] ==
                            ref_tensors["greedy_tokens"]).all().item())
        report["greedy_tokens_exact"] = greedy_same
        if not greedy_same:
            report["pass"] = False
    return report


def comparator_selftest(numpy):
    """Negative checks: the comparator must reject the right things."""
    checks = []

    def expect(name, condition):
        if not condition:
            raise CalibrationError(f"comparator negative check failed: {name}")
        checks.append(name)

    ref = numpy.array([[1.0, 0.5, 0.25, 0.1, 0.05, -0.2]], dtype=numpy.float32)
    atol, rtol = 1e-3, 2e-3
    expect("within_budget_passes", scalar_compare(ref, ref, atol, rtol)["pass"])
    over = ref.copy()
    over[0, 0] += 1.0
    expect("over_budget_fails", not scalar_compare(over, ref, atol, rtol)["pass"])
    nonfinite = ref.copy()
    nonfinite[0, 0] = numpy.nan
    expect("nonfinite_candidate_fails", not scalar_compare(nonfinite, ref, atol, rtol)["pass"])
    expect("nonfinite_reference_fails", not scalar_compare(ref, nonfinite, atol, rtol)["pass"])
    expect("shape_mismatch_fails",
           not scalar_compare(ref.reshape(1, 6), ref.reshape(6, 1), atol, rtol)["pass"])

    cls = classify_rows(ref, atol, rtol)
    expect("clear_top1_required", cls[0]["top1_required"])
    swapped = ref.copy()
    swapped[0, 0], swapped[0, 1] = ref[0, 1], ref[0, 0]
    expect("top1_swap_fails", not check_ranking(swapped, cls)["pass"])

    near = numpy.array([[1.0, 1.0 + 1e-6, 0.25, 0.1, 0.05, -0.2]], dtype=numpy.float32)
    near_cls = classify_rows(near, atol, rtol)
    expect("near_tie_not_required", not near_cls[0]["top1_required"])
    near_swapped = near.copy()
    near_swapped[0, 0], near_swapped[0, 1] = near[0, 1], near[0, 0]
    near_rank = check_ranking(near_swapped, near_cls)
    expect("near_tie_swap_recorded", near_rank["pass"] and 0 in near_rank["near_tie_rows"])

    # Classification must not depend on candidate error magnitude.
    huge = near.copy()
    huge[0, 0] += 100.0
    expect("classification_independent_of_candidate",
           classify_rows(near, atol, rtol) == near_cls)
    expect("huge_candidate_still_scalar_fails",
           not scalar_compare(huge, near, atol, rtol)["pass"])
    return {"negative_checks": checks}


# ---------------------------------------------------------------------------
# Asset loading and provenance
# ---------------------------------------------------------------------------

def load_bundle(bundle_root):
    from safetensors.torch import load_file
    bundle_root = Path(bundle_root)
    manifest = json.loads((bundle_root / "manifest.json").read_text())
    cases = {}
    for case_id in manifest["cases"]:
        case_dir = bundle_root / case_id
        tensors = load_file(str(case_dir / "reference.safetensors"))
        record = json.loads((case_dir / "reference.json").read_text())
        cases[case_id] = {"tensors": tensors, "record": record,
                          "model_dir": case_dir / "model"}
    return manifest, cases


def load_case_model(modules, model_dir):
    torch, _, _, _, LlamaConfig, LlamaForCausalLM = modules
    config = LlamaConfig.from_pretrained(model_dir, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(
        model_dir, config=config, local_files_only=True, use_safetensors=True,
        dtype=torch.float32, attn_implementation="eager")
    model.to(device="cpu", dtype=torch.float32).eval()
    geometry = {name: getattr(config, name) for name in golden.GEOMETRY_FIELDS}
    return model, geometry


def case_prompt(recipe, geometry, near_limit):
    length = geometry["max_position_embeddings"] - 2 if near_limit else len(recipe["short_prompt"])
    return [recipe["short_prompt"][i % len(recipe["short_prompt"])] for i in range(length)]


def run_id():
    sha = os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip() or "nogit"
    dirty = os.popen("git status --porcelain 2>/dev/null").read().strip()
    host = platform.node().split(".")[0].lower()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{stamp}_{sha}{'-dirty' if dirty else ''}_{host}_kv16-calibration"


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_selftest(modules):
    torch, _, numpy, _, _, _ = modules
    converter = converter_selftest(torch, numpy)
    comparator = comparator_selftest(numpy)
    return {"converter": converter, "comparator": comparator}


def mode_crosscheck(modules, bundle_root, recipe):
    torch = modules[0]
    manifest, cases = load_bundle(bundle_root)
    results = {}
    overall = True
    for spec in recipe["cases"]:
        case_id = spec["id"]
        case = cases[case_id]
        model, geometry = load_case_model(modules, case["model_dir"])
        near_limit = spec.get("near_limit", False)
        prompt = case_prompt(recipe, geometry, near_limit)
        tensors, trajectory = golden.trajectories(
            torch, model, geometry, prompt, recipe["decode_inputs"], near_limit,
            recipe["tolerance"])
        report = compare_trajectory(tensors, trajectory, case["tensors"],
                                    case["record"]["trajectory"], GOLDEN_TOLERANCE,
                                    require_exact_tokens=True)
        results[case_id] = report
        overall = overall and report["pass"]
        print(f"crosscheck {case_id}: {'PASS' if report['pass'] else 'FAIL'}", flush=True)
        del model, tensors
    return {"pass": overall, "cases": results,
            "golden_manifest_sha256": golden.file_identity(bundle_root / "manifest.json")["sha256"]}


def _run_config(torch, model, geometry, prompt, decode_inputs, near_limit,
                compute_dtype, kv_dtype):
    model.to(dtype=compute_dtype)
    if kv_dtype == "kv32":
        factory = None
    else:
        narrow_widen = make_narrow_widen(torch, kv_dtype)
        cache_class = make_quantized_cache_class(torch, narrow_widen)
        factory = cache_class
    tensors, trajectory = run_trajectories(
        torch, model, geometry, prompt, decode_inputs, near_limit, factory)
    return tensors, trajectory


def _writeback_f32(torch, tensors):
    """§2.8: the FP64-reduction oracle is written back to F32 for comparison."""
    return {name: (value.to(torch.float32) if value.is_floating_point() else value)
            for name, value in tensors.items()}


def mode_calibrate(modules, bundle_root, recipe, output_dir, cases_filter):
    torch, _, numpy, _, _, _ = modules
    manifest, cases = load_bundle(bundle_root)
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise CalibrationError("Calibration output must not already exist")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    rid = run_id()
    measurements = {"schema_version": SCHEMA_VERSION, "calibrator_version": CALIBRATOR_VERSION,
                    "run_id": rid, "cases": {}}
    with tempfile.TemporaryDirectory(prefix=".kv16-calibration-", dir=output_dir.parent) as tmp:
        staging = Path(tmp)
        for spec in recipe["cases"]:
            case_id = spec["id"]
            if cases_filter and case_id not in cases_filter:
                continue
            case = cases[case_id]
            near_limit = spec.get("near_limit", False)
            print(f"calibrating {case_id}", flush=True)
            case_out = staging / case_id
            case_out.mkdir()
            case_measure = {
                "model_files": {name: golden.file_identity(case["model_dir"] / name)
                                for name in ("config.json", "model.safetensors")},
                "combos": {},
            }
            for weight_dtype in ("f16", "bf16"):
                model, geometry = load_case_model(modules, case["model_dir"])
                converted = derive_math_w16(torch, model, weight_dtype)
                prompt = case_prompt(recipe, geometry, near_limit)
                # Quality reference: same W16, F32 compute, KV32.
                q_tensors, q_traj = _run_config(
                    torch, model, geometry, prompt, recipe["decode_inputs"], near_limit,
                    torch.float32, "kv32")
                for kv_dtype in ("f16", "bf16"):
                    combo = f"w{weight_dtype}-kv{kv_dtype}"
                    impl_tensors, impl_traj = _run_config(
                        torch, model, geometry, prompt, recipe["decode_inputs"], near_limit,
                        torch.float32, kv_dtype)
                    oracle_tensors, oracle_traj = _run_config(
                        torch, model, geometry, prompt, recipe["decode_inputs"], near_limit,
                        torch.float64, kv_dtype)
                    oracle_tensors = _writeback_f32(torch, oracle_tensors)
                    impl_budget = BUDGETS[("impl", kv_dtype)]
                    quality_budget = BUDGETS[("quality", kv_dtype)]
                    impl_report = compare_trajectory(
                        impl_tensors, impl_traj, oracle_tensors, oracle_traj,
                        impl_budget, require_exact_tokens=False)
                    quality_report = compare_trajectory(
                        impl_tensors, impl_traj, q_tensors, q_traj,
                        quality_budget, require_exact_tokens=False)
                    combo_dir = case_out / combo
                    combo_dir.mkdir()
                    tensor_records = {}
                    for label, tensors in (("impl_f32", impl_tensors),
                                           ("impl_f64_oracle", oracle_tensors),
                                           ("quality_kv32", q_tensors)):
                        encoded, records = golden.encode_tensors(torch, tensors)
                        (combo_dir / f"{label}.safetensors").write_bytes(encoded)
                        tensor_records[label] = records
                    combo_measure = {
                        "weight_dtype": weight_dtype, "kv_dtype": kv_dtype,
                        "converted_matrix_weights": converted,
                        "impl_correctness": impl_report,
                        "relative_kv32": quality_report,
                        "tensors": tensor_records,
                        "pass": impl_report["pass"] and quality_report["pass"],
                    }
                    case_measure["combos"][combo] = combo_measure
                    print(f"  {combo}: impl={'PASS' if impl_report['pass'] else 'FAIL'} "
                          f"quality={'PASS' if quality_report['pass'] else 'FAIL'}", flush=True)
                    del impl_tensors, oracle_tensors
                del model, q_tensors
            (case_out / "measurements.json").write_bytes(golden.json_bytes(case_measure))
            measurements["cases"][case_id] = case_measure
        identity = golden.reference_identity(modules, recipe["seed"])
        identity["execution"]["calibration_compute_dtypes"] = ["float32", "float64"]
        identity["execution"]["hf_precision_notes"] = (
            "HF eager softmax and RMSNorm variance reduce in float32 even in float64 "
            "model mode; GEMM, attention dot and value accumulation follow model dtype.")
        full_manifest = {
            "schema_version": SCHEMA_VERSION,
            "calibrator": {"version": CALIBRATOR_VERSION,
                           "sha256": golden.file_identity(__file__)["sha256"],
                           "recipe_sha256": golden.file_identity(
                               ROOT / "tests/fixtures/inference_numerical/recipe.json")["sha256"]},
            "reference": identity,
            "source_bundle_sha256": golden.file_identity(bundle_root / "manifest.json")["sha256"],
            "budgets": {f"{layer}-{kv}": v for (layer, kv), v in BUDGETS.items()},
            "run_id": rid,
            "measurements": measurements,
        }
        (staging / "manifest.json").write_bytes(golden.json_bytes(full_manifest))
        staging.rename(output_dir)
    return full_manifest


def mode_compare(bundle_a, bundle_b):
    """Repeatability: two independent runs must be bit-identical numerically.

    Compares every per-case measurements.json and safetensors payload, plus the
    durable manifest fields (calibrator/source/budgets/reference identity). The
    run_id timestamp is expected to differ and is excluded.
    """
    a, b = Path(bundle_a), Path(bundle_b)

    def canonical(rel):
        return rel.name == "measurements.json" or rel.suffix == ".safetensors"

    files_a = {p.relative_to(a): p for p in a.rglob("*")
               if p.is_file() and canonical(p.relative_to(a))}
    files_b = {p.relative_to(b): p for p in b.rglob("*")
               if p.is_file() and canonical(p.relative_to(b))}
    differences = []
    if set(files_a) != set(files_b):
        differences.append({"kind": "file_set",
                            "only_a": sorted(map(str, set(files_a) - set(files_b))),
                            "only_b": sorted(map(str, set(files_b) - set(files_a)))})
    for rel in sorted(set(files_a) & set(files_b)):
        if files_a[rel].read_bytes() != files_b[rel].read_bytes():
            differences.append({"kind": "content", "file": str(rel)})
    ma = json.loads((a / "manifest.json").read_text())
    mb = json.loads((b / "manifest.json").read_text())
    for key in ("calibrator", "source_bundle_sha256", "budgets", "reference"):
        if ma.get(key) != mb.get(key):
            differences.append({"kind": "manifest_field", "field": key})
    return {"pass": not differences,
            "compared_files": len(set(files_a) & set(files_b)),
            "differences": differences,
            "run_id_a": ma.get("run_id"), "run_id_b": mb.get("run_id")}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("selftest", "crosscheck", "calibrate", "compare"))
    parser.add_argument("--bundle", type=Path, default=ROOT / ".models/inference-numerical")
    parser.add_argument("--compare-bundle", type=Path, default=None,
                        help="second calibration bundle (compare mode)")
    parser.add_argument("--recipe", type=Path,
                        default=ROOT / "tests/fixtures/inference_numerical/recipe.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--report", type=Path, default=None,
                        help="optional JSON report path (selftest/crosscheck/compare)")
    options = parser.parse_args()
    try:
        if options.mode == "compare":
            if options.compare_bundle is None:
                parser.error("compare requires --bundle and --compare-bundle")
            result = mode_compare(options.bundle, options.compare_bundle)
        else:
            modules = golden.load_modules()
            recipe = golden.read_recipe(options.recipe)
            if options.mode == "selftest":
                result = mode_selftest(modules)
            elif options.mode == "crosscheck":
                result = mode_crosscheck(modules, options.bundle, recipe)
            else:
                if options.output_dir is None:
                    parser.error("calibrate requires --output-dir")
                result = mode_calibrate(modules, options.bundle, recipe, options.output_dir,
                                        options.cases)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"KV16 calibration failed: {error}\n")
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if options.report:
        options.report.write_text(text + "\n")
    if options.mode == "calibrate":
        print(options.output_dir / "manifest.json")
    else:
        print(text)
        if options.mode in ("crosscheck", "compare") and not result["pass"]:
            parser.exit(1, f"{options.mode} failed\n")


if __name__ == "__main__":
    main()