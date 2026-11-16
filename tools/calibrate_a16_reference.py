#!/usr/bin/env python3
"""P4 step-2 independent A16 numerical calibration.

Independent of AetherMind: the forward pass below is a from-scratch Llama
implementation with explicit quantization at every frozen A16 boundary. It is
validated against the accepted FP32 HF golden before it produces any budget.

Frozen A16 dataflow (proposal 11 section 4.5), BF16 preset shown:

    embed(W16) -> A16 gather                       (0 narrowing)
    RMSNorm        F32 variance/rsqrt -> A16       (1)
    QKV Linear     16-bit product, F32 accumulate -> A16 (1 each)
    RoPE           F32 angles/rotation -> A16      (1; Q and K were already
                                                    rounded at the Linear output,
                                                    so Q/K round twice in total)
    KVUpdate       A16 write-through copy          (0)
    Attention      F32 score/softmax/accumulate -> A16 (1)
    Add residual   F32 add of two A16 values -> A16 (1 per residual connection)
    SiLU/SiluMul   per-op A16 output               (1 each)
    LMHead         A16 x W16 -> F32                (0; logits stay F32)

The FP16 preset keeps FP32 residual and is not calibrated here; it is P4-B.

This tool does not implement or replace production kernels, allocate storage,
exercise O1/O2 execution, or measure performance. It produces the independent
references and measurements that freeze the A16 numerical budgets before any
production kernel (scalar, AVX2 fallback, or native ISA) is written.

Modes:
  selftest    converter bit-oracle checks + comparator negative checks
  crosscheck  reproduce the accepted FP32 HF golden with this implementation
  calibrate   two same-dtype A16 combinations against independent references
  compare     two calibration bundles must be bit-identical numerically
"""

import argparse
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import calibrate_kv16_reference as kv16
import generate_llama_numerical_reference as golden


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
CALIBRATOR_VERSION = 1
STAGES = kv16.STAGES

# Draft A16 budgets, frozen only after a passing calibration run. (atol, rtol)
# per comparison layer and activation/weight dtype. The implementation layer
# compares the F32 quantized dataflow against the same dataflow evaluated in
# FP64; the quality layer compares it against W16/A32 with KV32.
BUDGETS = {
    # The F16 implementation layer bounds F32-vs-F64 accumulation-order drift,
    # which grows with the 2046-token prefill; the measured worst case of the
    # corrected frozen dataflow is 1.72x the previous 2e-5/2e-4 cell, so this
    # cell is 1e-4/1e-3 (still tighter than the BF16 layer).
    ("impl", "f16"): {"logits": (1e-4, 1e-3), "kv": (2e-4, 2e-3)},
    ("impl", "bf16"): {"logits": (2e-4, 2e-3), "kv": (2e-3, 2e-2)},
    ("quality", "f16"): {"logits": (2e-3, 6e-3), "kv": (1e-3, 1e-2)},
    ("quality", "bf16"): {"logits": (1e-2, 3e-2), "kv": (5e-3, 3e-2)},
}
GOLDEN_TOLERANCE = kv16.GOLDEN_TOLERANCE


# ---------------------------------------------------------------------------
# Quantization boundary and parameter materialization
# ---------------------------------------------------------------------------

def narrow(torch, value, dtype):
    """Rounds a compute-dtype tensor to a 16-bit format and widens it back.

    Values live in the compute dtype (F32 impl, F64 oracle) so reductions stay
    in that precision; only the boundary rounding is 16-bit.
    """
    target = torch.float16 if dtype == "f16" else torch.bfloat16
    return value.to(target).to(value.dtype)


def load_model(modules, model_dir):
    torch, _, _, _, LlamaConfig, LlamaForCausalLM = modules
    config = LlamaConfig.from_pretrained(model_dir, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(
        model_dir, config=config, local_files_only=True, use_safetensors=True,
        dtype=torch.float32, attn_implementation="eager")
    model.to(device="cpu", dtype=torch.float32).eval()
    geometry = {name: getattr(config, name) for name in golden.GEOMETRY_FIELDS}
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    geometry["rope_theta"] = float(
        rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0) or 10000.0))
    return model, geometry


def extract_params(torch, model, geometry, weight_dtype):
    """Extracts F32 parameter tensors; matrices are rounded to W16 when asked.

    Norm weights stay F32: the projection owns their F32 execution.
    """
    state = model.state_dict()

    def norm(name):
        return state[name].to(torch.float32).detach().clone()

    def matrix(name):
        value = state[name].to(torch.float32)
        if weight_dtype is not None:
            value = narrow(torch, value, weight_dtype)
        return value.detach().clone()

    layers = []
    for index in range(geometry["num_hidden_layers"]):
        prefix = f"model.layers.{index}."
        layers.append({
            "input_norm": norm(prefix + "input_layernorm.weight"),
            "post_norm": norm(prefix + "post_attention_layernorm.weight"),
            "q": matrix(prefix + "self_attn.q_proj.weight"),
            "k": matrix(prefix + "self_attn.k_proj.weight"),
            "v": matrix(prefix + "self_attn.v_proj.weight"),
            "o": matrix(prefix + "self_attn.o_proj.weight"),
            "gate": matrix(prefix + "mlp.gate_proj.weight"),
            "up": matrix(prefix + "mlp.up_proj.weight"),
            "down": matrix(prefix + "mlp.down_proj.weight"),
        })
    return {"embed": matrix("model.embed_tokens.weight"),
            "layers": layers,
            "final_norm": norm("model.norm.weight"),
            "lm_head": matrix("lm_head.weight")}


# ---------------------------------------------------------------------------
# Quantized Llama forward (single request, no batch dimension)
# ---------------------------------------------------------------------------

def rms_norm(torch, hidden, weight, eps):
    """F32 variance/rsqrt and F32 weight multiply; the caller narrows once.

    The frozen placement keeps the normalized value in F32 while applying the
    F32 weight and narrows only the produced output, so this does not reproduce
    the HF mixed-precision pattern that narrows the normalized value first.
    """
    value = hidden.to(torch.float32)
    variance = value.pow(2).mean(-1, keepdim=True)
    value = value * torch.rsqrt(variance + eps)
    return weight * value


def rope(torch, q, k, position, head_dim, theta, compute_dtype):
    """Rotates q/k in the compute dtype with F32-or-better angles; one narrowing.

    The F32 implementation follows the frozen placement (F32 trigonometry and
    rotation, one narrowing of the produced Q/K) rather than narrowing cos/sin
    to the 16-bit dtype before the multiply. The FP64 oracle passes its compute
    dtype here so both runs evaluate the same dataflow.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(position, position + q.shape[1], dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(compute_dtype)
    sin = emb.sin().to(compute_dtype)

    def rotate_half(value):
        half = value.shape[-1] // 2
        return torch.cat((-value[..., half:], value[..., :half]), dim=-1)

    q_compute = q.to(compute_dtype)
    k_compute = k.to(compute_dtype)
    return (q_compute * cos + rotate_half(q_compute) * sin,
            k_compute * cos + rotate_half(k_compute) * sin)


def attention(torch, q, key, value, position, compute_dtype):
    """Eager causal attention in the compute dtype; the caller narrows once.

    The F32 implementation keeps score/softmax/accumulation in F32 as frozen;
    the FP64 oracle passes its compute dtype so only the evaluation precision
    differs between the two runs.
    """
    heads = q.shape[0]
    kv_heads = key.shape[0]
    if heads % kv_heads != 0:
        raise kv16.CalibrationError("A16 attention needs an integer GQA group size")
    repeat = heads // kv_heads
    q_compute = q.to(compute_dtype)
    key_compute = key.to(compute_dtype)
    value_compute = value.to(compute_dtype)
    if repeat > 1:
        key_compute = key_compute.repeat_interleave(repeat, dim=0)
        value_compute = value_compute.repeat_interleave(repeat, dim=0)

    queries = q.shape[1]
    keys = key.shape[1]
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q_compute * scale, key_compute.transpose(-1, -2))
    positions = torch.arange(position, position + queries, dtype=torch.int64)
    visible = torch.arange(keys, dtype=torch.int64)
    mask = visible.unsqueeze(0) > positions.unsqueeze(1)
    scores = scores.masked_fill(mask.unsqueeze(0), torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, value_compute)


def forward(torch, params, geometry, tokens, position, cache, compute_dtype,
            a16_dtype):
    """Runs one stage and returns (logits, updated cache planes).

    Logits are produced for every row so the trajectory token tensors match the
    accepted golden; the cache planes are [kv_heads, committed, head_dim].
    """
    dtype = compute_dtype

    def cast(value):
        return value.to(dtype)

    def boundary(value):
        return narrow(torch, value, a16_dtype) if a16_dtype is not None else value

    hidden = boundary(cast(params["embed"])[torch.tensor(tokens, dtype=torch.int64)])

    head_dim = geometry["head_dim"]
    heads = geometry["num_attention_heads"]
    kv_heads = geometry["num_key_value_heads"]
    eps = geometry["rms_norm_eps"]
    theta = geometry["rope_theta"]
    cache_out = {}
    for index, layer in enumerate(params["layers"]):
        residual = hidden
        normed = boundary(rms_norm(torch, hidden, cast(layer["input_norm"]), eps))

        q = boundary(torch.matmul(normed, cast(layer["q"]).t()))
        k = boundary(torch.matmul(normed, cast(layer["k"]).t()))
        v = boundary(torch.matmul(normed, cast(layer["v"]).t()))
        # Q/K were rounded at the Linear output; RoPE rounds them again.
        q, k = rope(torch, q.view(-1, heads, head_dim).transpose(0, 1),
                    k.view(-1, kv_heads, head_dim).transpose(0, 1),
                    position, head_dim, theta, dtype)
        q = boundary(q)
        k = boundary(k)
        v = v.view(-1, kv_heads, head_dim).transpose(0, 1)

        if index in cache:
            key = torch.cat((cache[index][0], k), dim=1)
            value = torch.cat((cache[index][1], v), dim=1)
        else:
            key, value = k, v
        cache_out[index] = (key, value)

        attended = boundary(attention(torch, q, key, value, position, dtype))
        projected = boundary(torch.matmul(
            attended.transpose(0, 1).reshape(-1, heads * head_dim),
            cast(layer["o"]).t()))
        hidden = boundary(residual + projected)

        residual = hidden
        normed = boundary(rms_norm(torch, hidden, cast(layer["post_norm"]), eps))
        gate = boundary(torch.matmul(normed, cast(layer["gate"]).t()))
        up = boundary(torch.matmul(normed, cast(layer["up"]).t()))
        # SiluMul is a single fused graph op: silu(gate) * up in the compute
        # dtype, narrowed once by the boundary.
        product = boundary(torch.nn.functional.silu(gate.to(dtype)) * up.to(dtype))
        down = boundary(torch.matmul(product, cast(layer["down"]).t()))
        hidden = boundary(residual + down)

    hidden = boundary(rms_norm(torch, hidden, cast(params["final_norm"]), eps))
    logits = torch.matmul(hidden, cast(params["lm_head"]).t())
    return logits, cache_out


def run_trajectories(torch, params, geometry, prompt, decode_inputs, near_limit,
                     compute_dtype, a16_dtype, greedy_steps):
    """Mirrors golden.trajectories with A16 cache planes and a greedy run."""
    tensors = {"prompt": torch.tensor(prompt, dtype=torch.int64),
               "decode_inputs": torch.tensor(decode_inputs, dtype=torch.int64)}
    positions = {}
    selected_rows = {}
    committed_positions = []
    stages = [("prefill", prompt, 0), ("decode_0", [decode_inputs[0]], len(prompt)),
              ("decode_1", [decode_inputs[1]], len(prompt) + 1)]
    with torch.inference_mode():
        cache = {}
        for stage, inputs, position in stages:
            logits, cache = forward(torch, params, geometry, inputs, position, cache,
                                    compute_dtype, a16_dtype)
            committed = position + len(inputs)
            rows = [len(inputs) - 1] if stage == "prefill" and near_limit else list(range(len(inputs)))
            tensors[stage + ".logit_rows"] = torch.tensor(rows, dtype=torch.int64)
            tensors[stage + ".logits"] = logits[rows].contiguous().clone()
            tensors[stage + ".tokens"] = logits.argmax(dim=-1).to(torch.int64).clone()
            tensors[stage + ".keys"] = torch.stack(
                [plane[0].permute(1, 0, 2) for plane in cache.values()]).clone()
            tensors[stage + ".values"] = torch.stack(
                [plane[1].permute(1, 0, 2) for plane in cache.values()]).clone()
            tensors[stage + ".position"] = torch.tensor([committed], dtype=torch.int64)
            positions[stage] = list(range(position, committed))
            selected_rows[stage] = rows
            committed_positions.append(committed)

        greedy = []
        margins = []
        inputs = prompt
        position = 0
        cache = {}
        for _ in range(greedy_steps):
            logits, cache = forward(torch, params, geometry, inputs, position, cache,
                                    compute_dtype, a16_dtype)
            last = logits[-1]
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


def writeback_f32(torch, tensors):
    """The FP64-reduction oracle is written back to F32 for comparison."""
    return {name: (value.to(torch.float32) if value.is_floating_point() else value)
            for name, value in tensors.items()}


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_selftest(modules):
    torch, _, numpy, _, _, _ = modules
    return {"converter": kv16.converter_selftest(torch, numpy),
            "comparator": kv16.comparator_selftest(numpy)}


def mode_crosscheck(modules, golden_bundle, recipe, cases_filter):
    """Validates this implementation against the accepted FP32 HF golden."""
    torch = modules[0]
    from safetensors.torch import load_file
    results = {}
    overall = True
    for spec in recipe["cases"]:
        case_id = spec["id"]
        if cases_filter and case_id not in cases_filter:
            continue
        case_dir = Path(golden_bundle) / case_id
        model, geometry = load_model(modules, case_dir / "model")
        params = extract_params(torch, model, geometry, None)
        near_limit = spec.get("near_limit", False)
        prompt = kv16.case_prompt(recipe, geometry, near_limit)
        tensors, trajectory = run_trajectories(
            torch, params, geometry, prompt, recipe["decode_inputs"], near_limit,
            torch.float32, None, recipe["greedy_steps"])
        reference = load_file(str(case_dir / "reference.safetensors"))
        record = json.loads((case_dir / "reference.json").read_text())
        report = kv16.compare_trajectory(tensors, trajectory, reference,
                                         record["trajectory"], GOLDEN_TOLERANCE,
                                         require_exact_tokens=True)
        results[case_id] = report
        overall = overall and report["pass"]
        print(f"crosscheck {case_id}: {'PASS' if report['pass'] else 'FAIL'}", flush=True)
        del model, params, tensors
    return {"pass": overall, "cases": results}


def mode_calibrate(modules, a16_bundle, recipe, output_dir, cases_filter):
    torch = modules[0]
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise kv16.CalibrationError("Calibration output must not already exist")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    rid = run_id()
    measurements = {"schema_version": SCHEMA_VERSION, "calibrator_version": CALIBRATOR_VERSION,
                    "run_id": rid, "cases": {}}
    with tempfile.TemporaryDirectory(prefix=".a16-calibration-", dir=output_dir.parent) as tmp:
        staging = Path(tmp)
        for spec in recipe["cases"]:
            case_id = spec["id"]
            if cases_filter and case_id not in cases_filter:
                continue
            near_limit = spec.get("near_limit", False)
            print(f"calibrating {case_id}", flush=True)
            case_out = staging / case_id
            case_out.mkdir()
            case_measure = {"model_files": {}, "combos": {}}
            for weight_dtype in ("f16", "bf16"):
                model_dir = Path(a16_bundle) / case_id / f"w{weight_dtype}"
                case_measure["model_files"][f"w{weight_dtype}"] = {
                    name: golden.file_identity(model_dir / name)
                    for name in ("config.json", "model.safetensors")}
                model, geometry = load_model(modules, model_dir)
                params = extract_params(torch, model, geometry, weight_dtype)
                prompt = kv16.case_prompt(recipe, geometry, near_limit)

                # Quality reference: same W16 checkpoint, F32 compute, KV32.
                quality_tensors, quality_traj = golden.trajectories(
                    torch, model, geometry, prompt, recipe["decode_inputs"],
                    near_limit, recipe["tolerance"])
                del model

                impl_tensors, impl_traj = run_trajectories(
                    torch, params, geometry, prompt, recipe["decode_inputs"], near_limit,
                    torch.float32, weight_dtype, recipe["greedy_steps"])
                oracle_tensors, oracle_traj = run_trajectories(
                    torch, params, geometry, prompt, recipe["decode_inputs"], near_limit,
                    torch.float64, weight_dtype, recipe["greedy_steps"])
                oracle_tensors = writeback_f32(torch, oracle_tensors)

                combo = f"w{weight_dtype}-a{weight_dtype}-kv{weight_dtype}"
                impl_report = kv16.compare_trajectory(
                    impl_tensors, impl_traj, oracle_tensors, oracle_traj,
                    BUDGETS[("impl", weight_dtype)], require_exact_tokens=False)
                quality_report = kv16.compare_trajectory(
                    impl_tensors, impl_traj, quality_tensors, quality_traj,
                    BUDGETS[("quality", weight_dtype)], require_exact_tokens=False)
                combo_dir = case_out / combo
                combo_dir.mkdir()
                tensor_records = {}
                for label, tensors in (("impl_f32", impl_tensors),
                                       ("impl_f64_oracle", oracle_tensors),
                                       ("quality_w16_a32", quality_tensors)):
                    encoded, records = golden.encode_tensors(torch, tensors)
                    (combo_dir / f"{label}.safetensors").write_bytes(encoded)
                    tensor_records[label] = records
                case_measure["combos"][combo] = {
                    "weight_dtype": weight_dtype,
                    "activation_dtype": weight_dtype,
                    "kv_dtype": weight_dtype,
                    "impl_correctness": impl_report,
                    "relative_w16_a32": quality_report,
                    "tensors": tensor_records,
                    "pass": impl_report["pass"] and quality_report["pass"],
                }
                print(f"  {combo}: impl={'PASS' if impl_report['pass'] else 'FAIL'} "
                      f"quality={'PASS' if quality_report['pass'] else 'FAIL'}", flush=True)
                del impl_tensors, oracle_tensors, quality_tensors, params
            (case_out / "measurements.json").write_bytes(golden.json_bytes(case_measure))
            measurements["cases"][case_id] = case_measure
        identity = golden.reference_identity(modules, recipe["seed"])
        identity["execution"]["calibration_compute_dtypes"] = ["float32", "float64"]
        identity["execution"]["a16_forward"] = (
            "from-scratch Llama implementation; explicit A16 rounding at every "
            "frozen boundary; F32 RMSNorm variance with an F32 weight and one "
            "output narrowing, F32 RoPE rotation, F32 attention accumulation, "
            "fused SiluMul (frozen placement, not HF mirroring)")
        full_manifest = {
            "schema_version": SCHEMA_VERSION,
            "calibrator": {"version": CALIBRATOR_VERSION,
                           "sha256": golden.file_identity(__file__)["sha256"],
                           "recipe_sha256": golden.file_identity(
                               ROOT / "tests/fixtures/inference_numerical/recipe.json")["sha256"]},
            "reference": identity,
            "source_bundle": str(a16_bundle),
            "budgets": {f"{layer}-{dtype}": value for (layer, dtype), value in BUDGETS.items()},
            "run_id": rid,
            "measurements": measurements,
        }
        (staging / "manifest.json").write_bytes(golden.json_bytes(full_manifest))
        staging.rename(output_dir)
    return full_manifest


def mode_compare(bundle_a, bundle_b):
    """Repeatability: two independent runs must be bit-identical numerically."""
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
    for key in ("calibrator", "budgets", "reference", "source_bundle"):
        if ma.get(key) != mb.get(key):
            differences.append({"kind": "manifest_field", "field": key})
    return {"pass": not differences,
            "compared_files": len(set(files_a) & set(files_b)),
            "differences": differences,
            "run_id_a": ma.get("run_id"), "run_id_b": mb.get("run_id")}


def run_id():
    sha = os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip() or "nogit"
    dirty = os.popen("git status --porcelain 2>/dev/null").read().strip()
    host = platform.node().split(".")[0].lower()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{stamp}_{sha}{'-dirty' if dirty else ''}_{host}_a16-calibration"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("selftest", "crosscheck", "calibrate", "compare"))
    parser.add_argument("--golden-bundle", type=Path,
                        default=ROOT / ".models/inference-numerical")
    parser.add_argument("--a16-bundle", type=Path, default=ROOT / ".models/kv16-reference")
    parser.add_argument("--compare-bundle", type=Path, default=None)
    parser.add_argument("--recipe", type=Path,
                        default=ROOT / "tests/fixtures/inference_numerical/recipe.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--report", type=Path, default=None)
    options = parser.parse_args()
    try:
        if options.mode == "compare":
            if options.compare_bundle is None:
                parser.error("compare requires --output-dir and --compare-bundle")
            result = mode_compare(options.output_dir, options.compare_bundle)
        else:
            modules = golden.load_modules()
            recipe = golden.read_recipe(options.recipe)
            if options.mode == "selftest":
                result = mode_selftest(modules)
            elif options.mode == "crosscheck":
                result = mode_crosscheck(modules, options.golden_bundle, recipe,
                                         options.cases)
            else:
                if options.output_dir is None:
                    parser.error("calibrate requires --output-dir")
                result = mode_calibrate(modules, options.a16_bundle, recipe,
                                        options.output_dir, options.cases)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"A16 calibration failed: {error}\n")
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
