#!/usr/bin/env python3
"""Generate the independent FP16 compute / FP32 residual Llama fixture."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import calibrate_a16_reference as a16
import calibrate_kv16_reference as kv16
import generate_llama_numerical_reference as golden
import prepare_a16_reference as fixture


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "tests/fixtures/inference_numerical/recipe.json"
IMPLEMENTATION = {"logits": (1e-4, 1e-3), "kv": (2e-4, 2e-3)}
QUALITY = {"logits": (2e-3, 6e-3), "kv": (1e-3, 1e-2)}


def forward(torch, params, geometry, tokens, position, cache, dtype):
    """Evaluate the same boundaries with F32 or FP64 arithmetic."""
    hidden = params["embed"][torch.tensor(tokens)].float()
    head_dim = geometry["head_dim"]
    heads = geometry["num_attention_heads"]
    kv_heads = geometry["num_key_value_heads"]
    eps = geometry["rms_norm_eps"]

    def to_branch(value):
        return value.to(torch.float16).to(dtype)

    def project(value, weight):
        return value @ weight.to(dtype).t()

    def norm_to_branch(residual, weight):
        f = residual.to(dtype)
        mean_sq = (f * f).mean(dim=-1, keepdim=True)
        normalized = f / torch.sqrt(mean_sq + eps) * weight.to(dtype)
        # The F32 RMSNorm output is itself a stored tensor before Cast.
        return to_branch(normalized.float())

    inv_freq = geometry["rope_theta"] ** (
        -torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    positions = torch.arange(position, position + len(tokens), dtype=torch.float64)
    angle = positions[:, None] * inv_freq
    cosine, sine = angle.cos().to(dtype), angle.sin().to(dtype)

    def rotate(x):
        first, second = x.chunk(2, dim=-1)
        return to_branch(torch.cat((first * cosine - second * sine,
                                   second * cosine + first * sine), dim=-1))

    updated = {}
    for index, layer in enumerate(params["layers"]):
        normed = norm_to_branch(hidden, layer["input_norm"])
        q = rotate(to_branch(project(normed, layer["q"])).view(-1, heads, head_dim).transpose(0, 1))
        k = rotate(to_branch(project(normed, layer["k"])).view(-1, kv_heads, head_dim).transpose(0, 1))
        v = to_branch(project(normed, layer["v"])).view(-1, kv_heads, head_dim).transpose(0, 1)
        if index in cache:
            k = torch.cat((cache[index][0], k), dim=1)
            v = torch.cat((cache[index][1], v), dim=1)
        updated[index] = (k, v)
        attended = to_branch(a16.attention(torch, q, k, v, position, dtype))
        projected = to_branch(project(attended.transpose(0, 1).reshape(len(tokens), -1), layer["o"]))
        hidden = hidden + projected.float()

        normed = norm_to_branch(hidden, layer["post_norm"])
        gate = to_branch(project(normed, layer["gate"]))
        up = to_branch(project(normed, layer["up"]))
        product = to_branch(torch.nn.functional.silu(gate) * up)
        down = to_branch(project(product, layer["down"]))
        hidden = hidden + down.float()
    logits = project(norm_to_branch(hidden, params["final_norm"]), params["lm_head"]).float()
    return logits, updated


def trajectory(torch, params, geometry, recipe, near_limit, dtype):
    prompt = kv16.case_prompt(recipe, geometry, near_limit)
    decode = recipe["decode_inputs"]
    tensors = {"prompt": torch.tensor(prompt), "decode_inputs": torch.tensor(decode)}
    record = {"prompt": prompt, "decode_inputs": decode, "positions": {},
              "logit_rows": {}, "commit_positions": [], "greedy_tokens": [], "greedy_margins": []}
    cache = {}
    position = 0
    for stage, inputs in zip(kv16.STAGES, (prompt, [decode[0]], [decode[1]])):
        logits, cache = forward(torch, params, geometry, inputs, position, cache, dtype)
        rows = [len(inputs) - 1] if stage == "prefill" and near_limit else list(range(len(inputs)))
        keys, values = map(list, zip(*cache.values()))
        tensors[stage + ".logit_rows"] = torch.tensor(rows)
        tensors[stage + ".logits"] = logits[rows].contiguous()
        tensors[stage + ".tokens"] = logits.argmax(-1)
        for name, plane in (("keys", keys), ("values", values)):
            tensors[stage + "." + name] = torch.stack([x.transpose(0, 1) for x in plane]).float()
        end = position + len(inputs)
        tensors[stage + ".position"] = torch.tensor([end])
        record["positions"][stage] = list(range(position, end))
        record["logit_rows"][stage] = rows
        record["commit_positions"].append(end)
        if stage == "prefill":
            greedy_cache, greedy_logits = cache, logits
        position = end

    position = len(prompt)
    for i in range(recipe["greedy_steps"]):
        top, tokens = torch.topk(greedy_logits[-1], 2)
        token = int(tokens[0])
        record["greedy_tokens"].append(token)
        record["greedy_margins"].append(float(top[0] - top[1]))
        if i + 1 < recipe["greedy_steps"]:
            greedy_logits, greedy_cache = forward(torch, params, geometry, [token],
                                                   position, greedy_cache, dtype)
            position += 1
    tensors["greedy_tokens"] = torch.tensor(record["greedy_tokens"])
    tensors["greedy_margins"] = torch.tensor(record["greedy_margins"], dtype=torch.float32)
    return tensors, record


def generate(source, output):
    modules = golden.load_modules()
    torch = modules[0]
    recipe = golden.read_recipe(RECIPE)
    measurements = {}
    with tempfile.TemporaryDirectory(prefix=".p4b-", dir=output.parent) as tmp:
        staged = Path(tmp) / "fixture"
        staged.mkdir()
        with torch.inference_mode():
            for spec in recipe["cases"]:
                name = spec["id"]
                case = source / name / "wf16"
                out = staged / name
                out.mkdir()
                for filename in fixture.CHECKPOINT_FILES:
                    shutil.copyfile(case / filename, out / filename)
                model, geometry = a16.load_model(modules, case)
                params = a16.extract_params(torch, model, geometry, "f16")
                long = spec.get("near_limit", False)
                actual, actual_record = trajectory(torch, params, geometry, recipe, long, torch.float32)
                oracle, oracle_record = trajectory(torch, params, geometry, recipe, long, torch.float64)
                quality, quality_record = golden.trajectories(
                    torch, model, geometry, oracle_record["prompt"], recipe["decode_inputs"],
                    long, recipe["tolerance"])
                implementation = kv16.compare_trajectory(actual, actual_record, oracle,
                                                         oracle_record, IMPLEMENTATION, False)
                quality = kv16.compare_trajectory(actual, actual_record, quality,
                                                  quality_record, QUALITY, False)
                if not implementation["pass"] or not quality["pass"]:
                    raise ValueError(f"{name}: P4-B reference calibration failed: "
                                     f"{json.dumps({'implementation': implementation, 'quality': quality})}")
                encoded, _ = golden.encode_tensors(torch, oracle)
                (out / "oracle.safetensors").write_bytes(encoded)
                measurements[name] = {"implementation": implementation, "quality": quality}
                print(name, "pass", flush=True)
        files = {p.relative_to(staged).as_posix(): golden.file_identity(p)
                 for p in sorted(staged.rglob("*")) if p.is_file()}
        manifest = {"files": files, "recipe": golden.file_identity(RECIPE),
                    "reference": golden.reference_identity(modules, recipe["seed"]),
                    "implementation_budget": IMPLEMENTATION, "quality_budget": QUALITY,
                    "measurements": measurements}
        (staged / "manifest.json").write_bytes(golden.json_bytes(manifest))
        staged.rename(output)


def verify(output):
    manifest = json.loads((output / "manifest.json").read_text())
    cases = {spec["id"] for spec in golden.read_recipe(RECIPE)["cases"]}
    files = {f"{case}/{name}" for case in cases
             for name in (*fixture.CHECKPOINT_FILES, "oracle.safetensors")}
    if manifest["recipe"] != golden.file_identity(RECIPE) or set(manifest["files"]) != files:
        raise ValueError("P4-B fixture differs from the frozen recipe")
    for name, identity in manifest["files"].items():
        if golden.file_identity(output / name) != identity:
            raise ValueError(f"P4-B fixture identity mismatch: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / ".models/kv16-reference")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify:
            verify(args.output)
        else:
            if args.output.exists() or args.output.is_symlink():
                parser.error("output must not already exist")
            generate(args.source, args.output)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"P4-B fixture failed: {error}\n")


if __name__ == "__main__":
    main()
