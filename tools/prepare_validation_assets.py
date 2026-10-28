#!/usr/bin/env python3
"""Prepare or verify externally stored validation assets without accepting new data.

The repository versions recipes and reviewed checksums only.  This command is
an explicit CI/developer bootstrapper: it never downloads by default and never
rewrites a reviewed checksum or an existing asset directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
ADMISSION_RECIPE = ROOT / "tests/fixtures/inference_admission/recipe.json"
ADMISSION_CHECKSUMS = ROOT / "tests/fixtures/inference_admission/checksums.json"
NUMERICAL_CHECKSUMS = ROOT / "tests/fixtures/inference_numerical/checksums.json"
ADMISSION_GENERATOR = ROOT / "tools/generate_fp32_admission_hf_fixtures.py"
NUMERICAL_GENERATOR = ROOT / "tools/generate_llama_numerical_reference.py"
NUMERICAL_VERIFIER = ROOT / "tools/verify_llama_numerical_reference.py"


class PreparationError(ValueError):
    """A requested bootstrap action would use unreviewed or unsafe assets."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PreparationError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_bytes(), object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              PreparationError(f"non-finite JSON value in {path}: {value}")))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PreparationError(f"cannot read {path}: {error}") from error


def absolute_path(value: str, name: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise PreparationError(f"{name} must be an absolute path")
    # Check the spelling supplied at the boundary before resolve() can erase a
    # link component.  Path.exists() is false for a dangling symlink, whereas
    # is_symlink() remains true, so inspect every existing lexical component.
    assert_no_symlink(path, name)
    return path.resolve(strict=False)


def interpreter_path(value: str, name: str) -> Path:
    """Accept a normal venv executable symlink without accepting linked parents.

    Model and generated-asset paths are storage boundaries and therefore reject
    every link.  A Python executable is an explicit invocation path: virtual
    environments conventionally expose ``bin/python3`` as a final symlink, so
    retain that lexical final component for subprocess instead of resolving it.
    """
    path = Path(value)
    if not path.is_absolute():
        raise PreparationError(f"{name} must be an absolute path")
    parent = path.parent
    assert_no_symlink(parent, name)
    return path


def assert_no_symlink(path: Path, name: str) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise PreparationError(f"{name} must not traverse a symlink: {current}")


def require_child(path: Path, parent: Path, name: str) -> None:
    try:
        path.relative_to(parent)
    except ValueError as error:
        raise PreparationError(f"{name} must stay under {parent}") from error


def path_record(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def run_checked(command: list[str], label: str) -> None:
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise PreparationError(f"{label} failed ({result.returncode}): {detail}")


def recipe_source() -> tuple[str, str, dict]:
    recipe = strict_json(ADMISSION_RECIPE)
    source = recipe.get("source")
    if not isinstance(source, dict):
        raise PreparationError("admission recipe has no source identity")
    repository, revision, files = source.get("repository"), source.get("revision"), source.get("files")
    if not isinstance(repository, str) or not isinstance(revision, str) or not isinstance(files, dict):
        raise PreparationError("admission recipe source identity is malformed")
    if set(files) != {"config.json", "model.safetensors"}:
        raise PreparationError("admission recipe must pin config.json and model.safetensors")
    return repository, revision, files


def source_matches(source: Path) -> bool:
    _, _, files = recipe_source()
    if not source.is_dir() or source.is_symlink():
        return False
    try:
        for name, expected in files.items():
            path = source / name
            if path.is_symlink() or not path.is_file() or path_record(path) != expected:
                return False
    except OSError:
        return False
    return True


def atomic_download_source(source: Path) -> None:
    repository, revision, files = recipe_source()
    if source.exists() or source.is_symlink():
        raise PreparationError(f"source already exists and will not be overwritten: {source}")
    source.parent.mkdir(parents=True, exist_ok=True)
    assert_no_symlink(source.parent, "source parent")
    with tempfile.TemporaryDirectory(prefix=".validation-source-", dir=source.parent) as temporary:
        staging = Path(temporary) / "source"
        staging.mkdir()
        for name, expected in files.items():
            url = f"https://huggingface.co/{repository}/resolve/{revision}/{name}"
            destination = staging / name
            try:
                with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as output:
                    shutil.copyfileobj(response, output)
            except OSError as error:
                raise PreparationError(f"download failed for {name}: {error}") from error
            if path_record(destination) != expected:
                raise PreparationError(f"downloaded source identity mismatch: {name}")
        os.replace(staging, source)


def verify_admission(source: Path, destination: Path) -> None:
    run_checked([sys.executable, str(ADMISSION_GENERATOR), "--source", str(source),
                 "--recipe", str(ADMISSION_RECIPE), "--output-dir", str(destination),
                 "--check", str(ADMISSION_CHECKSUMS)], "admission fixture verification")


def verify_numerical(destination: Path) -> None:
    run_checked([sys.executable, str(NUMERICAL_VERIFIER), "--root", str(destination),
                 "--expected", str(NUMERICAL_CHECKSUMS)], "numerical reference verification")


def publish_tree(staging: Path, destination: Path, label: str) -> None:
    if destination.exists() or destination.is_symlink():
        raise PreparationError(f"{label} destination already exists and will not be overwritten: {destination}")
    if not staging.is_dir() or staging.is_symlink():
        raise PreparationError(f"{label} staging output is invalid")
    os.replace(staging, destination)


def prepare(options: argparse.Namespace) -> dict:
    models = options.models_dir
    source = options.source
    admission = models / "inference-admission"
    numerical = models / "inference-numerical"
    for path, name in ((models, "models directory"), (source, "source"),
                       (admission, "admission destination"), (numerical, "numerical destination")):
        assert_no_symlink(path, name)
    require_child(source, models, "source")
    if source in (admission, numerical) or source in admission.parents or source in numerical.parents:
        raise PreparationError("source must not overlap generated validation assets")
    status = {"schema_version": 1, "status": "prepared", "models_dir": str(models),
              "source": "verified", "admission": "verified", "numerical": "verified",
              "downloaded_source": False}

    if not source_matches(source):
        if source.exists():
            raise PreparationError(f"source does not match the pinned recipe: {source}")
        if not options.download_source:
            raise PreparationError(
                f"pinned source is missing: {source}; rerun with --download-source or prepare it using "
                "tests/fixtures/inference_admission/README.md")
        atomic_download_source(source)
        status["downloaded_source"] = True
    if not source_matches(source):
        raise PreparationError("downloaded source failed its pinned identity check")

    if admission.exists() or admission.is_symlink():
        verify_admission(source, admission)
    else:
        with tempfile.TemporaryDirectory(prefix=".validation-admission-generate-", dir=models) as temporary:
            staging = Path(temporary) / "bundle"
            run_checked([sys.executable, str(ADMISSION_GENERATOR), "--source", str(source),
                         "--recipe", str(ADMISSION_RECIPE), "--output-dir", str(staging)],
                        "admission fixture generation")
            verify_admission(source, staging)
            publish_tree(staging, admission, "admission fixtures")
        status["admission"] = "generated_and_verified"

    if numerical.exists() or numerical.is_symlink():
        verify_numerical(numerical)
        return status
    if options.numerical_bundle is not None:
        bundle = options.numerical_bundle
        assert_no_symlink(bundle, "numerical bundle")
        if bundle == numerical or numerical in bundle.parents or bundle in numerical.parents:
            raise PreparationError("numerical bundle must not overlap its destination")
        verify_numerical(bundle)
        with tempfile.TemporaryDirectory(prefix=".validation-numerical-copy-", dir=models) as temporary:
            staging = Path(temporary) / "bundle"
            shutil.copytree(bundle, staging, symlinks=False)
            verify_numerical(staging)
            publish_tree(staging, numerical, "numerical bundle")
        status["numerical"] = "copied_and_verified"
        return status
    if options.reference_python is None:
        raise PreparationError(
            "numerical bundle is missing; supply --numerical-bundle with a reviewed bundle or "
            "--reference-python for the pinned isolated HF environment")
    interpreter = options.reference_python
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise PreparationError("reference Python must be an executable regular file")
    with tempfile.TemporaryDirectory(prefix=".validation-numerical-generate-", dir=models) as temporary:
        staging = Path(temporary) / "bundle"
        run_checked([str(interpreter), str(NUMERICAL_GENERATOR), "--admission-root", str(admission),
                     "--output-dir", str(staging)], "numerical candidate generation")
        verify_numerical(staging)
        publish_tree(staging, numerical, "numerical bundle")
    status["numerical"] = "generated_and_verified"
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=lambda value: absolute_path(value, "--models-dir"),
                        default=absolute_path(str(ROOT / ".models"), "--models-dir"))
    parser.add_argument("--source", type=lambda value: absolute_path(value, "--source"))
    parser.add_argument("--reference-python", type=lambda value: interpreter_path(value, "--reference-python"))
    parser.add_argument("--numerical-bundle", type=lambda value: absolute_path(value, "--numerical-bundle"))
    parser.add_argument("--download-source", action="store_true")
    options = parser.parse_args(argv)
    if options.source is None:
        options.source = absolute_path(str(options.models_dir / "tiny-random-LlamaForCausalLM"), "--source")
    try:
        result = prepare(options)
    except (PreparationError, OSError, subprocess.SubprocessError) as error:
        print(json.dumps({"schema_version": 1, "status": "failed", "error": str(error)},
                         sort_keys=True, allow_nan=False))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
