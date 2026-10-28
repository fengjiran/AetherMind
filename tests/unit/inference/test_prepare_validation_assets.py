"""Contracts for the no-overwrite validation asset bootstrapper."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("prepare_validation_assets",
                                              ROOT / "tools/prepare_validation_assets.py")
PREPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREPARE
SPEC.loader.exec_module(PREPARE)


class PrepareValidationAssetsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.models = Path(self.temporary.name) / "models"
        self.models.mkdir()
        self.source = self.models / "tiny-random-LlamaForCausalLM"
        self.source.mkdir()
        self.recipe = {"source": {"repository": "owner/repo", "revision": "a" * 40,
                                   "files": {}}}
        self.files = {"config.json": b"config", "model.safetensors": b"weights"}
        for name, data in self.files.items():
            (self.source / name).write_bytes(data)
            self.recipe["source"]["files"][name] = {"bytes": len(data),
                                                        "sha256": hashlib.sha256(data).hexdigest()}
        self.recipe_patch = mock.patch.object(PREPARE, "recipe_source",
                                               return_value=("owner/repo", "a" * 40,
                                                             self.recipe["source"]["files"]))
        self.recipe_patch.start()
        self.addCleanup(self.recipe_patch.stop)

    def options(self, **changes):
        result = argparse.Namespace(models_dir=self.models, source=self.source,
                                    reference_python=None, numerical_bundle=None,
                                    download_source=False)
        for name, value in changes.items():
            setattr(result, name, value)
        return result

    def create_admission(self):
        path = self.models / "inference-admission"
        path.mkdir()
        return path

    def create_numerical(self):
        path = self.models / "inference-numerical"
        path.mkdir()
        return path

    def test_existing_assets_only_verify(self):
        self.create_admission()
        self.create_numerical()
        with mock.patch.object(PREPARE, "verify_admission") as admission, \
             mock.patch.object(PREPARE, "verify_numerical") as numerical:
            result = PREPARE.prepare(self.options())
        self.assertEqual(result["status"], "prepared")
        admission.assert_called_once()
        numerical.assert_called_once()

    def test_missing_source_requires_explicit_download(self):
        for path in self.source.iterdir():
            path.unlink()
        self.source.rmdir()
        with self.assertRaisesRegex(PREPARE.PreparationError, "--download-source"):
            PREPARE.prepare(self.options())

    def test_bad_existing_source_is_not_overwritten(self):
        (self.source / "config.json").write_bytes(b"bad")
        with self.assertRaisesRegex(PREPARE.PreparationError, "does not match"):
            PREPARE.prepare(self.options(download_source=True))

    def test_missing_admission_calls_generator_then_verifies(self):
        self.create_numerical()
        calls = []
        def run(command, label):
            calls.append(label)
            if "generation" in label:
                Path(command[command.index("--output-dir") + 1]).mkdir()
        with mock.patch.object(PREPARE, "run_checked", side_effect=run), \
             mock.patch.object(PREPARE, "verify_numerical"):
            result = PREPARE.prepare(self.options())
        self.assertEqual(result["admission"], "generated_and_verified")
        self.assertEqual(calls, ["admission fixture generation", "admission fixture verification"])

    def test_existing_admission_is_never_regenerated(self):
        self.create_admission()
        self.create_numerical()
        with mock.patch.object(PREPARE, "verify_admission"), \
             mock.patch.object(PREPARE, "verify_numerical"), \
             mock.patch.object(PREPARE, "run_checked") as command:
            PREPARE.prepare(self.options())
        command.assert_not_called()

    def test_missing_numerical_requires_explicit_trusted_input(self):
        self.create_admission()
        with mock.patch.object(PREPARE, "verify_admission"):
            with self.assertRaisesRegex(PREPARE.PreparationError, "numerical bundle is missing"):
                PREPARE.prepare(self.options())

    def test_existing_numerical_is_only_verified(self):
        self.create_admission()
        self.create_numerical()
        with mock.patch.object(PREPARE, "verify_admission"), \
             mock.patch.object(PREPARE, "verify_numerical") as numerical:
            PREPARE.prepare(self.options())
        numerical.assert_called_once_with(self.models / "inference-numerical")

    def test_bundle_copy_verifies_source_and_staging_before_publish(self):
        self.create_admission()
        bundle = Path(self.temporary.name) / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text("{}")
        seen = []
        with mock.patch.object(PREPARE, "verify_admission"), \
             mock.patch.object(PREPARE, "verify_numerical", side_effect=lambda path: seen.append(path)):
            result = PREPARE.prepare(self.options(numerical_bundle=bundle))
        self.assertEqual(result["numerical"], "copied_and_verified")
        self.assertEqual(seen[0], bundle)
        self.assertTrue((self.models / "inference-numerical").is_dir())
        self.assertEqual(len(seen), 2)

    def test_bundle_target_is_never_overwritten(self):
        self.create_admission()
        self.create_numerical()
        bundle = Path(self.temporary.name) / "bundle"
        bundle.mkdir()
        with mock.patch.object(PREPARE, "verify_admission"), \
             mock.patch.object(PREPARE, "verify_numerical"):
            PREPARE.prepare(self.options(numerical_bundle=bundle))
        # Existing accepted target wins; bundle input is not copied over it.
        self.assertFalse((self.models / "inference-numerical" / "manifest.json").exists())

    def test_bundle_overlap_is_rejected(self):
        self.create_admission()
        bundle = self.models / "inference-numerical"
        bundle.mkdir()
        with mock.patch.object(PREPARE, "verify_admission"), \
             mock.patch.object(PREPARE, "verify_numerical"):
            PREPARE.prepare(self.options(numerical_bundle=bundle))

    def test_reference_python_must_be_regular_executable(self):
        self.create_admission()
        bad = Path(self.temporary.name) / "python"
        bad.write_text("not executable")
        with mock.patch.object(PREPARE, "verify_admission"):
            with self.assertRaisesRegex(PREPARE.PreparationError, "executable regular file"):
                PREPARE.prepare(self.options(reference_python=bad))

    def test_admission_generation_failure_does_not_publish_final_directory(self):
        self.create_numerical()
        with mock.patch.object(PREPARE, "run_checked", side_effect=PREPARE.PreparationError("generator failed")):
            with self.assertRaisesRegex(PREPARE.PreparationError, "generator failed"):
                PREPARE.prepare(self.options())
        self.assertFalse((self.models / "inference-admission").exists())

    def test_candidate_generation_verifies_before_publication(self):
        self.create_admission()
        executable = Path(self.temporary.name) / "python"
        executable.write_text("")
        executable.chmod(0o700)
        verified = []
        def run(command, label):
            Path(command[-1]).mkdir()
        with mock.patch.object(PREPARE, "verify_admission"), \
             mock.patch.object(PREPARE, "run_checked", side_effect=run), \
             mock.patch.object(PREPARE, "verify_numerical", side_effect=lambda path: verified.append(path)):
            result = PREPARE.prepare(self.options(reference_python=executable))
        self.assertEqual(result["numerical"], "generated_and_verified")
        self.assertEqual(len(verified), 1)
        self.assertTrue((self.models / "inference-numerical").is_dir())

    def test_source_must_be_under_models_directory(self):
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        with self.assertRaisesRegex(PREPARE.PreparationError, "stay under"):
            PREPARE.prepare(self.options(source=outside))

    def test_symlinked_models_path_is_rejected(self):
        target = Path(self.temporary.name) / "target"
        target.mkdir()
        alias = Path(self.temporary.name) / "alias"
        alias.symlink_to(target, target_is_directory=True)
        with self.assertRaises(SystemExit):
            PREPARE.main(["--models-dir", str(alias)])

    def test_cli_rejects_source_symlink_before_normalizing_it(self):
        target = Path(self.temporary.name) / "source-target"
        target.mkdir()
        alias = self.models / "source-link"
        alias.symlink_to(target, target_is_directory=True)
        with self.assertRaises(SystemExit):
            PREPARE.main(["--models-dir", str(self.models), "--source", str(alias)])

    def test_cli_rejects_numerical_bundle_symlink_before_normalizing_it(self):
        target = Path(self.temporary.name) / "bundle-target"
        target.mkdir()
        alias = Path(self.temporary.name) / "bundle-link"
        alias.symlink_to(target, target_is_directory=True)
        with self.assertRaises(SystemExit):
            PREPARE.main(["--models-dir", str(self.models), "--numerical-bundle", str(alias)])

    def test_reference_python_final_symlink_is_preserved_for_subprocess(self):
        self.create_admission()
        target = Path(self.temporary.name) / "real-python"
        target.write_text("")
        target.chmod(0o700)
        alias = Path(self.temporary.name) / "venv-python3"
        alias.symlink_to(target)
        seen = []
        def run(command, label):
            seen.append(command)
            Path(command[command.index("--output-dir") + 1]).mkdir()
        with mock.patch.object(PREPARE, "verify_admission"), \
             mock.patch.object(PREPARE, "run_checked", side_effect=run), \
             mock.patch.object(PREPARE, "verify_numerical"):
            result = PREPARE.prepare(self.options(reference_python=PREPARE.interpreter_path(str(alias), "--reference-python")))
        self.assertEqual(result["numerical"], "generated_and_verified")
        self.assertEqual(seen[0][0], str(alias))

    def test_download_validates_both_files_before_publication(self):
        for path in self.source.iterdir():
            path.unlink()
        self.source.rmdir()
        def download(path):
            path.mkdir()
            for name, data in self.files.items():
                (path / name).write_bytes(data)
        with mock.patch.object(PREPARE, "atomic_download_source", side_effect=download), \
             mock.patch.object(PREPARE, "verify_admission"), \
             mock.patch.object(PREPARE, "verify_numerical"):
            self.create_admission()
            self.create_numerical()
            result = PREPARE.prepare(self.options(download_source=True))
        self.assertTrue(result["downloaded_source"])

    def test_cli_rejects_relative_models_dir(self):
        with self.assertRaises(PREPARE.PreparationError):
            PREPARE.absolute_path("relative", "--models-dir")


if __name__ == "__main__":
    unittest.main()
