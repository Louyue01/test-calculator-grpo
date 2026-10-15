"""精简远程项目的 CPU 边界测试，不依赖本机报告或下载数据。"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), ROOT / "scripts/remote" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeploymentTests(unittest.TestCase):
    def test_standalone_manifest_verification_rejects_changed_and_extra_files(self):
        deploy = load_script("git_deploy.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = root / "code.py"
            code.write_bytes(b"pass\n")
            # 摘要取自固定夹具，不从待测校验器反向生成期待结果。
            manifest = {"schema_version": 1, "files": [{"path": "code.py", "size": 5,
                "sha256": hashlib.sha256(b"pass\n").hexdigest()}]}
            (root / "public-release.json").write_text(json.dumps(manifest))
            self.assertTrue(deploy.verify_public(root))
            (root / "extra.txt").write_text("not approved")
            with self.assertRaises(ValueError):
                deploy.verify_public(root)
            (root / "extra.txt").unlink()
            code.write_bytes(b"fail\n")
            with self.assertRaises(ValueError):
                deploy.verify_public(root)

    def test_manifest_path_escape_is_rejected(self):
        deploy = load_script("git_deploy.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "public-release.json").write_text(json.dumps({"schema_version": 1, "files": [
                {"path": "../outside.py", "size": 1, "sha256": "0" * 64}]}))
            with self.assertRaises(ValueError):
                deploy.verify_public(root)

    def test_install_and_clone_are_opt_in_and_do_not_create_paths(self):
        deploy = load_script("git_deploy.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("prepare-source", "install"):
                self.assertEqual(deploy.main(["--project-root", str(root), name,
                                              "--verl-source", str(root / "source")]), 1)
            self.assertEqual(list(root.iterdir()), [])

    def test_wrong_archive_is_rejected_before_directory_creation(self):
        deploy = load_script("git_deploy.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "wrong.tar"
            archive.write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                deploy.prepare_source(root, root / "source", base_archive=archive)
            self.assertFalse((root / "source").exists())

    def test_data_verification_rejects_unknown_files_without_downloading(self):
        inputs = load_script("prepare_inputs.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train.parquet").write_bytes(b"wrong data")
            with self.assertRaises((ValueError, FileNotFoundError)):
                inputs.verify_data(root)

    def test_model_download_requires_explicit_permission(self):
        inputs = load_script("prepare_inputs.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                inputs.download_model(root, allow_network=False)
            self.assertEqual(list(root.iterdir()), [])

    @unittest.skipIf(os.name == "nt", "Bash 用例在已有 Linux CPU 环境执行")
    def test_canary_without_inputs_does_not_create_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "run"
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith("CALCULATOR_") and key not in {"VERL_SOURCE_DIR", "PYTHON_BIN"}}
            env["CALCULATOR_RUN_ROOT"] = str(target)
            result = subprocess.run(["bash", str(ROOT / "scripts/remote/canary.sh")],
                                    env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
