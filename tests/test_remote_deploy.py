"""精简远程项目的 CPU 边界测试，不依赖本机报告或下载数据。"""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".", "_"), ROOT / "scripts/remote" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_fixture(root):
    """模拟已批准的源码与资产；不下载上游，也不读取历史报告。"""
    project, source = root / "project", root / "source"
    assets = project / "deploy/verl"
    shutil.copytree(ROOT / "deploy/verl", assets)
    names = ["pyproject.toml", "uv.lock", "verl/experimental/agent_loop/tool_agent_loop.py",
             "verl/experimental/agent_loop/tool_parser.py", "verl/trainer/ppo/v1/agent_loop_tq.py",
             "verl/workers/engine/fsdp/transformer_impl.py"]
    expected = {}
    for name in names:
        payload = (assets / name).read_bytes() if name in {"pyproject.toml", "uv.lock"} else b"# approved fixture\n"
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        expected[name] = hashlib.sha256(payload).hexdigest()
    upstream = json.loads((assets / "source.json").read_text())
    upstream["patched_files"] = expected
    (assets / "source.json").write_text(json.dumps(upstream))
    marker = {"schema_version": 2, "commit": "1876b06d0a3e4e71e06230be10af14492ca8a75b", "files": expected.copy()}
    (source / ".calculator-native.json").write_text(json.dumps(marker))
    # 只有测试资产进入此临时发布清单；SHA计算不复用生产校验器。
    files = [{"path": path.relative_to(project).as_posix(), "size": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in sorted(project.rglob("*")) if path.is_file()]
    (project / "public-release.json").write_text(json.dumps({"schema_version": 1, "files": files}))
    return project, source, marker


class DeploymentTests(unittest.TestCase):
    def test_marker_generation_records_actual_transformer_and_verifies_all_six_files(self):
        deploy = load_script("git_deploy.py")
        self.assertTrue(callable(getattr(deploy, "write_source_marker", None)), "缺少完整源码清单生成入口")
        with tempfile.TemporaryDirectory() as directory:
            project, source, expected = source_fixture(Path(directory))
            marker_path = source / ".calculator-native.json"
            marker_path.unlink()
            marker = deploy.write_source_marker(project, source)
            self.assertEqual(marker["files"], expected["files"])
            self.assertIn("verl/workers/engine/fsdp/transformer_impl.py", marker["files"])
            self.assertNotIn("verl/utils/fsdp_utils.py", marker["files"])
            self.assertEqual(deploy.verify_source(project, source), marker)
            before = marker_path.read_bytes()
            with self.assertRaises(FileExistsError):
                deploy.write_source_marker(project, source)
            self.assertEqual(marker_path.read_bytes(), before)

    def test_install_rejects_old_or_incomplete_marker_before_uv(self):
        deploy = load_script("git_deploy.py")
        for mutation in ("legacy", "missing-transformer", "empty"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                project, source, marker = source_fixture(Path(directory))
                if mutation == "legacy":
                    marker.pop("schema_version")
                elif mutation == "missing-transformer":
                    marker["files"].pop("verl/workers/engine/fsdp/transformer_impl.py")
                else:
                    marker["files"] = {}
                (source / ".calculator-native.json").write_text(json.dumps(marker))
                # 只替代平台门和真正的安装进程；实际执行资产、marker及源码校验。
                with patch.object(deploy, "native_linux"), patch.object(deploy.shutil, "which", return_value="uv"), \
                        patch.object(deploy.subprocess, "run") as installer:
                    with self.assertRaises(ValueError):
                        deploy.install(project, source, allow_network=True)
                    installer.assert_not_called()
                self.assertFalse((source / ".venv").exists())

    def test_install_rejects_source_and_marker_rehashed_together(self):
        deploy = load_script("git_deploy.py")
        with tempfile.TemporaryDirectory() as directory:
            project, source, marker = source_fixture(Path(directory))
            name = "verl/workers/engine/fsdp/transformer_impl.py"
            (source / name).write_bytes(b"# unapproved edit\n")
            marker["files"][name] = hashlib.sha256(b"# unapproved edit\n").hexdigest()
            (source / ".calculator-native.json").write_text(json.dumps(marker))
            with patch.object(deploy, "native_linux"), patch.object(deploy.shutil, "which", return_value="uv"), \
                    patch.object(deploy.subprocess, "run") as installer:
                with self.assertRaises(ValueError):
                    deploy.install(project, source, allow_network=True)
                installer.assert_not_called()

    def test_source_verification_rejects_each_missing_or_changed_target(self):
        deploy = load_script("git_deploy.py")
        names = ["pyproject.toml", "uv.lock", "verl/experimental/agent_loop/tool_agent_loop.py",
                 "verl/experimental/agent_loop/tool_parser.py", "verl/trainer/ppo/v1/agent_loop_tq.py",
                 "verl/workers/engine/fsdp/transformer_impl.py"]
        for name in names:
            for mutation in ("missing", "changed"):
                with self.subTest(name=name, mutation=mutation), tempfile.TemporaryDirectory() as directory:
                    project, source, marker = source_fixture(Path(directory))
                    if mutation == "missing":
                        (source / name).unlink()
                    else:
                        (source / name).write_bytes(b"# unapproved edit\n")
                    before = (source / ".calculator-native.json").read_bytes()
                    with self.assertRaises(ValueError):
                        deploy.verify_source(project, source)
                    with self.assertRaises(ValueError):
                        deploy.write_source_marker(project, source, replace=True)
                    self.assertEqual((source / ".calculator-native.json").read_bytes(), before)

    def test_valid_source_reaches_only_the_frozen_installer(self):
        deploy = load_script("git_deploy.py")
        with tempfile.TemporaryDirectory() as directory:
            project, source, marker = source_fixture(Path(directory))
            with patch.object(deploy, "native_linux"), patch.object(deploy.shutil, "which", return_value="uv"), \
                    patch.object(deploy.subprocess, "run") as installer:
                deploy.install(project, source, allow_network=True)
            installer.assert_called_once()
            self.assertEqual(installer.call_args.args[0], ["uv", "sync", "--frozen", "--all-packages", "--extra",
                "sglang", "--extra", "fsdp", "--python", "3.12", "--no-python-downloads"])
            self.assertEqual(installer.call_args.kwargs["cwd"], source.resolve())
            self.assertEqual(installer.call_args.kwargs["env"]["UV_PROJECT_ENVIRONMENT"], str(source.resolve() / ".venv"))

    def test_cli_repairs_old_marker_only_after_explicit_replace(self):
        deploy = load_script("git_deploy.py")
        with tempfile.TemporaryDirectory() as directory:
            project, source, marker = source_fixture(Path(directory))
            marker.pop("schema_version")
            marker_path = source / ".calculator-native.json"
            marker_path.write_text(json.dumps(marker))
            prefix = ["--project-root", str(project)]
            self.assertEqual(deploy.main(prefix + ["verify-source", "--verl-source", str(source)]), 1)
            before = marker_path.read_bytes()
            self.assertEqual(deploy.main(prefix + ["record-source", "--verl-source", str(source)]), 1)
            self.assertEqual(marker_path.read_bytes(), before)
            self.assertEqual(deploy.main(prefix + ["record-source", "--verl-source", str(source), "--replace-marker"]), 0)
            self.assertEqual(deploy.main(prefix + ["verify-source", "--verl-source", str(source)]), 0)

    @unittest.skipIf(os.name == "nt", "真实链接逃逸检查在 Linux CPU 环境执行")
    def test_source_link_is_rejected_even_when_content_hash_matches(self):
        deploy = load_script("git_deploy.py")
        with tempfile.TemporaryDirectory() as directory:
            project, source, marker = source_fixture(Path(directory))
            name = "verl/workers/engine/fsdp/transformer_impl.py"
            outside = Path(directory) / "outside.py"
            outside.write_bytes((source / name).read_bytes())
            (source / name).unlink()
            (source / name).symlink_to(outside)
            with self.assertRaises(ValueError):
                deploy.verify_source(project, source)

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
