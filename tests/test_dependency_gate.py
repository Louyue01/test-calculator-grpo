"""依赖例外门禁的 CPU 测试；不下载包或导入 GPU 库。"""

import importlib.util
import contextlib
import io
import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_gate():
    spec = importlib.util.spec_from_file_location("dependency_gate", ROOT / "scripts/remote/check_dependencies.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot():
    # 期待值来自已审查的实际 uv 输出，不调用生产代码构造例外。
    return {
        "python": "3.12.3", "system": "Linux", "release": "6.8.0-generic",
        "attention_backend": "torch_native", "rollout_engine": "sglang",
        "versions": {"verl": "0.9.1", "torch": "2.11.0+cu130", "sglang": "0.5.12",
                     "transformers": "5.9.0", "ray": "2.55.1", "numpy": "2.3.5",
                     "flash-attn": "2.8.3", "flash-attn-4": None},
        "returncode": 1,
        "stdout": "",
        "stderr": "Using Python 3.12.3 environment at: /tmp/test-env\n"
                  "Checked 249 packages in 54ms\nFound 2 incompatibilities\n"
                  "The package `sglang` requires `flash-attn-4>=4.0.0b9`, but it's not installed\n"
                  "The package `sglang` requires `transformers==5.6.0`, but `5.9.0` is installed\n",
    }


class DependencyGateTests(unittest.TestCase):
    def test_exact_exceptions_are_distinct_from_clean_and_keep_original_output(self):
        gate = load_gate()
        case = snapshot()
        report = gate.assess_snapshot(case)
        self.assertTrue(report["ok"])
        self.assertEqual(report["status"], "accepted_exceptions")
        self.assertEqual(report["raw"], {"returncode": 1, "stdout": "", "stderr": case["stderr"]})
        self.assertEqual(len(report["exceptions"]), 2)

    def test_unknown_or_missing_diagnostics_are_not_swallowed(self):
        gate = load_gate()
        for mutation in ("extra", "missing", "empty", "changed-requirement", "duplicate", "unknown-header"):
            with self.subTest(mutation=mutation):
                case = snapshot()
                if mutation == "extra":
                    case["stderr"] += "The package `mistral-common` requires `numpy<2.4`, but `2.4.6` is installed\n"
                elif mutation == "missing":
                    case["stderr"] = case["stderr"].split("The package `sglang` requires `transformers")[0]
                elif mutation == "empty":
                    case["stderr"] = ""
                elif mutation == "changed-requirement":
                    case["stderr"] = case["stderr"].replace("flash-attn-4>=4.0.0b9", "flash-attn-4>=5.0")
                elif mutation == "duplicate":
                    case["stderr"] += case["stderr"].splitlines()[-1] + "\n"
                else:
                    case["stderr"] += "warning: dependency inspection incomplete\n"
                self.assertFalse(gate.assess_snapshot(case)["ok"])

    def test_wrong_versions_backend_platform_and_fa4_presence_are_rejected(self):
        gate = load_gate()
        for package, value in (("numpy", "2.4.6"), ("sglang", "0.5.13"),
                               ("transformers", "5.6.0"), ("torch", "2.11.0+cpu"),
                               ("flash-attn", None), ("flash-attn-4", "4.0.0b9")):
            with self.subTest(package=package):
                case = snapshot()
                case["versions"][package] = value
                self.assertFalse(gate.assess_snapshot(case)["ok"])
        for field, value in (("attention_backend", "fa4"), ("attention_backend", None), ("rollout_engine", "vllm"),
                             ("python", "3.13.1"), ("system", "Windows"), ("release", "microsoft-standard-WSL2")):
            with self.subTest(field=field):
                case = snapshot()
                case[field] = value
                self.assertFalse(gate.assess_snapshot(case)["ok"])

    def test_command_failure_or_inconsistent_exit_code_is_rejected(self):
        gate = load_gate()
        for code in (0, 2, 124, -1):
            with self.subTest(code=code):
                case = snapshot()
                case["returncode"] = code
                self.assertFalse(gate.assess_snapshot(case)["ok"])

    def test_cli_reports_malformed_yaml_as_rejected_json(self):
        gate = load_gate()
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "invalid.yaml"
            config.write_text("actor_rollout_ref: [invalid\n", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(gate.main(["--config", str(config)]), 1)
            report = json.loads(output.getvalue())
            self.assertFalse(report["ok"])
            self.assertEqual(report["status"], "rejected")

    def test_split_attention_overrides_cannot_bypass_backend_policy(self):
        gate = load_gate()
        for key in ("prefill_attention_backend", "decode_attention_backend"):
            with self.subTest(key=key):
                case = snapshot()
                case[key] = "fa4"
                self.assertFalse(gate.assess_snapshot(case)["ok"])

    def test_collection_uses_actual_config_metadata_and_preserves_uv_output(self):
        gate = load_gate()
        # 只替代外部 uv 进程与包元数据读取，配置解析和结果判定保持真实执行。
        import subprocess
        case = snapshot()
        result = subprocess.CompletedProcess([], 1, case["stdout"], case["stderr"])
        with patch.object(gate.importlib.metadata, "version", side_effect=lambda name: case["versions"][name]), \
                patch.object(gate.shutil, "which", return_value="uv"), \
                patch.object(gate.subprocess, "run", return_value=result):
            collected = gate.collect_snapshot(ROOT / "configs/calculator_grpo_vast.yaml")
        self.assertEqual(collected["attention_backend"], "torch_native")
        self.assertEqual(collected["rollout_engine"], "sglang")
        self.assertEqual(collected["versions"], case["versions"])
        self.assertEqual(collected["stderr"], case["stderr"])
        self.assertEqual(collected["returncode"], 1)

    @unittest.skipIf(os.name == "nt", "远程 Bash 入口在 Linux CPU 环境验证")
    def test_canary_stops_on_dependency_gate_before_creating_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, data, run = root / "model", root / "data", root / "run"
            model.mkdir()
            data.mkdir()
            for path in (model / "config.json", model / "model.safetensors",
                         data / "train.parquet", data / "validation.parquet"):
                path.write_bytes(b"fixture")
            marker = root / "gate-invoked"
            python = root / "python"
            # 仅模拟外部输入校验程序；真实执行 canary 的调用顺序和失败退出。
            python.write_text('#!/bin/sh\ncase "$1" in\n'
                              '  */check_dependencies.py) touch "$GATE_MARKER"; exit 1;;\n'
                              '  -c) printf "[]";;\n'
                              'esac\nexit 0\n', encoding="utf-8")
            python.chmod(0o700)
            env = {**os.environ, "PYTHON_BIN": str(python), "VERL_SOURCE_DIR": str(root / "source"),
                   "CALCULATOR_MODEL_PATH": str(model), "CALCULATOR_DATA_DIR": str(data),
                   "CALCULATOR_RUN_ROOT": str(run), "GATE_MARKER": str(marker)}
            result = subprocess.run(["bash", str(ROOT / "scripts/remote/canary.sh"), "--train"],
                                    capture_output=True, text=True, env=env)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(marker.exists(), "训练入口没有执行依赖门禁")
            self.assertFalse(run.exists())


if __name__ == "__main__":
    unittest.main()
