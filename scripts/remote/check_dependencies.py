"""严格识别固定 SGLang 配置的两项依赖例外；不安装、不导入 GPU 库。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


PINNED = {"verl": "0.9.1", "torch": "2.11.0+cu130", "sglang": "0.5.12",
          "transformers": "5.9.0", "ray": "2.55.1", "numpy": "2.3.5",
          "flash-attn": "2.8.3", "flash-attn-4": None}
# 只接受已审查包版本下的完整诊断，不使用包名子串批量忽略报错。
EXCEPTIONS = {
    "The package `sglang` requires `flash-attn-4>=4.0.0b9`, but it's not installed",
    "The package `sglang` requires `transformers==5.6.0`, but `5.9.0` is installed",
}


def assess_snapshot(snapshot):
    """保留原始退出码；受控例外与原始检查零冲突是不同状态。"""
    errors = []
    if not re.fullmatch(r"3\.12\.\d+", snapshot.get("python", "")) \
            or snapshot.get("system") != "Linux" \
            or any(word in snapshot.get("release", "").lower() for word in ("microsoft", "wsl")):
        errors.append("仅接受已审查的原生 Linux Python 3.12 环境")
    if snapshot.get("attention_backend") != "torch_native" or snapshot.get("rollout_engine") != "sglang":
        errors.append("后端改变后不能沿用本组依赖例外")
    if any(snapshot.get(key) is not None for key in ("prefill_attention_backend", "decode_attention_backend")):
        errors.append("单独覆盖 prefill/decode 后端需要重新审查")
    versions = snapshot.get("versions", {})
    for name, expected in PINNED.items():
        if name not in versions or versions[name] != expected:
            errors.append(f"依赖版本不符合已审查组合: {name}")
    raw = {key: snapshot.get(key) for key in ("returncode", "stdout", "stderr")}
    diagnostics = []
    headers = []
    for line in ((raw["stdout"] or "") + "\n" + (raw["stderr"] or "")).splitlines():
        if not line.strip():
            continue
        if line in EXCEPTIONS:
            diagnostics.append(line)
        elif line == "Found 2 incompatibilities":
            headers.append(line)
        elif re.fullmatch(r"Using Python 3\.12\.\d+ environment at: .+", line) \
                or re.fullmatch(r"Checked \d+ packages? in .+", line):
            continue
        else:
            errors.append(f"未批准的依赖诊断: {line}")
    if raw["returncode"] != 1 or len(diagnostics) != 2 or set(diagnostics) != EXCEPTIONS \
            or headers != ["Found 2 incompatibilities"]:
        errors.append("检查失败模式、诊断数量或退出码与已审查例外不一致")
    return {"ok": not errors, "status": "rejected" if errors else "accepted_exceptions",
            "errors": errors, "exceptions": diagnostics, "versions": versions, "raw": raw}


def collect_snapshot(config):
    """只读取配置和已安装包元数据；pip check 不修复或安装任何包。"""
    import yaml

    try:
        settings = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"配置 YAML 无效: {error}") from error
    rollout = settings["actor_rollout_ref"]["rollout"]
    versions = {}
    for name in PINNED:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    uv = shutil.which("uv")
    if not uv:
        raise ValueError("缺少 uv，不能执行原始依赖检查")
    result = subprocess.run([uv, "--color", "never", "pip", "check", "--python", sys.executable],
                            capture_output=True, text=True, timeout=120,
                            env={**os.environ, "NO_COLOR": "1"})
    return {"python": platform.python_version(), "system": platform.system(), "release": platform.release(),
            "rollout_engine": rollout.get("name"),
            "attention_backend": rollout.get("engine_kwargs", {}).get("sglang", {}).get("attention_backend"),
            "prefill_attention_backend": rollout.get("engine_kwargs", {}).get("sglang", {}).get("prefill_attention_backend"),
            "decode_attention_backend": rollout.get("engine_kwargs", {}).get("sglang", {}).get("decode_attention_backend"),
            "versions": versions, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "configs/calculator_grpo_vast.yaml")
    args = parser.parse_args(argv)
    try:
        report = assess_snapshot(collect_snapshot(args.config))
    except (OSError, ValueError, KeyError, TypeError, ImportError, subprocess.SubprocessError) as error:
        report = {"ok": False, "status": "rejected", "errors": [str(error)]}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
