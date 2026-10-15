"""检查 Calculator Agent 本机 CPU 开发或远程原生 Linux CUDA 环境。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PINNED_PACKAGES = {
    "verl": "0.9.1",
    "torch": "2.11.0",
    "transformers": "5.9.0",
    "sglang": "0.5.12",
    "ray": "2.55.1",
    "datasets": "5.0.0",
    "pyarrow": "24.0.0",
}


def _check(name: str, status: str, detail: str, *, required: bool = True) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail, "required": required}


def _base_version(version: str | None) -> str | None:
    if version is None:
        return None
    return version.split("+", maxsplit=1)[0]


def assess_snapshot(snapshot: dict[str, Any], mode: str = "legacy") -> dict[str, Any]:
    """Assess a collected host or WSL2 snapshot without performing I/O."""
    # 历史快照仍可复核；命令行默认改为不启动 WSL/CUDA 的本机模式。
    if mode != "legacy":
        return _assess_migration_snapshot(snapshot, mode)
    checks: list[dict[str, Any]] = []
    host_os = snapshot.get("host_os")
    is_wsl = bool(snapshot.get("is_wsl"))
    gpu_name = snapshot.get("gpu_name")

    if host_os == "Windows" and not is_wsl:
        checks.append(_check("windows_host", "pass", "Windows host detected"))
        distributions = snapshot.get("wsl_distributions") or []
        has_wsl2 = snapshot.get("wsl_version_2") is True
        if distributions and has_wsl2:
            checks.append(
                _check("wsl_distribution", "pass", f"Installed WSL2 distributions: {', '.join(distributions)}")
            )
        elif distributions:
            checks.append(
                _check(
                    "wsl_distribution",
                    "fail",
                    f"WSL2 is required; installed distributions use WSL1: {', '.join(distributions)}",
                )
            )
        else:
            checks.append(_check("wsl_distribution", "fail", "No WSL distribution is installed"))
        checks.append(
            _check(
                "nvidia_gpu",
                "pass" if gpu_name else "fail",
                gpu_name or "nvidia-smi did not report a GPU",
            )
        )
        checks.append(
            _check(
                "training_stack",
                "skip",
                "Run this checker inside WSL2 to validate the Linux training stack",
                required=False,
            )
        )
        checks.append(
            _check(
                "torch_cuda",
                "skip",
                "CUDA availability is validated inside WSL2",
                required=False,
            )
        )
    elif host_os == "Linux" and is_wsl:
        checks.append(_check("wsl_runtime", "pass", "WSL2 Linux runtime detected"))
        python_version = str(snapshot.get("python_version") or "unknown")
        checks.append(
            _check(
                "python_version",
                "pass" if python_version.startswith("3.12.") else "fail",
                f"Expected Python 3.12.x, found {python_version}",
            )
        )
        checks.append(
            _check(
                "nvidia_gpu",
                "pass" if gpu_name else "fail",
                gpu_name or "nvidia-smi did not report a GPU inside WSL2",
            )
        )

        packages = snapshot.get("packages") or {}
        mismatches = []
        for package_name, expected_version in PINNED_PACKAGES.items():
            actual_version = packages.get(package_name)
            if _base_version(actual_version) != expected_version:
                mismatches.append(
                    f"{package_name} expected {expected_version}, found {actual_version or 'not installed'}"
                )
        checks.append(
            _check(
                "training_stack",
                "fail" if mismatches else "pass",
                "; ".join(mismatches) if mismatches else "Pinned verl training stack is installed",
            )
        )
        cuda_available = snapshot.get("torch_cuda_available") is True
        checks.append(
            _check(
                "torch_cuda",
                "pass" if cuda_available else "fail",
                "torch.cuda.is_available() is true" if cuda_available else "PyTorch cannot access CUDA",
            )
        )
    else:
        checks.append(
            _check(
                "supported_runtime",
                "fail",
                f"Expected Windows host or WSL2 Linux, found {host_os or 'unknown'}",
            )
        )

    ready = all(item["status"] == "pass" for item in checks if item["required"])
    return {"schema_version": 1, "ready": ready, "snapshot": snapshot, "checks": checks}


def _assess_migration_snapshot(snapshot: dict[str, Any], mode: str) -> dict[str, Any]:
    """区分开发检查与远程预检查，不把资源可见性当成训练验收。"""
    if mode not in {"local-cpu", "remote-cuda"}:
        raise ValueError(f"未知检查模式: {mode}")
    version = str(snapshot.get("python_version", "unknown"))
    checks = [_check("python_version", "pass" if version.startswith("3.12.") else "fail",
                     f"Expected Python 3.12.x, found {version}")]
    if mode == "local-cpu":
        supported = snapshot.get("host_os") in {"Windows", "Linux", "Darwin"}
        checks.append(_check("cpu_runtime", "pass" if supported else "fail", str(snapshot.get("host_os"))))
    else:
        native = snapshot.get("host_os") == "Linux" and not snapshot.get("is_wsl")
        checks.append(_check("native_linux", "pass" if native else "fail", "远程训练要求非 WSL Linux"))
        # 重用固定版本判定，但不沿用历史 WSL 就绪条件。
        training = assess_snapshot({**snapshot, "host_os": "Linux", "is_wsl": True})
        checks.extend(item for item in training["checks"] if item["name"] in {"nvidia_gpu", "training_stack", "torch_cuda"})
        checks.append(_check("cuda_runtime", "pass" if snapshot.get("torch_cuda_version") == "13.0" else "fail",
                             f"torch CUDA Runtime: {snapshot.get('torch_cuda_version') or 'unknown'}"))
        checks.append(_check("cuda_compiler", "pass" if "release 13.0," in str(snapshot.get("nvcc_version"))
                             or str(snapshot.get("nvcc_version")) == "release 13.0" else "fail",
                             str(snapshot.get("nvcc_version") or "nvcc unavailable")))
        checks.append(_check("driver_version", "pass" if snapshot.get("driver_version") else "fail",
                             str(snapshot.get("driver_version") or "unknown")))
        for name, field, minimum in (("disk_space", "disk_free_bytes", 10 * 1024**3),
                                     ("shared_memory", "shm_bytes", 1024**3)):
            value = snapshot.get(field, 0)
            checks.append(_check(name, "pass" if value >= minimum else "fail", f"{value} bytes; preflight minimum {minimum}"))
        # memlock 不是 CUDA 累计锁页额度；仅记录，真正的分配测试另行批准。
        checks.append(_check("memlock", "pass" if snapshot.get("memlock") is not None else "skip",
                             f"soft/hard: {snapshot.get('memlock')}; CUDA pinning not tested", required=False))
    return {"schema_version": 1, "mode": mode,
            "ready": all(item["status"] == "pass" for item in checks if item["required"]),
            "snapshot": snapshot, "checks": checks}


def _decode_process_output(raw: bytes) -> str:
    if not raw:
        return ""
    if b"\x00" in raw:
        return raw.decode("utf-16-le", errors="replace").lstrip("\ufeff")
    for encoding in ("utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode(errors="replace")


def _run_command(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, capture_output=True, check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return 127, ""
    combined = completed.stdout or completed.stderr
    return completed.returncode, _decode_process_output(combined).strip()


def _detect_wsl() -> bool:
    if platform.system() != "Linux":
        return False
    release = platform.release().lower()
    if "microsoft" in release or "wsl" in release:
        return True
    osrelease_path = Path("/proc/sys/kernel/osrelease")
    if osrelease_path.exists():
        return "microsoft" in osrelease_path.read_text(encoding="utf-8").lower()
    return False


def parse_wsl_verbose(output: str) -> tuple[list[str], bool]:
    """Parse the data rows from localized ``wsl --list --verbose`` output."""
    distributions = []
    has_wsl2 = False
    for raw_line in output.splitlines():
        columns = raw_line.lstrip("* ").split()
        if len(columns) < 2 or columns[-1] not in {"1", "2"}:
            continue
        distributions.append(columns[0])
        has_wsl2 = has_wsl2 or columns[-1] == "2"
    return distributions, has_wsl2


def _wsl_distributions() -> tuple[list[str], bool]:
    if platform.system() != "Windows":
        distro = os.environ.get("WSL_DISTRO_NAME")
        return ([distro], True) if distro else ([], False)
    return_code, output = _run_command(["wsl.exe", "--list", "--verbose"])
    if return_code != 0:
        return [], False
    return parse_wsl_verbose(output)


def _gpu_name() -> str | None:
    return_code, output = _run_command(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )
    if return_code != 0 or not output:
        return None
    return output.splitlines()[0].strip()


def _installed_packages() -> dict[str, str]:
    installed = {}
    for package_name in PINNED_PACKAGES:
        try:
            installed[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            installed[package_name] = ""
    return installed


def _torch_cuda_available() -> bool | None:
    try:
        import torch
    except (ImportError, OSError):
        return None
    return bool(torch.cuda.is_available())


def collect_snapshot(mode: str = "legacy", workspace: Path | None = None) -> dict[str, Any]:
    """Collect a side-effect-free snapshot of the current host or WSL process."""
    if mode == "local-cpu":
        return {"host_os": platform.system(), "python_version": platform.python_version()}
    if mode == "remote-cuda":
        # 在 Windows/WSL 误用远程模式时只报平台错误，不探测本机 GPU。
        snapshot = {"host_os": platform.system(), "is_wsl": _detect_wsl(),
                    "python_version": platform.python_version()}
        if snapshot["host_os"] != "Linux" or snapshot["is_wsl"]:
            return snapshot
        import resource
        snapshot.update({"gpu_name": _gpu_name(), "packages": _installed_packages(),
                         "torch_cuda_available": _torch_cuda_available(),
                         "nvcc_version": _run_command(["nvcc", "--version"])[1],
                         "driver_version": _run_command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])[1],
                         "memlock": list(resource.getrlimit(resource.RLIMIT_MEMLOCK)),
                         "disk_free_bytes": shutil.disk_usage(workspace or Path.cwd()).free,
                         "shm_bytes": shutil.disk_usage("/dev/shm").total})
        try:
            import torch
            snapshot["torch_cuda_version"] = torch.version.cuda
        except (ImportError, OSError):
            snapshot["torch_cuda_version"] = None
        return snapshot
    is_wsl = _detect_wsl()
    distributions, has_wsl2 = _wsl_distributions()
    return {
        "host_os": platform.system(),
        "is_wsl": is_wsl,
        "wsl_distributions": distributions,
        "wsl_version_2": has_wsl2,
        "gpu_name": _gpu_name(),
        "python_version": platform.python_version(),
        "packages": _installed_packages() if is_wsl else {},
        "torch_cuda_available": _torch_cuda_available() if is_wsl else None,
    }


def _render_text(report: dict[str, Any]) -> str:
    lines = [f"Calculator Agent environment ready: {report['ready']}"]
    for item in report["checks"]:
        lines.append(f"[{item['status'].upper()}] {item['name']}: {item['detail']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON report")
    parser.add_argument("--strict", action="store_true", help="Return exit code 1 when the environment is not ready")
    parser.add_argument("--mode", choices=("local-cpu", "remote-cuda", "legacy"), default="local-cpu")
    parser.add_argument("--workspace", type=Path, help="远程数据盘目录，用于磁盘空间检查")
    args = parser.parse_args(argv)

    try:
        report = assess_snapshot(collect_snapshot(mode=args.mode, workspace=args.workspace), mode=args.mode)
    except OSError as exc:
        # 挂载盘不存在或权限不足时仍输出稳定报告，不让远程预检查吐出堆栈。
        report = {"schema_version": 1, "mode": args.mode, "ready": False, "snapshot": {},
                  "checks": [_check("snapshot_collection", "fail", str(exc))]}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_render_text(report))
    return 1 if args.strict and not report["ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
