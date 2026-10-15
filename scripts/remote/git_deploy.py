"""Git 拉取后的分步初始化入口；默认只校验，不安装、不下载模型、不训练。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


BASE_ARCHIVE_SHA = "cf43327122bf9b33624dd1c3f5eac104d0d27ec8cbcf2ba57dc0e5c2dcb5a0f9"
BASE_COMMIT = "1876b06d0a3e4e71e06230be10af14492ca8a75b"
PATCH_NAMES = ["01-dependency-sources.patch", "02-hermes-tool-stop.patch",
               "03-deterministic-priority.patch", "04-single-gpu-state-dict.patch"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_public(root):
    """独立校验发布成员与内容，不依赖本机打包工具或历史目录。"""
    root = Path(root).resolve()
    record = json.loads((root / "public-release.json").read_text(encoding="utf-8"))
    expected = {}
    for item in record["files"]:
        name = item["path"]
        path = PurePosixPath(name)
        if not name or "\\" in name or ":" in name or path.is_absolute() \
                or any(part in {"", ".", "..", ".git"} for part in name.split("/")) \
                or name in expected or name == "public-release.json":
            raise ValueError("发布清单含重复或越界路径")
        current = root
        for part in path.parts:
            current = current / part
            if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
                raise ValueError("发布清单不能通过链接读取外部文件")
        if not current.is_file() or current.stat().st_size != item["size"] or sha(current) != item["sha256"]:
            raise ValueError(f"发布文件缺失或摘要不一致: {name}")
        expected[name] = item
    actual = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in relative.parts):
            continue
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise ValueError("公开项目不接纳链接")
        if path.is_file():
            actual.add(relative.as_posix())
    if not expected or actual != set(expected) | {"public-release.json"}:
        raise ValueError("发布成员集合不一致，存在额外或缺失文件")
    return True


def verify_assets(root):
    """检查实际携带的补丁和锁，拒绝不同基线或意外加入的补丁组。"""
    assets = Path(root) / "deploy/verl"
    record = json.loads((assets / "source.json").read_text(encoding="utf-8"))
    if record["commit"] != BASE_COMMIT or record["patches"] != PATCH_NAMES:
        raise ValueError("上游 commit 或原生补丁顺序与批准版本不一致")
    for name in PATCH_NAMES + ["pyproject.toml", "uv.lock"]:
        if sha(assets / name) != record["files"][name]:
            raise ValueError(f"部署资产摘要不一致: {name}")
    return assets


def native_linux():
    """安装仅在获批远程原生 Linux 执行，禁止误在本机 WSL 安装 GPU 栈。"""
    if platform.system() != "Linux" or any(word in platform.release().lower() for word in ("microsoft", "wsl")):
        raise ValueError("GPU 环境安装仅支持获批的远程原生 Linux")


def prepare_source(root, target, *, allow_network=False, base_archive=None):
    """从固定上游或已验收离线归档重建源码，不覆盖现有目录。"""
    target = Path(target).absolute()
    if not allow_network and base_archive is None:
        raise ValueError("联网重建需显式 --allow-network；离线需指定已核验 --base-archive")
    if base_archive is not None and sha(base_archive) != BASE_ARCHIVE_SHA:
        raise ValueError("离线归档不是已验收的 LF 上游基线")
    verify_public(root)
    assets = verify_assets(root)
    if target.exists():
        raise FileExistsError(target)
    if not shutil.which("patch"):
        raise ValueError("缺少 GNU patch，请先单独批准安装系统构建工具")
    if base_archive is None:
        native_linux()
        subprocess.run(["git", "clone", "--depth", "1", "--branch", "v0.9.1",
                        "https://github.com/verl-project/verl.git", str(target)], check=True)
        commit = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if commit != BASE_COMMIT:
            raise ValueError("上游 tag 的 commit 已改变，停止应用补丁")
    else:
        # 基线哈希先核验；data 过滤器额外拒绝越界成员和危险链接。
        with tarfile.open(base_archive) as archive:
            target.mkdir(parents=True, exist_ok=False)
            archive.extractall(target, filter="data")
    for name in PATCH_NAMES:
        command = ["patch", "--batch", "--forward", "-p1", "-d", str(target), "-i", str((assets / name).resolve())]
        subprocess.run(command + ["--dry-run"], check=True, capture_output=True, text=True)
        subprocess.run(command, check=True, capture_output=True, text=True)
    for name in ("uv.lock", "pyproject.toml"):
        if sha(target / name) != sha(assets / name):
            raise ValueError(f"应用补丁后的 {name} 与实际锁快照不一致")
    files = ["uv.lock", "pyproject.toml", "verl/experimental/agent_loop/tool_agent_loop.py",
             "verl/experimental/agent_loop/tool_parser.py", "verl/trainer/ppo/v1/agent_loop_tq.py",
             "verl/utils/fsdp_utils.py"]
    marker = {"commit": BASE_COMMIT, "files": {name: sha(target / name) for name in files}}
    (target / ".calculator-native.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    return marker


def install(root, source, allow_network=False):
    """只安装已冻结的原生 GPU 栈，不解析新依赖，不启动 GPU 工作负载。"""
    if not allow_network:
        raise ValueError("安装需要显式 --allow-network")
    native_linux()
    verify_public(root)
    assets = verify_assets(root)
    source = Path(source).resolve()
    marker = json.loads((source / ".calculator-native.json").read_text(encoding="utf-8"))
    if marker["commit"] != BASE_COMMIT:
        raise ValueError("源码未通过固定基线检查")
    for name, expected in marker["files"].items():
        if sha(source / name) != expected:
            raise ValueError(f"源码已改变: {name}")
    for name in ("uv.lock", "pyproject.toml"):
        if sha(source / name) != sha(assets / name):
            raise ValueError("实际依赖声明或锁已改变")
    if (source / ".venv").exists():
        raise FileExistsError("环境已存在；禁止隐式重同步或升级")
    if not shutil.which("uv"):
        raise ValueError("缺少 uv，请先单独批准安装")
    env = dict(os.environ, UV_PROJECT_ENVIRONMENT=str(source / ".venv"))
    subprocess.run(["uv", "sync", "--frozen", "--all-packages", "--extra", "sglang",
                    "--extra", "fsdp", "--python", "3.12", "--no-python-downloads"],
                   cwd=source, env=env, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("verify")
    for name in ("prepare-source", "install"):
        command = commands.add_parser(name)
        command.add_argument("--verl-source", type=Path, required=True)
        command.add_argument("--allow-network", action="store_true")
        if name == "prepare-source":
            command.add_argument("--base-archive", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-source":
            prepare_source(args.project_root, args.verl_source, allow_network=args.allow_network, base_archive=args.base_archive)
        elif args.command == "install":
            install(args.project_root, args.verl_source, args.allow_network)
        else:
            verify_public(args.project_root)
            verify_assets(args.project_root)
        print("本步骤完成；未下载模型、未启动训练")
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, tarfile.TarError) as exc:
        print(f"部署检查停止: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
