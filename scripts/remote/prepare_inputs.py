"""远程固定模型下载与 canary 输入校验；默认不联网，不执行生成或训练。"""

import argparse
import hashlib
import json
from pathlib import Path


MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_SHA = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
DATA_SHA = {
    "train.parquet": "f616d5efbaac995140ba089d799fa4601d87639925389b5802d290bf77e2161f",
    "validation.parquet": "34385507b0f826caa78ebde16863babb8eed128ddc5aaf0bac1da04775777923",
}


def file_sha(path):
    """分块读取模型，避免对宿主内存产生与模型体积相同的额外副本。"""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_model(directory):
    """核对固定权重与基本配置；不导入 torch 或加载模型。"""
    directory = Path(directory)
    if file_sha(directory / "model.safetensors") != MODEL_SHA:
        raise ValueError("模型权重不是已批准的 Qwen3-0.6B 快照")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise ValueError("模型 config.json 与 Qwen3 不一致")
    if not (directory / "tokenizer_config.json").is_file():
        raise ValueError("模型快照缺少 tokenizer_config.json")
    return directory


def verify_data(directory):
    """校验已验收的训练8题/验证32题原文件，禁止自动重抽样或替换。"""
    directory = Path(directory)
    for name, expected in DATA_SHA.items():
        if file_sha(directory / name) != expected:
            raise ValueError(f"canary 数据不在已批准版本中: {name}")
    return True


def download_model(cache_dir, *, allow_network=False):
    """只有显式许可才从官方 Hub 下载固定 revision，完成后仍核对权重。"""
    if not allow_network:
        raise ValueError("模型下载需要显式 --allow-network")
    # 延迟导入使 CPU 校验和命令帮助不依赖完整训练环境。
    from huggingface_hub import snapshot_download
    directory = snapshot_download(repo_id=MODEL_ID, revision=REVISION,
                                  cache_dir=str(Path(cache_dir).resolve() / "hub"))
    return verify_model(directory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("verify-data")
    data.add_argument("--data-dir", type=Path, required=True)
    model = commands.add_parser("verify-model")
    model.add_argument("--model-dir", type=Path, required=True)
    download = commands.add_parser("download-model")
    download.add_argument("--cache-dir", type=Path, required=True)
    download.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-data":
            verify_data(args.data_dir)
            print("已验收的 canary 8/32 数据摘要一致")
        elif args.command == "verify-model":
            verify_model(args.model_dir)
            print("固定 Qwen3 模型输入校验通过")
        else:
            print(download_model(args.cache_dir, allow_network=args.allow_network))
        return 0
    except (OSError, ValueError, ImportError) as exc:
        print(f"输入准备停止: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
