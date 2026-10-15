"""GSM8K 数据集探查工具（导入模块时不会加载数据）"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any


# 数据加载器类型，便于测试时注入本地假数据
DatasetLoader = Callable[[str, str], Mapping[str, Sequence[Mapping[str, Any]]]]


def _default_dataset_loader(name: str, configuration: str):
    """仅在实际执行探查命令时导入 datasets"""
    try:
        # 延迟导入，确保未安装训练依赖时仍可安全导入本模块
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The optional 'datasets' package is required to inspect GSM8K. "
            "Install the pinned training environment first."
        ) from exc
    return load_dataset(name, configuration)


def main(load_dataset_fn: DatasetLoader | None = None) -> int:
    """加载 GSM8K 并输出训练集、测试集和示例信息"""
    # 测试可传入假加载器，正常运行时使用 Hugging Face datasets
    loader = load_dataset_fn or _default_dataset_loader
    dataset = loader("gsm8k", "main")

    # 只输出数据集摘要，不在项目目录写入任何文件
    print(f"Training examples: {len(dataset['train'])}")
    print(f"Test examples: {len(dataset['test'])}")
    if dataset["train"]:
        print(f"Training example: {dataset['train'][0]}")
    return 0


# 仅直接运行脚本时加载远程数据集
if __name__ == "__main__":
    raise SystemExit(main())
