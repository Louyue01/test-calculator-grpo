"""将 GSM8K 转换为 verl v0.9.1 可读取的 Parquet 数据"""

import argparse
import copy
import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DATA_SOURCE = "openai/gsm8k"
# 中文副本仅供维护对照；保留调用轮与最终回答轮的完整协议。
SYSTEM_PROMPT_ZH = (
    "你是数学助手。用 <think>...</think> 推理，每轮选一种格式：\n"
    "调用轮：推理后输出单个工具 JSON。</tool_call> 后立即停止等待；"
    "勿生成工具结果或答案。\n"
    "最终回答轮：推理后，最后一行为 Final answer: <number>。"
    "只填整数或小数，无单位、货币符号、百分号、算式或末尾标点；不调用工具。\n"
    "calculator 仅作数值计算，不能求解方程、方程组。先化简；expression 不得包含变量、等号或代码。"
    "标签不嵌套。\n"
    "示例（勿输出角色名或照抄答案）：\n"
    "用户：17*19+4？\n"
    "助手：<think>先计算17*19，再加4。</think>\n"
    '<tool_call>{"name":"calculator","arguments":{"expression":"17*19+4"}}</tool_call>\n'
    "【程序返回】\n"
    "工具：327\n"
    "助手：<think>所以323+4=327。</think>\n"
    "Final answer: 327"
)
# 实际数据生成使用英文提示；工具名、JSON 字段和最终答案标签保持不变。
SYSTEM_PROMPT_EN = (
    "Math assistant: reason in <think>...</think>. Use one format per turn:\n"
    "Tool-call turn: after reasoning, output one JSON call. Stop immediately after </tool_call> "
    "and wait for the program; never generate tool results or answers.\n"
    "Final-answer turn: after reasoning, the last line must be Final answer: <number>. "
    "Use an integer or decimal only: no units, currency symbols, percent signs, expressions, "
    "or trailing punctuation. No tool calls.\n"
    "calculator is numerical only and cannot solve equations or systems of equations. "
    "Simplify first; expression forbids variables, equals signs, or code. Do not nest tags.\n"
    "Example (omit role labels; do not copy its answer):\n"
    "User: 17*19+4?\n"
    "Assistant: <think>First calculate 17*19, then add 4.</think>\n"
    '<tool_call>{"name":"calculator","arguments":{"expression":"17*19+4"}}</tool_call>\n'
    "[Program returns]\n"
    "Tool: 327\n"
    "Assistant: <think>Therefore, 323+4=327.</think>\n"
    "Final answer: 327"
)
# 统一入口供样本转换和已有数据刷新使用，不把中文副本拼入模型输入。
SYSTEM_PROMPT = SYSTEM_PROMPT_EN
OUTPUT_FILENAMES = {
    "train": "train.parquet",
    "validation": "validation.parquet",
    "final_test": "final_test.parquet",
}
ANSWER_PATTERN = re.compile(
    r"####\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*$"
)

# 可注入接口用于隔离网络和 Parquet 依赖
DatasetLoader = Callable[[str, str], Mapping[str, Iterable[Mapping[str, Any]]]]
ParquetWriter = Callable[[Sequence[Mapping[str, Any]], Path], Any]


def _default_dataset_loader(name: str, configuration: str):
    """仅在实际准备数据时导入 datasets"""
    try:
        # 延迟导入，避免 import prepare_data 时触发依赖错误或数据下载
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "准备 GSM8K 需要可选依赖 'datasets'，请先配置固定版本训练环境。"
        ) from exc
    return load_dataset(name, configuration)


def _default_parquet_writer(records: Sequence[Mapping[str, Any]], output_path: Path):
    """使用 pyarrow 写入一个 verl 数据分片"""
    try:
        # 延迟导入，离线单元测试不要求安装 pyarrow
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "写入 Parquet 需要可选依赖 'pyarrow'，请先配置固定版本训练环境。"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(records))
    pq.write_table(table, output_path)


def _default_parquet_reader(path: Path) -> list[dict[str, Any]]:
    """读取已有本地数据，刷新提示时不重新下载或划分数据集。"""
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def refresh_dataset_prompts(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    read_parquet_fn: Callable | None = None,
    write_parquet_fn: ParquetWriter | None = None,
) -> dict[str, int]:
    """仅替换三个分片的系统提示，并另存以保留已验收的原始数据。"""
    source_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()
    if source_root == output_root:
        raise ValueError("提示刷新必须另存，输入和输出目录不能相同")
    for filename in OUTPUT_FILENAMES.values():
        if (output_root / filename).exists():
            raise FileExistsError(f"拒绝覆盖已有数据: {output_root / filename}")

    reader = read_parquet_fn or _default_parquet_reader
    writer = write_parquet_fn or _default_parquet_writer
    refreshed = {}
    for split, filename in OUTPUT_FILENAMES.items():
        records = copy.deepcopy(reader(source_root / filename))
        for record in records:
            prompt = record.get("prompt")
            if not isinstance(prompt, list) or not prompt or prompt[0].get("role") != "system":
                raise ValueError(f"{filename} 缺少首条 system 消息")
            prompt[0]["content"] = SYSTEM_PROMPT
        refreshed[split] = records

    # 所有分片读取和结构检查通过后才写出，不改题目、答案、索引或 split。
    for split, records in refreshed.items():
        writer(records, output_root / OUTPUT_FILENAMES[split])
    return {split: len(records) for split, records in refreshed.items()}


def extract_gsm8k_answer(answer_text: str) -> str:
    """提取并规范化 GSM8K 最终数字答案"""
    if not isinstance(answer_text, str):
        raise TypeError("answer_text 必须是字符串")

    match = ANSWER_PATTERN.search(answer_text)
    if match is None:
        raise ValueError("GSM8K 答案缺少合法的 '#### 数字' 结尾")

    raw_answer = match.group(1).replace(",", "")
    try:
        number = Decimal(raw_answer)
    except InvalidOperation as exc:
        raise ValueError(f"无法解析 GSM8K 答案: {raw_answer}") from exc
    if not number.is_finite():
        raise ValueError("GSM8K 答案必须是有限数字")

    # 使用普通十进制字符串，避免奖励函数收到科学计数法或多余零
    normalized = format(number.normalize(), "f")
    return "0" if Decimal(normalized) == 0 else normalized


def convert_example(example: Mapping[str, Any], split: str, index: int) -> dict[str, Any]:
    """将一条 GSM8K 样本映射为 verl v0.9.1 记录"""
    if split not in OUTPUT_FILENAMES:
        raise ValueError(f"不支持的数据 split: {split}")

    question = example.get("question")
    answer_text = example.get("answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("GSM8K question 必须是非空字符串")
    if not isinstance(answer_text, str):
        raise ValueError("GSM8K answer 必须是字符串")

    ground_truth = extract_gsm8k_answer(answer_text)
    original_split = "test" if split == "final_test" else "train"

    # 字段与 verl v0.9.1 官方 GSM8K 预处理格式保持一致
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": split,
            "original_split": original_split,
            "index": index,
            "answer": answer_text,
            "question": question,
        },
    }


def split_training_examples(
    examples: Iterable[Mapping[str, Any]],
    validation_size: int,
    seed: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """使用固定随机种子将官方训练集划分为训练集和验证集"""
    rows = list(examples)
    if isinstance(validation_size, bool) or not isinstance(validation_size, int):
        raise TypeError("validation_size 必须是整数")
    if validation_size <= 0 or validation_size >= len(rows):
        raise ValueError("validation_size 必须大于 0 且小于训练样本数")

    # 只随机选择验证索引，两个输出仍保持原始数据顺序
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    validation_indices = set(indices[:validation_size])
    train_rows = [row for index, row in enumerate(rows) if index not in validation_indices]
    validation_rows = [row for index, row in enumerate(rows) if index in validation_indices]
    return train_rows, validation_rows


def prepare_gsm8k_dataset(
    output_dir: str | Path = "data",
    validation_size: int = 512,
    seed: int = 42,
    *,
    load_dataset_fn: DatasetLoader | None = None,
    write_parquet_fn: ParquetWriter | None = None,
) -> dict[str, int]:
    """生成训练、验证和最终测试三个用途隔离的 Parquet 文件"""
    loader = load_dataset_fn or _default_dataset_loader
    writer = write_parquet_fn or _default_parquet_writer
    source_dataset = loader(DATA_SOURCE, "main")

    train_source, validation_source = split_training_examples(
        source_dataset["train"],
        validation_size=validation_size,
        seed=seed,
    )
    final_test_source = list(source_dataset["test"])

    # 官方 test split 仅作为最终测试集，不参与训练或模型选择
    raw_splits = {
        "train": train_source,
        "validation": validation_source,
        "final_test": final_test_source,
    }
    output_root = Path(output_dir)
    counts = {}
    for split_name, source_rows in raw_splits.items():
        records = [
            convert_example(example, split=split_name, index=index)
            for index, example in enumerate(source_rows)
        ]
        writer(records, output_root / OUTPUT_FILENAMES[split_name])
        counts[split_name] = len(records)

    return counts


def build_argument_parser() -> argparse.ArgumentParser:
    """构造数据准备命令行参数"""
    parser = argparse.ArgumentParser(description="准备 verl GRPO 使用的 GSM8K Parquet 数据")
    parser.add_argument("--output-dir", default="data", help="Parquet 输出目录")
    parser.add_argument("--input-dir", help="仅刷新已有 Parquet 的系统提示，不下载或重新划分")
    parser.add_argument("--validation-size", type=int, default=512, help="验证集样本数")
    parser.add_argument("--seed", type=int, default=42, help="训练/验证划分随机种子")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并生成三个用途隔离的数据文件"""
    args = build_argument_parser().parse_args(argv)
    if args.input_dir is not None:
        counts = refresh_dataset_prompts(args.input_dir, args.output_dir)
    else:
        counts = prepare_gsm8k_dataset(
            output_dir=args.output_dir,
            validation_size=args.validation_size,
            seed=args.seed,
        )
    print(
        "数据准备完成: "
        f"train={counts['train']}, "
        f"validation={counts['validation']}, "
        f"final_test={counts['final_test']}"
    )
    return 0


# 防止导入模块时下载数据或写入文件
if __name__ == "__main__":
    raise SystemExit(main())
