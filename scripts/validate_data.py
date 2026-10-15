"""校验 Calculator Agent 的 verl Parquet 数据"""

import argparse
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


EXPECTED_SPLITS = ("train", "validation", "final_test")
REQUIRED_FIELDS = ("data_source", "prompt", "ability", "reward_model", "extra_info")
DEFAULT_FILENAMES = {
    "train": "train.parquet",
    "validation": "validation.parquet",
    "final_test": "final_test.parquet",
}
ParquetReader = Callable[[str | Path], list[dict[str, Any]]]
RAW_ANSWER_PATTERN = re.compile(
    r"####\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*$"
)


def _normalize_raw_answer(answer_text: str) -> str | None:
    """独立解析原始答案，用于交叉验证生产数据"""
    match = RAW_ANSWER_PATTERN.search(answer_text)
    if match is None:
        return None
    try:
        number = Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    normalized = format(number.normalize(), "f")
    return "0" if number == 0 else normalized


def _validate_prompt(prompt: Any, location: str, max_characters: int) -> list[str]:
    """校验消息列表结构和字符长度"""
    errors = []
    if not isinstance(prompt, list) or not prompt:
        return [f"{location}.prompt 必须是非空消息列表"]

    total_characters = 0
    has_user_message = False
    for message_index, message in enumerate(prompt):
        message_location = f"{location}.prompt[{message_index}]"
        if not isinstance(message, Mapping):
            errors.append(f"{message_location} 必须是对象")
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            errors.append(f"{message_location}.role 无效")
        if role == "user":
            has_user_message = True
        if not isinstance(content, str) or not content:
            errors.append(f"{message_location}.content 必须是非空字符串")
        else:
            total_characters += len(content)

    if not has_user_message:
        errors.append(f"{location}.prompt 缺少 user 消息")
    if total_characters > max_characters:
        errors.append(
            f"{location}.prompt 长度 {total_characters} 超过限制 {max_characters}"
        )
    return errors


def _validate_record(
    record: Any,
    split_name: str,
    index: int,
    max_prompt_characters: int,
) -> tuple[list[str], str | None]:
    """校验单条记录并返回用于查重的问题文本"""
    location = f"{split_name}[{index}]"
    if not isinstance(record, Mapping):
        return [f"{location} 必须是对象"], None

    errors = []
    for field_name in REQUIRED_FIELDS:
        if field_name not in record:
            errors.append(f"{location}.{field_name} 缺失")
    if errors:
        return errors, None

    if record["data_source"] != "openai/gsm8k":
        errors.append(f"{location}.data_source 必须是 openai/gsm8k")
    if record["ability"] != "math":
        errors.append(f"{location}.ability 必须是 math")
    errors.extend(
        _validate_prompt(record["prompt"], location, max_prompt_characters)
    )

    ground_truth = None
    reward_model = record["reward_model"]
    if not isinstance(reward_model, Mapping):
        errors.append(f"{location}.reward_model 必须是对象")
    else:
        if reward_model.get("style") != "rule":
            errors.append(f"{location}.reward_model.style 必须是 rule")
        ground_truth = reward_model.get("ground_truth")
        if not isinstance(ground_truth, str) or not ground_truth:
            errors.append(f"{location}.reward_model.ground_truth 必须是非空字符串")

    question = None
    answer_text = None
    extra_info = record["extra_info"]
    if not isinstance(extra_info, Mapping):
        errors.append(f"{location}.extra_info 必须是对象")
    else:
        if extra_info.get("split") != split_name:
            errors.append(
                f"{location}.extra_info.split 必须是 {split_name}，"
                f"实际为 {extra_info.get('split')!r}"
            )
        expected_original_split = "test" if split_name == "final_test" else "train"
        if extra_info.get("original_split") != expected_original_split:
            errors.append(
                f"{location}.extra_info.original_split 必须是 {expected_original_split}"
            )
        if isinstance(extra_info.get("index"), bool) or not isinstance(
            extra_info.get("index"), int
        ):
            errors.append(f"{location}.extra_info.index 必须是整数")
        question_value = extra_info.get("question")
        if not isinstance(question_value, str) or not question_value:
            errors.append(f"{location}.extra_info.question 必须是非空字符串")
        else:
            question = question_value
        answer_value = extra_info.get("answer")
        if not isinstance(answer_value, str):
            errors.append(f"{location}.extra_info.answer 必须是字符串")
        else:
            answer_text = answer_value

    # 交叉检查 prompt、原始样本和奖励标签，防止结构合法但内容错位
    if question is not None and isinstance(record["prompt"], list):
        user_messages = [
            message.get("content")
            for message in record["prompt"]
            if isinstance(message, Mapping) and message.get("role") == "user"
        ]
        if not user_messages or user_messages[-1] != question:
            errors.append(f"{location}.prompt 与 extra_info.question 不一致")
    if answer_text is not None and isinstance(ground_truth, str):
        normalized_answer = _normalize_raw_answer(answer_text)
        if normalized_answer is None:
            errors.append(f"{location}.extra_info.answer 无法解析")
        elif normalized_answer != ground_truth:
            errors.append(f"{location}.ground_truth 与原始答案不一致")

    return errors, question


def validate_splits(
    records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_prompt_characters: int = 32768,
) -> list[str]:
    """校验 schema、空值、重复问题和 split 泄漏"""
    if max_prompt_characters <= 0:
        raise ValueError("max_prompt_characters 必须大于 0")

    errors = []
    unexpected_splits = sorted(set(records_by_split) - set(EXPECTED_SPLITS))
    if unexpected_splits:
        errors.append(f"发现未知 split: {', '.join(unexpected_splits)}")

    # 记录问题首次出现位置，用于同时检测 split 内重复和跨 split 泄漏
    question_owners: dict[str, str] = {}
    for split_name in EXPECTED_SPLITS:
        records = records_by_split.get(split_name, [])
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            errors.append(f"{split_name} 必须是记录序列")
            continue
        if not records:
            errors.append(f"{split_name} 为空")
            continue

        split_indices = set()
        for index, record in enumerate(records):
            record_errors, question = _validate_record(
                record,
                split_name,
                index,
                max_prompt_characters,
            )
            errors.extend(record_errors)

            # index 是后续日志和轨迹审计的稳定样本标识，同一 split 内必须唯一
            if isinstance(record, Mapping) and isinstance(record.get("extra_info"), Mapping):
                record_index = record["extra_info"].get("index")
                if isinstance(record_index, int) and not isinstance(record_index, bool):
                    if record_index in split_indices:
                        errors.append(f"{split_name}[{index}].extra_info.index 重复")
                    else:
                        split_indices.add(record_index)
            if question is None:
                continue

            previous_split = question_owners.get(question)
            if previous_split is None:
                question_owners[question] = split_name
            elif previous_split == split_name:
                errors.append(f"{split_name}[{index}] 问题重复")
            else:
                errors.append(
                    f"{split_name}[{index}] 与 {previous_split} 存在 split 泄漏"
                )

    return errors


def read_parquet_records(path: str | Path) -> list[dict[str, Any]]:
    """读取 Parquet；仅在执行真实验证时导入 pyarrow"""
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "读取 Parquet 需要可选依赖 'pyarrow'，请先配置固定版本训练环境。"
        ) from exc
    return pq.read_table(Path(path)).to_pylist()


def build_argument_parser() -> argparse.ArgumentParser:
    """构造数据验证命令行参数"""
    parser = argparse.ArgumentParser(description="验证 verl GSM8K Parquet 数据")
    parser.add_argument("--data-dir", default="data", help="Parquet 数据目录")
    parser.add_argument(
        "--max-prompt-characters",
        type=int,
        default=32768,
        help="单条 prompt 的最大字符数（真实 token 限制在训练配置阶段验证）",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    read_parquet_fn: ParquetReader = read_parquet_records,
) -> int:
    """读取三个 Parquet 文件并返回可用于自动化的退出码"""
    args = build_argument_parser().parse_args(argv)
    data_directory = Path(args.data_dir)

    # 测试可注入内存读取器，正常运行时使用 pyarrow
    records_by_split = {
        split_name: read_parquet_fn(data_directory / filename)
        for split_name, filename in DEFAULT_FILENAMES.items()
    }
    errors = validate_splits(
        records_by_split,
        max_prompt_characters=args.max_prompt_characters,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    counts = ", ".join(
        f"{split_name}={len(records_by_split[split_name])}"
        for split_name in EXPECTED_SPLITS
    )
    print(f"数据验证通过: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
