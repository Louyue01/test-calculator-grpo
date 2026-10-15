"""数学推理 Agent 的抗投机规则奖励函数"""

import math
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any


NUMBER_PATTERN = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
FULL_NUMBER_PATTERN = re.compile(rf"^{NUMBER_PATTERN}$")
BOXED_ANSWER_PATTERN = re.compile(rf"\\boxed\{{\s*({NUMBER_PATTERN})\s*\}}")
LABELED_ANSWER_PATTERN = re.compile(
    rf"(?:最终答案|答案|final\s+answer|answer)\s*(?:是|为|is|=)?\s*[:：]?\s*({NUMBER_PATTERN})",
    re.IGNORECASE,
)
BARE_ANSWER_PATTERN = re.compile(rf"^\s*({NUMBER_PATTERN})\s*[。.]?\s*$")
THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
REASONING_CONNECTOR_PATTERN = re.compile(
    r"(?:因为|所以|首先|先|然后|接着|因此|可得|计算|分步|because|therefore|first|then|next)",
    re.IGNORECASE,
)
ARITHMETIC_STEP_PATTERN = re.compile(
    rf"{NUMBER_PATTERN}\s*(?:[+\-*/×÷]|==?|等于)\s*{NUMBER_PATTERN}"
)

ACCURACY_REWARD = 1.0
FORMAT_REWARD = 0.1
TOOL_REWARD = 0.1
THINK_REWARD = 0.05


def _normalize_number(value: object) -> str | None:
    """将有限十进制数字规范化为可精确比较的字符串"""
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return None

    raw_value = str(value).strip().replace(",", "")
    if FULL_NUMBER_PATTERN.fullmatch(raw_value) is None:
        return None
    try:
        number = Decimal(raw_value)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None

    normalized = format(number.normalize(), "f")
    return "0" if number == 0 else normalized


def _extract_answer_details(solution_str: object) -> tuple[str | None, bool]:
    """提取答案，并标记是否使用了明确的最终答案格式"""
    if not isinstance(solution_str, str):
        return None, False

    candidates = []
    for pattern in (BOXED_ANSWER_PATTERN, LABELED_ANSWER_PATTERN):
        for match in pattern.finditer(solution_str):
            candidates.append((match.start(), _normalize_number(match.group(1)), True))

    nonempty_lines = [line for line in solution_str.splitlines() if line.strip()]
    if nonempty_lines:
        final_line = nonempty_lines[-1]
        bare_match = BARE_ANSWER_PATTERN.fullmatch(final_line)
        if bare_match is not None:
            candidates.append(
                (
                    solution_str.rfind(final_line),
                    _normalize_number(bare_match.group(1)),
                    False,
                )
            )

    if not candidates:
        return None, False
    _, answer, has_explicit_format = max(candidates, key=lambda candidate: candidate[0])
    return answer, has_explicit_format


def extract_answer(solution_str: str) -> str | None:
    """从模型回复中提取并规范化明确的最终数字答案"""
    answer, _ = _extract_answer_details(solution_str)
    return answer


def _has_substantive_thinking(solution_str: str) -> bool:
    """检查 think 块是否同时包含步骤连接词和实际算式"""
    for think_content in THINK_BLOCK_PATTERN.findall(solution_str):
        compact_content = re.sub(r"\s+", "", think_content)
        if len(compact_content) < 8:
            continue
        if REASONING_CONNECTOR_PATTERN.search(think_content) is None:
            continue
        if ARITHMETIC_STEP_PATTERN.search(think_content) is None:
            continue
        return True
    return False


def _has_trusted_tool_success(extra_info: object) -> bool:
    """只接受 ToolAgentLoop 注入的正数工具奖励信号"""
    if not isinstance(extra_info, Mapping):
        return False
    tool_rewards = extra_info.get("tool_rewards")
    if not isinstance(tool_rewards, Sequence) or isinstance(tool_rewards, (str, bytes)):
        return False

    for value in tool_rewards:
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
            continue
        numeric_value = float(value)
        if math.isfinite(numeric_value) and numeric_value > 0:
            return True
    return False


def _empty_score(ground_truth_valid: bool = False) -> dict[str, float]:
    """构造 verl 可记录的全数值奖励分项"""
    return {
        "score": 0.0,
        "accuracy_reward": 0.0,
        "format_reward": 0.0,
        "tool_reward": 0.0,
        "think_reward": 0.0,
        "answer_extracted": 0.0,
        "ground_truth_valid": 1.0 if ground_truth_valid else 0.0,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """按单条 verl 样本计算正确性、格式、真实工具和有效推理奖励"""
    del kwargs
    normalized_ground_truth = _normalize_number(ground_truth)
    if normalized_ground_truth is None or not isinstance(data_source, str):
        return _empty_score()
    if not isinstance(solution_str, str):
        return _empty_score(ground_truth_valid=True)

    answer, has_explicit_format = _extract_answer_details(solution_str)
    result = _empty_score(ground_truth_valid=True)
    if answer is None:
        return result

    result["answer_extracted"] = 1.0
    if has_explicit_format:
        result["format_reward"] = FORMAT_REWARD

    is_correct = answer == normalized_ground_truth
    if is_correct:
        result["accuracy_reward"] = ACCURACY_REWARD
        if _has_trusted_tool_success(extra_info):
            result["tool_reward"] = TOOL_REWARD
        if _has_substantive_thinking(solution_str):
            result["think_reward"] = THINK_REWARD

    result["score"] = round(
        result["accuracy_reward"]
        + result["format_reward"]
        + result["tool_reward"]
        + result["think_reward"],
        6,
    )
    return result


def math_agent_reward(completions: list, ground_truths: list, **kwargs: Any) -> list[float]:
    """兼容原批量接口，并复用单样本抗投机奖励逻辑"""
    extra_infos = kwargs.get("extra_infos") or []
    data_sources = kwargs.get("data_sources") or []
    rewards = []
    for index, (completion, ground_truth) in enumerate(zip(completions, ground_truths)):
        extra_info = extra_infos[index] if index < len(extra_infos) else None
        data_source = data_sources[index] if index < len(data_sources) else "openai/gsm8k"
        score = compute_score(
            data_source=data_source,
            solution_str=completion,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        rewards.append(score["score"])
    return rewards


# 本地手工检查入口
if __name__ == "__main__":
    example = "<think>先计算 6 * 7 = 42，所以结果为 42。</think>\n最终答案是 42"
    print(compute_score("openai/gsm8k", example, "42", {"tool_rewards": [0.1]}))
