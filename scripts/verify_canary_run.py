#!/usr/bin/env python3
"""验证 Calculator Agent canary 训练的关键端到端证据。"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


GLOBAL_STEP_PATTERN = re.compile(
    r"[\"']?training/global_step[\"']?\s*[:=]\s*(\d+)"
)
GRAD_NORM_PATTERN = re.compile(
    r"[\"']?actor/grad_norm[\"']?\s*[:=]\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf))",
    re.IGNORECASE,
)


class CanaryVerificationError(ValueError):
    """表示 canary 证据缺失、损坏或未通过验证门。"""


def _parse_rollout_uid(uid: object) -> tuple[str, int]:
    """按 verl 的 ``sample_rollout_output`` 规则拆分 rollout UID。"""
    if not isinstance(uid, str):
        raise CanaryVerificationError("rollout uid 必须是字符串")
    parts = uid.rsplit("_", 2)
    if len(parts) != 3:
        raise CanaryVerificationError(f"无法解析 rollout uid: {uid}")
    try:
        rollout_index = int(parts[1])
        int(parts[2])
    except ValueError as error:
        raise CanaryVerificationError(f"无法解析 rollout uid: {uid}") from error
    return parts[0], rollout_index


def analyze_rollouts(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_prompts: int,
    expected_rollouts: int,
) -> dict[str, Any]:
    """验证 rollout 分组、奖励有限性、组内方差和工具调用可观测性。"""
    expected_count = expected_prompts * expected_rollouts
    if len(records) != expected_count:
        raise CanaryVerificationError(
            f"rollout 数量错误: 预期 {expected_count}，实际 {len(records)}"
        )

    grouped_rewards: dict[str, list[float]] = defaultdict(list)
    grouped_indices: dict[str, set[int]] = defaultdict(set)
    rewards = []
    tool_calls = 0
    invalid_calls = 0
    unknown_tool_calls = 0

    # 复用评估器的 Hermes 解析规则，避免训练与评估采用两套格式判定。
    from evaluate_model import parse_hermes_tool_calls

    for record in records:
        prompt_uid, rollout_index = _parse_rollout_uid(record.get("uid"))
        score = record.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise CanaryVerificationError("rollout 奖励必须是数值")
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise CanaryVerificationError("rollout 奖励必须全部有限")

        grouped_rewards[prompt_uid].append(numeric_score)
        grouped_indices[prompt_uid].add(rollout_index)
        rewards.append(numeric_score)

        output = record.get("output", "")
        if not isinstance(output, str):
            raise CanaryVerificationError("rollout output 必须是字符串")
        _, attempts = parse_hermes_tool_calls(output)
        for attempt in attempts:
            if attempt.error is not None:
                invalid_calls += 1
            elif attempt.name == "calculator":
                tool_calls += 1
            else:
                unknown_tool_calls += 1

    if len(grouped_rewards) != expected_prompts:
        raise CanaryVerificationError(
            f"prompt 分组数量错误: 预期 {expected_prompts}，实际 {len(grouped_rewards)}"
        )
    expected_indices = set(range(expected_rollouts))
    for prompt_uid, prompt_rewards in grouped_rewards.items():
        if len(prompt_rewards) != expected_rollouts:
            raise CanaryVerificationError(
                f"prompt {prompt_uid} 的 rollout 数量不是 {expected_rollouts}"
            )
        if grouped_indices[prompt_uid] != expected_indices:
            raise CanaryVerificationError(
                f"prompt {prompt_uid} 的 rollout 索引不完整"
            )

    reward_variance = statistics.pvariance(rewards)
    group_variances = {
        prompt_uid: statistics.pvariance(prompt_rewards)
        for prompt_uid, prompt_rewards in grouped_rewards.items()
    }
    nonzero_variance_groups = sum(value > 0 for value in group_variances.values())
    if reward_variance <= 0 or nonzero_variance_groups == 0:
        raise CanaryVerificationError("rollout 奖励方差必须大于 0")

    return {
        "rollout_count": len(records),
        "prompt_groups": len(grouped_rewards),
        "rollouts_per_prompt": expected_rollouts,
        "reward_variance": reward_variance,
        "nonzero_variance_groups": nonzero_variance_groups,
        "tool_calls": tool_calls,
        "invalid_calls": invalid_calls,
        "unknown_tool_calls": unknown_tool_calls,
    }


def analyze_training_log(log_text: str) -> dict[str, float | int]:
    """从 console 日志中提取已完成 step 和 actor 梯度范数。"""
    step_matches = GLOBAL_STEP_PATTERN.findall(log_text)
    grad_matches = GRAD_NORM_PATTERN.findall(log_text)
    if not step_matches:
        raise CanaryVerificationError("训练日志缺少 training/global_step")
    if not grad_matches:
        raise CanaryVerificationError("训练日志缺少 actor/grad_norm")

    global_step = int(step_matches[-1])
    actor_grad_norm = float(grad_matches[-1])
    if global_step < 1:
        raise CanaryVerificationError("训练至少需要完成一个优化 step")
    if not math.isfinite(actor_grad_norm) or actor_grad_norm <= 0:
        raise CanaryVerificationError("actor/grad_norm 必须是有限正数")
    return {"global_step": global_step, "actor_grad_norm": actor_grad_norm}


def inspect_actor_checkpoint(actor_path: Path | str) -> dict[str, int]:
    """检查单个 verl FSDP actor checkpoint 是否可用于恢复和合并。"""
    actor_path = Path(actor_path)
    fsdp_config_path = actor_path / "fsdp_config.json"
    huggingface_config_path = actor_path / "huggingface" / "config.json"
    if not fsdp_config_path.is_file():
        raise CanaryVerificationError(f"缺少 FSDP 配置: {fsdp_config_path}")
    if not huggingface_config_path.is_file():
        raise CanaryVerificationError(f"缺少 Hugging Face 配置: {huggingface_config_path}")

    try:
        fsdp_config = json.loads(fsdp_config_path.read_text(encoding="utf-8"))
        world_size = int(fsdp_config["world_size"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise CanaryVerificationError("fsdp_config.json 缺少有效 world_size") from error
    if world_size < 1:
        raise CanaryVerificationError("checkpoint world_size 必须大于 0")

    counts = {}
    for prefix, report_name in (
        ("model", "model_shards"),
        ("optim", "optimizer_shards"),
        ("extra_state", "extra_state_shards"),
    ):
        expected_paths = [
            actor_path / f"{prefix}_world_size_{world_size}_rank_{rank}.pt"
            for rank in range(world_size)
        ]
        missing_paths = [str(path) for path in expected_paths if not path.is_file()]
        if missing_paths:
            raise CanaryVerificationError(
                f"checkpoint 缺少 {prefix} 分片: {', '.join(missing_paths)}"
            )
        counts[report_name] = len(expected_paths)

    return {"world_size": world_size, **counts}


def _safetensor_weight_map(model_path: Path) -> dict[str, Path]:
    """建立参数名到 safetensors 分片的映射。"""
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise CanaryVerificationError(f"模型索引损坏: {index_path}") from error
        if not isinstance(weight_map, Mapping):
            raise CanaryVerificationError(f"模型索引缺少 weight_map: {index_path}")
        return {str(name): model_path / str(filename) for name, filename in weight_map.items()}

    from safetensors import safe_open

    result = {}
    for shard_path in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(shard_path), framework="np") as shard:
            for name in shard.keys():
                if name in result:
                    raise CanaryVerificationError(f"参数重复出现在多个分片中: {name}")
                result[name] = shard_path
    if not result:
        raise CanaryVerificationError(f"目录中没有 safetensors 模型: {model_path}")
    return result


def _read_parameter_array(shard_path: Path, name: str):
    """按需读取 BF16；转换为 FP32 是精确表示，不引入比较舍入误差。"""
    from safetensors import safe_open

    with safe_open(str(shard_path), framework="np") as shard:
        if shard.get_slice(name).get_dtype() != "BF16":
            return shard.get_tensor(name)
    # NumPy 没有内置 bfloat16；仅该分支按需使用训练环境已有的 PyTorch。
    with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
        return shard.get_tensor(name).float().numpy()


def compare_safetensor_models(
    base_model_path: Path | str,
    trained_model_path: Path | str,
) -> dict[str, Any]:
    """逐参数比较基础模型和合并模型，避免依赖分片文件布局。"""
    import numpy as np

    base_model_path = Path(base_model_path)
    trained_model_path = Path(trained_model_path)
    base_weights = _safetensor_weight_map(base_model_path)
    trained_weights = _safetensor_weight_map(trained_model_path)
    if set(base_weights) != set(trained_weights):
        missing = sorted(set(base_weights) - set(trained_weights))
        added = sorted(set(trained_weights) - set(base_weights))
        raise CanaryVerificationError(
            f"模型参数集合不一致: 缺少 {len(missing)}，新增 {len(added)}"
        )

    changed_parameter_names = []
    for name in sorted(base_weights):
        base_tensor = _read_parameter_array(base_weights[name], name)
        trained_tensor = _read_parameter_array(trained_weights[name], name)
        if base_tensor.shape != trained_tensor.shape:
            raise CanaryVerificationError(f"参数形状不一致: {name}")
        if not np.array_equal(base_tensor, trained_tensor):
            changed_parameter_names.append(name)

    return {
        "common_parameters": len(base_weights),
        "changed_parameters": len(changed_parameter_names),
        "changed_parameter_names": changed_parameter_names,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 verl 产生的 JSONL，并保留行号错误信息。"""
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise CanaryVerificationError(f"JSONL 第 {line_number} 行损坏: {path}") from error
        if not isinstance(record, dict):
            raise CanaryVerificationError(f"JSONL 第 {line_number} 行不是对象: {path}")
        records.append(record)
    return records


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    """汇总所有阶段 9 验证门，并在任何门失败时抛出稳定错误。"""
    rollout_report = analyze_rollouts(
        _load_jsonl(args.rollout_jsonl),
        expected_prompts=args.expected_prompts,
        expected_rollouts=args.expected_rollouts,
    )
    log_report = analyze_training_log(args.training_log.read_text(encoding="utf-8"))
    checkpoint_report = inspect_actor_checkpoint(args.actor_checkpoint)
    parameter_report = compare_safetensor_models(
        args.base_model_dir,
        args.trained_model_dir,
    )
    if parameter_report["changed_parameters"] < 1:
        raise CanaryVerificationError("合并模型中没有检测到发生变化的 actor 参数")
    return {
        "schema_version": 1,
        "passed": True,
        "rollout": rollout_report,
        "training": log_report,
        "checkpoint": checkpoint_report,
        "parameters": parameter_report,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-jsonl", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--trained-model-dir", type=Path, required=True)
    parser.add_argument("--expected-prompts", type=int, default=8)
    parser.add_argument("--expected-rollouts", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        report = build_report(args)
    except (CanaryVerificationError, OSError) as error:
        print(f"canary 验证失败: {error}", file=sys.stderr)
        return 1

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    # 独立脚本的搜索根为 scripts/，显式加入项目目录以加载共享解析规则。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
