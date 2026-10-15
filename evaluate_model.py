"""对普通 Hugging Face checkpoint 执行确定性数学 Agent 评估"""

import argparse
import json
import math
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


TOOL_CALL_START = "<tool_call>"
TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
# 无工具对照保留同样的推理与最终答案格式，不混入 Agent 的调用示例。
PLAIN_SYSTEM_PROMPT = (
    "Math assistant: reason in <think>...</think>. Solve without tools. "
    "After reasoning, the last line must be Final answer: <number>. "
    "Use an integer or decimal only: no units, currency symbols, percent signs, "
    "expressions, or trailing punctuation."
)
STRICT_FINAL_PATTERN = re.compile(r"Final answer: [+-]?\d+(?:\.\d+)?")
COMPLETE_THINK_PATTERN = re.compile(r"<think>\s*[^<\s].*?</think>", re.DOTALL)
CALCULATOR_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "计算数学表达式。输入一个数学表达式字符串，返回计算结果。",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "待计算的数学表达式",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class EvaluationConfig:
    """控制可复现评估的纯数据配置"""

    seed: int = 42
    batch_size: int = 4
    limit: int | None = None
    max_new_tokens: int = 512
    max_assistant_turns: int = 4
    do_sample: bool = False
    mode: str = "agent"
    max_total_new_tokens: int | None = None
    max_tool_rounds: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    enable_thinking: bool | None = None

    def __post_init__(self):
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed 必须是整数")
        for field_name in ("batch_size", "max_new_tokens", "max_assistant_turns"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} 必须是正整数")
        if self.limit is not None:
            if isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit <= 0:
                raise ValueError("limit 必须是正整数或 None")
        if not isinstance(self.do_sample, bool):
            raise ValueError("do_sample 必须为布尔值")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)) or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature 必须为有限正数")
        if isinstance(self.top_p, bool) or not isinstance(self.top_p, (int, float)) or not 0 < self.top_p <= 1:
            raise ValueError("top_p 必须在 (0, 1] 范围内")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k 必须为非负整数")
        if self.enable_thinking is not None and not isinstance(self.enable_thinking, bool):
            raise ValueError("enable_thinking 必须为布尔值或 None")
        if self.mode not in ("agent", "plain"):
            raise ValueError("mode 必须为 agent 或 plain")
        for field_name in ("max_total_new_tokens", "max_tool_rounds"):
            value = getattr(self, field_name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{field_name} 必须为正整数或 None")

    def to_dict(self) -> dict[str, Any]:
        """转换为可直接写入 JSON 的配置"""
        return asdict(self)


@dataclass(frozen=True)
class GeneratedTurn:
    """一次 assistant 生成结果及其 token 数"""

    text: str
    token_count: int
    finish_reason: str | None = None
    assistant_prefix: str = ""


@dataclass(frozen=True)
class ToolCallAttempt:
    """一段 Hermes 工具调用的解析结果"""

    name: str | None
    arguments: dict[str, Any] | None
    error: str | None
    raw: str


class GenerationBackend(Protocol):
    """评估核心依赖的最小模型生成接口"""

    def generate_batch(
        self,
        conversations: Sequence[Sequence[Mapping[str, Any]]],
        tools: Sequence[Mapping[str, Any]],
        max_new_tokens: int,
    ) -> Sequence[GeneratedTurn]:
        """为一批对话生成下一个 assistant 回合"""


def select_records(
    records: Sequence[Mapping[str, Any]],
    limit: int | None,
    seed: int,
) -> list[Mapping[str, Any]]:
    """按固定 seed 选择评估子集，并保持原始数据顺序"""
    selected = list(records)
    if limit is None:
        return selected
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit 必须是正整数或 None")
    if limit >= len(selected):
        return selected

    indices = sorted(random.Random(seed).sample(range(len(selected)), limit))
    return [selected[index] for index in indices]


def parse_hermes_tool_calls(text: str) -> tuple[str, list[ToolCallAttempt]]:
    """解析 verl Hermes 格式，并保留无效调用用于统计"""
    if not isinstance(text, str):
        raise TypeError("模型输出必须是字符串")

    attempts = []
    for raw_call in TOOL_CALL_PATTERN.findall(text):
        try:
            payload = json.loads(raw_call)
        except json.JSONDecodeError:
            attempts.append(ToolCallAttempt(None, None, "invalid_json", raw_call))
            continue

        if not isinstance(payload, Mapping):
            attempts.append(ToolCallAttempt(None, None, "invalid_structure", raw_call))
            continue
        name = payload.get("name")
        arguments = payload.get("arguments")
        if not isinstance(name, str) or not name or not isinstance(arguments, Mapping):
            attempts.append(ToolCallAttempt(None, None, "invalid_structure", raw_call))
            continue
        attempts.append(ToolCallAttempt(name, dict(arguments), None, raw_call))

    content = TOOL_CALL_PATTERN.sub("", text)
    unclosed_count = content.count(TOOL_CALL_START)
    if unclosed_count:
        for _ in range(unclosed_count):
            attempts.append(ToolCallAttempt(None, None, "unclosed_tool_call", content))
        content = content.split(TOOL_CALL_START, 1)[0]
    return content, attempts


def _append_failure(state: dict[str, Any], reason: str) -> None:
    """为单个样本追加不重复的失败原因"""
    if reason not in state["failure_reasons"]:
        state["failure_reasons"].append(reason)


def _build_tool_messages(
    state: dict[str, Any],
    content: str,
    attempt: ToolCallAttempt,
) -> bool:
    """执行首个工具调用并把 assistant/tool 消息追加到轨迹"""
    from calculator_tool import calculator

    if attempt.error is not None:
        state["invalid_calls"] += 1
        _append_failure(state, "invalid_tool_call")
        state["messages"].append({"role": "assistant", "content": content})
        return False

    name = attempt.name
    arguments = attempt.arguments or {}
    state["messages"].append(
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        }
    )

    if name != "calculator":
        state["invalid_calls"] += 1
        _append_failure(state, "unknown_tool")
        state["messages"].append({"role": "tool", "content": f"未知工具: {name}"})
        return True

    expression = arguments.get("expression")
    if set(arguments) != {"expression"} or not isinstance(expression, str):
        state["invalid_argument_calls"] += 1
        state["invalid_calls"] += 1
        _append_failure(state, "invalid_tool_arguments")
        state["messages"].append(
            {
                "role": "tool",
                "content": "计算错误: expression 必须是唯一的字符串参数",
            }
        )
        return True

    state["tool_executions"] += 1
    result = calculator(expression)
    state["messages"].append({"role": "tool", "content": result})
    if result.startswith("计算错误:"):
        state["invalid_calls"] += 1
        _append_failure(state, "calculator_error")
    else:
        state["tool_successes"] += 1
    return True


def _initial_state(record: Mapping[str, Any], selected_position: int) -> dict[str, Any]:
    """校验一条 Parquet 记录并创建评估状态"""
    prompt = record.get("prompt")
    reward_model = record.get("reward_model")
    if not isinstance(prompt, Sequence) or isinstance(prompt, (str, bytes)):
        raise ValueError(f"第 {selected_position} 条记录缺少合法 prompt")
    if not isinstance(reward_model, Mapping) or "ground_truth" not in reward_model:
        raise ValueError(f"第 {selected_position} 条记录缺少 ground_truth")

    messages = []
    for message in prompt:
        if not isinstance(message, Mapping):
            raise ValueError(f"第 {selected_position} 条记录包含非法消息")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(f"第 {selected_position} 条记录包含非法消息字段")
        messages.append({"role": role, "content": content})

    extra_info = record.get("extra_info")
    sample_index = (
        extra_info.get("index", selected_position)
        if isinstance(extra_info, Mapping)
        else selected_position
    )
    return {
        "sample_index": sample_index,
        "data_source": record.get("data_source", ""),
        "ground_truth": reward_model["ground_truth"],
        "messages": messages,
        "assistant_parts": [],
        "final_assistant_text": None,
        "assistant_tokens": 0,
        "tool_attempts": 0,
        "tool_successes": 0,
        "invalid_calls": 0,
        "used_tool": False,
        "failure_reasons": [],
        "error_details": [],
        "raw_assistant_outputs": [],
        "turn_finish_reasons": [],
        "turn_token_counts": [],
        "template_assistant_prefixes": [],
        "tool_executions": 0,
        "invalid_argument_calls": 0,
        "tool_rounds": 0,
        "generation_limit_hits": 0,
        "tool_call_stop_violations": 0,
    }


def _generate_active_batch(
    states: list[dict[str, Any]],
    active_indices: Sequence[int],
    backend: GenerationBackend,
    config: EvaluationConfig,
    assistant_turn: int,
) -> list[int]:
    """生成一轮回复，执行工具，并返回仍需继续的样本索引"""
    next_active = []
    # 不同样本的剩余预算可能不同，按预算分组，避免批处理突破单条总上限。
    budget_groups = {}
    for index in active_indices:
        budget = config.max_new_tokens
        if config.max_total_new_tokens is not None:
            budget = min(budget, config.max_total_new_tokens - states[index]["assistant_tokens"])
        if budget <= 0:
            _append_failure(states[index], "total_token_budget")
            continue
        budget_groups.setdefault(budget, []).append(index)
    chunks = [
        (budget, indices[offset : offset + config.batch_size])
        for budget, indices in budget_groups.items()
        for offset in range(0, len(indices), config.batch_size)
    ]
    for budget, chunk_indices in chunks:
        conversations = [states[index]["messages"] for index in chunk_indices]
        try:
            outputs = list(
                backend.generate_batch(
                    conversations,
                    [CALCULATOR_TOOL_SCHEMA] if config.mode == "agent" else [],
                    budget,
                )
            )
            if len(outputs) != len(chunk_indices):
                raise RuntimeError("模型返回数量与输入 batch 不一致")
        except Exception as exc:
            error_detail = f"{type(exc).__name__}: {exc}"
            for index in chunk_indices:
                _append_failure(states[index], "generation_error")
                states[index]["error_details"].append(error_detail)
            continue

        for index, output in zip(chunk_indices, outputs):
            state = states[index]
            if (
                not isinstance(output, GeneratedTurn)
                or not isinstance(output.text, str)
                or isinstance(output.token_count, bool)
                or not isinstance(output.token_count, int)
                or output.token_count < 0
                or output.token_count > budget
                or not isinstance(output.assistant_prefix, str)
            ):
                _append_failure(state, "generation_error")
                state["error_details"].append("模型返回了非法 GeneratedTurn")
                continue

            # 外部前缀仅作诊断，不能补齐模型输出中的think标签或参与评分。
            logical_text = output.text
            content, attempts = parse_hermes_tool_calls(logical_text)
            state["assistant_tokens"] += output.token_count
            state["raw_assistant_outputs"].append(output.text)
            state["turn_finish_reasons"].append(output.finish_reason)
            state["turn_token_counts"].append(output.token_count)
            state["template_assistant_prefixes"].append(output.assistant_prefix)
            state["generation_limit_hits"] += int(output.finish_reason == "length")
            if content.strip():
                state["assistant_parts"].append(content.strip())

            if not attempts:
                state["messages"].append({"role": "assistant", "content": content})
                state["final_assistant_text"] = content.strip()
                continue

            state["used_tool"] = True
            state["tool_attempts"] += len(attempts)
            matches = list(TOOL_CALL_PATTERN.finditer(output.text))
            if matches and output.text[matches[0].end() :].strip():
                state["tool_call_stop_violations"] += 1
            if config.mode == "plain":
                state["messages"].append({"role": "assistant", "content": logical_text})
                _append_failure(state, "tools_disabled")
                continue
            if config.max_tool_rounds is not None and state["tool_rounds"] >= config.max_tool_rounds:
                state["messages"].append({"role": "assistant", "content": logical_text})
                _append_failure(state, "tool_round_limit")
                continue
            first_attempt = attempts[0]
            for _ in attempts[1:]:
                state["invalid_calls"] += 1
                _append_failure(state, "parallel_call_limit")

            can_continue = _build_tool_messages(state, content, first_attempt)
            if not can_continue:
                continue
            state["tool_rounds"] += 1
            if assistant_turn + 1 >= config.max_assistant_turns:
                _append_failure(state, "assistant_turn_limit")
            else:
                next_active.append(index)
    return next_active


def _safe_rate(numerator: int, denominator: int) -> float:
    """计算零分母安全的评估比例"""
    return numerator / denominator if denominator else 0.0


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    backend: GenerationBackend,
    config: EvaluationConfig,
) -> dict[str, Any]:
    """运行批量多回合评估并返回可序列化报告"""
    from reward_function import compute_score, extract_answer

    all_records = list(records)
    selected_records = select_records(all_records, config.limit, config.seed)
    states = [
        _initial_state(record, selected_position)
        for selected_position, record in enumerate(selected_records)
    ]
    if config.mode == "plain":
        for state in states:
            state["messages"] = [
                {"role": "system", "content": PLAIN_SYSTEM_PROMPT},
                *[message for message in state["messages"] if message["role"] != "system"],
            ]

    active_indices = list(range(len(states)))
    for assistant_turn in range(config.max_assistant_turns):
        if not active_indices:
            break
        active_indices = _generate_active_batch(
            states,
            active_indices,
            backend,
            config,
            assistant_turn,
        )

    samples = []
    failure_counts = Counter()
    correct_samples = 0
    format_pass_samples = 0
    tool_call_samples = 0
    tool_attempts = 0
    tool_successes = 0
    invalid_calls = 0
    assistant_tokens = 0
    diagnostics = Counter()

    for state in states:
        solution = state["final_assistant_text"] or ""
        score = compute_score(
            data_source=state["data_source"],
            solution_str=solution,
            ground_truth=state["ground_truth"],
            extra_info={},
        )
        extracted_answer = extract_answer(solution)
        is_correct = score["accuracy_reward"] > 0
        format_passed = score["format_reward"] > 0
        last_line = solution.strip().splitlines()[-1] if solution.strip() else ""
        strict_final = STRICT_FINAL_PATTERN.fullmatch(last_line) is not None
        raw_complete_think = any(COMPLETE_THINK_PATTERN.search(text) is not None for text in state["raw_assistant_outputs"])
        complete_think = raw_complete_think
        diagnostics["answer_extracted_samples"] += int(extracted_answer is not None)
        diagnostics["strict_final_format_samples"] += int(strict_final)
        diagnostics["complete_think_samples"] += int(complete_think)
        diagnostics["raw_complete_think_samples"] += int(raw_complete_think)
        diagnostics["length_limited_samples"] += int(state["generation_limit_hits"] > 0)
        for field_name in ("tool_executions", "invalid_argument_calls", "generation_limit_hits", "tool_call_stop_violations"):
            diagnostics[field_name] += state[field_name]
        if extracted_answer is None:
            _append_failure(state, "missing_final_answer")
        elif not is_correct:
            _append_failure(state, "wrong_final_answer")

        correct_samples += int(is_correct)
        format_pass_samples += int(format_passed)
        tool_call_samples += int(state["used_tool"])
        tool_attempts += state["tool_attempts"]
        tool_successes += state["tool_successes"]
        invalid_calls += state["invalid_calls"]
        assistant_tokens += state["assistant_tokens"]
        failure_counts.update(state["failure_reasons"])
        samples.append(
            {
                "sample_index": state["sample_index"],
                "ground_truth": str(state["ground_truth"]),
                "predicted_answer": extracted_answer,
                "is_correct": is_correct,
                "format_passed": format_passed,
                "used_tool": state["used_tool"],
                "tool_attempts": state["tool_attempts"],
                "tool_successes": state["tool_successes"],
                "invalid_calls": state["invalid_calls"],
                "assistant_tokens": state["assistant_tokens"],
                "assistant_text": "\n".join(state["assistant_parts"]),
                "final_assistant_text": solution,
                "failure_reasons": list(state["failure_reasons"]),
                "error_details": list(state["error_details"]),
                "strict_final_format": strict_final,
                "complete_think": complete_think,
                "raw_complete_think": raw_complete_think,
                "template_assistant_prefixes": list(state["template_assistant_prefixes"]),
                "raw_assistant_outputs": list(state["raw_assistant_outputs"]),
                "turn_finish_reasons": list(state["turn_finish_reasons"]),
                "turn_token_counts": list(state["turn_token_counts"]),
                "messages": state["messages"],
                **{field_name: state[field_name] for field_name in ("tool_executions", "invalid_argument_calls", "generation_limit_hits", "tool_call_stop_violations")},
            }
        )

    evaluated_samples = len(states)
    summary = {
        "dataset_size": len(all_records),
        "selected_samples": len(selected_records),
        "evaluated_samples": evaluated_samples,
        "correct_samples": correct_samples,
        "format_pass_samples": format_pass_samples,
        "tool_call_samples": tool_call_samples,
        "tool_attempts": tool_attempts,
        "tool_successes": tool_successes,
        "invalid_calls": invalid_calls,
        "assistant_tokens": assistant_tokens,
        "final_answer_accuracy": _safe_rate(correct_samples, evaluated_samples),
        "format_pass_rate": _safe_rate(format_pass_samples, evaluated_samples),
        "tool_call_rate": _safe_rate(tool_call_samples, evaluated_samples),
        "tool_success_rate": _safe_rate(tool_successes, tool_attempts),
        "invalid_call_rate": _safe_rate(invalid_calls, tool_attempts),
        "average_assistant_tokens": _safe_rate(assistant_tokens, evaluated_samples),
        "failure_reasons": dict(sorted(failure_counts.items())),
        **{field_name: diagnostics[field_name] for field_name in (
            "answer_extracted_samples", "strict_final_format_samples", "complete_think_samples",
            "raw_complete_think_samples", "length_limited_samples", "tool_executions", "invalid_argument_calls",
            "generation_limit_hits", "tool_call_stop_violations",
        )},
        "answer_extraction_rate": _safe_rate(diagnostics["answer_extracted_samples"], evaluated_samples),
        "strict_final_format_rate": _safe_rate(diagnostics["strict_final_format_samples"], evaluated_samples),
        "complete_think_rate": _safe_rate(diagnostics["complete_think_samples"], evaluated_samples),
        "raw_complete_think_rate": _safe_rate(diagnostics["raw_complete_think_samples"], evaluated_samples),
        "length_limited_rate": _safe_rate(diagnostics["length_limited_samples"], evaluated_samples),
        "invalid_argument_rate": _safe_rate(diagnostics["invalid_argument_calls"], tool_attempts),
    }
    return {
        "config": config.to_dict(),
        "summary": summary,
        "samples": samples,
    }


def load_parquet_records(data_file: str | Path) -> list[Mapping[str, Any]]:
    """延迟加载一个本地 verl Parquet 评估文件"""
    data_path = Path(data_file).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"评估数据文件不存在: {data_path}")
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("评估 Parquet 需要可选依赖 'datasets'") from exc

    dataset = load_dataset(
        "parquet",
        data_files={"evaluation": str(data_path)},
        split="evaluation",
    )
    return [dict(record) for record in dataset]


def resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    """将 CLI 精度名称转换为 Transformers 接受的 torch dtype"""
    if dtype_name == "auto":
        return "auto"
    dtype = getattr(torch_module, dtype_name, None)
    if dtype is None or dtype_name not in {"float16", "bfloat16", "float32"}:
        raise ValueError(f"不支持的 torch dtype: {dtype_name}")
    return dtype


def configure_generation_tokenizer(tokenizer: Any) -> None:
    """为 decoder-only 模型配置安全的批量生成填充方式"""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer 同时缺少 pad_token 和 eos_token")
        tokenizer.pad_token = tokenizer.eos_token


class HuggingFaceBackend:
    """普通 Hugging Face causal LM 的批量生成适配器"""

    def __init__(self, model: Any, tokenizer: Any, torch_module: Any, *, evaluation_config: EvaluationConfig | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.evaluation_config = evaluation_config or EvaluationConfig()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        seed: int,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        evaluation_config: EvaluationConfig | None = None,
    ) -> "HuggingFaceBackend":
        """延迟加载模型，避免普通源码检查依赖 PyTorch"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError("加载模型需要完整的 PyTorch 与 Transformers 环境") from exc

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        configure_generation_tokenizer(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=resolve_torch_dtype(torch, torch_dtype),
            device_map=device_map,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        model.eval()
        return cls(model, tokenizer, torch, evaluation_config=evaluation_config)

    def _count_generated_tokens(self, token_ids: Sequence[int], stop_token_ids: set[int] | None = None) -> int:
        """统计到首个 EOS 为止的真实生成 token"""
        ids = list(token_ids)
        eos_ids = self._generation_eos_ids() if stop_token_ids is None else stop_token_ids
        for position, token_id in enumerate(ids):
            if token_id in eos_ids:
                return position + 1
        pad_token_id = self.tokenizer.pad_token_id
        while ids and pad_token_id is not None and ids[-1] == pad_token_id:
            ids.pop()
        return len(ids)

    def _generation_eos_ids(self) -> set[int]:
        """结合模型与 tokenizer 的结束标记，避免把正常 EOS 误报为长度截断。"""
        configured = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        values = configured if isinstance(configured, (list, tuple)) else [configured]
        return {value for value in [*values, self.tokenizer.eos_token_id] if value is not None}

    def _tool_end_token_id(self) -> int:
        """校验当前模型的闭合标签是单个token，禁止跨模型硬编码停止编号。"""
        ids = self.tokenizer.encode("</tool_call>", add_special_tokens=False)
        if len(ids) != 1 or isinstance(ids[0], bool) or not isinstance(ids[0], int) or self.tokenizer.decode(ids, skip_special_tokens=False) != "</tool_call>":
            raise ValueError("工具停止要求 </tool_call> 为可原样解码的单个token")
        if self.tokenizer.decode(ids, skip_special_tokens=True) != "</tool_call>":
            raise ValueError("工具闭合标签不能被skip_special_tokens移除")
        return ids[0]

    def generate_batch(
        self,
        conversations: Sequence[Sequence[Mapping[str, Any]]],
        tools: Sequence[Mapping[str, Any]],
        max_new_tokens: int,
    ) -> list[GeneratedTurn]:
        """使用配置中的greedy或固定种子采样生成，保留原生assistant前缀。"""
        template_kwargs = {}
        if self.evaluation_config.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.evaluation_config.enable_thinking
        rendered = [
            self.tokenizer.apply_chat_template(
                list(messages),
                tools=list(tools),
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            for messages in conversations
        ]
        prefixes = []
        for messages, text in zip(conversations, rendered):
            base = self.tokenizer.apply_chat_template(list(messages), tools=list(tools), tokenize=False, add_generation_prompt=False, **template_kwargs)
            suffix = text[len(base) :] if text.startswith(base) else ""
            marker = "<|im_start|>assistant\n"
            prefix = suffix[len(marker) :] if suffix.startswith(marker) else ""
            # 用户要求标签由模型生成；遇到预填模板明确拒绝，不静默删改模板。
            if "<think>" in prefix or "</think>" in prefix:
                raise ValueError("当前评估禁止模板预填think标签，须由模型自行生成")
            prefixes.append("")
        encoded = self.tokenizer(
            rendered,
            padding=True,
            return_tensors="pt",
        )
        encoded = {
            name: tensor.to(self.model.device)
            for name, tensor in encoded.items()
        }
        input_width = encoded["input_ids"].shape[1]
        generation_kwargs = {"do_sample": self.evaluation_config.do_sample}
        eos_ids = self._generation_eos_ids()
        tool_end_id = None
        if self.evaluation_config.mode == "agent" and tools:
            tool_end_id = self._tool_end_token_id()
            # 仅本次生成增加停止条件；HF逐行结束，保留模型产生的闭合token与原有EOS。
            generation_kwargs["eos_token_id"] = sorted(eos_ids | {tool_end_id})
        if self.evaluation_config.do_sample:
            generation_kwargs.update(temperature=self.evaluation_config.temperature, top_p=self.evaluation_config.top_p, top_k=self.evaluation_config.top_k)
        with self.torch.inference_mode():
            sequences = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                **generation_kwargs,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        results = []
        for sequence, prefix in zip(sequences, prefixes):
            generated_ids = sequence[input_width:].tolist()
            token_count = self._count_generated_tokens(generated_ids, eos_ids | ({tool_end_id} if tool_end_id is not None else set()))
            generated_ids = generated_ids[:token_count]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            ended_with_eos = bool(generated_ids and generated_ids[-1] in eos_ids)
            ended_with_tool = bool(generated_ids and tool_end_id is not None and generated_ids[-1] == tool_end_id)
            finish_reason = "tool_call" if ended_with_tool else ("eos" if ended_with_eos else ("length" if token_count >= max_new_tokens else None))
            results.append(GeneratedTurn(text=text, token_count=token_count, finish_reason=finish_reason, assistant_prefix=prefix))
        return results


def write_json_report(report: Mapping[str, Any], output_file: str | Path) -> None:
    """以 UTF-8 JSON 保存完整评估报告"""
    output_path = Path(output_file).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """构造确定性评估命令行参数"""
    parser = argparse.ArgumentParser(description="评估 Calculator Agent 的普通 HF checkpoint")
    parser.add_argument("--model-path", required=True, help="本地 HF 目录或 Hugging Face 模型 ID")
    parser.add_argument(
        "--data-file",
        default="data/final_test.parquet",
        help="verl Parquet 评估文件",
    )
    parser.add_argument(
        "--output-file",
        default="reports/evaluation.json",
        help="完整 JSON 报告输出路径",
    )
    parser.add_argument("--seed", type=int, default=42, help="固定样本选择和模型随机种子")
    parser.add_argument("--batch-size", type=int, default=4, help="每次生成的样本数")
    parser.add_argument("--limit", type=int, default=None, help="固定 seed 抽取的样本上限")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="每回合最大生成 token")
    parser.add_argument("--max-assistant-turns", type=int, default=4, help="最大 assistant 回合数")
    parser.add_argument("--mode", choices=("agent", "plain"), default="agent", help="工具 Agent 或无工具数学对照")
    parser.add_argument("--max-total-new-tokens", type=int, help="整条轨迹的 assistant 生成总预算，不含工具消息")
    parser.add_argument("--max-tool-rounds", type=int, help="工具交互轮上限，省略时保持原评估行为")
    parser.add_argument("--do-sample", action="store_true", help="固定seed采样；默认仍为greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--thinking", choices=("auto", "on", "off"), default="auto", help="原生模板thinking开关")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Hugging Face 模型加载精度",
    )
    parser.add_argument("--device-map", default="auto", help="Transformers device_map")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="禁止 Hugging Face 联网下载",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="允许执行模型仓库自定义代码，默认关闭",
    )
    return parser


def _print_summary(summary: Mapping[str, Any]) -> None:
    """打印便于终端查看的核心评估指标"""
    print(
        "评估完成: "
        f"samples={summary['evaluated_samples']}, "
        f"accuracy={summary['final_answer_accuracy']:.2%}, "
        f"format={summary['format_pass_rate']:.2%}, "
        f"tool_call={summary['tool_call_rate']:.2%}, "
        f"tool_success={summary['tool_success_rate']:.2%}, "
        f"invalid_call={summary['invalid_call_rate']:.2%}, "
        f"avg_assistant_tokens={summary['average_assistant_tokens']:.2f}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """加载数据和普通 HF checkpoint，执行评估并保存报告"""
    args = build_argument_parser().parse_args(argv)
    config = EvaluationConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        max_assistant_turns=args.max_assistant_turns,
        mode=args.mode,
        max_total_new_tokens=args.max_total_new_tokens,
        max_tool_rounds=args.max_tool_rounds,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        enable_thinking={"auto": None, "on": True, "off": False}[args.thinking],
    )
    records = load_parquet_records(args.data_file)
    backend = HuggingFaceBackend.from_pretrained(
        args.model_path,
        seed=config.seed,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        evaluation_config=config,
    )
    report = evaluate_records(records, backend, config)
    report["config"].update(
        {
            "model_path": args.model_path,
            "data_file": str(Path(args.data_file).expanduser()),
            "torch_dtype": args.torch_dtype,
            "device_map": args.device_map,
            "local_files_only": args.local_files_only,
            "trust_remote_code": args.trust_remote_code,
        }
    )
    write_json_report(report, args.output_file)
    _print_summary(report["summary"])
    print(f"完整报告: {Path(args.output_file).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
