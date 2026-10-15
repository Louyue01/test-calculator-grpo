import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATE_PATH = PROJECT_ROOT / "evaluate_model.py"
MERGE_SCRIPT = PROJECT_ROOT / "scripts" / "merge_checkpoint.sh"


def make_record(index, ground_truth, question=None):
    """构造与 final_test.parquet 一致的最小评估记录"""
    return {
        "data_source": "openai/gsm8k",
        "prompt": [
            {"role": "system", "content": "你是一个数学助手。"},
            {"role": "user", "content": question or f"问题 {index}"},
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(ground_truth)},
        "extra_info": {
            "split": "final_test",
            "original_split": "test",
            "index": index,
            "question": question or f"问题 {index}",
        },
    }


class ScriptedBackend:
    """只替换外部模型生成，保留真实评估与工具执行逻辑"""

    def __init__(self, scripts):
        self.scripts = scripts

    def generate_batch(self, conversations, tools, max_new_tokens):
        del tools, max_new_tokens
        from evaluate_model import GeneratedTurn

        outputs = []
        for messages in conversations:
            question = next(
                message["content"] for message in messages if message["role"] == "user"
            )
            assistant_turn = sum(message["role"] == "assistant" for message in messages)
            turn = self.scripts[question][assistant_turn]

            # 第二回合只有收到真实 calculator 结果后才返回正确答案
            if callable(turn):
                turn = turn(messages)
            outputs.append(GeneratedTurn(*turn))
        return outputs


class EvaluateImportSafetyTests(unittest.TestCase):
    def test_import_does_not_load_optional_model_dependencies_or_write_cache(self):
        """导入评估器时不应加载模型、访问网络或创建缓存"""
        script = textwrap.dedent(
            f"""
            import importlib.util
            import sys

            spec = importlib.util.spec_from_file_location("evaluate_model", {str(EVALUATE_PATH)!r})
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            """
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            cache_path = temporary_path / "cache"
            environment = os.environ.copy()
            environment.update(
                {
                    "HF_HOME": str(cache_path),
                    "HF_DATASETS_CACHE": str(cache_path / "datasets"),
                    "TRANSFORMERS_CACHE": str(cache_path / "transformers"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            completed = subprocess.run(
                [sys.executable, "-B", "-c", script],
                cwd=temporary_path,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "")
            self.assertEqual(list(temporary_path.iterdir()), [])


class DeterministicSelectionTests(unittest.TestCase):
    def test_same_seed_selects_same_subset_without_mutating_input(self):
        from evaluate_model import select_records

        records = [make_record(index, index) for index in range(12)]
        original_indices = [record["extra_info"]["index"] for record in records]

        first = select_records(records, limit=5, seed=7)
        repeated = select_records(records, limit=5, seed=7)
        different_seed = select_records(records, limit=5, seed=8)

        first_indices = [record["extra_info"]["index"] for record in first]
        self.assertEqual(
            first_indices,
            [record["extra_info"]["index"] for record in repeated],
        )
        self.assertNotEqual(
            first_indices,
            [record["extra_info"]["index"] for record in different_seed],
        )
        self.assertEqual(
            [record["extra_info"]["index"] for record in records],
            original_indices,
        )

    def test_limit_none_keeps_the_complete_dataset_order(self):
        from evaluate_model import select_records

        records = [make_record(index, index) for index in range(4)]

        selected = select_records(records, limit=None, seed=99)

        self.assertEqual(
            [record["extra_info"]["index"] for record in selected],
            [0, 1, 2, 3],
        )

    def test_boolean_limit_is_rejected_even_when_dataset_has_one_record(self):
        from evaluate_model import select_records

        with self.assertRaisesRegex(ValueError, "limit"):
            select_records([make_record(0, 0)], limit=True, seed=42)


class HermesToolCallTests(unittest.TestCase):
    def test_parser_accepts_valid_calculator_call_and_removes_tag_from_content(self):
        from evaluate_model import parse_hermes_tool_calls

        content, attempts = parse_hermes_tool_calls(
            '先计算。<tool_call>{"name":"calculator","arguments":{"expression":"6 * 7"}}</tool_call>'
        )

        self.assertEqual(content, "先计算。")
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0].name, "calculator")
        self.assertEqual(attempts[0].arguments, {"expression": "6 * 7"})
        self.assertIsNone(attempts[0].error)

    def test_parser_counts_malformed_and_unclosed_blocks_as_invalid_attempts(self):
        from evaluate_model import parse_hermes_tool_calls

        _, malformed = parse_hermes_tool_calls("<tool_call>{bad json}</tool_call>")
        _, unclosed = parse_hermes_tool_calls("<tool_call>{\"name\":\"calculator\"")

        self.assertEqual(len(malformed), 1)
        self.assertEqual(malformed[0].error, "invalid_json")
        self.assertEqual(len(unclosed), 1)
        self.assertEqual(unclosed[0].error, "unclosed_tool_call")


class EvaluateRecordsTests(unittest.TestCase):
    def test_reports_each_turn_finish_reason_and_count(self):
        """工具结束与最终EOS必须逐轮记录，不能由整条轨迹的长度反推。"""
        from evaluate_model import EvaluationConfig, GeneratedTurn, evaluate_records
        class Backend:
            def generate_batch(self, conversations, tools, max_new_tokens):
                if conversations[0][-1]["role"] == "tool":
                    return [GeneratedTurn("Final answer: 42", 6, finish_reason="eos")]
                return [GeneratedTurn('<tool_call>{"name":"calculator","arguments":{"expression":"6*7"}}</tool_call>', 15, finish_reason="tool_call")]
        sample = evaluate_records([make_record(0, 42)], Backend(), EvaluationConfig())["samples"][0]
        self.assertEqual(sample["turn_finish_reasons"], ["tool_call", "eos"])
        self.assertEqual(sample["turn_token_counts"], [15, 6])

    def test_think_tags_split_across_tool_turns_do_not_count_as_complete_block(self):
        """不能把两个assistant轮次各半个标签拼成一个完整思考块。"""
        from evaluate_model import EvaluationConfig, evaluate_records

        backend = ScriptedBackend({"split-think": [
            ('<think>First calculate 6*7.<tool_call>{"name":"calculator","arguments":{"expression":"6*7"}}</tool_call>', 20),
            ("</think>\nFinal answer: 42", 8),
        ]})
        sample = evaluate_records([make_record(0, 42, "split-think")], backend, EvaluationConfig())["samples"][0]
        self.assertFalse(sample["complete_think"])
        self.assertFalse(sample["raw_complete_think"])

    def test_disabled_tool_trace_does_not_prepend_external_think_prefix(self):
        """即使后端报告模板前缀，也不能将其补成模型生成的消息。"""
        from evaluate_model import EvaluationConfig, GeneratedTurn, evaluate_records

        raw = 'First calculate.</think><tool_call>{"name":"calculator","arguments":{"expression":"6*7"}}</tool_call>'
        class PrefixBackend:
            def generate_batch(self, conversations, tools, max_new_tokens):
                return [GeneratedTurn(raw, 20, assistant_prefix="<think>\n")]
        sample = evaluate_records([make_record(0, 42)], PrefixBackend(), EvaluationConfig(mode="plain"))["samples"][0]
        self.assertEqual(sample["messages"][-1]["content"], raw)
        self.assertEqual(sample["raw_assistant_outputs"], [raw])

    def test_template_think_prefix_is_not_falsely_recorded_as_model_output(self):
        """外部前缀不能把模型只生成的闭合标签补成完整think或计入指标。"""
        from evaluate_model import EvaluationConfig, GeneratedTurn, evaluate_records

        raw = "First calculate 6*7=42.</think>\nFinal answer: 42"

        class PrefixBackend:
            def generate_batch(self, conversations, tools, max_new_tokens):
                return [GeneratedTurn(raw, 18, assistant_prefix="<think>\n")]

        report = evaluate_records([make_record(0, 42)], PrefixBackend(), EvaluationConfig())
        sample = report["samples"][0]
        self.assertEqual(sample["raw_assistant_outputs"], [raw])
        self.assertEqual(sample["template_assistant_prefixes"], ["<think>\n"])
        self.assertFalse(sample["complete_think"])
        self.assertFalse(sample["raw_complete_think"])
        self.assertTrue(sample["is_correct"])
        self.assertEqual(sample["assistant_tokens"], 18)
        self.assertEqual(sample["messages"][-1]["content"], raw)

    def test_backend_cannot_report_more_tokens_than_requested_budget(self):
        """生成后端若超出预算，必须报告接口错误，不能产生失真的公平对照。"""
        from evaluate_model import EvaluationConfig, evaluate_records

        backend = ScriptedBackend({"over-budget": [("Final answer: 42", 6)]})
        sample = evaluate_records([make_record(0, 42, "over-budget")], backend, EvaluationConfig(max_new_tokens=5))["samples"][0]
        self.assertIn("generation_error", sample["failure_reasons"])
        self.assertEqual(sample["assistant_tokens"], 0)

    def test_empty_evaluation_has_zero_diagnostic_counts(self):
        """空评估也要提供稳定的诊断字段，不能让汇总消费者因缺键失败。"""
        from evaluate_model import EvaluationConfig, evaluate_records

        summary = evaluate_records([], ScriptedBackend({}), EvaluationConfig())["summary"]
        self.assertEqual(summary["strict_final_format_samples"], 0)
        self.assertEqual(summary["tool_executions"], 0)
        self.assertEqual(summary["generation_limit_hits"], 0)

    def test_plain_mode_removes_tools_and_does_not_execute_generated_calls(self):
        """无工具对照不得继承 Agent 提示，也不得执行模型自行生成的调用。"""
        from evaluate_model import EvaluationConfig, GeneratedTurn, evaluate_records

        record = make_record(0, 42, "plain-question")
        record["prompt"][0]["content"] = "Agent-only calculator instructions"

        class PlainBackend:
            def generate_batch(self, conversations, tools, max_new_tokens):
                if tools or "Agent-only" in conversations[0][0]["content"]:
                    raise RuntimeError("无工具对照仍收到 Agent 协议")
                return [GeneratedTurn('<tool_call>{"name":"calculator","arguments":{"expression":"6*7"}}</tool_call>', 12)]

        report = evaluate_records([record], PlainBackend(), EvaluationConfig(mode="plain"))
        sample = report["samples"][0]
        self.assertEqual(sample["tool_executions"], 0)
        self.assertEqual(sample["tool_successes"], 0)
        self.assertIn("tools_disabled", sample["failure_reasons"])
        self.assertNotIn("generation_error", sample["failure_reasons"])
        self.assertEqual(record["prompt"][0]["content"], "Agent-only calculator instructions")

    def test_reports_strict_format_thinking_and_real_invalid_arguments_separately(self):
        """宽松格式奖励、思考标签和参数错误不可冒充严格格式或真实执行。"""
        from evaluate_model import EvaluationConfig, evaluate_records

        backend = ScriptedBackend({
            "strict": [("<think>First compute 6*7=42.</think>\nFinal answer: 42", 12)],
            "loose": [("Final answer: 42%", 8)],
            "bad-args": [('<tool_call>{"name":"calculator","arguments":{"expression":"6*7","x":1}}</tool_call>', 10), ("Final answer: 42", 6)],
        })
        report = evaluate_records([make_record(0, 42, "strict"), make_record(1, 42, "loose"), make_record(2, 42, "bad-args")], backend, EvaluationConfig())
        summary = report["summary"]
        self.assertEqual(summary["strict_final_format_samples"], 2)
        self.assertEqual(summary["complete_think_samples"], 1)
        self.assertEqual(summary["answer_extracted_samples"], 3)
        self.assertEqual(summary["invalid_argument_calls"], 1)
        self.assertEqual(summary["tool_executions"], 0)
        self.assertAlmostEqual(summary["strict_final_format_rate"], 2 / 3)

    def test_total_budget_and_one_tool_round_prevent_unfair_additional_generation(self):
        """第二轮只获得剩余预算，到达工具轮上限后不能再次执行计算器。"""
        from evaluate_model import EvaluationConfig, GeneratedTurn, evaluate_records

        class BudgetBackend:
            def generate_batch(self, conversations, tools, max_new_tokens):
                if conversations[0][-1]["role"] == "tool":
                    if max_new_tokens != 5:
                        raise RuntimeError("第二轮预算未扣除首轮已生成的7个token")
                    return [GeneratedTurn('<tool_call>{"name":"calculator","arguments":{"expression":"1+1"}}</tool_call>', 5, finish_reason="length")]
                return [GeneratedTurn('<tool_call>{"name":"calculator","arguments":{"expression":"6*7"}}</tool_call>', 7)]

        report = evaluate_records([make_record(0, 42)], BudgetBackend(), EvaluationConfig(max_new_tokens=12, max_total_new_tokens=12, max_tool_rounds=1))
        sample = report["samples"][0]
        self.assertEqual(sample["assistant_tokens"], 12)
        self.assertEqual(sample["tool_executions"], 1)
        self.assertEqual(sample["generation_limit_hits"], 1)
        self.assertIn("tool_round_limit", sample["failure_reasons"])
        self.assertNotIn("generation_error", sample["failure_reasons"])

    def test_stop_violation_and_raw_messages_are_retained(self):
        """真实工具返回之前的续写必须可见，不能在解析时从审计轨迹中消失。"""
        from evaluate_model import EvaluationConfig, evaluate_records

        raw = '<tool_call>{"name":"calculator","arguments":{"expression":"6*7"}}</tool_call>\nTool: 999'
        backend = ScriptedBackend({"stop": [(raw, 20), ("Final answer: 42", 5)]})
        sample = evaluate_records([make_record(0, 42, "stop")], backend, EvaluationConfig())["samples"][0]
        self.assertEqual(sample["tool_call_stop_violations"], 1)
        self.assertEqual(sample["raw_assistant_outputs"], [raw, "Final answer: 42"])
        self.assertIn({"role": "tool", "content": "42"}, sample["messages"])

    def test_aggregates_answers_tool_calls_tokens_and_failure_reasons(self):
        from evaluate_model import EvaluationConfig, evaluate_records

        records = [
            make_record(0, 42, "tool-success"),
            make_record(1, 7, "malformed-tool"),
            make_record(2, 9, "wrong-answer"),
        ]

        def finish_after_tool(messages):
            if messages[-1] == {"role": "tool", "content": "42"}:
                return ("最终答案是 42", 3)
            return ("最终答案是 -1", 3)

        backend = ScriptedBackend(
            {
                "tool-success": [
                    (
                        '<tool_call>{"name":"calculator","arguments":{"expression":"6 * 7"}}</tool_call>',
                        4,
                    ),
                    finish_after_tool,
                ],
                "malformed-tool": [("<tool_call>{bad json}</tool_call>", 2)],
                "wrong-answer": [("最终答案是 8", 5)],
            }
        )
        config = EvaluationConfig(
            seed=7,
            batch_size=2,
            limit=None,
            max_new_tokens=64,
            max_assistant_turns=4,
        )

        report = evaluate_records(records, backend, config)

        summary = report["summary"]
        self.assertEqual(summary["dataset_size"], 3)
        self.assertEqual(summary["selected_samples"], 3)
        self.assertEqual(summary["evaluated_samples"], 3)
        self.assertEqual(summary["correct_samples"], 1)
        self.assertEqual(summary["format_pass_samples"], 2)
        self.assertEqual(summary["tool_call_samples"], 2)
        self.assertEqual(summary["tool_attempts"], 2)
        self.assertEqual(summary["tool_successes"], 1)
        self.assertEqual(summary["invalid_calls"], 1)
        self.assertEqual(summary["assistant_tokens"], 14)
        self.assertAlmostEqual(summary["final_answer_accuracy"], 1 / 3)
        self.assertAlmostEqual(summary["format_pass_rate"], 2 / 3)
        self.assertAlmostEqual(summary["tool_call_rate"], 2 / 3)
        self.assertEqual(summary["tool_success_rate"], 0.5)
        self.assertEqual(summary["invalid_call_rate"], 0.5)
        self.assertAlmostEqual(summary["average_assistant_tokens"], 14 / 3)
        self.assertEqual(
            summary["failure_reasons"],
            {
                "invalid_tool_call": 1,
                "missing_final_answer": 1,
                "wrong_final_answer": 1,
            },
        )

    def test_unknown_tool_and_calculator_error_are_invalid_but_loop_can_finish(self):
        from evaluate_model import EvaluationConfig, evaluate_records

        records = [
            make_record(0, 1, "unknown-tool"),
            make_record(1, 2, "calculator-error"),
        ]
        backend = ScriptedBackend(
            {
                "unknown-tool": [
                    (
                        '<tool_call>{"name":"shell","arguments":{"expression":"1"}}</tool_call>',
                        2,
                    ),
                    ("最终答案是 1", 3),
                ],
                "calculator-error": [
                    (
                        '<tool_call>{"name":"calculator","arguments":{"expression":"1 / 0"}}</tool_call>',
                        2,
                    ),
                    ("最终答案是 2", 3),
                ],
            }
        )

        report = evaluate_records(records, backend, EvaluationConfig(batch_size=2))

        self.assertEqual(report["summary"]["correct_samples"], 2)
        self.assertEqual(report["summary"]["tool_attempts"], 2)
        self.assertEqual(report["summary"]["tool_successes"], 0)
        self.assertEqual(report["summary"]["invalid_calls"], 2)
        self.assertEqual(
            report["summary"]["failure_reasons"],
            {"calculator_error": 1, "unknown_tool": 1},
        )

    def test_text_before_invalid_tool_call_is_not_scored_as_final_answer(self):
        from evaluate_model import EvaluationConfig, evaluate_records

        records = [make_record(0, 42, "premature-answer")]
        backend = ScriptedBackend(
            {
                "premature-answer": [
                    ("最终答案是 42<tool_call>{bad json}</tool_call>", 6),
                ]
            }
        )

        report = evaluate_records(records, backend, EvaluationConfig())

        self.assertEqual(report["summary"]["correct_samples"], 0)
        self.assertEqual(report["summary"]["format_pass_samples"], 0)
        self.assertIsNone(report["samples"][0]["predicted_answer"])
        self.assertEqual(report["samples"][0]["final_assistant_text"], "")

    def test_generation_failure_is_reported_for_each_affected_sample(self):
        from evaluate_model import EvaluationConfig, evaluate_records

        class FailingBackend:
            def generate_batch(self, conversations, tools, max_new_tokens):
                del conversations, tools, max_new_tokens
                raise RuntimeError("显存不足")

        records = [make_record(0, 1), make_record(1, 2)]

        report = evaluate_records(records, FailingBackend(), EvaluationConfig(batch_size=2))

        self.assertEqual(report["summary"]["evaluated_samples"], 2)
        self.assertEqual(
            report["summary"]["failure_reasons"],
            {"generation_error": 2, "missing_final_answer": 2},
        )
        self.assertEqual(
            report["samples"][0]["error_details"],
            ["RuntimeError: 显存不足"],
        )


class ReportAndCliTests(unittest.TestCase):
    def test_sampling_configuration_and_cli_preserve_exact_parameters(self):
        """固定种子采样必须在配置与CLI中记录，非法概率或非有限温度被拒绝。"""
        from evaluate_model import EvaluationConfig, build_argument_parser

        config = EvaluationConfig(do_sample=True, temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True)
        self.assertEqual(config.to_dict()["temperature"], 0.6)
        args = build_argument_parser().parse_args(["--model-path", "local", "--do-sample", "--temperature", "0.6", "--top-p", "0.95", "--top-k", "20", "--thinking", "on"])
        self.assertTrue(args.do_sample)
        self.assertEqual(args.thinking, "on")
        self.assertEqual(args.top_k, 20)
        for kwargs in ({"temperature": 0}, {"temperature": float("nan")}, {"top_p": 1.1}, {"top_k": True}, {"enable_thinking": "yes"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EvaluationConfig(**kwargs)

    def test_cli_exposes_controlled_baseline_modes_and_budgets(self):
        """命令行参数必须可传入公平对照的模式、总预算和单工具轮限制。"""
        from evaluate_model import EvaluationConfig, build_argument_parser

        args = build_argument_parser().parse_args(["--model-path", "local-model", "--mode", "plain", "--max-total-new-tokens", "512", "--max-tool-rounds", "1"])
        config = EvaluationConfig(mode=args.mode, max_total_new_tokens=args.max_total_new_tokens, max_tool_rounds=args.max_tool_rounds)
        self.assertEqual(config.mode, "plain")
        self.assertEqual(config.max_total_new_tokens, 512)
        self.assertEqual(config.max_tool_rounds, 1)
        for kwargs in ({"mode": "invalid"}, {"max_total_new_tokens": True}, {"max_tool_rounds": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EvaluationConfig(**kwargs)

    def test_generation_tokenizer_uses_left_padding_and_eos_fallback(self):
        from evaluate_model import configure_generation_tokenizer

        class FakeTokenizer:
            padding_side = "right"
            pad_token_id = None
            pad_token = None
            eos_token = "<eos>"

        tokenizer = FakeTokenizer()

        configure_generation_tokenizer(tokenizer)

        self.assertEqual(tokenizer.padding_side, "left")
        self.assertEqual(tokenizer.pad_token, "<eos>")

    def test_torch_dtype_names_resolve_to_real_torch_dtype_objects(self):
        from evaluate_model import resolve_torch_dtype

        class FakeTorch:
            float16 = object()
            bfloat16 = object()
            float32 = object()

        self.assertEqual(resolve_torch_dtype(FakeTorch, "auto"), "auto")
        self.assertIs(resolve_torch_dtype(FakeTorch, "float16"), FakeTorch.float16)
        self.assertIs(resolve_torch_dtype(FakeTorch, "bfloat16"), FakeTorch.bfloat16)
        self.assertIs(resolve_torch_dtype(FakeTorch, "float32"), FakeTorch.float32)

    def test_json_report_preserves_deterministic_configuration(self):
        from evaluate_model import EvaluationConfig, write_json_report

        report = {
            "config": EvaluationConfig(seed=11, batch_size=3).to_dict(),
            "summary": {"evaluated_samples": 0},
            "samples": [],
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "nested" / "report.json"
            write_json_report(report, output_path)
            loaded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["config"]["seed"], 11)
        self.assertEqual(loaded["config"]["batch_size"], 3)
        self.assertFalse(loaded["config"]["do_sample"])

    def test_cli_uses_final_test_and_requires_model_path(self):
        from evaluate_model import build_argument_parser

        parser = build_argument_parser()
        args = parser.parse_args(["--model-path", "exports/calculator-step-1"])

        self.assertEqual(args.data_file, "data/final_test.parquet")
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.max_assistant_turns, 4)


class HuggingFaceBackendTests(unittest.TestCase):
    def test_real_hf_generate_stops_batch_rows_independently_and_keeps_tool_tag(self):
        """用真实CPU generate验证一行工具停止不截断其它行，EOS和长度结束仍有效。"""
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessor
        from evaluate_model import CALCULATOR_TOOL_SCHEMA, EvaluationConfig, HuggingFaceBackend

        class Schedule(LogitsProcessor):
            def __call__(self, ids, scores):
                step = ids.shape[1] - 1
                scores.fill_(-float("inf"))
                tokens = [4 if step == 0 else 7, 5 if step == 0 else (6 if step == 1 else 9), 5]
                for row, token in enumerate(tokens):
                    scores[row, token] = 0
                return scores

        class Model(GPT2LMHeadModel):
            def generate(self, **kwargs):
                return super().generate(**kwargs, logits_processor=[Schedule()])

        class Tokenizer:
            pad_token_id, eos_token_id = 0, 9
            def apply_chat_template(self, messages, **kwargs):
                return "context"
            def __call__(self, rendered, **kwargs):
                return {"input_ids": torch.ones((3, 1), dtype=torch.long), "attention_mask": torch.ones((3, 1), dtype=torch.long)}
            def encode(self, text, **kwargs):
                return [7]
            def decode(self, ids, **kwargs):
                return "".join({0: "", 4: "a", 5: "b", 6: "c", 7: "</tool_call>", 9: ""}[token] for token in ids)

        model = Model(GPT2Config(vocab_size=12, n_positions=16, n_embd=8, n_layer=1, n_head=1, eos_token_id=9, pad_token_id=0)).eval()
        conversations = [[{"role": "user", "content": "question"}]] * 3
        backend = HuggingFaceBackend(model, Tokenizer(), torch)
        outputs = backend.generate_batch(conversations, [CALCULATOR_TOOL_SCHEMA], 4)
        self.assertEqual([row.finish_reason for row in outputs], ["tool_call", "eos", "length"])
        self.assertEqual([row.token_count for row in outputs], [2, 3, 4])
        self.assertEqual(outputs[0].text, "a</tool_call>")
        backend.evaluation_config = EvaluationConfig(mode="plain")
        plain = backend.generate_batch(conversations, [], 4)
        self.assertEqual(plain[0].token_count, 4)
        self.assertEqual(plain[0].finish_reason, "length")
        self.assertEqual(model.generation_config.eos_token_id, 9)

    def test_tool_stop_rejects_split_or_hidden_closing_tag(self):
        """不能静默使用半个标签停止，也不能把闭合标签从解析文本中移除。"""
        from evaluate_model import HuggingFaceBackend

        class Tokenizer:
            def __init__(self, ids, hidden=False):
                self.ids, self.hidden = ids, hidden
            def encode(self, text, **kwargs):
                return self.ids
            def decode(self, ids, skip_special_tokens):
                return "" if self.hidden and skip_special_tokens else "</tool_call>"

        for ids, hidden in (([7, 8], False), ([], False), ([7], True)):
            with self.subTest(ids=ids, hidden=hidden):
                backend = HuggingFaceBackend(None, Tokenizer(ids, hidden), None)
                with self.assertRaises(ValueError):
                    backend._tool_end_token_id()

    def test_backend_forwards_sampling_and_rejects_prefilled_think_tags(self):
        """采样真实生效；模型自行生成标签，模板预填必须在生成前拒绝。"""
        from types import SimpleNamespace
        from evaluate_model import EvaluationConfig, HuggingFaceBackend

        class Tensor:
            shape = (1, 2)
            def to(self, device):
                return self

        class Sequence(list):
            def __getitem__(self, key):
                value = super().__getitem__(key)
                return Sequence(value) if isinstance(key, slice) else value
            def tolist(self):
                return list(self)

        class Tokenizer:
            eos_token_id, pad_token_id = 9, 0
            prefill = False
            def apply_chat_template(self, messages, **kwargs):
                if kwargs.get("enable_thinking") is not True:
                    raise RuntimeError("未传入原生thinking设置")
                suffix = "<|im_start|>assistant\n" + ("<think>\n" if self.prefill else "")
                return "context" + (suffix if kwargs["add_generation_prompt"] else "")
            def __call__(self, rendered, **kwargs):
                return {"input_ids": Tensor(), "attention_mask": Tensor()}
            def decode(self, ids, **kwargs):
                return "<think>Therefore, 6*7=42.</think>\nFinal answer: 42"

        class Context:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False

        seen = {}
        def generate(**kwargs):
            seen.update(kwargs)
            return [Sequence([1, 2, 3, 9])]
        model = SimpleNamespace(device="cpu", generate=generate)
        config = EvaluationConfig(do_sample=True, temperature=0.6, top_p=0.95, top_k=20, enable_thinking=True)
        backend = HuggingFaceBackend(model, Tokenizer(), SimpleNamespace(inference_mode=Context), evaluation_config=config)
        output = backend.generate_batch([[{"role": "user", "content": "6*7?"}]], [], 512)[0]
        self.assertEqual(output.assistant_prefix, "")
        self.assertTrue(output.text.startswith("<think>"))
        self.assertEqual(output.token_count, 2)
        self.assertEqual((seen["do_sample"], seen["temperature"], seen["top_p"], seen["top_k"]), (True, 0.6, 0.95, 20))
        tokenizer = Tokenizer()
        tokenizer.prefill = True
        backend = HuggingFaceBackend(model, tokenizer, SimpleNamespace(inference_mode=Context), evaluation_config=config)
        with self.assertRaisesRegex(ValueError, "预填"):
            backend.generate_batch([[{"role": "user", "content": "6*7?"}]], [], 512)

    def test_model_eos_list_counts_secondary_end_token(self):
        """模型的第二个 EOS 也应正常结束，不能把 padding 算入实际生成长度。"""
        from types import SimpleNamespace
        from evaluate_model import HuggingFaceBackend

        backend = HuggingFaceBackend(
            SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[9, 10])),
            SimpleNamespace(eos_token_id=9, pad_token_id=0), None,
        )
        self.assertEqual(backend._count_generated_tokens([4, 10, 0, 0]), 2)

    def test_batch_generation_uses_greedy_decoding_and_counts_real_tokens(self):
        from evaluate_model import CALCULATOR_TOOL_SCHEMA, HuggingFaceBackend

        class FakeTensor:
            shape = (2, 3)

            def to(self, device):
                self.device = device
                return self

        class FakeSequence(list):
            def __getitem__(self, item):
                value = super().__getitem__(item)
                return FakeSequence(value) if isinstance(item, slice) else value

            def tolist(self):
                return list(self)

        class FakeTokenizer:
            pad_token_id = 0
            eos_token_id = 9

            def __init__(self):
                self.tool_arguments = []

            def apply_chat_template(self, messages, tools, **kwargs):
                del messages, kwargs
                self.tool_arguments.append(tools)
                return "rendered"

            def __call__(self, rendered, **kwargs):
                del rendered, kwargs
                return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}

            def decode(self, token_ids, **kwargs):
                del kwargs
                if token_ids == [8]:
                    return "</tool_call>"
                return ",".join(str(token_id) for token_id in token_ids)

            def encode(self, text, **kwargs):
                return [8]

        class FakeModel:
            device = "cuda:0"

            def generate(self, **kwargs):
                self.generation_kwargs = kwargs
                return [
                    FakeSequence([10, 11, 12, 4, 9, 0]),
                    FakeSequence([20, 21, 22, 5, 6, 7]),
                ]

        class FakeInferenceMode:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc_value, traceback):
                del exc_type, exc_value, traceback
                return False

        class FakeTorch:
            @staticmethod
            def inference_mode():
                return FakeInferenceMode()

        model = FakeModel()
        tokenizer = FakeTokenizer()
        backend = HuggingFaceBackend(model, tokenizer, FakeTorch)

        outputs = backend.generate_batch(
            [
                [{"role": "user", "content": "问题 1"}],
                [{"role": "user", "content": "问题 2"}],
            ],
            [CALCULATOR_TOOL_SCHEMA],
            max_new_tokens=32,
        )

        self.assertEqual(outputs[0].text, "4,9")
        self.assertEqual(outputs[0].token_count, 2)
        self.assertEqual(outputs[1].text, "5,6,7")
        self.assertEqual(outputs[1].token_count, 3)
        self.assertFalse(model.generation_kwargs["do_sample"])
        self.assertEqual(model.generation_kwargs["max_new_tokens"], 32)
        # 前缀核验会额外渲染无生成提示的模板，两条真正输入仍携带同一工具schema。
        self.assertEqual(tokenizer.tool_arguments[:2], [[CALCULATOR_TOOL_SCHEMA]] * 2)
        self.assertTrue(all(tools == [CALCULATOR_TOOL_SCHEMA] for tools in tokenizer.tool_arguments))


class MergeCheckpointScriptTests(unittest.TestCase):
    def setUp(self):
        # dry-run 只检查入口结构，用临时源码夹具，不依赖开发者的 verl checkout。
        from unittest.mock import patch
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        source = Path(temporary.name) / "verl-source"
        merger = source / "verl/model_merger/__main__.py"
        merger.parent.mkdir(parents=True)
        merger.write_text("# CPU dry-run fixture\n", encoding="utf-8")
        environment = patch.dict(os.environ, {"VERL_SOURCE_DIR": str(source)})
        environment.start()
        self.addCleanup(environment.stop)

    def _create_actor_checkpoint(self, root, world_size=2):
        actor_path = root / "actor checkpoint"
        huggingface_path = actor_path / "huggingface"
        huggingface_path.mkdir(parents=True)
        (actor_path / "fsdp_config.json").write_text(
            json.dumps({"world_size": world_size}),
            encoding="utf-8",
        )
        (huggingface_path / "config.json").write_text("{}", encoding="utf-8")
        for rank in range(world_size):
            (actor_path / f"model_world_size_{world_size}_rank_{rank}.pt").write_bytes(b"test")
        return actor_path

    def test_dry_run_validates_actor_checkpoint_without_creating_target(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            actor_path = self._create_actor_checkpoint(temporary_path)
            target_path = temporary_path / "merged model"

            completed = subprocess.run(
                ["bash", str(MERGE_SCRIPT), "--dry-run", str(actor_path), str(target_path)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("python -m verl.model_merger merge", completed.stdout)
            self.assertIn(str(actor_path), completed.stdout)
            self.assertIn(str(target_path), completed.stdout)
            self.assertFalse(target_path.exists())

    def test_dry_run_reports_missing_actor_checkpoint_clearly(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            missing_path = temporary_path / "missing actor"

            completed = subprocess.run(
                [
                    "bash",
                    str(MERGE_SCRIPT),
                    "--dry-run",
                    str(missing_path),
                    str(temporary_path / "merged"),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("actor checkpoint 不存在", completed.stderr)

    def test_dry_run_rejects_incomplete_actor_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            actor_path = self._create_actor_checkpoint(temporary_path, world_size=2)
            (actor_path / "model_world_size_2_rank_1.pt").unlink()

            completed = subprocess.run(
                [
                    "bash",
                    str(MERGE_SCRIPT),
                    "--dry-run",
                    str(actor_path),
                    str(temporary_path / "merged"),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("缺少 FSDP 分片", completed.stderr)

    def test_dry_run_refuses_nonempty_export_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            actor_path = self._create_actor_checkpoint(temporary_path)
            target_path = temporary_path / "merged"
            target_path.mkdir()
            (target_path / "existing.json").write_text("{}", encoding="utf-8")

            completed = subprocess.run(
                ["bash", str(MERGE_SCRIPT), "--dry-run", str(actor_path), str(target_path)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("目标目录非空", completed.stderr)

    def test_dry_run_uses_configured_python_without_uv(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            actor_path = self._create_actor_checkpoint(temporary_path)
            environment = os.environ.copy()
            environment.update(
                {
                    "VERL_USE_UV": "0",
                    "PYTHON_BIN": "/opt/calculator-python",
                }
            )

            completed = subprocess.run(
                [
                    "bash",
                    str(MERGE_SCRIPT),
                    "--dry-run",
                    str(actor_path),
                    str(temporary_path / "merged"),
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("/opt/calculator-python -m verl.model_merger merge", completed.stdout)


if __name__ == "__main__":
    unittest.main()
