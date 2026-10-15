import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path


# 按文件路径加载模块，确保测试不依赖当前工作目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "prepare_data.py"


def load_prepare_data():
    """加载待测数据准备模块"""
    spec = importlib.util.spec_from_file_location("prepare_data_stage3", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AnswerExtractionTests(unittest.TestCase):
    def setUp(self):
        self.prepare_data = load_prepare_data()

    def test_extracts_and_normalizes_supported_gsm8k_answers(self):
        cases = {
            "work\n#### 42": "42",
            "work\n#### -17": "-17",
            "work\n#### +3.50": "3.5",
            "work\n#### 1,234.00": "1234",
        }

        for answer_text, expected in cases.items():
            with self.subTest(answer_text=answer_text):
                self.assertEqual(
                    self.prepare_data.extract_gsm8k_answer(answer_text),
                    expected,
                )

    def test_rejects_missing_or_malformed_final_answer(self):
        for answer_text in ("no marker", "#### NaN", "#### 12 apples"):
            with self.subTest(answer_text=answer_text):
                with self.assertRaises(ValueError):
                    self.prepare_data.extract_gsm8k_answer(answer_text)


class RecordConversionTests(unittest.TestCase):
    def setUp(self):
        self.prepare_data = load_prepare_data()

    def test_prompt_example_call_produces_demonstrated_final_answer(self):
        """示例调用必须可执行，工具结果和最终答案不得矛盾或污染当前题目。"""
        from calculator_tool import CalculatorTool

        record = self.prepare_data.convert_example(
            {"question": "9 / 3?", "answer": "work\n#### 3"},
            split="train", index=0,
        )
        prompt = record["prompt"][0]["content"]
        calls = re.findall(r"<tool_call>(.*?)</tool_call>", prompt, re.DOTALL)
        self.assertEqual(len(calls), 1)
        call = json.loads(calls[0])
        self.assertEqual(call["name"], "calculator")
        result = CalculatorTool().execute(call["arguments"]["expression"])
        self.assertEqual(result, "327")
        self.assertIn("Tool: 327", prompt)
        self.assertRegex(prompt, r"<think>[^<]+</think>")
        self.assertIn("Final answer: 327", prompt)
        self.assertEqual(record["reward_model"]["ground_truth"], "3")
        self.assertEqual(record["prompt"][1]["content"], "9 / 3?")

    def test_generated_prompt_states_final_answer_and_calculator_contract(self):
        """新数据缺少输出协议时，真实 rollout 的正确答案也可能被评分器漏掉。"""
        record = self.prepare_data.convert_example(
            {"question": "How much is 8 minus 3?", "answer": "work\n#### 5"},
            split="train", index=0,
        )
        prompt = record["prompt"][0]["content"]
        self.assertIn("Final answer: <number>", prompt)
        self.assertIn("<think>...</think>", prompt)
        self.assertIn("cannot solve equations or systems of equations", prompt)
        self.assertIn("variables, equals signs, or code", prompt)
        # 实际记录只携带英文提示；中文副本仅供维护，不能混入模型输入。
        self.assertTrue(prompt.isascii())
        self.assertEqual(prompt, self.prepare_data.SYSTEM_PROMPT_EN)
        self.assertIn("调用轮", self.prepare_data.SYSTEM_PROMPT_ZH)
        self.assertEqual(record["prompt"][1]["content"], "How much is 8 minus 3?")
        self.assertEqual(record["reward_model"]["ground_truth"], "5")

    def test_convert_example_builds_verl_v091_record(self):
        convert_example = getattr(self.prepare_data, "convert_example", None)
        self.assertIsNotNone(convert_example, "convert_example 尚未实现")

        record = convert_example(
            {
                "question": "A box has 1,234 balls. How many balls?",
                "answer": "reasoning\n#### 1,234",
            },
            split="validation",
            index=7,
        )

        self.assertEqual(
            set(record),
            {"data_source", "prompt", "ability", "reward_model", "extra_info"},
        )
        self.assertEqual(record["data_source"], "openai/gsm8k")
        self.assertEqual(record["ability"], "math")
        self.assertEqual(record["reward_model"], {"style": "rule", "ground_truth": "1234"})
        self.assertEqual([message["role"] for message in record["prompt"]], ["system", "user"])
        self.assertEqual(record["prompt"][1]["content"], "A box has 1,234 balls. How many balls?")
        self.assertEqual(
            record["extra_info"],
            {
                "split": "validation",
                "original_split": "train",
                "index": 7,
                "answer": "reasoning\n#### 1,234",
                "question": "A box has 1,234 balls. How many balls?",
            },
        )


class DatasetSplitTests(unittest.TestCase):
    def setUp(self):
        self.prepare_data = load_prepare_data()

    def test_split_is_deterministic_complete_and_disjoint(self):
        split_training_examples = getattr(self.prepare_data, "split_training_examples", None)
        self.assertIsNotNone(split_training_examples, "split_training_examples 尚未实现")
        examples = [{"question": f"q{index}", "answer": f"#### {index}"} for index in range(8)]

        first = split_training_examples(examples, validation_size=2, seed=17)
        second = split_training_examples(examples, validation_size=2, seed=17)

        self.assertEqual(first, second)
        train_rows, validation_rows = first
        train_questions = {row["question"] for row in train_rows}
        validation_questions = {row["question"] for row in validation_rows}
        self.assertEqual(len(train_rows), 6)
        self.assertEqual(len(validation_rows), 2)
        self.assertFalse(train_questions & validation_questions)
        self.assertEqual(train_questions | validation_questions, {f"q{index}" for index in range(8)})

    def test_split_rejects_validation_size_outside_training_range(self):
        split_training_examples = getattr(self.prepare_data, "split_training_examples", None)
        self.assertIsNotNone(split_training_examples, "split_training_examples 尚未实现")
        examples = [{"question": "q0", "answer": "#### 0"}]

        for validation_size in (0, 1, 2):
            with self.subTest(validation_size=validation_size):
                with self.assertRaises(ValueError):
                    split_training_examples(examples, validation_size=validation_size, seed=42)


class DatasetPreparationTests(unittest.TestCase):
    def test_refresh_existing_data_only_changes_system_prompt(self):
        """离线刷新不得改动题目、答案、split、索引或传入的原始记录。"""
        import copy

        prepare_data = load_prepare_data()
        refresh = getattr(prepare_data, "refresh_dataset_prompts", None)
        self.assertIsNotNone(refresh, "缺少无下载的已有 Parquet 提示刷新入口")
        source = prepare_data.convert_example(
            {"question": "6 / 2?", "answer": "reasoning\n#### 3"}, split="train", index=4,
        )
        source["prompt"][0]["content"] = "old system prompt"
        untouched = copy.deepcopy(source)
        written = {}
        with tempfile.TemporaryDirectory() as directory:
            counts = refresh(
                Path(directory) / "old", Path(directory) / "new",
                read_parquet_fn=lambda path: [source],
                write_parquet_fn=lambda records, path: written.update({path.name: records}),
            )
        self.assertEqual(counts, {"train": 1, "validation": 1, "final_test": 1})
        self.assertEqual(source, untouched)
        for records in written.values():
            changed = copy.deepcopy(records[0])
            self.assertIn("Final answer: <number>", changed["prompt"][0]["content"])
            changed["prompt"][0]["content"] = "old system prompt"
            self.assertEqual(changed, untouched)

    def test_prompt_refresh_refuses_in_place_or_existing_outputs(self):
        """提示刷新必须另存，不得覆盖已验收的数据或以前的实验。"""
        refresh = getattr(load_prepare_data(), "refresh_dataset_prompts", None)
        self.assertIsNotNone(refresh, "refresh_dataset_prompts 尚未实现")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                refresh(root, root)
            target = root / "new"
            target.mkdir()
            (target / "validation.parquet").write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                refresh(root / "old", target)
            self.assertEqual((target / "validation.parquet").read_bytes(), b"existing")

    def test_pipeline_writes_train_validation_and_final_test_separately(self):
        prepare_data = load_prepare_data()
        prepare_gsm8k_dataset = getattr(prepare_data, "prepare_gsm8k_dataset", None)
        self.assertIsNotNone(prepare_gsm8k_dataset, "prepare_gsm8k_dataset 尚未实现")
        source_dataset = {
            "train": [
                {"question": f"train-{index}", "answer": f"#### {index}"}
                for index in range(4)
            ],
            "test": [
                {"question": f"test-{index}", "answer": f"#### {index + 10}"}
                for index in range(2)
            ],
        }
        written = {}

        def fake_loader(name, configuration):
            self.assertEqual((name, configuration), ("openai/gsm8k", "main"))
            return source_dataset

        def fake_writer(records, path):
            written[Path(path).name] = list(records)

        with tempfile.TemporaryDirectory() as temporary_directory:
            counts = prepare_gsm8k_dataset(
                output_dir=temporary_directory,
                validation_size=1,
                seed=23,
                load_dataset_fn=fake_loader,
                write_parquet_fn=fake_writer,
            )

        self.assertEqual(counts, {"train": 3, "validation": 1, "final_test": 2})
        self.assertEqual(
            set(written),
            {"train.parquet", "validation.parquet", "final_test.parquet"},
        )
        for split_name, filename in (
            ("train", "train.parquet"),
            ("validation", "validation.parquet"),
            ("final_test", "final_test.parquet"),
        ):
            self.assertTrue(written[filename])
            self.assertTrue(
                all(record["extra_info"]["split"] == split_name for record in written[filename])
            )
        self.assertTrue(
            all(
                record["extra_info"]["original_split"] == "test"
                for record in written["final_test.parquet"]
            )
        )


if __name__ == "__main__":
    unittest.main()
