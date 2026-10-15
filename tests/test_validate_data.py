import importlib.util
import contextlib
import inspect
import io
import unittest
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "validate_data.py"


def load_validator(test_case):
    """加载验证器；文件缺失时生成清晰的测试失败"""
    test_case.assertTrue(MODULE_PATH.is_file(), "scripts/validate_data.py 尚未实现")
    spec = importlib.util.spec_from_file_location("validate_data_stage3", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_record(question, split, index=0):
    """构造完整的 verl v0.9.1 测试记录"""
    original_split = "test" if split == "final_test" else "train"
    return {
        "data_source": "openai/gsm8k",
        "prompt": [
            {"role": "system", "content": "你是一个数学助手。"},
            {"role": "user", "content": question},
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": "42"},
        "extra_info": {
            "split": split,
            "original_split": original_split,
            "index": index,
            "answer": "reasoning\n#### 42",
            "question": question,
        },
    }


def make_valid_splits():
    return {
        "train": [make_record("train question", "train")],
        "validation": [make_record("validation question", "validation")],
        "final_test": [make_record("test question", "final_test")],
    }


class DataValidationTests(unittest.TestCase):
    def test_accepts_complete_disjoint_verl_records(self):
        validator = load_validator(self)

        errors = validator.validate_splits(make_valid_splits())

        self.assertEqual(errors, [])

    def test_reports_missing_required_field_with_location(self):
        validator = load_validator(self)
        splits = make_valid_splits()
        del splits["train"][0]["reward_model"]

        errors = validator.validate_splits(splits)

        self.assertTrue(any("train[0].reward_model" in error for error in errors), errors)

    def test_reports_duplicate_question_across_splits(self):
        validator = load_validator(self)
        splits = make_valid_splits()
        duplicate = deepcopy(splits["train"][0])
        duplicate["extra_info"]["split"] = "validation"
        splits["validation"] = [duplicate]

        errors = validator.validate_splits(splits)

        self.assertTrue(any("split 泄漏" in error for error in errors), errors)

    def test_reports_empty_split_and_metadata_mismatch(self):
        validator = load_validator(self)
        splits = make_valid_splits()
        splits["validation"] = []
        splits["final_test"][0]["extra_info"]["split"] = "test"

        errors = validator.validate_splits(splits)

        self.assertTrue(any("validation 为空" in error for error in errors), errors)
        self.assertTrue(any("final_test[0].extra_info.split" in error for error in errors), errors)

    def test_reports_prompt_over_character_limit(self):
        validator = load_validator(self)
        splits = make_valid_splits()
        splits["train"][0]["prompt"][1]["content"] = "x" * 11

        errors = validator.validate_splits(splits, max_prompt_characters=10)

        self.assertTrue(any("prompt 长度" in error for error in errors), errors)

    def test_reports_prompt_and_ground_truth_mismatch(self):
        validator = load_validator(self)
        splits = make_valid_splits()
        splits["train"][0]["prompt"][1]["content"] = "different question"
        splits["train"][0]["reward_model"]["ground_truth"] = "41"

        errors = validator.validate_splits(splits)

        self.assertTrue(any("prompt 与 extra_info.question 不一致" in error for error in errors), errors)
        self.assertTrue(any("ground_truth 与原始答案不一致" in error for error in errors), errors)

    def test_reports_duplicate_index_within_split(self):
        validator = load_validator(self)
        splits = make_valid_splits()
        splits["train"].append(make_record("another train question", "train", index=0))

        errors = validator.validate_splits(splits)

        self.assertTrue(any("extra_info.index 重复" in error for error in errors), errors)


class ValidatorCliTests(unittest.TestCase):
    def test_main_returns_zero_for_valid_files(self):
        validator = load_validator(self)
        self.assertIn(
            "read_parquet_fn",
            inspect.signature(validator.main).parameters,
            "main 尚未提供可注入的 Parquet 读取器",
        )
        splits = make_valid_splits()

        def fake_reader(path):
            return splits[Path(path).stem]

        standard_output = io.StringIO()
        with contextlib.redirect_stdout(standard_output):
            return_code = validator.main(
                ["--data-dir", "unused"],
                read_parquet_fn=fake_reader,
            )

        self.assertEqual(return_code, 0)
        self.assertIn("数据验证通过", standard_output.getvalue())

    def test_main_returns_one_and_reports_invalid_file(self):
        validator = load_validator(self)
        self.assertIn(
            "read_parquet_fn",
            inspect.signature(validator.main).parameters,
            "main 尚未提供可注入的 Parquet 读取器",
        )
        splits = make_valid_splits()
        del splits["validation"][0]["ability"]

        def fake_reader(path):
            return splits[Path(path).stem]

        standard_error = io.StringIO()
        with contextlib.redirect_stderr(standard_error):
            return_code = validator.main(
                ["--data-dir", "unused"],
                read_parquet_fn=fake_reader,
            )

        self.assertEqual(return_code, 1)
        self.assertIn("validation[0].ability", standard_error.getvalue())


if __name__ == "__main__":
    unittest.main()
