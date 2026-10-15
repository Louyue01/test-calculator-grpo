import math
import unittest
from pathlib import Path
from unittest.mock import patch

import reward_function


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "calculator_grpo_vast.yaml"


def require_reward_api(test_case):
    """取得新版奖励接口；缺失时给出清晰的 RED 失败"""
    compute_score = getattr(reward_function, "compute_score", None)
    test_case.assertIsNotNone(compute_score, "compute_score 尚未实现")
    return compute_score


class ExtractAnswerTests(unittest.TestCase):
    def test_extracts_and_normalizes_supported_final_answer_formats(self):
        cases = {
            r"推理过程。\boxed{1,234.00}": "1234",
            "计算完成，最终答案是 -42.500": "-42.5",
            "Therefore, answer is 7.0": "7",
            "计算完成\n18": "18",
        }

        for solution, expected in cases.items():
            with self.subTest(solution=solution):
                self.assertEqual(reward_function.extract_answer(solution), expected)

    def test_uses_explicit_final_answer_instead_of_other_numbers(self):
        solution = "先计算 20 - 3 = 17，再检查 2 个步骤。\n最终答案：17"

        self.assertEqual(reward_function.extract_answer(solution), "17")

    def test_later_bare_final_answer_overrides_labeled_reasoning_answer(self):
        solution = "<think>试算时曾写答案是 42。</think>\n43"

        self.assertEqual(reward_function.extract_answer(solution), "43")

    def test_does_not_treat_arbitrary_last_reasoning_number_as_final_answer(self):
        self.assertIsNone(reward_function.extract_answer("先计算 20 - 3 = 17"))

    def test_rejects_non_string_solution(self):
        self.assertIsNone(reward_function.extract_answer(None))


class ComputeScoreTests(unittest.TestCase):
    def test_combines_correct_format_real_tool_and_substantive_think_rewards(self):
        compute_score = require_reward_api(self)
        solution = "<think>先计算 6 * 7 = 42，所以结果为 42。</think>\n最终答案是 42"

        result = compute_score(
            data_source="openai/gsm8k",
            solution_str=solution,
            ground_truth="42",
            extra_info={"tool_rewards": [0.1]},
        )

        self.assertEqual(
            result,
            {
                "score": 1.25,
                "accuracy_reward": 1.0,
                "format_reward": 0.1,
                "tool_reward": 0.1,
                "think_reward": 0.05,
                "answer_extracted": 1.0,
                "ground_truth_valid": 1.0,
            },
        )

    def test_correct_bare_answer_receives_only_accuracy_reward(self):
        compute_score = require_reward_api(self)

        result = compute_score("openai/gsm8k", "42", "42")

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["accuracy_reward"], 1.0)
        self.assertEqual(result["format_reward"], 0.0)
        self.assertEqual(result["tool_reward"], 0.0)
        self.assertEqual(result["think_reward"], 0.0)

    def test_wrong_formatted_answer_cannot_claim_tool_or_think_reward(self):
        compute_score = require_reward_api(self)
        solution = "<think>先计算 6 * 8 = 48，所以结果为 48。</think>\n最终答案是 48"

        result = compute_score(
            "openai/gsm8k",
            solution,
            "42",
            {"tool_rewards": [0.1, 0.1]},
        )

        self.assertEqual(result["score"], 0.1)
        self.assertEqual(result["accuracy_reward"], 0.0)
        self.assertEqual(result["format_reward"], 0.1)
        self.assertEqual(result["tool_reward"], 0.0)
        self.assertEqual(result["think_reward"], 0.0)

    def test_spoofed_tool_and_empty_think_text_receive_no_auxiliary_reward(self):
        compute_score = require_reward_api(self)
        solution = "<think></think> calculator('6 * 7')\n最终答案是 42"

        result = compute_score("openai/gsm8k", solution, "42", extra_info={})

        self.assertEqual(result["score"], 1.1)
        self.assertEqual(result["tool_reward"], 0.0)
        self.assertEqual(result["think_reward"], 0.0)

    def test_multiple_successful_tool_calls_are_capped_at_one_bonus(self):
        compute_score = require_reward_api(self)

        result = compute_score(
            "openai/gsm8k",
            "最终答案是 42",
            "42",
            extra_info={"tool_rewards": [0.1, 0.1, 0.1]},
        )

        self.assertEqual(result["tool_reward"], 0.1)
        self.assertEqual(result["score"], 1.2)

    def test_invalid_inputs_return_finite_zero_score_components(self):
        compute_score = require_reward_api(self)
        cases = [
            {"solution_str": None, "ground_truth": "42", "extra_info": None},
            {"solution_str": "最终答案是 42", "ground_truth": "not-a-number", "extra_info": {}},
            {"solution_str": "最终答案是 42", "ground_truth": "NaN", "extra_info": []},
        ]

        for case in cases:
            with self.subTest(case=case):
                result = compute_score(
                    data_source="openai/gsm8k",
                    **case,
                )
                self.assertEqual(result["score"], 0.0)
                self.assertTrue(all(type(value) is float for value in result.values()))
                self.assertTrue(all(math.isfinite(value) for value in result.values()))

    def test_legacy_batch_interface_uses_new_antigaming_scores(self):
        require_reward_api(self)
        completions = [
            "calculator('6 * 7')\n最终答案是 42",
            "<think>先计算 6 * 8 = 48，所以得到 48。</think>\n最终答案是 48",
        ]

        rewards = reward_function.math_agent_reward(
            completions,
            ["42", "42"],
            extra_infos=[{"tool_rewards": []}, {"tool_rewards": [0.1]}],
        )

        self.assertEqual(rewards, [1.1, 0.1])


class RewardConfigTests(unittest.TestCase):
    def test_config_selects_custom_score_and_official_tool_reward_channel(self):
        try:
            from omegaconf import OmegaConf
        except ImportError:
            self.skipTest("未安装 OmegaConf，跳过奖励配置集成测试")

        # 训练从 verl checkout 启动，因此路径必须解析回当前项目而非依赖工作目录。
        with patch.dict("os.environ", {"CALCULATOR_AGENT_ROOT": str(PROJECT_ROOT)}):
            config = OmegaConf.load(CONFIG_PATH)
            self.assertEqual(
                config.reward.custom_reward_function.path,
                str(PROJECT_ROOT / "reward_function.py"),
            )
            self.assertEqual(config.reward.custom_reward_function.name, "compute_score")
            self.assertEqual(config.reward.reward_manager.source, "register")
            self.assertEqual(config.reward.reward_manager.name, "naive")


if __name__ == "__main__":
    unittest.main()
