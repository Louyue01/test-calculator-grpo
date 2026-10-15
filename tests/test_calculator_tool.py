import unittest

import calculator_tool


def make_limits(test_case, **overrides):
    """构造限制配置；接口缺失时生成清晰的测试失败"""
    limits_class = getattr(calculator_tool, "CalculatorLimits", None)
    test_case.assertIsNotNone(limits_class, "CalculatorLimits 尚未实现")
    return limits_class(**overrides)


class SafeEvalTests(unittest.TestCase):
    def test_evaluates_supported_real_number_arithmetic(self):
        cases = {
            "2 + 3 * 4": 14,
            "(8 - 3) // 2": 2,
            "17 % 5": 2,
            "-2.5 + +1": -1.5,
            "2 ** 8": 256,
        }

        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                self.assertEqual(calculator_tool.safe_eval(expression), expected)

    def test_rejects_non_real_numeric_constants(self):
        for expression in ("'x'", "True", "1j", "None"):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    calculator_tool.safe_eval(expression)

    def test_rejects_unsupported_ast_nodes(self):
        for expression in ("abs(1)", "value + 1", "[1][0]", "(1).__class__"):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    calculator_tool.safe_eval(expression)

    def test_rejects_expression_over_character_limit(self):
        limits = make_limits(self, max_expression_length=5)

        with self.assertRaises(ValueError):
            calculator_tool.safe_eval("1 + 234", limits)

    def test_rejects_expression_over_node_limit(self):
        limits = make_limits(self, max_ast_nodes=5)

        with self.assertRaises(ValueError):
            calculator_tool.safe_eval("1 + 2 + 3", limits)

    def test_rejects_expression_over_depth_limit(self):
        limits = make_limits(self, max_ast_depth=3)

        with self.assertRaises(ValueError):
            calculator_tool.safe_eval("-(-(-1))", limits)

    def test_rejects_integer_over_digit_limit(self):
        limits = make_limits(self, max_integer_digits=4)

        with self.assertRaises(ValueError):
            calculator_tool.safe_eval("12345", limits)

    def test_rejects_exponent_outside_limit(self):
        limits = make_limits(self, max_exponent_abs=10)

        for expression in ("2 ** 11", "2 ** -11"):
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    calculator_tool.safe_eval(expression, limits)

    def test_rejects_result_outside_magnitude_limit(self):
        limits = make_limits(self, max_abs_result=100)

        with self.assertRaises(ValueError):
            calculator_tool.safe_eval("10 * 11", limits)

    def test_rejects_non_finite_float(self):
        with self.assertRaises(ValueError):
            calculator_tool.safe_eval("1e309")


class CalculatorInterfaceTests(unittest.TestCase):
    def test_formats_supported_results(self):
        calculator = calculator_tool.CalculatorTool()

        self.assertEqual(calculator.execute("1,234 + 6"), "1240")
        self.assertEqual(calculator.execute("100 / 3"), "33.333333")

    def test_returns_stable_errors_for_syntax_and_division(self):
        calculator = calculator_tool.CalculatorTool()

        self.assertEqual(calculator.execute("1 +"), "计算错误: 表达式语法无效")
        self.assertEqual(calculator.execute("1 / 0"), "计算错误: 除数不能为零")

    def test_rejects_string_multiplication_without_building_output(self):
        result = calculator_tool.CalculatorTool().execute("'x' * 3")

        self.assertTrue(result.startswith("计算错误:"), result)

    def test_rejects_long_raw_input_before_whitespace_cleanup(self):
        limits = make_limits(self, max_expression_length=5)
        calculator = calculator_tool.CalculatorTool(limits=limits)

        self.assertEqual(
            calculator.execute("      1"),
            "计算错误: 表达式长度超出限制",
        )

    def test_enforces_output_length_limit(self):
        limits = make_limits(self, max_output_length=3)
        calculator = calculator_tool.CalculatorTool(limits=limits)

        self.assertEqual(calculator.execute("1234"), "计算错误: 输出长度超出限制")

    def test_function_interface_matches_tool_interface(self):
        calculator = getattr(calculator_tool, "calculator", None)
        self.assertIsNotNone(calculator, "calculator 函数接口尚未实现")

        self.assertEqual(calculator("6 * 7"), "42")


if __name__ == "__main__":
    unittest.main()
