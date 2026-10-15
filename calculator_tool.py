"""计算器工具 —— 供 Agent 在 RL 训练中调用"""
import ast
import math
import operator
from dataclasses import dataclass

# 允许的运算符白名单（安全过滤）
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


@dataclass(frozen=True)
class CalculatorLimits:
    """计算器的资源与数值边界"""

    max_expression_length: int = 256
    max_ast_nodes: int = 64
    max_ast_depth: int = 16
    max_integer_digits: int = 100
    max_exponent_abs: int = 12
    max_abs_result: int | float = 1e100
    max_output_length: int = 128


def _ast_depth(node: ast.AST) -> int:
    """计算 AST 的最大嵌套深度"""
    child_depths = [_ast_depth(child) for child in ast.iter_child_nodes(node)]
    return 1 + max(child_depths, default=0)


def _validate_number(value: object, limits: CalculatorLimits) -> int | float:
    """确保中间值为有限实数，并处于允许范围内"""
    if type(value) not in (int, float):
        raise ValueError("仅支持整数和有限浮点数")

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("不允许非有限浮点数")

    if isinstance(value, int) and len(str(abs(value))) > limits.max_integer_digits:
        raise ValueError("整数位数超出限制")

    if abs(value) > limits.max_abs_result:
        raise ValueError("结果绝对值超出限制")

    return value


def safe_eval(
    expression: str,
    limits: CalculatorLimits | None = None,
) -> int | float:
    """安全地计算数学表达式（不使用 eval，防止代码注入）"""
    active_limits = limits or CalculatorLimits()
    if not isinstance(expression, str):
        raise ValueError("表达式必须是字符串")
    if not expression.strip():
        raise ValueError("表达式不能为空")
    if len(expression) > active_limits.max_expression_length:
        raise ValueError("表达式长度超出限制")

    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > active_limits.max_ast_nodes:
        raise ValueError("表达式节点数超出限制")
    if _ast_depth(tree.body) > active_limits.max_ast_depth:
        raise ValueError("表达式嵌套深度超出限制")

    def _eval_node(node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant):
            return _validate_number(node.value, active_limits)

        if isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            op_type = type(node.op)
            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"不允许的运算符: {op_type.__name__}")
            if op_type is ast.Pow and abs(right) > active_limits.max_exponent_abs:
                raise ValueError("指数绝对值超出限制")
            result = ALLOWED_OPERATORS[op_type](left, right)
            return _validate_number(result, active_limits)

        if isinstance(node, ast.UnaryOp):
            operand = _eval_node(node.operand)
            op_type = type(node.op)
            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"不允许的运算符: {op_type.__name__}")
            result = ALLOWED_OPERATORS[op_type](operand)
            return _validate_number(result, active_limits)

        raise ValueError(f"不支持的表达式节点: {type(node).__name__}")

    return _eval_node(tree.body)


class CalculatorTool:
    """计算器工具"""

    name = "calculator"
    description = "计算数学表达式。输入一个数学表达式字符串，返回计算结果。"

    def __init__(self, limits: CalculatorLimits | None = None):
        self.limits = limits or CalculatorLimits()

    def execute(self, expression: str) -> str:
        """执行计算

        Args:
            expression: 数学表达式，如 "234 * 567 + 89"

        Returns:
            计算结果字符串，或错误信息
        """
        try:
            if not isinstance(expression, str):
                raise ValueError("表达式必须是字符串")
            if len(expression) > self.limits.max_expression_length:
                raise ValueError("表达式长度超出限制")

            # 清理输入：去除空格、千位逗号等
            cleaned = expression.replace(",", "").replace(" ", "")
            result = safe_eval(cleaned, self.limits)

            # 整数结果不保留小数点，其他结果最多保留六位小数
            if isinstance(result, int) or result.is_integer():
                output = str(int(result))
            else:
                output = str(round(result, 6))

            if len(output) > self.limits.max_output_length:
                raise ValueError("输出长度超出限制")
            return output
        except SyntaxError:
            return "计算错误: 表达式语法无效"
        except ZeroDivisionError:
            return "计算错误: 除数不能为零"
        except OverflowError:
            return "计算错误: 数值超出范围"
        except ValueError as error:
            return f"计算错误: {error}"


def calculator(expression: str) -> str:
    """提供给后续 verl 工具注册使用的稳定函数接口"""
    return CalculatorTool().execute(expression)


# 本地手工检查入口
if __name__ == "__main__":
    calc = CalculatorTool()
    print(calc.execute("234 * 567 + 89"))        # 132767
    print(calc.execute("(123 + 456) * 789"))     # 456831
    print(calc.execute("100 / 3"))               # 33.333333
    print(calc.execute("__import__('os')"))      # 计算错误（安全拦截）
