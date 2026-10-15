"""将纯计算器函数注册为 verl FunctionTool"""

from calculator_tool import calculator as evaluate_expression
from verl.tools.function_tool import function_tool


@function_tool("calculator")
def calculator(expression: str) -> tuple[str, float, dict[str, float]]:
    """计算数学表达式并返回结果字符串。

    Args:
        expression: 待计算的数学表达式，例如 "234 * 567 + 89"。

    Returns:
        计算结果、真实调用奖励和成功指标；输入无效时奖励为零。
    """
    result = evaluate_expression(expression)
    is_success = not result.startswith("计算错误:")
    reward = 0.1 if is_success else 0.0
    metrics = {"tool_success": 1.0 if is_success else 0.0}
    return result, reward, metrics
