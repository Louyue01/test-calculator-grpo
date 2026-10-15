import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException
from yaml import YAMLError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "calculator_grpo_vast.yaml"
# verl 将后端专用参数定义为空映射；其内部键由对应推理后端继续校验。
OPEN_MAPPING_PATHS = {"actor_rollout_ref.rollout.engine_kwargs"}


class ConfigValidationError(ValueError):
    """表示训练配置无法安全交给 verl 启动"""


@contextmanager
def temporary_environment(values: dict[str, str]) -> Iterator[None]:
    """临时设置 Hydra 插值需要的环境变量，并在退出时恢复"""
    original = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _official_config_dir(verl_source_dir: Path) -> Path:
    """返回固定版本 verl 的训练配置目录"""
    config_dir = verl_source_dir / "verl" / "trainer" / "config"
    if not (config_dir / "ppo_trainer.yaml").is_file():
        raise ConfigValidationError(f"找不到 verl ppo_trainer 配置: {config_dir}")
    return config_dir


def _validate_known_keys(project_node, official_node, prefix: str = "") -> None:
    """递归拒绝项目覆盖文件中不存在于上游配置的键"""
    if not isinstance(project_node, DictConfig):
        return
    if prefix.rstrip(".") in OPEN_MAPPING_PATHS:
        return
    if not isinstance(official_node, DictConfig):
        raise ConfigValidationError(f"配置结构与 verl 不一致: {prefix.rstrip('.')}")

    for key in project_node.keys():
        if not prefix and key in {"defaults", "hydra"}:
            continue
        path = f"{prefix}{key}"
        # OmegaConf 对值为 ??? 的必填键执行成员测试会返回 False，因此检查真实键集合。
        if key not in official_node.keys():
            raise ConfigValidationError(f"未知配置键: {path}")
        child = project_node._get_node(key)
        if isinstance(child, DictConfig):
            _validate_known_keys(child, official_node._get_node(key), f"{path}.")


def _compose_official_config(config_dir: Path) -> DictConfig:
    """组合 verl 官方基础配置，用作严格键空间"""
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="ppo_trainer")


def _validate_overrides(overrides: Sequence[str]) -> None:
    """只允许覆盖已有配置值，禁止通过 Hydra 语法改变配置结构"""
    for override in overrides:
        if override.startswith(("+", "~")):
            raise ConfigValidationError(f"命令行覆盖不得新增或删除配置键: {override}")


def _compose_project_config(
    config_path: Path,
    environment: dict[str, str],
    overrides: Sequence[str],
) -> DictConfig:
    """组合项目配置及其上游默认项"""
    with temporary_environment(environment):
        with initialize_config_dir(version_base=None, config_dir=str(config_path.parent)):
            config = compose(config_name=config_path.stem, overrides=list(overrides))
        OmegaConf.resolve(config)
    return config


def _as_path_list(value, field_name: str) -> list[Path]:
    """将 Hydra 文件字段规范化为路径列表"""
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, ListConfig)):
        values = list(value)
    else:
        raise ConfigValidationError(f"{field_name} 必须是路径或路径列表")
    if not values:
        raise ConfigValidationError(f"{field_name} 不能为空")
    return [Path(str(item)).expanduser().resolve() for item in values]


def _require_file(path: Path, field_name: str) -> None:
    """确保训练入口依赖的文件存在"""
    if not path.is_file():
        raise ConfigValidationError(f"{field_name} 文件不存在: {path}")


def _validate_runtime_contract(config: DictConfig, project_root: Path) -> None:
    """验证 Calculator Agent 的 GRPO、数据和单卡 Canary 约束"""
    if config.algorithm.adv_estimator != "grpo":
        raise ConfigValidationError("algorithm.adv_estimator 必须为 grpo")
    if config.algorithm.use_kl_in_reward:
        raise ConfigValidationError("KL 不得混入规则奖励")
    if not config.actor_rollout_ref.actor.use_kl_loss:
        raise ConfigValidationError("actor KL loss 必须启用")
    if config.actor_rollout_ref.rollout.name != "sglang":
        raise ConfigValidationError("多轮工具 rollout 必须使用 sglang")
    if config.actor_rollout_ref.rollout.mode != "async":
        raise ConfigValidationError("ToolAgentLoop rollout 必须使用 async 模式")
    if not config.actor_rollout_ref.rollout.multi_turn.enable:
        raise ConfigValidationError("multi_turn 必须启用")
    if config.actor_rollout_ref.rollout.n < 2:
        raise ConfigValidationError("GRPO 的 rollout.n 必须至少为 2")

    train_batch_size = int(config.data.train_batch_size)
    mini_batch_size = int(config.actor_rollout_ref.actor.ppo_mini_batch_size)
    micro_batch_size = int(config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu)
    rollout_n = int(config.actor_rollout_ref.rollout.n)
    if min(train_batch_size, mini_batch_size, micro_batch_size) <= 0:
        raise ConfigValidationError("batch 和 micro-batch 必须为正整数")
    if train_batch_size % mini_batch_size != 0:
        raise ConfigValidationError("train_batch_size 必须能被 ppo_mini_batch_size 整除")
    if train_batch_size * rollout_n % micro_batch_size != 0:
        raise ConfigValidationError("展开后的轨迹数必须能被 micro-batch 整除")

    gpu_memory_utilization = float(config.actor_rollout_ref.rollout.gpu_memory_utilization)
    if not 0 < gpu_memory_utilization < 1:
        raise ConfigValidationError("gpu_memory_utilization 必须位于 0 和 1 之间")

    train_files = _as_path_list(config.data.train_files, "data.train_files")
    val_files = _as_path_list(config.data.val_files, "data.val_files")
    for path in [*train_files, *val_files]:
        _require_file(path, "训练数据")
    final_test_path = (project_root / "data" / "final_test.parquet").resolve()
    if final_test_path in [*train_files, *val_files]:
        raise ConfigValidationError("final_test.parquet 不得用于训练或训练期验证")

    reward_path = Path(str(config.reward.custom_reward_function.path)).expanduser().resolve()
    tool_path = Path(str(config.actor_rollout_ref.rollout.multi_turn.function_tool_path)).expanduser().resolve()
    _require_file(reward_path, "reward.custom_reward_function.path")
    _require_file(tool_path, "rollout.multi_turn.function_tool_path")


def load_and_validate_config(
    config_path: Path,
    verl_source_dir: Path,
    project_root: Path = PROJECT_ROOT,
    overrides: Sequence[str] = (),
) -> DictConfig:
    """组合并验证项目训练配置，成功时返回已解析配置"""
    config_path = Path(config_path).expanduser().resolve()
    verl_source_dir = Path(verl_source_dir).expanduser().resolve()
    project_root = Path(project_root).expanduser().resolve()
    _require_file(config_path, "训练配置")

    official_config_dir = _official_config_dir(verl_source_dir)
    official_config = _compose_official_config(official_config_dir)
    project_overrides = OmegaConf.load(config_path)
    _validate_known_keys(project_overrides, official_config)

    environment = {
        "CALCULATOR_AGENT_ROOT": str(project_root),
        "VERL_CONFIG_DIR": str(official_config_dir),
    }
    _validate_overrides(overrides)
    config = _compose_project_config(config_path, environment, overrides)
    _validate_runtime_contract(config, project_root)
    return config


def build_argument_parser() -> argparse.ArgumentParser:
    """构造无 GPU 配置验证命令行"""
    parser = argparse.ArgumentParser(description="验证 Calculator Agent 的 verl GRPO 配置")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--verl-source-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--print-resolved", action="store_true")
    parser.add_argument("overrides", nargs="*", help="传给 Hydra 的 key=value 参数覆盖")
    return parser


def main(argv=None) -> int:
    """验证配置并按需打印 Hydra 展开结果"""
    arguments = build_argument_parser().parse_args(argv)
    try:
        config = load_and_validate_config(
            config_path=arguments.config,
            verl_source_dir=arguments.verl_source_dir,
            project_root=arguments.project_root,
            overrides=arguments.overrides,
        )
    except (
        ConfigValidationError,
        HydraException,
        OmegaConfBaseException,
        OSError,
        TypeError,
        ValueError,
        YAMLError,
    ) as exc:
        print(f"训练配置验证失败: {exc}", file=sys.stderr)
        return 1

    if arguments.print_resolved:
        print(OmegaConf.to_yaml(config, resolve=True), end="")
    else:
        print("训练配置验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
