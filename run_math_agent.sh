#!/usr/bin/env bash
# run_math_agent.sh —— 验证配置；仅在远程原生 Linux 启动 GRPO

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# 源码路径由发布环境显式提供，禁止回退到开发者机器的私有目录。
VERL_SOURCE_DIR=${VERL_SOURCE_DIR:?请设置经过审计的 VERL_SOURCE_DIR}
# 即使当前 shell 未激活环境，也固定使用已由 uv 锁定并验收的解释器。
PYTHON_BIN=${PYTHON_BIN:-${VERL_SOURCE_DIR}/.venv/bin/python3}
CONFIG_PATH=${CONFIG_PATH:-${PROJECT_ROOT}/configs/calculator_grpo_vast.yaml}
CONFIG_DIR=$(cd "$(dirname "${CONFIG_PATH}")" && pwd)
CONFIG_NAME=$(basename "${CONFIG_PATH}" .yaml)
RUN_DIR=${RUN_DIR:-${PROJECT_ROOT}}

export CALCULATOR_AGENT_ROOT=${PROJECT_ROOT}
export VERL_CONFIG_DIR=${VERL_SOURCE_DIR}/verl/trainer/config
# Ray 子进程继承该路径后才能导入项目根目录中的工具与奖励模块。
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# 云端不继承 WSL 专用绕过；历史补丁保留但启动入口明确关闭。
export CALCULATOR_AGENT_WSL_VMM_IPC=0
export CALCULATOR_AGENT_WSL_REF_GPU=0
export CALCULATOR_AGENT_WSL_BLOCKING_OFFLOAD=0
# Dense Qwen 不使用 DeepGEMM；显式关闭该可选 JIT，保留 SGLang CUDA Graph 所需的其余编译路径。
export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
# 工具闭合标签触发运行时停止，保留模型原始token；不依赖模型再生成EOS。
export CALCULATOR_AGENT_HERMES_TOOL_STOP=${CALCULATOR_AGENT_HERMES_TOOL_STOP:-1}
# PyTorch 使用 CUDA 13.0，SGLang 的 CUDA Graph 预热必须调用同版本 nvcc 编译内核。
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="${CUDA_HOME}/bin:${PATH}"
# PyTorch CUDA 13 运行库由 uv 环境提供；SGLang 新建进程不会自动继承 wheel 的 rpath。
CUDA_RUNTIME_LIB_DIR=${CUDA_RUNTIME_LIB_DIR:-}
# 独立远程环境未必位于 verl/.venv；需要时显式提供实际 wheel 运行库路径。
# 不注入 WSL 驱动 stub；原生 Linux 的宿主驱动由容器运行时提供。
if [[ -n ${CUDA_RUNTIME_LIB_DIR} ]]; then
    export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

validate_config() {
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/validate_training_config.py" \
        --config "${CONFIG_PATH}" \
        --verl-source-dir "${VERL_SOURCE_DIR}" \
        --project-root "${PROJECT_ROOT}" \
        --print-resolved \
        "$@"
}

if [[ ${1:-} == "--validate-only" ]]; then
    shift
    validate_config "$@"
    exit 0
fi

# 本机训练已终止；WSL 仍可使用上面的纯配置验证分支。
KERNEL_RELEASE=$(uname -r)
if [[ $(uname -s) != Linux || ${KERNEL_RELEASE,,} == *microsoft* || ${KERNEL_RELEASE,,} == *wsl* ]]; then
    echo "拒绝在 WSL 或非 Linux 环境训练，请在已批准的远程原生 Linux 实例启动。" >&2
    exit 1
fi
# 正式训练前先拒绝缺失路径和不兼容配置，避免 Ray 启动后才失败。
validate_config "$@" >/dev/null

cd "${VERL_SOURCE_DIR}"

if [[ ${VERL_USE_UV:-1} == 1 ]]; then
    # 环境已由冻结锁文件完成同步；启动时不得再次增删已验收的 GPU wheel。
    # 使用预检查的明确解释器，不让 uv 隐式选择另一套虚拟环境。
    exec uv run --frozen --no-sync --all-packages --extra sglang --extra fsdp "${PYTHON_BIN}" \
        -m verl.trainer.main_ppo \
        --config-dir "${CONFIG_DIR}" \
        --config-name "${CONFIG_NAME}" \
        "hydra.run.dir=${RUN_DIR}" \
        "$@"
fi

exec "${PYTHON_BIN}" -m verl.trainer.main_ppo \
    --config-dir "${CONFIG_DIR}" \
    --config-name "${CONFIG_NAME}" \
    "hydra.run.dir=${RUN_DIR}" \
    "$@"
