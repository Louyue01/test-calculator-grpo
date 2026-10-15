#!/usr/bin/env bash
# 默认只验证；真实单步训练必须显式 --train，数据/模型不随 Git 下载。
set -euo pipefail
MODE=${1:---validate-only}
if [[ $# -gt 1 || ( ${MODE} != --validate-only && ${MODE} != --train ) ]]; then
    echo '用法: bash scripts/remote/canary.sh [--validate-only|--train]' >&2
    exit 2
fi
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${VERL_SOURCE_DIR:?请指定已验收的原生 verl 源码目录}"
: "${CALCULATOR_MODEL_PATH:?请指定固定 revision 的已核验本地模型快照}"
: "${CALCULATOR_DATA_DIR:?请指定已验收的 canary 8/32 数据目录}"
: "${CALCULATOR_RUN_ROOT:?请指定新的实验目录}"
export PYTHON_BIN=${PYTHON_BIN:-${VERL_SOURCE_DIR}/.venv/bin/python3}
export VERL_USE_UV=0
export CONFIG_PATH=${PROJECT_ROOT}/configs/calculator_grpo_vast.yaml
export RUN_DIR=${CALCULATOR_RUN_ROOT}
for path in "${CALCULATOR_MODEL_PATH}/config.json" "${CALCULATOR_MODEL_PATH}/model.safetensors" \
    "${CALCULATOR_DATA_DIR}/train.parquet" "${CALCULATOR_DATA_DIR}/validation.parquet"; do
    if [[ ! -s ${path} ]]; then
        echo "缺少已核验输入: ${path}" >&2
        exit 1
    fi
done
# 入口直接核对已验收权重和8/32数据，不能仅靠文件存在便开始训练。
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/remote/prepare_inputs.py" verify-data --data-dir "${CALCULATOR_DATA_DIR}"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/remote/prepare_inputs.py" verify-model --model-dir "${CALCULATOR_MODEL_PATH}"
# OmegaConf 点列表由 JSON 编码，数据目录包含空格时也不会被拆成多个参数。
TRAIN_FILES=$("${PYTHON_BIN}" -c 'import json,sys; print(json.dumps([sys.argv[1]]))' "${CALCULATOR_DATA_DIR}/train.parquet")
VAL_FILES=$("${PYTHON_BIN}" -c 'import json,sys; print(json.dumps([sys.argv[1]]))' "${CALCULATOR_DATA_DIR}/validation.parquet")
OVERRIDES=("data.train_files=${TRAIN_FILES}" "data.val_files=${VAL_FILES}")
if [[ ${MODE} == --validate-only ]]; then
    exec bash "${PROJECT_ROOT}/run_math_agent.sh" --validate-only "${OVERRIDES[@]}"
fi
if [[ -e ${CALCULATOR_RUN_ROOT} ]]; then
    echo '拒绝复用实验目录，请指定新的 CALCULATOR_RUN_ROOT。' >&2
    exit 1
fi
KERNEL_RELEASE=$(uname -r)
if [[ $(uname -s) != Linux || ${KERNEL_RELEASE,,} == *microsoft* || ${KERNEL_RELEASE,,} == *wsl* ]]; then
    echo '训练仅允许在已批准的远程原生 Linux 执行。' >&2
    exit 1
fi
# 配置验证先于目录创建；不会因 --train 绕过缺失数据或无效配置。
bash "${PROJECT_ROOT}/run_math_agent.sh" --validate-only "${OVERRIDES[@]}" >/dev/null
mkdir -p -- "${CALCULATOR_RUN_ROOT}"
exec bash "${PROJECT_ROOT}/run_math_agent.sh" "${OVERRIDES[@]}"
