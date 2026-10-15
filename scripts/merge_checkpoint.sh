#!/usr/bin/env bash
set -euo pipefail

# 将 verl FSDP actor checkpoint 合并为普通 Hugging Face 模型目录。
usage() {
    cat >&2 <<'EOF'
用法:
  scripts/merge_checkpoint.sh [--dry-run] ACTOR_CHECKPOINT TARGET_DIR

环境变量:
  VERL_SOURCE_DIR  必须显式指定已验收的 verl v0.9.1 源码目录
  VERL_USE_UV      1 使用官方 uv lock，0 使用当前 Python 环境，默认 1
  PYTHON_BIN       VERL_USE_UV=0 时使用的 Python，默认 python3
EOF
}

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi
if [[ $# -ne 2 ]]; then
    usage
    exit 2
fi

actor_input=$1
if [[ ! -e "$actor_input" ]]; then
    echo "错误: actor checkpoint 不存在: $actor_input" >&2
    exit 1
fi
if [[ ! -d "$actor_input" ]]; then
    echo "错误: actor checkpoint 不是目录: $actor_input" >&2
    exit 1
fi

actor_checkpoint=$(realpath -e -- "$actor_input")
target_dir=$(realpath -m -- "$2")
# 不回退到开发者机器的私有路径，防止远端误用其他运行副本。
verl_source_dir=${VERL_SOURCE_DIR:?请设置经过审计的 VERL_SOURCE_DIR}
verl_use_uv=${VERL_USE_UV:-1}
python_bin=${PYTHON_BIN:-python3}

if [[ ! -f "$actor_checkpoint/fsdp_config.json" ]]; then
    echo "错误: 缺少 fsdp_config.json: $actor_checkpoint" >&2
    exit 1
fi
if [[ ! -f "$actor_checkpoint/huggingface/config.json" ]]; then
    echo "错误: 缺少 Hugging Face config.json: $actor_checkpoint/huggingface" >&2
    exit 1
fi
if [[ ! -f "$verl_source_dir/verl/model_merger/__main__.py" ]]; then
    echo "错误: VERL_SOURCE_DIR 不是有效的 verl v0.9.1 源码目录: $verl_source_dir" >&2
    exit 1
fi

# 从官方 fsdp_config.json 读取 world_size，并逐个确认 rank 分片存在。
world_size=$(python3 -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle).get("world_size")
if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    raise SystemExit("world_size 必须是正整数")
print(value)
' "$actor_checkpoint/fsdp_config.json")

for ((rank = 0; rank < world_size; rank++)); do
    shard="$actor_checkpoint/model_world_size_${world_size}_rank_${rank}.pt"
    if [[ ! -s "$shard" ]]; then
        echo "错误: 缺少 FSDP 分片或分片为空: $shard" >&2
        exit 1
    fi
done

if [[ -e "$target_dir" && ! -d "$target_dir" ]]; then
    echo "错误: 目标路径存在但不是目录: $target_dir" >&2
    exit 1
fi
if [[ -d "$target_dir" ]] && [[ -n "$(find "$target_dir" -mindepth 1 -print -quit)" ]]; then
    echo "错误: 目标目录非空，拒绝覆盖: $target_dir" >&2
    exit 1
fi

merger_args=(
    -m verl.model_merger merge
    --backend fsdp
    --local_dir "$actor_checkpoint"
    --target_dir "$target_dir"
)
if [[ "$verl_use_uv" == "1" ]]; then
    command=(uv run --frozen --all-packages --extra sglang --extra fsdp python "${merger_args[@]}")
elif [[ "$verl_use_uv" == "0" ]]; then
    command=("$python_bin" "${merger_args[@]}")
else
    echo "错误: VERL_USE_UV 只能是 0 或 1" >&2
    exit 1
fi

printf 'verl 目录: %s\n' "$verl_source_dir"
printf 'actor checkpoint: %s\n' "$actor_checkpoint"
printf '导出目录: %s\n' "$target_dir"
printf '合并命令: '
printf '%q ' "${command[@]}"
printf '\n'

if [[ "$dry_run" == "1" ]]; then
    echo "dry-run 完成：checkpoint 结构有效，未创建导出目录。"
    exit 0
fi

cd "$verl_source_dir"
exec "${command[@]}"
