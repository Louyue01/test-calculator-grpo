# Calculator Agent：远程 GRPO 最小项目

本地修改代码，通过公开 Git 仓库同步；仅在 Vast.ai 原生 Linux 安装 GPU 环境、生成和训练。
不包含完整 verl 源码、本机 WSL/IPC 诊断、历史报告、数据集、模型或缓存。
业务代码通过 CPU 测试不等于远程 GPU 训练成功；首次只执行获批的单步技术 canary。

## 文件说明

- `calculator_tool.py`、`tools/`：真实计算器与 verl 工具注册。
- `prepare_data.py`：英文 system prompt、中文副本与 GSM8K 转换；不在导入时下载数据。
- `reward_function.py`：包含工具和 think 奖励，不移除。
- `evaluate_model.py`、`scripts/merge_checkpoint.sh`：评估及 checkpoint 导出。
- `configs/calculator_grpo_vast.yaml`：唯一运行配置。
- `deploy/verl/`：固定上游 commit、四个原生补丁、实际 pyproject/uv.lock 和上游许可声明，不是源码副本。
- `scripts/remote/`：分阶段安装、输入校验和运行入口；默认不训练。
- `tests/`：少量业务安全与部署 CPU 测试，使用临时夹具，不要求代码目录里存在真实数据。

## 1. 拉取代码并校验

前提：Ubuntu 24.04、x86_64、Python 3.12、CUDA 13.0 devel 工具链与兼容宿主驱动。
GPU 推荐单张 4090 24GB，约 64GB RAM、8 vCPU、100GB SSD；这是预留建议，不是实测最低需求。
容器需要已有 git、GNU patch、uv 和编译工具；不要在容器安装宿主显卡驱动。

```bash
git clone https://github.com/Louyue01/test-calculator-grpo.git /workspace/calculator-agent
cd /workspace/calculator-agent
git checkout --detach <已批准的代码commit>
python3 scripts/remote/git_deploy.py
```

`/workspace` 是约定路径，需要创建并确认其磁盘生命周期；不是默认外部备份。
最后一条只核验发布清单、补丁和锁，不安装、不下载、不训练。
公开仓库 clone 不需要 GitHub token；不要上传 SSH 私钥或访问令牌。
项目尚未选择开源许可证；公开可见不等于授予任意使用许可。上游材料保留原许可与 Notice。

## 2. 固定源码与环境（分别获批后执行）

```bash
python3 scripts/remote/git_deploy.py prepare-source \
  --verl-source /workspace/verl/calculator-native-v1 --allow-network

python3 scripts/remote/git_deploy.py install \
  --verl-source /workspace/verl/calculator-native-v1 --allow-network
```

第一条只下载固定 verl v0.9.1，验证 commit `1876b06d0a3e4e71e06230be10af14492ca8a75b`，
按清单应用四个补丁并核对实际锁。第二条以 Python 3.12 执行冻结的 uv sync，创建源码下 `.venv`。
现有目录/环境不覆盖，失败不自动清理、不换版本、不扩大资源预算。
离线重建可使用 `--base-archive <已验收的LF基线.tar>` 替代联网选项。
不上传本机虚拟环境，不应用 WSL VMM/ref/blocking 补丁。

源码完整性清单采用 schema_version=2，必须覆盖四个补丁实际修改的全部六个文件。
安装前同时核对发布资产中的预期摘要与现场文件，不接受空清单、漏项或现场自行更新的摘要。
纯只读复核可执行：

```bash
python3 scripts/remote/git_deploy.py verify-source --verl-source /workspace/verl/calculator-native-v1
```

旧部署清单会被拒绝。修复代码发布、拉取并获批后，可显式重建旧清单：

```bash
python3 scripts/remote/git_deploy.py record-source \
  --verl-source /workspace/verl/calculator-native-v1 --replace-marker
```

此命令先核对全部六个源码文件的已批准摘要，通过才替换清单；不会重置源码、安装或训练。
源码已改动时不能靠重新计算现场摘要放行，需另行审查和发布。

安装组合：PyTorch 2.11.0+cu130、SGLang 0.5.12、Transformers 5.9.0、Ray 2.55.1。
`deploy/verl/uv.lock` 是实际锁，根目录 pyproject 只声明项目及 CPU 开发依赖。
首次安装仍须验证 wheel 来源、GPU 架构和运行库；不能假定本机锁在云端必然成功。

## 3. 数据和固定模型（独立于 Git）

本次 canary 继续使用已验收的英文训练8题/验证32题，不重新随机抽样。
将两份原始 Parquet 经 SSH 单独传到 `/workspace/data/approved-canary/` 后：

```bash
export VERL_SOURCE_DIR=/workspace/verl/calculator-native-v1
export PYTHON_BIN="$VERL_SOURCE_DIR/.venv/bin/python3"
export PYTHONPATH="$VERL_SOURCE_DIR:$PWD"
export VERL_USE_UV=0
export HF_HOME=/workspace/cache/huggingface
export CALCULATOR_DATA_DIR=/workspace/data/approved-canary

"$PYTHON_BIN" scripts/remote/prepare_inputs.py verify-data --data-dir "$CALCULATOR_DATA_DIR"

# 只有获准下载后才执行；输出已核验的本地模型快照路径。
"$PYTHON_BIN" scripts/remote/prepare_inputs.py download-model \
  --cache-dir "$HF_HOME" --allow-network

export CALCULATOR_MODEL_PATH=<上述命令输出的实际快照路径>
export CALCULATOR_RUN_ROOT=/workspace/runs/<新的实验ID>
```

模型固定为 Qwen/Qwen3-0.6B，revision `c1899de289a04d12100db370d81485cdf75e47ca`。
入口核对 safetensors SHA256；`CALCULATOR_MODEL_PATH` 必须是实际路径，不是 Hub ID。
数据校验固定两份已验收文件的 SHA256，不以“文件存在”或“前8题”代替身份验收。
全量新实验才使用 prepare_data.py，必须单独批准；它不负责复制历史 canary 子集。
不要把官方 final_test split 用于训练或调参。

## 4. 环境验证与单步 canary（分别获批）

```bash
# 远程 CUDA 检查，不做分配压力测试或训练。
"$PYTHON_BIN" scripts/check_environment.py --mode remote-cuda --workspace /workspace --json --strict

# 默认只校验模型、数据和配置，不创建实验目录。
bash scripts/remote/canary.sh

# 只有在环境与冒烟验收后单独批准才执行。
bash scripts/remote/canary.sh --train
```

保留既定 8 prompt × n4、actor mini-batch8、512 token 总生成预算、actor KL loss，KL 不加入奖励。
think 由模型生成；calculator 仅计算数值表达式，不能直接求解方程；Hermes 工具闭合后停止等待真实工具结果。
配置仅训练1步，resume 关闭；它不是完整训练计划，不保证能力提升。
32题n1基线、恢复训练、更多步数和最终评估须另行批准，不由安装入口隐式启动。
文件描述符软上限和 memlock 只检查记录，必要调整先核对容器权限；不自动修改全局设置。
若 SGLang 子进程找不到 CUDA wheel 库，先定位实际库目录再设置 CUDA_RUNTIME_LIB_DIR，不使用 WSL 驱动 stub。

## 5. 导出、评估与后续同步

```bash
bash scripts/merge_checkpoint.sh --dry-run <actor-checkpoint目录> <新HF导出目录>
"$PYTHON_BIN" evaluate_model.py --help
```

移除 --dry-run 才实际导出；运行评估前展示并批准模型、题目身份和生成预算。
日志/checkpoint/导出放在仓库外，按 run-id 下载至本地备份。停止实例仍可能收取存储费用，销毁需另行批准。
本地修改和 CPU 测试后重新生成发布摘要，验收后再 commit/push；远端只在无训练任务时 checkout 获批 commit。
不要直接推送外层开发工作区历史，也不要训练过程中 git pull。

## CPU 验证

Python 3.12 的已有环境中：

```bash
python -m pytest -q
```

这些测试不下载模型、不启动 Ray/CUDA 工作负载。Windows 会跳过 Bash 行为用例；Linux CPU 环境应执行这些用例。
实际 GPU rollout、权重同步、梯度、checkpoint 恢复需远端单独验收。
