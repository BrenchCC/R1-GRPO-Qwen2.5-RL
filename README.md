# R1-GRPO-Qwen2.5-RL

> 基于 Hugging Face TRL/Accelerate/DeepSpeed 的 Qwen2.5 小模型 R1-Zero 风格强化学习复现项目，核心目标是用 **GRPO（Group Relative Policy Optimization）** 在可验证问答数据上激发模型的长链式推理、答案格式遵循与数学/医学可验证任务求解能力。

本仓库适合作为简历项目展示：它不仅包含训练入口、奖励函数、GRPO Trainer 定制、ZeRO-3 配置、vLLM 加速采样和评测脚本，也沉淀了完整的工程流程说明。更详细的技术解读见 [`docs/grpo_training_deep_dive.md`](docs/grpo_training_deep_dive.md)。

## 项目亮点

- **R1-Zero 风格 RL 训练闭环**：从数据集加载、对话模板构造、批内多样本采样、奖励函数打分、组内优势归一化，到 PPO-style clip loss 更新策略，形成端到端训练链路。
- **自定义 `BrenchGRPOTrainer`**：在 TRL `GRPOTrainer` 基础上扩展/复写采样器、vLLM 生成、奖励聚合、多轮 policy update、KL 约束和日志指标。
- **多奖励函数组合**：支持准确率奖励、格式奖励、推理步骤奖励、长度奖励、余弦长度缩放奖励、N-gram 重复惩罚，并可通过 YAML/CLI 灵活组合。
- **可验证任务导向**：针对数学 LaTeX 答案使用 `math-verify` 做符号级验证；对文本型答案提供 `<answer>` 抽取和归一化匹配兜底。
- **工程化训练配置**：提供 Qwen2.5-0.5B/1.5B 配置、DeepSpeed ZeRO-3、bfloat16、梯度检查点、vLLM 采样、W&B 日志和 checkpoint 保存策略。
- **评测与结果留存**：`src/benchmark.py` 支持用 vLLM 在 MATH-500 等可验证数据集上批量生成，并输出准确率、格式率与 JSON 明细。

## 仓库结构

```text
.
├── README.md                         # 项目说明与快速启动
├── docs/
│   └── grpo_training_deep_dive.md    # 训练全流程与奖励函数详细解读
├── recipes/
│   ├── zero3.yaml                    # DeepSpeed ZeRO-3/Accelerate 配置
│   ├── R1_Zero_0dot5B_config_v1.yaml # Qwen2.5-0.5B GRPO 配置
│   └── R1_Zero_1dot5B_config_v1.yaml # Qwen2.5-1.5B GRPO 配置
├── scripts/
│   ├── run_r1_zero_v1.sh             # 1.5B 训练启动脚本
│   └── run_r1_zero_v2.sh             # 另一组训练启动脚本
├── src/
│   ├── grpo.py                       # 主训练入口：参数解析、数据准备、Trainer 初始化
│   ├── brench_grpo_trainer.py        # 自定义 GRPO Trainer 核心实现
│   ├── rewards.py                    # 奖励函数集合
│   ├── config.py                     # GRPO/SFT 配置扩展
│   ├── prompts.py                    # R1 格式 system prompt
│   ├── benchmark.py                  # vLLM 评测脚本
│   └── utils/                        # callbacks、Hub、evaluation 工具
└── tests/                            # 单元测试
```

## 环境准备

建议使用 Python 3.10+ 与 CUDA 环境。核心依赖见 `requirements.txt`，其中包含：

- `torch==2.5.1`
- `transformers==4.48.2`
- `trl==0.15.2` 及一个指定 commit 的 TRL Git 依赖
- `accelerate`、`deepspeed`、`vllm`、`peft`、`math-verify`、`wandb`

安装示例：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

> 如果需要 `flash_attention_2`，请根据本机 CUDA/PyTorch 版本单独安装 `flash-attn`，并在配置文件中保留 `attn_implementation: flash_attention_2`。

## 数据与 Prompt 构造

训练入口会通过 `load_dataset` 读取配置中的数据集，例如 `Brench/R1-Zero-GRPO-750`。每条样本会被转换成对话格式：

```python
[
  {"role": "system", "content": SYSTEM_PROMPT},
  {"role": "user", "content": example["problem"]},
]
```

`SYSTEM_PROMPT` 明确要求模型输出：

```text
<think> reasoning process here </think><answer> answer here </answer>
```

这使奖励函数可以同时约束 **推理过程** 与 **最终答案**，也便于后续评测时解析 `<answer>`。

## GRPO 训练流程概览

1. **解析配置**：`TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))` 同时读取脚本参数、训练参数和模型参数。
2. **加载数据集**：根据 `dataset_name`/`dataset_config` 加载数据，并映射成 system/user prompt。
3. **注册奖励函数**：根据 `reward_funcs` 选择 accuracy、format、reasoning_steps、cosine、repetition_penalty、length 等函数。
4. **初始化模型与 PEFT**：从 `model_name_or_path` 加载 Qwen2.5，按需配置 LoRA/PEFT、bf16、attention implementation、cache 和梯度检查点。
5. **批内重复采样**：对每个 prompt 生成 `num_generations` 个 completion，形成一个 group。
6. **奖励打分**：每个 completion 分别进入多个奖励函数，按 `reward_weights` 聚合成标量 reward。
7. **组内优势归一化**：对同一 prompt 的多个 completion 计算均值/标准差，得到相对优势：

   ```text
   A_i = (r_i - mean(group_rewards)) / (std(group_rewards) + 1e-4)
   ```

8. **策略更新**：使用 old policy log-prob 与 current policy log-prob 的 ratio，结合 clip 范围 `epsilon` 计算 GRPO loss；若 `beta > 0`，额外加入 reference model KL 惩罚。
9. **日志与保存**：记录 completion length、各奖励分量、总 reward、reward std、clip ratio、KL 等指标，并保存模型与 model card。

## 奖励函数设计

| 奖励函数 | 目标 | 典型用途 |
| --- | --- | --- |
| `accuracy_reward` | 校验最终答案是否正确 | 数学题、可验证问答主奖励 |
| `format_reward` | 检查是否严格输出 `<think>...</think><answer>...</answer>` | 保证 R1 格式可解析 |
| `reasoning_steps_reward` | 检测 Step/编号/项目符号/转折词 | 鼓励显式分步推理 |
| `len_reward` | 在正确前提下鼓励更短答案；错误答案不奖励冗长 | 抑制 overthinking |
| `get_cosine_scaled_reward` | 用余弦曲线按长度缩放正确/错误答案奖励 | 更平滑的长度控制 |
| `get_repetition_penalty_reward` | 按重复 N-gram 比例给负奖励 | 减少循环复读 |

默认 `GRPOScriptArguments.reward_funcs` 为 `['accuracy', 'format']`，可以在配置或命令行中扩展。

## 快速启动训练

### 1. 修改配置

根据机器环境修改 `recipes/R1_Zero_0dot5B_config_v1.yaml` 或 `recipes/R1_Zero_1dot5B_config_v1.yaml`：

- `model_name_or_path`：本地模型路径或 Hugging Face Hub 模型名。
- `dataset_name`：训练数据集。
- `per_device_train_batch_size`、`num_generations`、`max_completion_length`：显存相关参数。
- `use_vllm`、`vllm_gpu_memory_utilization`：是否使用 vLLM 生成。
- `report_to`、`wandb_project`、`wandb_entity`：W&B 日志配置。

### 2. 启动训练

```bash
bash scripts/run_r1_zero_v1.sh
```

脚本内部会调用：

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/zero3.yaml \
  --num_processes=1 src/grpo.py \
  --config recipes/R1_Zero_1dot5B_config_v1.yaml
```

### 3. 输出产物

训练结果默认保存在配置中的 `output_dir`，通常包含：

- 模型权重与 tokenizer/config
- trainer state
- 训练指标 JSON
- 自动生成的 model card
- shell 重定向保存的训练日志

## 评测

使用 `src/benchmark.py` 可以在 MATH-500 等数据集上做 vLLM 推理评测：

```bash
CUDA_VISIBLE_DEVICES=0 python src/benchmark.py \
  --model_name=Brench/Qwen2.5-R1-Zero-GRPO-1.5B-V1 \
  --dataset_name=HuggingFaceH4/MATH-500 \
  --output_name=./output/result_benchmark_math500 \
  --max_output_tokens=8192 \
  --num_gpus=1
```

脚本会输出：

- `eval_acc`：基于 `accuracy_answer_reward` 的答案正确率。
- `eval_format`：是否满足 `<think>...</think><answer>...</answer>` 的格式率。
- `*.json`：每条样本的 prompt、completion、gold answer、accuracy score、format score。

## 简历表述建议

可以将本项目提炼为如下经历：

> 复现并工程化实现 Qwen2.5 小模型 R1-Zero 风格 GRPO 强化学习训练框架，基于 TRL/Accelerate/DeepSpeed/vLLM 构建从数据集对话化、批内多样本采样、数学可验证奖励、格式/长度/重复惩罚奖励、组内优势归一化到 clipped policy optimization 的完整 RLHF 训练闭环；支持 ZeRO-3、bf16、W&B 日志、checkpoint 管理和 MATH-500 vLLM 评测，提升模型按 `<think>/<answer>` 格式进行长链式推理与可验证答案输出的能力。

更详细的“项目背景、算法公式、代码路径、指标解读、面试问答”请阅读 [`docs/grpo_training_deep_dive.md`](docs/grpo_training_deep_dive.md)。

## 常见问题

### 为什么用 GRPO 而不是 PPO？

GRPO 不需要单独训练 value model，而是在同一 prompt 的多个 completion 内做相对奖励归一化，降低工程复杂度和显存开销，尤其适合小模型 R1-Zero 风格可验证任务训练。

### 为什么需要 format reward？

因为准确率奖励只关心最终答案，模型可能输出不可解析或无结构文本。格式奖励把 `<think>` 和 `<answer>` 变成可学习约束，保证后续自动评测、答案抽取和简历展示中的可解释性。

### 为什么要加长度/重复惩罚？

长 CoT 训练容易出现 overthinking、空转和重复。长度奖励与余弦长度缩放鼓励“正确且简洁”的推理；N-gram 重复惩罚降低循环生成概率。

## License

当前仓库未显式声明 License。如需开源发布，请在使用的数据集、基础模型、依赖库许可证兼容的前提下补充许可证文件。
