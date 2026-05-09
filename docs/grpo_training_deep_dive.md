# Qwen2.5 R1-Zero GRPO 训练全流程详细解读

本文档面向后续写简历、答辩或面试复盘，系统梳理本仓库如何构建 Qwen2.5 小模型的 R1-Zero 风格 GRPO 强化学习训练流程，重点覆盖：项目目标、数据构造、Prompt 设计、GRPO 算法、奖励函数、Trainer 实现、配置参数、评测方法与可包装的简历表达。

## 1. 项目定位

### 1.1 背景

DeepSeek-R1/R1-Zero 类工作证明：只要任务具备可验证反馈，模型可以通过强化学习自发学习更长、更结构化的推理行为。本项目聚焦小参数 Qwen2.5-Instruct 模型，在数学/医学等可验证问答数据上进行 GRPO 训练，目标不是单纯监督模型“模仿 CoT”，而是通过 reward signal 让模型探索并强化能够得到正确答案的推理轨迹。

### 1.2 核心目标

- 让模型稳定输出 `<think>...</think><answer>...</answer>`。
- 用可验证答案奖励提升最终答案正确率。
- 用组内相对优势替代 value model，降低 PPO 式 RLHF 的工程成本。
- 用长度和重复惩罚控制长 CoT 训练中的 overthinking 与复读问题。
- 形成可复用、可评测、可写入简历的 LLM RL 工程闭环。

## 2. 代码入口与模块分工

| 模块 | 职责 |
| --- | --- |
| `src/grpo.py` | 主入口：解析参数、加载数据、选择奖励函数、初始化 `BrenchGRPOTrainer`、训练与保存。 |
| `src/brench_grpo_trainer.py` | 自定义 Trainer：多 completion 采样、奖励聚合、优势归一化、GRPO loss、vLLM/DeepSpeed/PEFT 适配。 |
| `src/rewards.py` | 奖励函数：准确率、格式、推理步骤、长度、余弦长度缩放、重复惩罚。 |
| `src/config.py` | 扩展 TRL 配置：benchmark、callback、W&B、Hub、`num_iterations`、`epsilon` 等。 |
| `src/prompts.py` | 统一 system prompt，强制 R1 输出格式。 |
| `src/benchmark.py` | vLLM 推理评测，统计答案准确率和格式通过率。 |
| `recipes/*.yaml` | 实验配置：模型路径、数据集、显存参数、GRPO 超参数、日志与保存策略。 |

## 3. 数据与对话构造

### 3.1 数据读取

训练脚本通过 `load_dataset(script_args.dataset_name, name=script_args.dataset_config)` 加载 Hugging Face 数据集。配置示例中使用：

```yaml
dataset_name: Brench/R1-Zero-GRPO-750
```

对于 `FreedomIntelligence/medical-o1-verifiable-problem`，脚本会把原始字段重命名为统一字段：

```python
"Open-ended Verifiable Question" -> "problem"
"Ground-True Answer" -> "solution"
```

这样后续训练只依赖 `problem` 与 `solution` 两个核心字段。

### 3.2 Prompt 构造

每条样本被映射成 conversational prompt：

```python
{
    "prompt": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["problem"]},
    ]
}
```

`SYSTEM_PROMPT` 要求模型先在 `<think>` 中给出推理过程，再在 `<answer>` 中给最终答案。这是整个训练设计的关键：

- `<think>` 承载推理过程，便于观察模型是否学会分步推理。
- `<answer>` 承载可验证最终答案，便于正则抽取和自动打分。
- 格式可解析后，评测脚本可以稳定计算准确率和格式率。

### 3.3 训练样本在 GRPO 中如何使用

GRPO 并不是对每个 prompt 只生成一个答案，而是为同一个 prompt 生成 `num_generations` 个候选 completion。例如配置中：

```yaml
num_generations: 4
per_device_train_batch_size: 4
```

含义是每个问题在一个 group 内采样 4 条不同回答。之后对这 4 条回答分别打分，再用组内均值和标准差计算相对优势。

## 4. GRPO 算法核心

### 4.1 为什么选择 GRPO

传统 PPO-RLHF 通常需要：

1. policy model；
2. reference model；
3. reward model/rule reward；
4. value model。

其中 value model 带来额外显存、训练和稳定性成本。GRPO 的思想是：同一个 prompt 下采样多个 completion，以组内 reward 的相对好坏估计 advantage，从而不需要单独 value model。

### 4.2 组内奖励与优势

设同一个 prompt 生成 \(G\) 个回答，每个回答的聚合奖励为 \(r_i\)。组内均值和标准差为：

\[
\mu = \frac{1}{G}\sum_{i=1}^{G} r_i
\]

\[
\sigma = \sqrt{\frac{1}{G}\sum_{i=1}^{G}(r_i-\mu)^2}
\]

本项目中的 advantage 计算为：

\[
A_i = \frac{r_i - \mu}{\sigma + 10^{-4}}
\]

直观理解：

- 同一题中更好的回答得到正 advantage。
- 更差的回答得到负 advantage。
- 只比较同题候选，减少不同题难度不同带来的 reward 尺度问题。

### 4.3 Policy ratio 与 clip

Trainer 会计算当前策略与旧策略在 completion token 上的 log probability：

\[
\rho_i = \exp(\log \pi_\theta(y_i|x) - \log \pi_{old}(y_i|x))
\]

然后采用 PPO-style clipped objective：

\[
L_i = -\min(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon)A_i)
\]

代码中 `epsilon` 来自 `GRPOConfig`，用于限制新旧策略变化幅度，防止训练一步把 policy 推得过远。

### 4.4 KL 约束

如果配置中 `beta != 0.0`，Trainer 会计算当前模型与 reference model 的近似 KL：

\[
D_{KL} \approx \exp(\log \pi_{ref} - \log \pi_\theta) - (\log \pi_{ref} - \log \pi_\theta) - 1
\]

并将 `beta * KL` 加入 token loss，避免模型在 RL 过程中偏离初始指令模型过多。如果使用 PEFT/LoRA，reference 可以通过禁用 adapter 得到；否则会创建 reference model。

## 5. 奖励函数详解

奖励函数位于 `src/rewards.py`。训练脚本通过 `get_reward_funcs` 将配置中的字符串映射到具体函数。

### 5.1 `accuracy_reward`：答案正确性奖励

这是最核心的奖励。逻辑如下：

1. 从 completion 中取出文本。
2. 用 `math_verify.parse` 解析 gold solution。
3. 如果 gold solution 可解析为 LaTeX/数学表达式，则用相同解析器解析模型答案。
4. 用 `math_verify.verify(answer_parsed, gold_parsed)` 判断符号等价，正确给 `1.0`，错误给 `0.0`。
5. 如果 gold solution 不可解析，则抽取 `<answer>` 标签内容，做 lowercase/空白归一化后与 solution 精确匹配。

优点：

- 对数学答案更鲁棒，例如等价分数、表达式可被判为正确。
- 对医学/文本型答案有兜底逻辑。
- 奖励离散清晰，便于 R1-Zero 风格 RL。

潜在改进：

- 文本答案可以引入更强的语义相似度模型或 LLM-as-judge。
- 可以对部分正确推理设计过程奖励，但需要防止 reward hacking。

### 5.2 `format_reward`：格式奖励

正则：

```python
r"^<think>.*?</think><answer>.*?</answer>$"
```

满足格式给 `1.0`，否则 `0.0`。

它解决的问题：

- 强制模型输出结构化推理。
- 确保 `<answer>` 可抽取。
- 避免模型只输出最终答案或输出闲聊文本。

在简历中可表述为：设计格式约束奖励，提高 RL 训练中输出可解析性和自动评测稳定性。

### 5.3 `reasoning_steps_reward`：推理步骤奖励

检测如下模式：

- `Step 1:`、`Step 2:`
- 行首编号：`1.`、`2.`
- bullet：`-`、`*`
- 过渡词：`First,`、`Second,`、`Next,`、`Finally,`

奖励为：

```python
min(1.0, count / 3)
```

也就是检测到 3 个及以上推理步骤给满分。该奖励适合在训练早期增强分步推理倾向，但权重过大可能诱导模型机械堆步骤，因此建议和 accuracy reward 搭配使用。

### 5.4 `len_reward`：长度奖励

该奖励先判断每条 completion 是否正确，然后根据同一 batch 中 completion 的字符长度计算：

\[
\lambda = 0.5 - \frac{len - min\_len}{max\_len - min\_len}
\]

- 如果答案正确：reward = \(\lambda\)。
- 如果答案错误：reward = \(\min(0, \lambda)\)。

直观效果：

- 正确且更短的回答奖励更高。
- 错误回答不会因为短而获得正奖励。
- 可缓解长 CoT 空转问题。

### 5.5 `get_cosine_scaled_reward`：余弦长度缩放奖励

该函数根据答案是否正确选择不同奖励区间，再用 completion 长度计算余弦进度：

\[
progress = \frac{gen\_len}{max\_len}
\]

\[
cosine = \cos(progress \cdot \pi)
\]

最终奖励：

\[
reward = min\_value + 0.5(max\_value - min\_value)(1 + cosine)
\]

默认设计思路：

- 正确答案：短回答更接近 `max_value_correct`，长回答逐渐接近 `min_value_correct`。
- 错误答案：通过交换 wrong 区间，让错误答案的长度也受到不同程度惩罚。

相比 `len_reward`，余弦缩放更平滑，适合对长度进行连续控制。

### 5.6 `get_repetition_penalty_reward`：N-gram 重复惩罚

逻辑：

1. 将 completion 按空格分词。
2. 统计所有 N-gram 总数 `total` 与去重 N-gram 数 `len(ngrams)`。
3. 重复比例：

\[
scaling = 1 - \frac{|unique\_ngrams|}{total}
\]

4. 奖励：

\[
reward = scaling \cdot max\_penalty
\]

其中 `max_penalty` 必须小于等于 0。重复越多，负奖励越大。该奖励用于降低 RL 采样中常见的循环输出、重复句式和无意义 token 堆叠。

## 6. 自定义 Trainer 训练链路

### 6.1 初始化阶段

`BrenchGRPOTrainer` 初始化时完成：

- 加载 causal LM。
- 如果传入 PEFT 配置，则用 `get_peft_model` 包装模型。
- 如果开启 gradient checkpointing，则禁用 cache 并启用 checkpoint。
- 根据 `beta` 决定是否创建 reference model。
- 加载 tokenizer，并将 padding side 设置为 left。
- 校验 `reward_funcs` 与 `reward_weights` 数量一致。
- 设置 `num_generations`、`num_iterations`、`epsilon`、`use_vllm` 等关键参数。

### 6.2 Sampler：保证同一 prompt 成组出现

GRPO 需要同一个 prompt 重复采样多个 completion，所以 Trainer 中的 sampler 会按结构化方式重复 dataset index。这样一个 batch 内可以形成：

```text
prompt_A, prompt_A, prompt_A, prompt_A,
prompt_B, prompt_B, prompt_B, prompt_B,
...
```

之后 reward 可以 reshape 成 `(-1, num_generations)` 做组内均值/标准差。

### 6.3 生成阶段

Trainer 支持两条生成路径：

- **vLLM 路径**：用于高吞吐采样，适合长 completion 和多 generation。
- **Transformers generate 路径**：直接用当前模型生成，更简单但吞吐较低。

生成后会：

1. 截取 prompt 后面的 completion ids。
2. 在第一个 EOS 后 mask token，避免 padding 或 EOS 后 token 参与 loss。
3. 解码 completion 文本，封装成 conversational completion。

### 6.4 奖励计算与聚合

每个 reward function 返回一个长度为 batch size 的列表。Trainer 将它们组织成矩阵：

```text
rewards_per_func: [batch_size, num_reward_funcs]
```

如果配置了 `reward_weights`，则按权重加权求和：

```text
reward = sum(rewards_per_func[j] * reward_weights[j])
```

然后对 `reward.view(-1, num_generations)` 做组内归一化。

### 6.5 Loss 计算

`compute_loss` 中只对 completion token 计算 loss：

1. 拼接 prompt ids 与 completion ids。
2. 取 completion 部分 log-prob。
3. 计算 current/old policy ratio。
4. 计算 clipped policy loss。
5. 如 `beta != 0`，加入 KL。
6. 用 completion mask 做平均。

记录的关键指标包括：

- `completion_length`
- `rewards/<reward_func_name>`
- `reward`
- `reward_std`
- `clip_ratio`
- `kl`（如果启用）

## 7. 配置文件解读

以 `recipes/R1_Zero_0dot5B_config_v1.yaml` 为例：

### 7.1 模型参数

```yaml
model_name_or_path: /path/to/Qwen2.5-0.5B-Instruct
model_revision: main
torch_dtype: bfloat16
attn_implementation: flash_attention_2
```

含义：加载 Qwen2.5 instruct 模型，使用 bf16 和 FlashAttention 提升训练效率。

### 7.2 数据参数

```yaml
dataset_name: Brench/R1-Zero-GRPO-750
dataset_configs:
- train
```

训练脚本会从数据集中读取 `problem` 与 `solution` 字段，并构造成 prompt。

### 7.3 训练/显存参数

```yaml
bf16: true
use_vllm: true
vllm_gpu_memory_utilization: 0.10
gradient_checkpointing: true
per_device_train_batch_size: 4
num_generations: 4
max_prompt_length: 1024
max_completion_length: 4096
```

需要重点关注：

- `num_generations` 越大，组内比较越稳定，但生成成本越高。
- `max_completion_length` 决定 CoT 最大长度，对显存和吞吐影响很大。
- `gradient_checkpointing` 降低显存但增加计算。
- vLLM 与训练进程共用 GPU 时需谨慎设置显存占比。

### 7.4 优化参数

```yaml
learning_rate: 1.0e-05
lr_scheduler_type: cosine
num_train_epochs: 2
warmup_ratio: 0.1
```

RL 阶段通常学习率不宜过大，否则容易破坏指令模型能力或造成 reward collapse。

### 7.5 日志与保存

```yaml
report_to:
- wandb
logging_steps: 1
save_strategy: "steps"
save_steps: 200
```

建议在简历或项目展示中放出 W&B 曲线，例如：reward 上升、format reward 上升、clip ratio 稳定、completion length 收敛等。

## 8. 评测流程

`src/benchmark.py` 的评测步骤：

1. 加载 tokenizer。
2. 读取测试集，例如 `HuggingFaceH4/MATH-500`。
3. 将测试样本转换成与训练一致的 system/user prompt。
4. 使用 vLLM 批量生成 completion。
5. 用 `accuracy_answer_reward` 判断答案正确性。
6. 用正则计算 format reward。
7. 输出平均 `eval_acc` 和 `eval_format`。
8. 将每条样本写入 JSON，便于人工分析 case。

推荐额外做三类分析：

- **正确但格式错**：说明答案能力有，但格式奖励不足。
- **格式对但答案错**：说明模型学会外壳，但推理/计算能力不足。
- **长且错/重复**：说明需要加强长度或重复惩罚。

## 9. 可写入简历的技术点

### 9.1 一句话版本

基于 TRL/Accelerate/DeepSpeed/vLLM 复现 Qwen2.5 小模型 R1-Zero 风格 GRPO 强化学习训练框架，通过数学可验证奖励、格式奖励、长度控制和 N-gram 重复惩罚，构建从数据对话化到组内优势归一化策略优化的完整 LLM RL 训练闭环。

### 9.2 项目经历版本

- 搭建 Qwen2.5-Instruct 的 GRPO 强化学习训练管线，完成 Hugging Face 数据集加载、R1 格式 prompt 构造、vLLM 多 completion 采样、reward 聚合、advantage normalization、clipped policy loss 与 checkpoint 保存。
- 设计多维规则奖励体系：使用 `math-verify` 做 LaTeX/数学符号等价验证，结合 `<think>/<answer>` 格式奖励、推理步骤奖励、长度奖励和 N-gram 重复惩罚，提高模型推理答案正确性、输出可解析性与生成稳定性。
- 自定义 `BrenchGRPOTrainer`，支持 PEFT/LoRA、DeepSpeed ZeRO-3、bf16、gradient checkpointing、reference KL、W&B 日志和 MATH-500 vLLM 评测，沉淀可复现实验配置与自动化评测脚本。

### 9.3 面试可展开点

**Q：GRPO 和 PPO 的主要区别是什么？**

A：PPO 通常依赖 value model 估计 advantage，而 GRPO 对同一 prompt 采样多个 completion，通过组内 reward 标准化得到相对 advantage，省去了 value model，降低显存和训练复杂度。它尤其适合数学题这类 reward 可直接验证的任务。

**Q：为什么需要同一个 prompt 生成多个 completion？**

A：因为 GRPO 的 advantage 来自组内比较。如果每个 prompt 只有一个 completion，就无法判断它相对同题其他解法是好还是差。多个 completion 也能鼓励模型探索不同推理路径。

**Q：为什么 accuracy reward 之外还要 format reward？**

A：只用 accuracy reward 时，模型可能输出不可解析格式，导致训练和评测不稳定。format reward 明确约束 `<think>/<answer>` 结构，让最终答案可抽取，也让推理过程更可展示。

**Q：如何防止模型为了拿奖励而无限变长？**

A：本项目提供 `len_reward`、`get_cosine_scaled_reward` 和 `get_repetition_penalty_reward`。它们分别从正确答案长度、余弦长度衰减、重复 N-gram 比例三个角度抑制 overthinking、复读和无效长输出。

**Q：训练时重点看哪些指标？**

A：重点看总 reward、accuracy reward、format reward、completion length、reward std、clip ratio 和 KL。理想状态是 accuracy/format 上升，completion length 不无限增长，clip ratio 不长期过高，KL 在合理范围内。

## 10. 潜在优化方向

- **奖励权重网格搜索**：对 accuracy、format、length、repetition 设置不同权重组合，观察稳定性和最终评测。
- **课程学习**：先训练 format/简单数学，再逐渐加入更难题目。
- **更强文本答案评测**：医学问答可引入语义匹配模型或 LLM judge。
- **采样策略调参**：调节 temperature、top_p、num_generations，提高探索质量。
- **错误样本回流**：将评测中错误但高价值的样本加入下一轮 RL 数据。
- **消融实验**：分别去掉 format/length/repetition reward，验证各奖励贡献。

## 11. 端到端流程总结

```text
原始数据集(problem, solution)
        ↓
构造 system/user prompt，要求 <think>/<answer>
        ↓
同一 prompt 采样 num_generations 个 completion
        ↓
accuracy / format / reasoning / length / repetition 等奖励打分
        ↓
奖励加权求和，按 prompt group 计算 mean/std
        ↓
得到 normalized advantage
        ↓
计算 current policy 与 old policy ratio
        ↓
clipped GRPO loss + optional KL penalty
        ↓
反向传播更新 Qwen2.5 policy
        ↓
保存 checkpoint，使用 benchmark.py 做 vLLM 评测
```

这条链路就是本项目最核心的工程价值：把 R1-Zero 风格“可验证奖励驱动推理能力涌现”的思想，落到了可运行、可配置、可评测、可复盘的代码工程中。
