import re
import torch
from datasets import load_dataset,Dataset
from transformers import AutoTokenizer,AutoModelForCausalLM

from trl import GRPOTrainer,GRPOConfig


system_prompt = """你的名字是Brench-AI，是由Brench创造出的深度推理AI助手,专注于各种推理问题的解答和分析，拥有强大的推理能力和分析能力以及反思能力，可以帮助用户解决各种推理性问题。
Your name is Brench-AI, a deep reasoning AI assistant created by Brench, focusing on the answer and analysis of various reasoning problems. You focus on the solution and analysis of various reasoning problems. At the same time, you have strong reasoning, analytical and reflective abilities, which can help users solve various reasoning problems.
Please respond reasoning question in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

cot_format = """
<think>
{think_content}
</think>
<answer>
{answer_content}
</answer>
"""

def extract_tag_answer(content: str) -> str:
    answer = content.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()
def extract_data_answer(content:str) -> str:
    idx = content.rfind(r"\boxed")
    if idx < 0:
        return None
    
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(content):
        if content[i] == '{':
            num_left_braces_open += 1
        if content[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx is None:
        boxed_ans = None
    else:
        boxed_ans = content[idx:right_brace_idx+1]
    
    return boxed_ans

def remove_boxed(s:str) -> str:
    left = r"\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except Exception:
        return None

def extract_boxed_answer(content:str) -> str:
    boxed_ans = extract_data_answer(content)
    if boxed_ans is None:
        return None
    answer = remove_boxed(boxed_ans)
    if answer is None:
        return None
    return answer


def get_logic_questions() -> Dataset:
    data  = load_dataset("parquet",data_files="./data/R1-Zero-GRPO-750/data/train-00000-of-00001.parquet")['train']
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': x['problem']}
        ],
        'answer': extract_boxed_answer(x['solution'])
    })
    return data


dataset = get_logic_questions()

print(dataset[99])

def correctness_reward_func(prompts,completions,answer,**kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    query = prompts[0][-1]['content']
    extracted_responses = [extract_tag_answer(r) for r in responses]
    recording_item = {
        'Question': query,
        'Answer': answer[0],
        'Response': responses[0],
        'Extracted': extracted_responses[0]
    }
    print('-'*20, f"Question:\n{query}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if res == ans else 0.0 for res, ans in zip(extracted_responses, answer)]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_cot(content) -> float:
    count = 0.0
    if content.count("<think>\n") == 1:
        count += 0.125
    if content.count("\n</think>\n") == 1:
        count += 0.125
    if content.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(content.split("\n</answer>\n")[-1])*0.001
    if content.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(content.split("\n</answer>")[-1]) - 1)*0.001
    return count

def cotcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_cot(c) for c in contents]

model_path = "./models/Qwen2.5-1.5B-Instruct"

output_dir = "./outputs/Qwen2.5-1.5B-R1-GRPO-DEMO"
run_name = "Qwen2.5-1.5B-R1-GRPO-DEMO-TEST"

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name = run_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    num_generations=4,
    max_prompt_length=1024,
    max_completion_length=8192,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=False,
    vllm_gpu_memory_utilization=.3,
    vllm_device="cuda:0",
    report_to="wandb",
    
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=None
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token