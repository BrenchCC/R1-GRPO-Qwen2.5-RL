"""Reward functions definition for GRPO training."""

import re
from typing import Dict
import os
from openai import OpenAI
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig,parse,verify

client = OpenAI(
    api_key='',
    base_url=''
)

def normalize_text(text):
    if text is None:
        return ""
    text = re.sub(r'\s+',' ',text.strip().lower())
    return text

def extract_answer(text):
    if text is None:
        return ""
    match = re.search(r'<answer>(.*?)</answer>',text,re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def evaluate_answer_similarity(answer, solution):
    return 1.0 if normalize_text(answer) == normalize_text(solution) else 0.0

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) != 0:
            answer_parsed = parse(
                content,
                extraction_config = [
                    LatexExtractionConfig(
                        normalization_config = NormalizationConfig(
                            nits = False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units = True,
                        )
                    )
                ],
                extraction_mode = "first_match",
            )

            reward = float(verify(answer_parsed, gold_parsed))
            print('#'*100)
            print('\nanswer_parsed:', answer_parsed, '\ngold_parsed:', gold_parsed, '\nreward:', reward)
        else :
            # For medical text answers, extract from <answer> tags and use GPT4O-mini for evaluation
            answer_content = extract_answer(content)
            normalized_content = normalize_text(answer_content)
            normalized_solution = normalize_text(sol)
            reward = evaluate_answer_similarity(normalized_content, normalized_solution)
            print('#' * 100)
            print('\nanswer_parsed:', normalized_content, '\ngold_parsed:', normalized_solution, '\nreward:', reward)
        rewards.append(reward)

        print('\naccuracy rewards:', rewards)

        return rewards

def accuracy_answer_reward(completion,answer,**kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    '''
    Input is completion string, and the answer is extracted gold answer
    '''
    gold_parsed = answer
    reward = 0.0
    if len(gold_parsed) != 0:
        answer_parsed = parse(
            completion,
            extraction_config = [
                LatexExtractionConfig(
                    normalization_config = NormalizationConfig(
                        nits = False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units = True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode = "first_match",
        )
        reward = float(verify(answer_parsed, gold_parsed))
        print('#' * 100)
        print('\nanswer_parsed:', answer_parsed,'\ngold_parsed:', gold_parsed, '\nreward:', reward)
    return reward

def format_reward(completions,**kwargs):
    """Reward function that checks if the completion has a specific format"""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern,content) for content in completion_contents]

    rewards = [1.0 if match else 0.0 for match in matches]

    print('#' * 100)
    print("\nformat rewards:",rewards)
    return rewards

def reasoning_steps_reward(completions,**kwargs):
    """
    Reward function that checks for clear step by step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3)for count in matches]

def len_reward(completions:list[Dict[str,str]], solutions:list[str],**kwargs) -> float:
    """
    Compute length-based rewards to discourage overthinking and promote token efficiency.
    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solutions: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check conrrectness of answers
    correctness = []
    for content, sol in zip(contents, solutions):
        gold_parsed = parse(
            sol,
            extraction_mode= "first_match",
            extraction_config= [LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable example
            correctness.append(True)
            print("Failed to parse gold solution from ",sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config = [
                LatexExtractionConfig(
                    normalization_config = NormalizationConfig(
                        nits = False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                ),
            ],
            extraction_mode = "first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    if max_len == min_len:
        return [0.0] * len(completions)
    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards
