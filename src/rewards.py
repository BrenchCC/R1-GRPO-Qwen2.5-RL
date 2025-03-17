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