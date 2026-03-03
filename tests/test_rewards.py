import sys
import types
import unittest
from unittest.mock import patch


# Provide lightweight stubs for optional third-party dependencies so this unit test
# can run in minimal CI environments.
latex_stub = types.ModuleType("latex2sympy2_extended")
class _NormalizationConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

latex_stub.NormalizationConfig = _NormalizationConfig
sys.modules.setdefault("latex2sympy2_extended", latex_stub)

math_verify_stub = types.ModuleType("math_verify")
class _LatexExtractionConfig:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

math_verify_stub.LatexExtractionConfig = _LatexExtractionConfig
math_verify_stub.parse = lambda *args, **kwargs: []
math_verify_stub.verify = lambda *args, **kwargs: False
sys.modules.setdefault("math_verify", math_verify_stub)

from src import rewards


class RewardsTests(unittest.TestCase):
    def test_extract_answer_prefers_answer_tag(self):
        text = "<think>x</think><answer>42</answer>"
        self.assertEqual(rewards.extract_answer(text), "42")

    def test_accuracy_reward_processes_full_batch(self):
        completions = [
            [{"content": "resp-a"}],
            [{"content": "resp-b"}],
            [{"content": "resp-c"}],
        ]
        solutions = ["sol-a", "sol-b", "sol-c"]

        with patch("src.rewards.parse", side_effect=[["g1"], ["a1"], ["g2"], ["a2"], ["g3"], ["a3"]]), patch(
            "src.rewards.verify", side_effect=[True, False, True]
        ):
            output = rewards.accuracy_reward(completions, solutions)

        self.assertEqual(len(output), 3)
        self.assertEqual(output, [1.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
