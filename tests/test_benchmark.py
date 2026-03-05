import sys
import types
import unittest


# Stubs for optional dependencies imported via src.rewards
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

from src.benchmark import _pick_first_existing


class BenchmarkHelpersTests(unittest.TestCase):
    def test_pick_first_existing_prefers_first_present_key(self):
        example = {"problem": "p", "question": "q"}
        self.assertEqual(_pick_first_existing(example, ["content", "problem", "question"]), "p")


    def test_pick_first_existing_skips_none_and_uses_next(self):
        example = {"content": None, "question": "q"}
        self.assertEqual(_pick_first_existing(example, ["content", "question"]), "q")

    def test_pick_first_existing_requires_mapping(self):
        with self.assertRaises(TypeError):
            _pick_first_existing([("content", "x")], ["content"])

    def test_pick_first_existing_raises_if_missing(self):
        with self.assertRaises(KeyError):
            _pick_first_existing({}, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
