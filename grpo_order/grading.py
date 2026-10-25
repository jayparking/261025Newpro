"""One binary math reward; extraction errors are observable, never bonus rewards."""
from functools import lru_cache

from .core import packages


class GraderEnvironmentError(RuntimeError):
    """Dependency/import/basic parser failures are never dataset exclusions."""


class GoldIssue(ValueError):
    def __init__(self, category, reason):
        super().__init__(reason)
        self.category = category


def grader_identity():
    versions = packages()
    return {"policy": "math_verify_no_fallback_v3_fail_on_grader_exception", "packages": {
        k: versions[k] for k in ("math-verify", "sympy", "latex2sympy2_extended", "antlr4-python3-runtime")}}


def check_grader():
    try:
        from math_verify import LatexExtractionConfig, parse, verify
        import antlr4  # Import failures must fail preparation, not exclude every row.
        left = parse(r"$\frac{1}{2}$", extraction_config=[LatexExtractionConfig()], fallback_mode="no_fallback")
        right = parse("$0.5$", extraction_config=[LatexExtractionConfig()], fallback_mode="no_fallback")
        if not left or not right or not verify(left, right):
            raise RuntimeError("basic fraction/decimal parser self-test failed")
    except Exception as exc:
        raise GraderEnvironmentError(f"grader installation/runtime failure: {type(exc).__name__}: {exc}") from exc
    return grader_identity()


def gold_parse(answer):
    if not isinstance(answer, str) or not answer.strip():
        raise GoldIssue("question_problem", "missing_or_empty_gold")
    return _gold_parse(answer)


@lru_cache(maxsize=8192)
def _gold_parse(answer):
    try:
        from math_verify import LatexExtractionConfig, parse
        parsed = parse("$" + answer.strip().strip("$") + "$",
                       extraction_config=[LatexExtractionConfig()], fallback_mode="no_fallback")
    except (ImportError, ModuleNotFoundError) as exc:
        raise GraderEnvironmentError(str(exc)) from exc
    except Exception as exc:
        raise GoldIssue("parser_limitation", f"parser_exception:{type(exc).__name__}:{exc}") from exc
    if not parsed:
        raise GoldIssue("parser_limitation", "no_symbolic_parse; requires review, not evidence of an invalid question")
    return parsed


def grade(response, answer):
    from math_verify import parse, verify
    gold = gold_parse(answer)
    try:
        extracted = parse(response, fallback_mode="no_fallback")
        if not extracted:
            return {"reward": 0, "extracted_answer": None, "failure_reason": "answer_extraction_failed"}
        correct = bool(verify(gold, extracted))
        return {"reward": int(correct), "extracted_answer": [str(x) for x in extracted],
                "failure_reason": None if correct else "incorrect_answer"}
    except Exception as e:
        raise GraderEnvironmentError(f"Unexpected grader failure; refusing to label it incorrect: {type(e).__name__}: {e}") from e
