"""Confirmed sources. Paths are configured at the TOP of configs/experiment.json."""

# No Math/Instruct/7B substitutions are allowed for this experiment.
MODEL_IDS = {
    "base": "Qwen/Qwen2.5-1.5B",
    "rl": "hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo",
}
MATH_ID = "EleutherAI/hendrycks_math"
MATH500_ID = "HuggingFaceH4/MATH-500"
SUBJECTS = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
)
LEVEL_MAPPING = {"Level 1": "Easy", "Level 2": "Easy", "Level 3": "Medium",
                 "Level 4": "Hard", "Level 5": "Hard"}
LINEAGE_SOURCES = [
    "https://huggingface.co/Qwen/Qwen2.5-1.5B",
    "https://huggingface.co/hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo",
    "https://github.com/hkust-nlp/simpleRL-reason/tree/v1",
    "https://arxiv.org/html/2503.18892v3",
]
