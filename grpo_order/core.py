import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import tempfile

from .sources import LEVEL_MAPPING, MATH500_ID, MATH_ID, MODEL_IDS

DIFFICULTIES = ("Easy", "Medium", "Hard")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def packages():
    names = ("torch", "transformers", "peft", "accelerate", "numpy", "matplotlib", "math-verify", "huggingface-hub", "tokenizers", "safetensors", "sympy", "latex2sympy2_extended", "antlr4-python3-runtime")
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def load_config(path):
    c = read_json(path)
    for role, spec in c["models"].items():
        if spec.get("hub_id") and spec["hub_id"] != MODEL_IDS.get(role):
            raise ValueError(f"Model substitution forbidden for {role}: expected {MODEL_IDS.get(role)}")
        if spec.get("tokenizer_revision") and spec["tokenizer_revision"] != spec.get("revision"):
            raise ValueError("Model and tokenizer revisions must name the same snapshot")
    if "assets" in c:
        if c["assets"]["datasets"]["math"]["hub_id"] != MATH_ID or c["assets"]["datasets"]["math500"]["hub_id"] != MATH500_ID:
            raise ValueError("Dataset substitution forbidden for this experiment")
        if c["data"]["difficulty_mapping"] != LEVEL_MAPPING or c["data"]["difficulty_field"] != "level":
            raise ValueError("Use original level: Easy=1-2, Medium=3, Hard=4-5")
    t = c["train"]
    fixed = {"optimizer": "AdamW", "scheduler": "constant", "warmup": 0, "epochs": 1,
             "responses_per_problem": 4, "problems_per_update": 1, "group_reuse": 1,
             "top_p": 1.0, "kl_beta": 0.0, "advantage_std": "population",
             "loss": "grpo_response_mean_token_mean",
             "truncation": "grade_partial_include_all_generated_tokens", "precision": "bf16"}
    for key, value in fixed.items():
        if t[key] != value:
            raise ValueError(f"Unsupported protocol change train.{key}: expected {value!r}")
    checks = {
        "positive train temperature/advantage epsilon": t["temperature"] > 0 and t["advantage_epsilon"] > 0,
        "response microbatch 1/2/4": t["response_microbatch"] in (1, 2, 4),
        "checkpoints 0/100/200/300": t["checkpoints"] == [0, 100, 200, 300],
        "100 training problems per difficulty": c["data"]["train_per_difficulty"] == 100,
        "positive diagnostic target": c["data"]["diagnostic_per_difficulty"] > 0,
        "dedup threshold (0,1]": 0 < c["data"]["near_duplicate_threshold"] <= 1,
        "eight samples plus greedy": c["evaluation"]["samples"] == 8 and c["evaluation"]["greedy"],
        "fixed evaluation groups": c["evaluation"]["mixed_groups"] == [[0, 1, 2, 3], [4, 5, 6, 7]],
        "diagnostic schedule 0/300 or 0/100/200/300": c["evaluation"]["diagnostic_steps"] in ([0, 300], [0, 100, 200, 300]),
        "MATH500 schedule": c["evaluation"]["math500_steps"] == [0, 300],
        "LoRA dropout/initialization": c["lora"]["dropout"] == 0 and c["lora"]["init"] == "kaiming_uniform_A_zero_B",
        "offline models without remote code": c["runtime"]["local_files_only"] and not c["runtime"]["trust_remote_code"],
        "one or two allocated CUDA slots": c["runtime"]["allocated_gpu_slots"] in ([0], [0, 1]),
        "scoped top-p determinism policy": c["runtime"]["top_p_determinism"] == "scoped_warn_only",
        "deterministic baseline": c["runtime"]["deterministic_algorithms"] is True,
        "validation-only pilot": c["pilot"]["selection_split"] == "validation",
        "batched generation": c["runtime"]["generation_batching"] == "one_problem_num_return_sequences",
        "supported attention backend": c["runtime"].get("attention_implementation", "eager") in ("eager", "sdpa"),
        "positive evaluation problem batch size": type(c["evaluation"].get("problem_batch_size", 1)) is int
            and 1 <= c["evaluation"].get("problem_batch_size", 1) <= 32,
        "step0 seed sharing flag": type(c["evaluation"].get("step0_share_across_seeds", False)) is bool,
        "pilot updates 0..300": isinstance(c["pilot"]["updates"], int) and 0 <= c["pilot"]["updates"] <= 300,
        "training seeds [1] or [1,2,3]": c["seeds"]["training"] in ([1], [1, 2, 3]),
        "positive output budgets": t["max_new_tokens"] > 0 and c["evaluation"]["max_new_tokens"] > 0,
        "valid evaluation sampling": c["evaluation"]["temperature"] > 0 and 0 < c["evaluation"]["top_p"] <= 1,
        "valid optimizer values": t["lr"] > 0 and t["epsilon"] > 0 and t["max_grad_norm"] > 0
            and len(t["betas"]) == 2 and all(0 <= b < 1 for b in t["betas"]),
    }
    for name, valid in checks.items():
        if not valid:
            raise ValueError("Invalid protocol setting: " + name)
    return c


def seed_plan(c, seed):
    return {"training": seed, "data_selection": c["data"]["selection_seed"],
            "order": c["seeds"]["order_base"] + seed,
            "lora": c["seeds"]["lora_base"] + seed,
            "generation": c["seeds"]["generation_base"] + seed,
            "evaluation": c["seeds"]["evaluation"]}


def ordered_problems(rows, order, seed):
    rng = random.Random(seed)
    if order == "random":
        out = sorted(rows, key=lambda r: r["id"])
        rng.shuffle(out)
    elif order == "curriculum":
        out = []
        for difficulty in DIFFICULTIES:
            group = sorted((r for r in rows if r["difficulty"] == difficulty), key=lambda r: r["id"])
            rng.shuffle(group)
            out.extend(group)
    else:
        raise ValueError(order)
    assert len(out) == len(rows) == len({r["id"] for r in out})
    return out


def advantages(rewards, epsilon=1e-8):
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    adv = [(r - mean) / (std + epsilon) for r in rewards]
    kind = "all_wrong" if mean == 0 else "all_correct" if mean == 1 else "mixed"
    return mean, std, adv, kind


def question_metrics(rewards, greedy_reward):
    if len(rewards) != 8:
        raise ValueError("Expected eight sampled responses for ONE question")
    kinds = [advantages(rewards[:4])[3], advantages(rewards[4:])[3]]
    return {"avg_at_8": sum(rewards) / 8, "pass_at_8": float(any(rewards)),
            "greedy": float(greedy_reward),
            **{k + "_fraction": kinds.count(k) / 2 for k in ("mixed", "all_wrong", "all_correct")}}


def capture_rng():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def isolated_rng(seed):
    state = capture_rng()
    try:
        seed_all(seed)
        yield
    finally:
        restore_rng(state)


def per_question_seed(base, split, question_id):
    return int(digest([base, split, question_id])[:8], 16)


def evaluation_batch_seed(base, split, question_ids):
    if len(question_ids) == 1:
        return per_question_seed(base, split, question_ids[0])
    return int(digest(["evaluation_batch_v1", base, split, question_ids])[:8], 16)


def code_hash():
    root = Path(__file__).resolve().parents[1]
    files = list((root / "grpo_order").glob("*.py")) + list((root / "scripts").glob("*.py"))
    files += list((root / "scripts").glob("*.sh")) + list(root.glob("requirements*.txt"))
    files += list(root.glob("*.sh"))
    files += list((root / "poc_Test").glob("*.py")) + list((root / "poc_Test").glob("*.sh"))
    return digest({str(p.relative_to(root)): file_hash(p) for p in sorted(files)})
