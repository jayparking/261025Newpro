import time
from pathlib import Path

from .core import atomic_json, digest, isolated_rng, per_question_seed, question_metrics, read_json
from .grading import grade
from .model import GENERATION_POLICY, adapter_hash, generate, prompt_ids
from .step0 import shared_evaluate
from .data import origin_metadata


def summarize(questions):
    keys = ("avg_at_8", "pass_at_8", "greedy", "mixed_fraction", "all_wrong_fraction", "all_correct_fraction",
            "sampled_truncation_fraction", "sampled_mean_tokens", "greedy_truncation", "greedy_tokens")
    return {"questions": len(questions), **{k: sum(q[k] for q in questions) / len(questions) for k in keys}}


def evaluate(model, tok, config, rows, split, step, root, contract, *, subset=None):
    # Enclose grading, filesystem recovery and all other evaluation work as well.
    with isolated_rng(config["seeds"]["evaluation"]):
        if contract.get("mode") in ("smoke", "pilot") and split == "math500":
            raise ValueError("MATH500 is reserved for frozen full runs, never pilot tuning")
        selected = rows if subset is None else rows[:subset]
        dst = Path(root) / "eval" / f"{split}_step-{step:06d}.json"
        if step == 0 and config["evaluation"].get("step0_cache", False) and "model" in contract and not dst.exists():
            started = time.monotonic()
            result = shared_evaluate(model, tok, config, selected, split, root, contract, _evaluate,
                                     identity(config, selected, split, step, contract))
            result["wall_seconds_this_attempt"] = time.monotonic() - started
            atomic_json(dst, result)
            return result
        return _evaluate(model, tok, config, rows, split, step, root, contract, subset=subset)


def identity(config, rows, split, step, contract):
    return digest({"contract": contract, "split": split, "step": step, "rows": rows,
                   "prompt": config["prompt"], "seed": config["seeds"]["evaluation"],
                   "settings": config["evaluation"], "generation_policy": GENERATION_POLICY})


def _evaluate(model, tok, config, rows, split, step, root, contract, *, subset=None):
    root = Path(root)
    dst = root / "eval" / f"{split}_step-{step:06d}.json"
    rows = rows if subset is None else rows[:subset]
    evaluation_id = identity(config, rows, split, step, contract)
    if dst.exists():
        result = read_json(dst)
        if result["evaluation_id"] != evaluation_id or result["adapter_hash"] != adapter_hash(model):
            raise ValueError(f"Evaluation output already exists with different inputs: {dst}")
        return result
    import torch
    before = adapter_hash(model)
    was_training = model.training
    e = config["evaluation"]
    started = time.monotonic()
    records = []
    generated_this_attempt = 0
    # Resumable per-question files: evaluation interruption never repeats completed questions.
    qdir = dst.with_suffix("")
    qdir.mkdir(parents=True, exist_ok=True)
    try:
        for row in rows:
            qpath = qdir / (digest(row["id"]) + ".json")
            if qpath.exists():
                q = read_json(qpath)
                if q["evaluation_id"] != evaluation_id:
                    raise ValueError("Stale partial evaluation")
            else:
                seed = per_question_seed(config["seeds"]["evaluation"], split, row["id"])
                with isolated_rng(seed), torch.inference_mode():
                    ids = prompt_ids(tok, config, row["problem"])
                    sampled = generate(model, tok, ids, count=8, max_new_tokens=e["max_new_tokens"],
                                       temperature=e["temperature"], top_p=e["top_p"])
                    greedy = generate(model, tok, ids, count=1, max_new_tokens=e["max_new_tokens"],
                                      temperature=1, top_p=1, greedy=True)[0]
                for response in sampled + [greedy]:
                    response.update(grade(response["text"], row["answer"]))
                    generated_this_attempt += response["generated_tokens"]
                q = {"evaluation_id": evaluation_id, "id": row["id"], "difficulty": row["difficulty"],
                     "origin": origin_metadata(row),
                     "evaluation_seed": seed, "sampled": sampled, "greedy_response": greedy,
                     **question_metrics([r["reward"] for r in sampled], greedy["reward"]),
                     "sampled_truncation_fraction": sum(r["truncated"] for r in sampled) / 8,
                     "sampled_mean_tokens": sum(r["generated_tokens"] for r in sampled) / 8,
                     "greedy_truncation": float(greedy["truncated"]), "greedy_tokens": greedy["generated_tokens"]}
                atomic_json(qpath, q)
            records.append(q)
    finally:
        model.train(was_training)
    if adapter_hash(model) != before:
        raise AssertionError("Evaluation modified LoRA weights")
    by_difficulty = {d: summarize([r for r in records if r["difficulty"] == d])
                     for d in sorted({r["difficulty"] for r in records})}
    result = {"evaluation_id": evaluation_id, "contract": contract, "split": split, "step": step,
              "generation_policy": GENERATION_POLICY, "eos": tok.grpo_eos,
              "adapter_hash": before, "settings": e, "metrics": summarize(records),
              "by_difficulty": by_difficulty, "questions": records,
              "generated_tokens": sum(sum(r["generated_tokens"] for r in q["sampled"]) + q["greedy_tokens"] for q in records),
              "generated_tokens_this_attempt": generated_this_attempt,
              "wall_seconds_this_attempt": time.monotonic() - started,
              "note": "Two fixed groups per question are averaged within question, never counted as independent questions"}
    atomic_json(dst, result)
    return result
