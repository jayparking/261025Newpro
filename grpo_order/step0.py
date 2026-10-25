"""Share verified initial evaluation, optionally using the canonical unadapted model."""
import copy
import os
from contextlib import nullcontext
from pathlib import Path

from .core import atomic_json, digest, isolated_rng, evaluation_batch_seed, read_json
from .grading import grader_identity
from .model import GENERATION_POLICY, adapter_hash, generate_batch, prompt_ids
from .storage import exclusive_lock


def verify_zero_adapter(model):
    """Verify the real tensors before disabling the exactly-zero LoRA contribution."""
    import torch
    all_params = list(model.named_parameters())
    if any(p.requires_grad and "lora_" not in name for name, p in all_params):
        raise ValueError("Canonical step-0 requires a frozen original model")
    params = [(name, p) for name, p in all_params if "lora_" in name]
    if not params or not any("lora_B" in name for name, _ in params):
        raise ValueError("Canonical step-0 needs actual LoRA A/B parameters")
    for name, p in params:
        if not any(part in name for part in ("lora_A", "lora_B")) or not bool(torch.isfinite(p).all()):
            raise ValueError("Canonical step-0 requires finite, LoRA-only trainable parameters")
        if "lora_B" in name and bool(torch.count_nonzero(p)):
            raise ValueError("Cannot share step-0 across seeds with nonzero LoRA B")


def cache_identity(model, tok, c, rows, split, contract):
    import torch
    initial = getattr(model, "grpo_initial_adapter_hash", None)
    if initial is None or adapter_hash(model) != initial:
        raise ValueError("Step-0 cache requires the verified, unchanged initial adapter")
    device = next(model.parameters()).device
    hardware = {"device": device.type, "torch_cuda": torch.version.cuda,
                "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                "capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None}
    e = c["evaluation"]
    canonical = e.get("step0_share_across_seeds", False)
    if canonical:
        verify_zero_adapter(model)
    first_batch = rows[:e.get("problem_batch_size", 1)]
    with isolated_rng(evaluation_batch_seed(c["seeds"]["evaluation"], split, [r["id"] for r in first_batch])):
        ids = [prompt_ids(tok, c, r["problem"]) for r in first_batch]
        samples = generate_batch(model, tok, ids, count=8, max_new_tokens=e["max_new_tokens"],
                                 temperature=e["temperature"], top_p=e["top_p"])
        greedy = generate_batch(model, tok, ids, count=1, max_new_tokens=e["max_new_tokens"],
                                temperature=1, top_p=1, greedy=True)
        witness = [r for sample, g in zip(samples, greedy) for r in sample + g]
    return {"schema": 2, "model": contract["model"],
            "initial_adapter": "verified_zero_B_disabled_adapter" if canonical else initial,
            "tokenizer": {"vocab": digest(tok.get_vocab()), "eos": tok.grpo_eos,
                          "chat_template": getattr(tok, "chat_template", None),
                          "backend": tok.backend_tokenizer.to_str() if hasattr(tok, "backend_tokenizer") else None},
            "rows": rows, "split": split, "prompt": c["prompt"],
            "prompt_ids": [prompt_ids(tok, c, row["problem"]) for row in rows],
            "evaluation_seed": c["seeds"]["evaluation"], "settings": e,
            "policy": GENERATION_POLICY, "runtime": c["runtime"], "hardware": hardware,
            "cublas_workspace": os.getenv("CUBLAS_WORKSPACE_CONFIG"),
            "dtype": sorted({str(p.dtype) for p in model.parameters()}),
            "lora": c["lora"], "code": contract["code"], "packages": contract["packages"],
            "grader": grader_identity(), "initial_output_witness": [r["token_ids"] for r in witness]}


def shared_evaluate(model, tok, c, rows, split, root, contract, evaluate_fn, evaluation_id):
    initial = getattr(model, "grpo_initial_adapter_hash", None)
    if initial is None or adapter_hash(model) != initial:
        raise ValueError("Step-0 evaluation requires the unchanged initial adapter")
    canonical = c["evaluation"].get("step0_share_across_seeds", False)
    if canonical:
        verify_zero_adapter(model)
    # Disabling a verified zero adapter gives every seed the SAME kernel path.
    # Base/RL retain separate original-model content hashes and never share.
    with model.disable_adapter() if canonical else nullcontext():
        return _shared_evaluate(model, tok, c, rows, split, root, contract, evaluate_fn, evaluation_id)


def _shared_evaluate(model, tok, c, rows, split, root, contract, evaluate_fn, evaluation_id):
    print(f"Step-0 {split}: verifying initial output before cache lookup", flush=True)
    identity = cache_identity(model, tok, c, rows, split, contract)
    key = digest(identity)
    cache = Path(c["output_dir"]) / "step0_cache" / key
    with exclusive_lock(cache.with_suffix(".lock"), wait=True):
        receipt = cache / "receipt.json"
        evaluation_tokens_this_attempt = 0
        if receipt.exists():
            saved = read_json(receipt)
            result = read_json(cache / "eval" / f"{split}_step-000000.json")
            if saved != {"identity": identity, "result_hash": digest(result)}:
                raise ValueError("Step-0 cache identity/checksum mismatch")
            hit = True
        else:
            complete_output_exists = (cache / "eval" / f"{split}_step-000000.json").exists()
            result = evaluate_fn(model, tok, c, rows, split, 0, cache,
                                 {"step0_cache": key,
                                  "canonical_initial_model": c["evaluation"].get("step0_share_across_seeds", False)})
            if not complete_output_exists:
                evaluation_tokens_this_attempt = result.get("generated_tokens_this_attempt", 0)
            atomic_json(receipt, {"identity": identity, "result_hash": digest(result)})
            hit = False
    result = copy.deepcopy(result)
    result.update(contract=contract, evaluation_id=evaluation_id, adapter_hash=adapter_hash(model),
                  cache={"key": key, "source": str(cache), "hit": hit,
                         "initial_output_witness_verified": True,
                         "sharing": "verified zero LoRA B, adapter disabled; shared across seeds per original model"
                         if c["evaluation"].get("step0_share_across_seeds", False)
                         else "identical model and initial adapter only"})
    result["source_evaluation_wall_seconds"] = result.pop("wall_seconds_this_attempt")
    result["generated_tokens_this_attempt"] = (sum(len(tokens) for tokens in identity["initial_output_witness"])
                                             + evaluation_tokens_this_attempt)
    for question in result["questions"]:
        question["evaluation_id"] = evaluation_id
    return result
