"""Conservative step-0 sharing: identical initial adapter bytes AND output witness.

Different LoRA seeds do not share, even with zero B. This avoids assuming that
different adapters have identical numerical outputs on every input.
"""
import copy
import os
from pathlib import Path

from .core import atomic_json, digest, isolated_rng, per_question_seed, read_json
from .grading import grader_identity
from .model import GENERATION_POLICY, adapter_hash, generate, prompt_ids
from .storage import exclusive_lock


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
    # Validate actual initial sampled + greedy outputs as well as the complete
    # model/adapter identity. Batch/seed are exactly the evaluation policy.
    with isolated_rng(per_question_seed(c["seeds"]["evaluation"], split, rows[0]["id"])):
        ids = prompt_ids(tok, c, rows[0]["problem"])
        witness = generate(model, tok, ids, count=8, max_new_tokens=e["max_new_tokens"],
                           temperature=e["temperature"], top_p=e["top_p"])
        witness += generate(model, tok, ids, count=1, max_new_tokens=e["max_new_tokens"],
                            temperature=1, top_p=1, greedy=True)
    return {"schema": 1, "model": contract["model"], "initial_adapter": initial,
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
    identity = cache_identity(model, tok, c, rows, split, contract)
    key = digest(identity)
    cache = Path(c["output_dir"]) / "step0_cache" / key
    with exclusive_lock(cache.with_suffix(".lock"), wait=True):
        receipt = cache / "receipt.json"
        if receipt.exists():
            saved = read_json(receipt)
            result = read_json(cache / "eval" / f"{split}_step-000000.json")
            if saved != {"identity": identity, "result_hash": digest(result)}:
                raise ValueError("Step-0 cache identity/checksum mismatch")
            hit = True
        else:
            result = evaluate_fn(model, tok, c, rows, split, 0, cache, {"step0_cache": key})
            atomic_json(receipt, {"identity": identity, "result_hash": digest(result)})
            hit = False
    result = copy.deepcopy(result)
    result.update(contract=contract, evaluation_id=evaluation_id,
                  cache={"key": key, "source": str(cache), "hit": hit,
                         "initial_output_witness_verified": True,
                         "sharing": "identical model and initial adapter only; different LoRA seeds do not share"})
    result["source_evaluation_wall_seconds"] = result.pop("wall_seconds_this_attempt")
    result["generated_tokens_this_attempt"] = sum(len(tokens) for tokens in identity["initial_output_witness"]) + (
        0 if hit else result.get("generated_tokens_this_attempt", 0))
    for question in result["questions"]:
        question["evaluation_id"] = evaluation_id
    return result
