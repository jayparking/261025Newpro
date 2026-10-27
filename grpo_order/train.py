import math
import os
from pathlib import Path
import time

from .core import (DIFFICULTIES, advantages, atomic_json, capture_rng, code_hash, digest,
                   ordered_problems, packages, read_json, restore_rng, seed_all, seed_plan)
from .data import load_prepared, origin_metadata
from .evaluation import evaluate
from .freeze import verify_freeze
from .grading import gold_parse, grade
from .model import GENERATION_POLICY, adapter_hash, adapter_norms, generate, load_model, load_tokenizer, prompt_ids, validate_context
from .objective import accumulate_group
from .integrity import validate_completion
from .preflight import dependency_errors, gpu_errors, hardware, resolve_model
from .presmoke import select_one
from .storage import (checkpoints, exclusive_lock, load_checkpoint, reconcile_event,
                      run_path, save_checkpoint)
from poc_Test.protocol import select_splits as select_poc_splits, validate_config as validate_poc_config


def build_contract(c, manifest, model_meta, role, order, seed, mode, rows):
    return {"config": digest(c), "data": manifest["hash"], "model": model_meta["content_hash"],
            "model_revision": model_meta.get("resolved_model_revision"),
            "tokenizer_revision": model_meta.get("resolved_tokenizer_revision"),
            "model_card": model_meta["model_card_hash"], "code": code_hash(), "packages": packages(),
            "role": role, "order": order, "seed": seed, "mode": mode,
            "sequence": digest([r["id"] for r in rows])}


def check_pair_initialization(root, role, seed, mode, c, contract, lora_meta):
    path = Path(c["output_dir"]) / mode / "paired_initialization" / f"{role}_seed{seed}.json"
    expected = {"initial_adapter_hash": lora_meta["initial_adapter_hash"],
                "model_hash": contract["model"], "config_hash": contract["config"],
                "targets": lora_meta["targets"], "data_hash": contract["data"], "code_hash": contract["code"]}
    with exclusive_lock(path.with_suffix(".lock"), wait=True):
        if path.exists() and read_json(path) != expected:
            raise ValueError("Random/Curriculum initial state mismatch")
        if not path.exists():
            atomic_json(path, expected)


def train(c, role, order, seed, mode="pilot", resume=False, stop_after=None):
    if mode == "poc":
        validate_poc_config(c)
        if role not in ("base", "rl") or (order, seed) != ("random", 1):
            raise ValueError("POC uses Base/RL with the same random order and seed 1")
    elif "poc" in c:
        raise ValueError("A POC config cannot be used for main-experiment training")
    if mode == "presmoke" and (role, order, seed) != ("base", "random", 1):
        raise ValueError("presmoke runs only Base/random/seed1; use smoke for four conditions")
    if stop_after is not None and (mode != "smoke" or stop_after not in (1, 2)):
        raise ValueError("--stop-after is only for smoke at exposure 1 or 2")
    validate_assignment(order, c)
    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise ValueError("DDP is not supported: launch one independent process per physical GPU")
    if mode == "full":
        verify_freeze(c)
    root = run_path(c, mode, role, order, seed)
    if root.exists() and not resume:
        raise FileExistsError(f"Run exists: {root}; use --resume or a new output_dir")
    h = check_environment(c)
    uuid = h["uuid_mapping"][0]["uuid"]
    gpu_lock = Path("/tmp") / f"grpo-order-{os.getuid()}-{uuid}.lock"
    with exclusive_lock(root.parent / (root.name + ".lock")), exclusive_lock(gpu_lock):
        if root.exists() and not resume:
            raise FileExistsError(root)
        root.mkdir(parents=True, exist_ok=True)
        try:
            return _train(c, role, order, seed, mode, resume, root, h, stop_after)
        except BaseException as e:
            atomic_json(root / "failure.json", {"error": repr(e), "time": time.time()})
            raise


def validate_assignment(order, config):
    single_gpu = config["runtime"]["allocated_gpu_slots"] == [0]
    slot = "0" if single_gpu or order == "random" else "1"
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if os.getenv("GRPO_ALLOCATED_SLOT", slot) != slot or not visible or "," in visible:
        raise ValueError(f"{order} requires allocated slot {slot} and exactly one visible GPU; use launch")
    if not single_gpu and visible.isdecimal() and visible != slot:
        raise ValueError(f"{order} requires allocated slot {slot}; prefer launch for UUID mapping")
    return slot


def configure_torch(c):
    """Same numerical settings for training and for saved-checkpoint evaluation."""
    import torch
    torch.use_deterministic_algorithms(c["runtime"]["deterministic_algorithms"], warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = c["runtime"]["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = c["runtime"]["allow_tf32"]


def check_environment(c):
    h = hardware()
    errors = gpu_errors(h, c, single=True) + dependency_errors()
    if errors:
        raise RuntimeError("; ".join(errors))
    return h


def select_rows(splits, order, plan, mode, c=None):
    rows = splits["train"]
    if mode == "presmoke":
        return select_one(rows)
    if mode == "smoke":
        # Same 3 problems for both orders; smoke is never included in scientific analysis.
        rows = [next(r for r in rows if r["difficulty"] == d) for d in DIFFICULTIES]
    elif mode == "pilot":
        # Select a common balanced prefix BEFORE ordering, including step-0-only pilots.
        groups = [sorted((r for r in rows if r["difficulty"] == d), key=lambda r: r["id"]) for d in DIFFICULTIES]
        rows = [row for triple in zip(*groups) for row in triple][:c["pilot"]["updates"]]
    return ordered_problems(rows, order, plan["order"])


def open_run(c, role, order, seed, mode):
    """Resolve original model, prepared data, problem order and identity contract of one run."""
    spec = c["models"][role]
    if not spec["role_verified"]:
        raise ValueError(f"Verify models.{role}.role_verified from model provenance first")
    path, model_meta = resolve_model(spec)
    if mode == "full":
        verify_freeze(c, {role: model_meta})
    if Path(c["output_dir"]).resolve().is_relative_to(path.resolve()):
        raise ValueError("output_dir must be outside the immutable original model directory")
    manifest, splits = load_prepared(c)
    if mode == "poc":
        splits = select_poc_splits(c, splits)
    if mode == "presmoke":
        # Validate and tokenize only the two questions actually used by this check.
        splits = {name: select_one(splits[name]) for name in ("train", "diagnostic")}
        if splits["train"][0]["id"] == splits["diagnostic"][0]["id"]:
            raise ValueError("presmoke train and diagnostic questions must be distinct")
    for rows in splits.values():
        for row in rows:
            gold_parse(row["answer"])
    plan = seed_plan(c, seed)
    rows = select_rows(splits, order, plan, mode, c)
    contract = build_contract(c, manifest, model_meta, role, order, seed, mode, rows)
    return path, model_meta, manifest, splits, plan, rows, contract


def _train(c, role, order, seed, mode, resume, root, h, stop_after=None):
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    configure_torch(c)
    path, model_meta, manifest, splits, plan, rows, contract = open_run(c, role, order, seed, mode)
    metadata_path = root / "metadata.json"
    if metadata_path.exists() and read_json(metadata_path)["contract"] != contract:
        raise ValueError("Existing run configuration/data/model/code/package identity changed; refusing resume")
    if (root / "complete.json").exists():
        validate_completion(read_json(metadata_path), read_json(root / "complete.json"), len(rows))
        print(f"Already complete and identity verified: {root}", flush=True)
        return read_json(root / "complete.json")
    tok, context = load_tokenizer(path, c["models"][role])
    length_report = validate_context(tok, context, c, splits)
    model, lora_meta = load_model(path, c, plan["lora"])
    check_pair_initialization(root, role, seed, mode, c, contract, lora_meta)
    if not metadata_path.exists():
        atomic_json(metadata_path, {"contract": contract, "config": c, "seeds": plan,
                                   "hardware": h, "model": model_meta, "lora": lora_meta,
                                   "generation_policy": GENERATION_POLICY, "eos": tok.grpo_eos,
                                   "context_validation": length_report,
                                   "expected_exposures": len(rows), "expected_responses": 4 * len(rows),
                                   "expected_optimizer_updates": len(rows),
                                   "sequence": [{"id": r["id"], "difficulty": r["difficulty"]} for r in rows]})
    t = c["train"]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=t["lr"],
                                 betas=tuple(t["betas"]), eps=t["epsilon"], weight_decay=t["weight_decay"],
                                 foreach=False, fused=False)
    seed_all(plan["generation"])
    state = {"contract": contract, "cursor": 0, "optimizer_steps": 0, "skipped_steps": 0,
             "generated_responses": 0, "generated_tokens": 0, "training_seconds": 0.0,
             "group_counts": {"all_wrong": 0, "mixed": 0, "all_correct": 0}, "last_record": None}
    saved = checkpoints(root)
    if saved:
        if not resume:
            raise ValueError("Checkpoint exists; explicit --resume required")
        state = load_checkpoint(saved[-1], contract)
        set_peft_model_state_dict(model, state["adapter"])
        if adapter_hash(model) != state["adapter_hash"]:
            raise ValueError("Restored adapter checksum mismatch")
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        reconcile_event(root, state)
        if state["skipped_steps"]:
            raise ValueError("Checkpoint contains skipped updates; refusing to continue an invalid experiment")
    milestones = sorted(set([0, len(rows)] + ([] if mode in ("presmoke", "smoke") else t["checkpoints"])))

    def commit():
        state.update(adapter={k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()},
                     adapter_hash=adapter_hash(model), optimizer=optimizer.state_dict(), rng=capture_rng())
        return save_checkpoint(root, state, milestones)

    def evaluate_due(step):
        if mode == "poc":
            if step == len(rows):
                destination = root / "eval" / f"validation_step-{step:06d}.json"
                already_evaluated = destination.exists()
                torch.cuda.reset_peak_memory_stats()
                result = evaluate(model, tok, c, splits["validation"], "validation", step, root, contract)
                if not already_evaluated:
                    result["poc_gpu_memory"] = {
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    }
                    atomic_json(destination, result)
            return
        if mode == "presmoke":
            if step == len(rows):
                evaluate(model, tok, c, splits["diagnostic"], "diagnostic", step, root, contract, subset=1)
            return
        if mode == "pilot":
            if step in c["evaluation"]["diagnostic_steps"] or step == len(rows):
                evaluate(model, tok, c, splits["validation"], "validation", step, root, contract)
            return
        if step in ([0, len(rows)] if mode == "smoke" else c["evaluation"]["diagnostic_steps"]):
            evaluate(model, tok, c, splits["diagnostic"], "diagnostic", step, root, contract,
                     subset=3 if mode == "smoke" else None)
        if mode == "full" and step in c["evaluation"]["math500_steps"]:
            evaluate(model, tok, c, splits["math500"], "math500", step, root, contract)

    if not saved:
        commit()
    evaluate_due(state["cursor"])
    for index in range(state["cursor"], len(rows)):
        row = rows[index]
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        ids = prompt_ids(tok, c, row["problem"])
        responses = generate(model, tok, ids, count=4, max_new_tokens=t["max_new_tokens"],
                             temperature=t["temperature"], top_p=t["top_p"])
        torch.cuda.synchronize()
        generated_at = time.monotonic()
        for r in responses:
            r.update(grade(r["text"], row["answer"]))
        mean, std, adv, kind = advantages([r["reward"] for r in responses], t["advantage_epsilon"])
        graded_at = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        metrics = accumulate_group(model, ids, responses, adv, t, tok.pad_token_id)
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], t["max_grad_norm"])
        finite = bool(torch.isfinite(grad_norm))
        if not finite or not all(math.isfinite(metrics[k]) for k in ("loss", "token_entropy_nats")):
            raise FloatingPointError(f"Non-finite loss/entropy/gradient at exposure {index + 1}; "
                                     "stopped before optimizer/cursor commit. Inspect before resuming.")
        optimizer.step()  # Exactly one call, even for zero-advantage groups.
        state["optimizer_steps"] += 1
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        state["cursor"] = index + 1
        state["generated_responses"] += 4
        state["generated_tokens"] += sum(r["generated_tokens"] for r in responses)
        state["training_seconds"] += elapsed
        state["group_counts"][kind] += 1
        state["last_record"] = {"exposure": index + 1, "problem_id": row["id"], "difficulty": row["difficulty"],
                                "prompt_tokens": len(ids), "responses": responses, "origin": origin_metadata(row),
                                "reward_mean": mean, "reward_std_population": std, "advantages": adv,
                                "group_type": kind, "cumulative_group_fractions": {k: v / (index + 1) for k, v in state["group_counts"].items()},
                                **metrics, "gradient_norm_pre_clip": float(grad_norm) if finite else None,
                                "optimizer_step_performed": finite, "skip_reason": None if finite else "nonfinite_gradient",
                                "optimizer_steps": state["optimizer_steps"], "skipped_steps": state["skipped_steps"],
                                "learning_rate": optimizer.param_groups[0]["lr"],
                                "adapter_norms": adapter_norms(model),
                                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                                "gpu_free_bytes": torch.cuda.mem_get_info()[0], "seconds": elapsed,
                                "generation_seconds": generated_at - started,
                                "grading_seconds": graded_at - generated_at,
                                "optimization_seconds": started + elapsed - graded_at,
                                "cumulative_training_seconds": state["training_seconds"],
                                "cumulative_generated_tokens": state["generated_tokens"]}
        commit()
        tokens = sum(r["generated_tokens"] for r in responses)
        print(f"{role}/{order}/seed{seed}: exposure {index + 1}/{len(rows)}, reward={mean:.2f}, "
              f"optimizer_steps={state['optimizer_steps']}, {elapsed:.1f}s/update, "
              f"{tokens / elapsed:.1f} training tok/s, "
              f"peak={state['last_record']['gpu_peak_allocated_bytes'] / 1024**3:.2f} GiB", flush=True)
        evaluate_due(state["cursor"])
        if stop_after is not None and state["cursor"] == stop_after:
            evaluate(model, tok, c, splits["diagnostic"], "smoke_pause", state["cursor"], root, contract, subset=1)
            print(f"Smoke stopped after checkpoint {stop_after}; resume with --resume-failed", flush=True)
            return {"stopped_at": stop_after, "contract": contract}
    assert state["cursor"] == len(rows)
    assert state["generated_responses"] == 4 * len(rows)
    assert state["optimizer_steps"] + state["skipped_steps"] == len(rows)
    result = {k: state[k] for k in ("cursor", "optimizer_steps", "skipped_steps", "generated_responses", "generated_tokens", "training_seconds", "group_counts")}
    result.update(contract=contract, expected_updates=len(rows), mode=mode,
                  valid_without_step_skips=state["skipped_steps"] == 0)
    if mode == "smoke":
        # Real serialization/reload path, including optimizer/RNG/cursor, on GPU.
        restored = load_checkpoint(checkpoints(root)[-1], contract)
        set_peft_model_state_dict(model, restored["adapter"])
        optimizer.load_state_dict(restored["optimizer"])
        restore_rng(restored["rng"])
        if adapter_hash(model) != restored["adapter_hash"] or restored["cursor"] != len(rows):
            raise AssertionError("Smoke checkpoint round-trip failed")
        evaluate(model, tok, c, splits["diagnostic"], "smoke_resume", len(rows), root, contract, subset=1)
        result["checkpoint_reload_evaluation"] = "passed"
    atomic_json(root / "complete.json", result)
    return result
