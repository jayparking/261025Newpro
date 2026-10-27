import argparse
import copy
import os
from pathlib import Path
import time

from .analysis import analyze
from .core import atomic_json, load_config
from .data import prepare
from .evaluation import evaluate
from .freeze import freeze
from .launcher import launch
from .model import adapter_hash, load_model, load_tokenizer, validate_context
from .preflight import check
from .presmoke import quick_config
from .storage import exclusive_lock, load_checkpoint, run_path
from .train import check_environment, configure_torch, open_run, train, validate_assignment


def main():
    parser = argparse.ArgumentParser(description="Fixed-set 2 x 2 additional GRPO experiment with configured training seeds")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "prepare-data", "freeze", "train", "launch", "evaluate", "analyze"):
        p = sub.add_parser(command)
        p.add_argument("--config", default="configs/experiment.json")
        if command == "preflight":
            p.add_argument("--environment-only", action="store_true")
            p.add_argument("--report")
        if command in ("train", "evaluate"):
            p.add_argument("--model", choices=("base", "rl"), required=True)
            p.add_argument("--order", choices=("random", "curriculum"), required=True)
            p.add_argument("--seed", type=int, choices=(1, 2, 3), default=1)
        if command in ("train", "launch", "evaluate", "analyze"):
            modes = ("poc", "presmoke", "smoke", "pilot", "full") if command == "train" else ("smoke", "pilot", "full")
            p.add_argument("--mode", choices=modes, default="pilot")
        if command == "train":
            p.add_argument("--resume", action="store_true")
        if command in ("train", "launch"):
            p.add_argument("--stop-after", type=int, choices=(1, 2), help="smoke only: save, evaluate, exit; then resume")
        if command == "launch":
            p.add_argument("--resume-failed", action="store_true")
            p.add_argument("--seed", type=int, choices=(1, 2, 3), default=1)
        if command == "evaluate":
            p.add_argument("--step", type=int, required=True)
            p.add_argument("--split", choices=("diagnostic", "math500", "validation"), default="validation")
        if command == "analyze":
            p.add_argument("--output", required=True)
            p.add_argument("--allow-incomplete", action="store_true")
            p.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    c = load_config(args.config)
    # Before torch import / CUDA initialization, including direct single-run launches.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.command in ("launch", "preflight", "freeze"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(map(str, c["runtime"]["allocated_gpu_slots"])))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.command == "train" and args.mode == "presmoke":
        c = quick_config(c)
    if args.command in ("train", "launch", "evaluate") and args.mode == "smoke":
        c = copy.deepcopy(c)
        c["train"]["max_new_tokens"] = 32
        c["evaluation"]["max_new_tokens"] = 32
    if args.command == "preflight":
        report = check(c, environment_only=args.environment_only)
        dst = Path(args.report or f"reports/preflight-{time.time_ns()}.json")
        if dst.exists():
            raise FileExistsError(dst)
        atomic_json(dst, report)
        print(f"Report: {dst}")
        for error in report["errors"]:
            print("ERROR:", error)
        return int(bool(report["errors"]))
    if args.command == "prepare-data":
        print(prepare(c))
    elif args.command == "freeze":
        print(freeze(c))
    elif args.command == "train":
        train(c, args.model, args.order, args.seed, args.mode, args.resume, args.stop_after)
    elif args.command == "launch":
        launch(c, args.config, args.mode, args.resume_failed, args.seed, args.stop_after)
    elif args.command == "evaluate":
        evaluate_saved(c, args)
    elif args.command == "analyze":
        analyze(c, args.mode, args.output, args.allow_incomplete, not args.no_plots)
    return 0


def evaluate_saved(c, args):
    """Evaluate one retained checkpoint of an existing run; shares setup with training."""
    validate_assignment(args.order, c)
    if args.mode != "full" and args.split == "math500":
        raise ValueError("MATH500 evaluation is only allowed for frozen full runs")
    root = run_path(c, args.mode, args.model, args.order, args.seed)
    if not root.is_dir():
        raise FileNotFoundError(f"No run directory to evaluate: {root}")
    h = check_environment(c)
    uuid = h["uuid_mapping"][0]["uuid"]
    with exclusive_lock(root.parent / (root.name + ".lock")), exclusive_lock(f"/tmp/grpo-order-{os.getuid()}-{uuid}.lock"):
        configure_torch(c)
        path, _, _, splits, plan, _, contract = open_run(c, args.model, args.order, args.seed, args.mode)
        state = load_checkpoint(root / "checkpoints" / f"step-{args.step:06d}", contract)
        tok, limit = load_tokenizer(path, c["models"][args.model])
        validate_context(tok, limit, c, splits)
        from peft import set_peft_model_state_dict
        model, _ = load_model(path, c, plan["lora"])
        set_peft_model_state_dict(model, state["adapter"])
        if adapter_hash(model) != state["adapter_hash"]:
            raise ValueError("Checkpoint adapter hash differs after load")
        evaluate(model, tok, c, splits[args.split], args.split, args.step, root, contract,
                 subset=3 if args.mode == "smoke" and args.split == "diagnostic" else None)
