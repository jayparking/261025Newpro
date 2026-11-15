"""Atomic state commits. A completed checkpoint is the authoritative exposure cursor."""
import contextlib
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

from .core import atomic_json, file_hash, read_json


def append_training_exports(root, record):
    """Derived logs; committed steps/checkpoints remain authoritative on restart."""
    root = Path(root)
    flat = {"step": record["exposure"], "problem_id": record["problem_id"],
            "difficulty": record["difficulty"], "group_type": record["group_type"],
            "zero_advantage": int(record["group_type"] != "mixed"),
            "rewards": json.dumps([r["reward"] for r in record["responses"]]),
            "reward_mean": record["reward_mean"], "loss": record["loss"],
            "seconds": record["seconds"]}
    path = root / "training.csv"
    header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat))
        if header:
            writer.writeheader()
        writer.writerow(flat)
    with (root / "training.jsonl").open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def rebuild_training_exports(root, cursor):
    # Recover a missing line, truncated append or duplicate after a process crash.
    root = Path(root)
    for name in ("training.csv", "training.jsonl"):
        (root / name).write_text("")
    for step in range(1, cursor + 1):
        append_training_exports(root, read_json(root / "steps" / f"{step:06d}.json"))


@contextlib.contextmanager
def exclusive_lock(path, *, wait=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError as e:
            raise RuntimeError(f"Another process holds {path}") from e
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def run_path(c, mode, role, order, seed):
    return Path(c["output_dir"]) / mode / f"{role}_seed{seed}_{order}"


def checkpoints(root):
    return sorted(p for p in (Path(root) / "checkpoints").glob("step-*") if (p / "commit.json").is_file())


def save_checkpoint(root, state, milestones):
    import torch
    root = Path(root)
    parent = root / "checkpoints"
    parent.mkdir(exist_ok=True)
    dst = parent / f"step-{state['cursor']:06d}"
    if dst.exists():
        raise FileExistsError(f"Refusing to replace committed checkpoint {dst}")
    tmp = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
    with (tmp / "state.pt").open("wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    atomic_json(tmp / "commit.json", {"cursor": state["cursor"], "contract": state["contract"],
                                      "sha256": file_hash(tmp / "state.pt")})
    os.rename(tmp, dst)
    reconcile_event(root, state)
    # Retain required snapshots plus newest restart state. Never touch other runs.
    for old in checkpoints(root):
        step = int(old.name.split("-")[1])
        if step not in milestones and step < state["cursor"]:
            shutil.rmtree(old)
    return dst


def reconcile_event(root, state):
    record = state.get("last_record")
    if record is None:
        return
    dst = Path(root) / "steps" / f"{state['cursor']:06d}.json"
    if dst.exists() and read_json(dst) != record:
        raise ValueError(f"Step log disagrees with committed checkpoint: {dst}")
    if not dst.exists():
        atomic_json(dst, record)


def load_checkpoint(path, contract):
    import torch
    path = Path(path)
    commit = read_json(path / "commit.json")
    if commit["contract"] != contract:
        raise ValueError("Checkpoint contract mismatch (config/data/model/code/packages/order)")
    if file_hash(path / "state.pt") != commit["sha256"]:
        raise ValueError("Checkpoint bytes are corrupted")
    # Only load checkpoints created by this local pipeline, never untrusted pickle files.
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    if state["contract"] != contract or state["cursor"] != commit["cursor"]:
        raise ValueError("Checkpoint metadata disagrees with state")
    return state
