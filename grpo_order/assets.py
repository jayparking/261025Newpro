"""Explicit download ONLY. No torch import, GPU allocation, or training invocation.

Default destinations: ./models/{Qwen2.5-1.5B,Qwen-2.5-1.5B-SimpleRL-Zoo}/
and ./data/raw/{hendrycks_math,MATH-500}/. See configs/experiment.json first.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct

from .core import atomic_json, digest, file_hash, load_config, read_json
from .sources import MATH_ID, MATH500_ID, MODEL_IDS, SUBJECTS
from .storage import exclusive_lock

RECEIPT = ".asset.json"


def inventory(root):
    root = Path(root)
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != RECEIPT
            and not any(part in (".cache", ".git") for part in p.relative_to(root).parts)}


def safetensor_dtypes(root):
    """Inspect actual storage headers, not config.torch_dtype or inferred GPU dtype."""
    counts = Counter()
    for path in sorted(Path(root).glob("*.safetensors")):
        with path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 100_000_000:
                raise ValueError(f"Invalid safetensors header size: {path}")
            header = json.loads(f.read(n))
        for name, meta in header.items():
            if name != "__metadata__":
                counts[meta["dtype"]] += math.prod(meta["shape"])
    return dict(counts)


def validate_receipt(path, repo_id, revision=None):
    root = Path(path)
    rpath = root / RECEIPT
    if not rpath.is_file():
        raise ValueError(f"Unverified local asset at {root}: missing {RECEIPT}. Use scripts/run.sh download; do not rename another model here")
    r = read_json(rpath)
    payload = {k: v for k, v in r.items() if k != "receipt_hash"}
    if r.get("receipt_hash") != digest(payload):
        raise ValueError(f"Asset receipt modified: {root}")
    if r["repo_id"] != repo_id or (revision is not None and r["revision"] != revision):
        raise ValueError(f"Wrong asset source/revision at {root}")
    if not re.fullmatch(r"[0-9a-f]{40}", r["revision"]):
        raise ValueError("Asset revision must be an immutable 40-character commit")
    if r["repo_type"] == "model" and (r.get("tokenizer_revision") != r["revision"]
                                      or r.get("tokenizer_repo_id") != repo_id):
        raise ValueError("Tokenizer must come from the same model repository and revision")
    if inventory(root) != r["files"]:
        raise ValueError(f"Asset files changed or missing: {root}")
    return r


def validate_layout(root, repo_id, kind):
    root = Path(root)
    if kind == "model":
        if repo_id not in MODEL_IDS.values():
            raise ValueError(f"Model substitution forbidden: {repo_id}")
        cfg = read_json(root / "config.json")
        expected = {"model_type": "qwen2", "hidden_size": 1536, "num_hidden_layers": 28}
        if any(cfg.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Unexpected architecture for specified 1.5B model: {root}")
        # Architecture check alone never establishes provenance; receipt also pins exact repo.
        if not list(root.glob("*.safetensors")):
            raise ValueError("Complete safetensors weights required")
        for name in ("tokenizer.json", "tokenizer_config.json"):
            if not (root / name).is_file():
                raise ValueError(f"Missing tokenizer from the SAME model snapshot: {name}")
        idx = root / "model.safetensors.index.json"
        if list(root.glob("model-*-of-*.safetensors")) and not idx.exists():
            raise ValueError("Sharded checkpoint is missing its safetensors index")
        if idx.exists():
            missing = [f for f in set(read_json(idx)["weight_map"].values()) if not (root / f).is_file()]
            if missing:
                raise ValueError(f"Missing model shards: {missing}")
    elif repo_id == MATH_ID:
        actual = {p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")}
        if actual != set(SUBJECTS):
            raise ValueError(f"Expected exactly seven subject subsets, found {sorted(actual)}")
        for subject in SUBJECTS:
            for split in ("train", "test"):
                if not list((root / subject).glob(f"{split}-*.parquet")):
                    raise ValueError(f"Missing {subject}/{split} parquet shard")
    elif repo_id == MATH500_ID:
        if not (root / "test.jsonl").is_file():
            raise ValueError("Missing original MATH500 test.jsonl")
    else:
        raise ValueError(f"Unexpected dataset: {repo_id}")


def fetch_asset(repo_id, kind, destination, lock_entry=None, revision=None, offline=False, on_revision=None):
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    dst = Path(destination)
    if lock_entry and (lock_entry["repo_id"] != repo_id or lock_entry["repo_type"] != kind):
        raise ValueError(f"Asset lock source differs for {destination}")
    pinned = revision or (lock_entry or {}).get("revision")
    if revision and lock_entry and revision != lock_entry["revision"]:
        raise ValueError("Requested revision differs from lock; use a new asset destination and lock")
    if dst.exists():
        receipt = validate_receipt(dst, repo_id, pinned)
        validate_layout(dst, repo_id, kind)
        print(f"Verified existing local asset: {dst}", flush=True)
        return receipt
    # Check previously cached exact repository first; never choose a similar model name.
    cached = None
    cache_roots = dict.fromkeys([HF_HUB_CACHE, str(Path.home() / ".cache/huggingface/hub")])
    for cache in cache_roots:
        try:
            cached = Path(snapshot_download(repo_id, repo_type=kind, revision=pinned or "main",
                                            cache_dir=cache, local_files_only=True))
            break
        except Exception:
            continue
    if cached is None and offline:
        raise FileNotFoundError(f"No cached {repo_id}; rerun without --offline on a networked server")
    if cached is not None:
        resolved = cached.name
        if not re.fullmatch(r"[0-9a-f]{40}", resolved):
            raise ValueError(f"Cannot identify cached snapshot revision: {cached}")
        if pinned and pinned != resolved:
            raise ValueError("Cached revision mismatch")
        pinned = resolved
    elif pinned is None:
        pinned = HfApi().repo_info(repo_id, repo_type=kind, revision="main").sha
    if not re.fullmatch(r"[0-9a-f]{40}", pinned):
        raise ValueError("Could not resolve immutable Hub revision")
    if on_revision is not None:
        on_revision(pinned)  # Pin BEFORE transfer, so retries cannot silently move main.
    dst.parent.mkdir(parents=True, exist_ok=True)
    stage = dst.with_name(f".{dst.name}.download-{pinned}")
    stage.mkdir(exist_ok=True)
    if cached is not None:
        shutil.copytree(cached, stage, dirs_exist_ok=True, ignore=shutil.ignore_patterns('.cache', '.git', RECEIPT))
    try:
        validate_layout(stage, repo_id, kind)
        # A local cache may have only config/tokenizer. In online mode fetch every file.
        complete_cache = cached is not None and offline
    except (ValueError, FileNotFoundError):
        complete_cache = False
    if not complete_cache:
        if offline:
            raise ValueError(f"Cached snapshot is incomplete: {repo_id}")
        snapshot_download(repo_id, repo_type=kind, revision=pinned, local_dir=stage)
    validate_layout(stage, repo_id, kind)
    receipt = {"repo_id": repo_id, "repo_type": kind, "revision": pinned,
               "tokenizer_revision": pinned if kind == "model" else None,
               "tokenizer_repo_id": repo_id if kind == "model" else None,
               "files": inventory(stage), "stored_parameter_dtypes": safetensor_dtypes(stage) if kind == "model" else None}
    receipt["receipt_hash"] = digest(receipt)
    atomic_json(stage / RECEIPT, receipt)
    os.rename(stage, dst)
    print(f"Prepared {repo_id}@{pinned} -> {dst}", flush=True)
    return receipt


def main():
    p = argparse.ArgumentParser(description="Download specified assets only; NEVER starts training")
    p.add_argument("--config", default="./configs/experiment.json")
    p.add_argument("--only", choices=("models", "data", "all"), default="all")
    p.add_argument("--offline", action="store_true", help="Copy verified existing Hub cache only; no network")
    args = p.parse_args()
    c = load_config(args.config)
    a = c["assets"]
    # Keep any NEW download cache inside the project; inspect an existing default cache first.
    if not args.offline:
        os.environ.setdefault("HF_HOME", str(Path(a["cache_dir"]).resolve()))
    lock_path = Path(a["lock_file"])
    with exclusive_lock(lock_path.with_suffix(".lock")):
        lock = read_json(lock_path) if lock_path.exists() else {}
        specs = []
        if args.only in ("models", "all"):
            specs += [(role, "model", spec["hub_id"], spec["path"], spec.get("revision")) for role, spec in c["models"].items()]
        if args.only in ("data", "all"):
            specs += [(key, "dataset", spec["hub_id"], spec["path"], spec.get("revision")) for key, spec in a["datasets"].items()]
        for key, kind, repo_id, path, revision in specs:
            def pin_before_download(sha):
                lock[key] = {"repo_id": repo_id, "repo_type": kind, "path": path,
                             "revision": sha, "tokenizer_revision": sha if kind == "model" else None,
                             "status": "downloading"}
                atomic_json(lock_path, lock)
            receipt = fetch_asset(repo_id, kind, path, lock.get(key), revision, args.offline, pin_before_download)
            lock[key] = {"repo_id": repo_id, "repo_type": kind, "path": path,
                         "revision": receipt["revision"], "tokenizer_revision": receipt["tokenizer_revision"],
                         "receipt_hash": receipt["receipt_hash"], "status": "complete"}
            atomic_json(lock_path, lock)
    print("Asset preparation finished. No GPU training was started.")


if __name__ == "__main__":
    main()
