"""Explicit code/config/data/model/package freeze; never starts training."""
from pathlib import Path

from .core import atomic_json, code_hash, digest, packages, read_json
from .data import load_prepared


def freeze_identity(c):
    manifest, _ = load_prepared(c)
    return {"config": digest(c), "code": code_hash(), "data": manifest["hash"], "packages": packages()}


def freeze_path(c):
    return Path(c["output_dir"]) / "protocol-freeze.json"


def verify_freeze(c, models=None):
    if not c["evaluation"]["length_finalized"]:
        raise ValueError("Finalize the common output length before full runs")
    path = freeze_path(c)
    if not path.is_file():
        raise ValueError(f"Code/config/data are not frozen. Run freeze after smoke/pilot validation: {path}")
    record = read_json(path)
    if record["identity"] != freeze_identity(c):
        raise ValueError("Frozen code/config/data/packages changed; create a new experiment output directory")
    if models is not None:
        for role, meta in models.items():
            if record["models"][role] != meta["content_hash"]:
                raise ValueError(f"Frozen model/tokenizer changed: {role}")
    return record


def freeze(c):
    from .preflight import check
    from .storage import exclusive_lock
    if not c["evaluation"]["length_finalized"]:
        raise ValueError("Select common prompt/length/LR from validation and finalize length first")
    dst = freeze_path(c)
    with exclusive_lock(dst.with_suffix(".lock")):
        if dst.exists():
            raise FileExistsError(dst)
        report = check(c)
        if report["errors"]:
            raise ValueError(f"Freeze preflight failed: {report['errors']}")
        atomic_json(dst, {"identity": freeze_identity(c),
                          "models": {role: m["content_hash"] for role, m in report["models"].items()},
                          "preflight": report})
    return dst
