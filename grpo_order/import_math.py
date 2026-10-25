"""Convert the seven local EleutherAI/hendrycks_math subsets; no downloads.

Preserve source, source_revision, original split, subject/subset, level, solution,
and stable content ID. Experimental difficulty is a grouping of dataset levels.
"""
import json
import os
from pathlib import Path
import random
import re
import tempfile
import argparse
import importlib.metadata

from .assets import validate_layout, validate_receipt
from .core import DIFFICULTIES, atomic_json, digest, file_hash, jsonl, load_config
from .data import normalize
from .sources import LEVEL_MAPPING, MATH_ID, MATH500_ID, SUBJECTS


def boxed_answer(solution):
    """Extract the final boxed/fbox answer with balanced nested and escaped braces."""
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", solution))
    if not matches:
        raise ValueError("missing_boxed_gold")
    start = matches[-1].end()
    depth = 1
    for i in range(start, len(solution)):
        # Braces escaped by an odd number of backslashes are literal, not delimiters.
        backslashes, j = 0, i - 1
        while j >= 0 and solution[j] == "\\":
            backslashes += 1
            j -= 1
        if backslashes % 2:
            continue
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
            if depth == 0:
                result = solution[start:i].strip()
                if not result:
                    raise ValueError("empty_boxed_gold")
                return result
    raise ValueError("unbalanced_boxed_gold")


def convert_math_row(row, subset, split, revision, source_file, source_row):
    if subset not in SUBJECTS or split not in ("train", "test"):
        raise ValueError("Unexpected source subset or split")
    problem = row.get("problem")
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("Missing source problem text")
    if not isinstance(row.get("solution"), str):
        raise ValueError("Missing source solution")
    level = row.get("level")
    # ID deliberately excludes row index and revision: reordering the source cannot change ID.
    stable = digest([MATH_ID, split, subset, normalize(problem)])
    result = {**row, "id": f"{MATH_ID}/{split}/{subset}/{stable}",
              "source": MATH_ID, "source_revision": revision, "split": split,
              "subject": row.get("type", subset), "subset": subset,
              "source_file": source_file, "source_row": source_row,
              "original_id": row.get("id"), "level": level,
              "difficulty": LEVEL_MAPPING.get(level), "normalized_problem": normalize(problem)}
    reasons = []
    if level not in LEVEL_MAPPING:
        reasons.append("unknown_original_level")
    try:
        result["answer"] = boxed_answer(row["solution"])
    except ValueError as e:
        result["answer"] = None
        reasons.append(str(e))
    result["conversion_exclusion_reasons"] = reasons
    return result


def convert_benchmark(row, revision, index):
    for name in ("problem", "answer", "unique_id"):
        if not isinstance(row.get(name), str) or not row[name].strip():
            raise ValueError(f"MATH500 row {index}: missing {name}")
    level = row.get("level")
    original_level = f"Level {level}" if isinstance(level, int) else level
    if original_level not in LEVEL_MAPPING:
        raise ValueError(f"MATH500 row {index}: unexpected level {level!r}")
    return {**row, "id": f"{MATH500_ID}/{row['unique_id']}", "original_id": row["unique_id"],
            "source": MATH500_ID, "source_revision": revision, "split": "test",
            "level": original_level, "original_level": level, "difficulty": LEVEL_MAPPING[original_level],
            "subject": row.get("subject"), "subset": "default", "source_row": index,
            "source_file": "test.jsonl", "normalized_problem": normalize(row["problem"])}


def split_validation(rows, per_difficulty, seed):
    """Choose dev questions from original train only, keeping their original split."""
    eligible = [r for r in rows if not r["conversion_exclusion_reasons"]]
    if any(r["split"] != "train" for r in eligible):
        raise ValueError("Original test problems must not enter additional training/dev pool")
    rng = random.Random(seed)
    chosen = set()
    validation = []
    for difficulty in DIFFICULTIES:
        group = sorted((r for r in eligible if r["difficulty"] == difficulty), key=lambda r: r["id"])
        # Identical normalized questions are never sampled twice for validation.
        group = list({r["normalized_problem"]: r for r in group}.values())
        rng.shuffle(group)
        if len(group) < per_difficulty:
            raise ValueError(f"Insufficient unique {difficulty} validation candidates")
        validation.extend(group[:per_difficulty])
        chosen.update(r["normalized_problem"] for r in group[:per_difficulty])
    pool = [r for r in eligible if r["normalized_problem"] not in chosen]
    return sorted(pool, key=lambda r: r["id"]), sorted(validation, key=lambda r: r["id"])


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def import_local(c):
    import pyarrow.parquet as pq
    a = c["assets"]
    dst = Path(a["converted_dir"])
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite converted data: {dst}")
    math = a["datasets"]["math"]
    benchmark = a["datasets"]["math500"]
    mreceipt = validate_receipt(math["path"], MATH_ID, math.get("revision"))
    breceipt = validate_receipt(benchmark["path"], MATH500_ID, benchmark.get("revision"))
    validate_layout(math["path"], MATH_ID, "dataset")
    validate_layout(benchmark["path"], MATH500_ID, "dataset")
    merged, counts = {"train": [], "test": []}, {}
    for subset in SUBJECTS:
        counts[subset] = {}
        for split in ("train", "test"):
            rows = []
            for path in sorted((Path(math["path"]) / subset).glob(f"{split}-*.parquet")):
                for index, row in enumerate(pq.read_table(path).to_pylist()):
                    rows.append(convert_math_row(row, subset, split, mreceipt["revision"],
                                                 str(path.relative_to(math["path"])), index))
            counts[subset][split] = len(rows)
            merged[split].extend(rows)
    if {k: len(v) for k, v in merged.items()} != {"train": 7500, "test": 5000}:
        raise ValueError(f"Unexpected original MATH split sizes: {counts}; do not silently omit subjects/shards")
    bdata = [convert_benchmark(r, breceipt["revision"], i) for i, r in enumerate(jsonl(Path(benchmark["path"]) / "test.jsonl"))]
    if len(bdata) != 500 or len({r["id"] for r in bdata}) != 500:
        raise ValueError("MATH500 must have exactly 500 unique original IDs")
    pool, validation = split_validation(merged["train"], a["validation_per_difficulty"], a["validation_selection_seed"])
    exclusions = [{"id": r["id"], "split": r["split"], "reasons": r["conversion_exclusion_reasons"]}
                  for rows in merged.values() for r in rows if r["conversion_exclusion_reasons"]]
    dst.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".convert-", dir=dst.parent))
    for name, rows in {"all_train": merged["train"], "all_test": merged["test"],
                       "pool": pool, "validation": validation, "math500": bdata}.items():
        write_jsonl(stage / f"{name}.jsonl", rows)
    atomic_json(stage / "conversion_exclusions.json", exclusions)
    report = {"schema": 1, "sources": {"math": mreceipt, "math500": breceipt},
              "conversion_packages": {p: importlib.metadata.version(p) for p in ("pyarrow", "huggingface-hub")},
              "subject_counts": counts, "level_mapping": LEVEL_MAPPING,
              "difficulty_definition": "Original dataset level grouping, NOT model empirical accuracy",
              "validation_selection_seed": a["validation_selection_seed"],
              "validation_per_difficulty_before_near_dedup": a["validation_per_difficulty"],
              "files": {p.name: file_hash(p) for p in sorted(stage.iterdir()) if p.is_file()},
              "eligible_training_candidates": len(pool), "conversion_exclusions": len(exclusions),
              "test_policy": "Original 5000 test rows archived only; never used to populate training pool",
              "id_policy": "SHA256 of source/split/subset/normalized problem; independent of row order and random seed"}
    report["hash"] = digest(report)
    atomic_json(stage / "import_manifest.json", report)
    os.rename(stage, dst)
    return report


def main():
    p = argparse.ArgumentParser(description="Convert local MATH subsets; no network or training")
    p.add_argument("--config", default="./configs/experiment.json")
    args = p.parse_args()
    report = import_local(load_config(args.config))
    print(json.dumps({k: report[k] for k in ("hash", "subject_counts", "eligible_training_candidates", "conversion_exclusions")}, indent=2))
    print("Converted only. Next: python -m grpo_order prepare-data")


if __name__ == "__main__":
    main()
