"""Fixed selection, conservative duplicate exclusions, immutable prepared manifest."""
from collections import Counter, defaultdict
import os
from pathlib import Path
import random
import tempfile
import time
import unicodedata

from .core import DIFFICULTIES, atomic_json, digest, file_hash, jsonl, read_json
from .grading import GoldIssue, GraderEnvironmentError, check_grader, gold_parse, grader_identity


def origin_metadata(row):
    return {key: row.get(key) for key in ("source", "source_revision", "split", "subject", "subset", "level", "difficulty", "original_id", "source_file", "source_row")}


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def shingles(text):
    # Character shingles preserve numeric differences; no invented difficulty score.
    return {text[i:i + 5] for i in range(max(1, len(text) - 4))}


def read_source(path, c, require_difficulty):
    out = []
    for row in jsonl(path):
        for key in ("id", "problem"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{path}: nonempty string {key} required: {row.get('id')}")
        d = c["difficulty_mapping"].get(str(row.get(c["difficulty_field"])))
        if require_difficulty and d not in DIFFICULTIES:
            raise ValueError(f"{path}: unmapped difficulty for {row['id']}")
        if c.get("require_origin_metadata"):
            for key in ("source", "source_revision", "split", "subject", "subset", "level"):
                if not row.get(key):
                    raise ValueError(f"{path}/{row['id']}: missing original {key}")
            if row.get("difficulty") != d:
                raise ValueError(f"{path}/{row['id']}: difficulty disagrees with original level")
            if require_difficulty and row["split"] != "train":
                raise ValueError("Training pool contains original test problems")
        # Preserve original provenance; do not replace the dataset ID with a local filename.
        out.append({**row, "difficulty": d or "Unknown", "normalized_problem": normalize(row["problem"]),
                    "source": row.get("source", str(path)), "input_file": str(path)})
    return sorted(out, key=lambda r: r["id"])


def prepare(config):
    c = config["data"]
    for key in ("pool", "validation", "math500", "difficulty_rationale", "source_provenance"):
        if not c.get(key):
            raise ValueError(f"Set data.{key}; no automatic data download or difficulty inference")
    dst = Path(c["prepared_dir"])
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite {dst}; use a new prepared_dir")
    audit_path = dst.parent / f"{dst.name}-gold-audit-{time.time_ns()}.json"
    try:
        grader = check_grader()
    except GraderEnvironmentError as exc:
        atomic_json(audit_path, {"status": "blocked", "category": "installation_error", "error": str(exc)})
        raise GraderEnvironmentError(f"{exc}; report: {audit_path}") from exc
    source_hashes = {k: file_hash(c[k]) for k in ("pool", "validation", "math500")}
    import_manifest = None
    if c.get("import_manifest"):
        ipath = Path(c["import_manifest"])
        import_manifest = read_json(ipath)
        if digest({k: v for k, v in import_manifest.items() if k != "hash"}) != import_manifest["hash"]:
            raise ValueError("Import manifest checksum mismatch")
        for name, expected in import_manifest["files"].items():
            if file_hash(ipath.parent / name) != expected:
                raise ValueError(f"Converted original data changed: {name}")
    by_split = {k: read_source(c[k], c, k == "pool") for k in source_hashes}
    if len(by_split["math500"]) != 500:
        raise ValueError("MATH500 source must contain exactly 500 records; source identity needs provenance")
    accepted, exclusions = {}, []
    failures = []
    eligible = {}
    for split, rows in by_split.items():
        eligible[split] = []
        for row in rows:
            try:
                gold_parse(row.get("answer"))
            except GoldIssue as exc:
                item = {"excluded_id": row["id"], "split": split, "category": exc.category,
                        "reason": str(exc), "answer": row.get("answer"), "origin": origin_metadata(row)}
                failures.append(item)
                if split != "math500":
                    exclusions.append(item)
            except GraderEnvironmentError as exc:
                atomic_json(audit_path, {"status": "blocked", "category": "installation_error", "error": str(exc)})
                raise
            else:
                eligible[split].append(row)
    gold_audit = {"grader": grader, "source_hashes": source_hashes, "failures": failures,
                  "math500_count": len(by_split["math500"]),
                  "math500_failed_ids": [r["excluded_id"] for r in failures if r["split"] == "math500"]}
    atomic_json(audit_path, gold_audit)
    if gold_audit["math500_failed_ids"]:
        raise ValueError(f"MATH500 gold audit failed; no items removed or scored. Review {audit_path}")
    ids, texts, index, previous = {}, {}, defaultdict(set), []
    # Preserve evaluation first, then validation, then training candidates.
    for split in ("math500", "validation", "pool"):
        accepted[split] = []
        for row in eligible[split]:
            text = row["normalized_problem"]
            grams = shingles(text)
            match = ids.get(row["id"]) or texts.get(text)
            reason, score = "exact_id_or_text", 1.0
            if match is None:
                candidates = Counter(j for g in grams for j in index[g])
                for j, overlap in candidates.items():
                    other, other_grams, other_split = previous[j]
                    sim = overlap / (len(grams) + len(other_grams) - overlap)
                    if sim >= c["near_duplicate_threshold"]:
                        match, reason, score = (other["id"], other_split), "near_char5_jaccard", sim
                        break
            if match is not None:
                item = {"excluded_id": row["id"], "split": split, "retained": match,
                        "reason": reason, "similarity": score}
                if split == "math500":
                    raise ValueError(f"Duplicate inside MATH500; review source without dropping benchmark items: {item}")
                exclusions.append(item)
                continue
            ids[row["id"]] = texts[text] = (row["id"], split)
            j = len(previous)
            previous.append((row, grams, split))
            for gram in grams:
                index[gram].add(j)
            accepted[split].append(row)
    rng = random.Random(c["selection_seed"])
    train, diagnostic, counts = [], [], {}
    for d in DIFFICULTIES:
        pool = [r for r in accepted["pool"] if r["difficulty"] == d]
        rng.shuffle(pool)
        n = c["train_per_difficulty"]
        if len(pool) < n:
            raise ValueError(f"{d}: only {len(pool)} unique eligible problems, need {n}")
        train.extend(pool[:n])
        selected = pool[n:n + c["diagnostic_per_difficulty"]]
        diagnostic.extend(selected)
        counts[d] = {"train": n, "diagnostic": len(selected), "eligible_pool": len(pool)}
        if not selected:
            raise ValueError(f"{d}: zero diagnostic problems; provide more data")
    if not accepted["validation"]:
        raise ValueError("No independent validation problems remain after duplicate checks")
    splits = {"train": train, "diagnostic": diagnostic, "validation": accepted["validation"], "math500": accepted["math500"]}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".prepare-", dir=dst.parent))
    for name, rows in splits.items():
        atomic_json(tmp / f"{name}.json", sorted(rows, key=lambda r: r["id"]))
    atomic_json(tmp / "exclusions.json", exclusions)
    atomic_json(tmp / "gold_audit.json", gold_audit)
    manifest = {"schema": 3, "data_config": c, "source_hashes": source_hashes, "counts": counts,
                "gold_audit": gold_audit,
                "import_manifest": import_manifest,
                "split_counts": {k: len(v) for k, v in splits.items()},
                "files": {p.name: file_hash(p) for p in sorted(tmp.glob("*.json"))},
                "dedup": "global IDs, NFKC/casefold/whitespace text, character 5-gram Jaccard; semantic paraphrases not guaranteed",
                "diagnostic_claim": "not used in this additional training; historical exposure unknown"}
    manifest["hash"] = digest(manifest)
    atomic_json(tmp / "manifest.json", manifest)
    os.rename(tmp, dst)
    return manifest


def load_prepared(config):
    root = Path(config["data"]["prepared_dir"])
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema") != 3 or manifest.get("gold_audit", {}).get("grader") != grader_identity():
        raise ValueError("Prepared data needs a fresh gold audit with this grader; use a new prepared_dir")
    payload = {k: v for k, v in manifest.items() if k != "hash"}
    if digest(payload) != manifest["hash"]:
        raise ValueError("Manifest hash mismatch")
    if config["data"] != manifest["data_config"]:
        raise ValueError("Data configuration differs from prepared manifest")
    for name, expected in manifest["files"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"Prepared file changed: {name}")
    splits = {k: read_json(root / f"{k}.json") for k in ("train", "diagnostic", "validation", "math500")}
    return manifest, splits
