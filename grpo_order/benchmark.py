"""Separate, resumable large-k evaluation profiles; never change a training contract."""
import copy
import time
from pathlib import Path

from .core import atomic_json, digest, isolated_rng, read_json
from .grading import grade
from .model import generate_batch


def benchmark_config(config, samples=None):
    c = copy.deepcopy(config)
    profile = c.get("benchmark", {})
    e = c["evaluation"]
    n = profile.get("samples", 64) if samples is None else samples
    ks = profile.get("pass_k", [1, 4, 16, 64])
    chunk = profile.get("sample_batch_size", 8)
    batch = profile.get("problem_batch_size", e.get("problem_batch_size", 4))
    if (type(n) is not int or n < 1 or not isinstance(ks, list) or not ks
            or any(type(k) is not int or not 1 <= k <= n for k in ks)
            or len(set(ks)) != len(ks) or 1 not in ks
            or type(chunk) is not int or not 1 <= chunk <= n
            or type(batch) is not int or not 1 <= batch <= 32):
        raise ValueError("Invalid benchmark sample count, pass@k list or batch size")
    e.update(samples=n, pass_k=sorted(ks), greedy=False, sample_batch_size=chunk,
             problem_batch_size=batch, benchmark_schema="chunked-passk-v1")
    # Sampling prompt/temperature/top-p/length and seed remain exactly frozen.
    return c


def benchmark_root(root, config, samples=None):
    e = benchmark_config(config, samples)["evaluation"]
    return Path(root) / "benchmarks" / f"math500-n{e['samples']}-{digest(e)[:16]}"


def sampled_chunks(model, tok, ids, rows, config, split, evaluation_id, directory):
    """Commit a whole question-batch/sample-chunk, then resume without RNG drift."""
    from .integrity import validate_responses
    e = config["evaluation"]
    batch_ids = [row["id"] for row in rows]
    output = [[] for _ in rows]
    generated_tokens = 0
    directory = Path(directory) / digest(batch_ids)
    for offset in range(0, e["samples"], e["sample_batch_size"]):
        count = min(e["sample_batch_size"], e["samples"] - offset)
        path = directory / f"samples-{offset:06d}.json"
        seed = int(digest(["sample_chunk_v1", config["seeds"]["evaluation"], split, batch_ids, offset])[:8], 16)
        identity = {"evaluation_id": evaluation_id, "batch_question_ids": batch_ids,
                    "offset": offset, "count": count, "seed": seed}
        if path.exists():
            saved = read_json(path)
            payload = {k: v for k, v in saved.items() if k != "hash"}
            if saved.get("hash") != digest(payload) or saved.get("identity") != identity:
                raise ValueError(f"Corrupt or mismatched evaluation chunk: {path}")
            batches = saved["responses"]
        else:
            started = time.monotonic()
            with isolated_rng(seed):
                batches = generate_batch(model, tok, ids, count=count, max_new_tokens=e["max_new_tokens"],
                                         temperature=e["temperature"], top_p=e["top_p"])
            if len(batches) != len(rows):
                raise ValueError("Evaluation chunk has the wrong number of questions")
            for row, responses in zip(rows, batches):
                for response in responses:
                    response.update(grade(response["text"], row["answer"]))
                    generated_tokens += response["generated_tokens"]
                # Never commit invalid responses: a retry must be able to regenerate
                # the failed chunk instead of loading a permanently invalid cache.
                validate_responses(responses, count)
            payload = {"identity": identity, "responses": batches,
                       "wall_seconds": time.monotonic() - started}
            atomic_json(path, {**payload, "hash": digest(payload)})
            print(f"    sampled {offset + count}/{e['samples']} per question; "
                  f"chunk={payload['wall_seconds']:.1f}s", flush=True)
        if len(batches) != len(rows):
            raise ValueError("Missing question in evaluation chunk")
        for target, responses in zip(output, batches):
            validate_responses(responses, count)
            target.extend({**r, "sample_index": offset + i, "sampling_seed": seed}
                          for i, r in enumerate(responses))
    return output, generated_tokens
