"""Pair conditions by training seed and evaluation question, never by response."""
import csv
from pathlib import Path
import random
import statistics

from .core import atomic_json, digest, ordered_problems, read_json, seed_plan
from .data import load_prepared
from .integrity import validate_completion, validate_evaluation, validate_steps
from .storage import run_path


METRICS = ("avg_at_8", "pass_at_8", "greedy")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_differences(random_questions, curriculum_questions, metric):
    a = {q["id"]: q[metric] for q in random_questions}
    b = {q["id"]: q[metric] for q in curriculum_questions}
    if len(a) != len(random_questions) or len(b) != len(curriculum_questions) or a.keys() != b.keys():
        raise ValueError("Evaluation question identities differ or contain duplicates")
    return {k: b[k] - a[k] for k in sorted(a)}


def bootstrap_question_means(seed_maps, seed, repeats=2000):
    # Average over the observed training seeds first. Resample question clusters,
    # jointly across all seeds/conditions. This CI excludes training-seed uncertainty.
    ids = sorted(seed_maps[0])
    if any(set(m) != set(ids) for m in seed_maps):
        raise ValueError("Bootstrap requires shared question IDs")
    values = [statistics.mean(m[k] for m in seed_maps) for k in ids]
    rng = random.Random(seed)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    return draws[int(0.025 * repeats)], draws[min(repeats - 1, int(0.975 * repeats))]


def analyze(c, mode, output, allow_incomplete=False, plots=True, bootstrap_repeats=2000):
    if mode != "full":
        raise ValueError("This analysis requires full MATH500 results; smoke/pilot are validation only")
    manifest, splits = load_prepared(c)
    if len(splits["train"]) != 300 or len(splits["math500"]) != 500:
        raise ValueError("Full analysis requires 300 training and 500 benchmark questions")
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite analysis {out}")
    all_runs, missing, curves, costs = {}, [], [], []
    initial_adapters = {}
    signatures = set()
    role_models = {"base": set(), "rl": set()}
    selected_seeds = c["seeds"]["training"]
    for seed in selected_seeds:
        for role in ("base", "rl"):
            for order in ("random", "curriculum"):
                root = run_path(c, mode, role, order, seed)
                if not (root / "complete.json").exists():
                    missing.append(str(root))
                    continue
                metadata = read_json(root / "metadata.json")
                completion = read_json(root / "complete.json")
                validate_completion(metadata, completion, 300)
                contract = metadata["contract"]
                if any(contract[k] != v for k, v in {"role": role, "order": order, "seed": seed,
                                                     "mode": mode, "data": manifest["hash"]}.items()):
                    raise ValueError(f"Run identity/condition mismatch: {root}")
                pair = (role, seed)
                initial = metadata["lora"]["initial_adapter_hash"]
                if initial_adapters.setdefault(pair, initial) != initial:
                    raise ValueError("Random/Curriculum started from different adapters")
                if metadata["contract"]["config"] != digest(c):
                    raise ValueError(f"Analysis config differs from {root}")
                signatures.add((metadata["contract"]["data"], metadata["contract"]["config"], metadata["contract"]["code"], digest(metadata["contract"]["packages"])))
                role_models[role].add(metadata["contract"]["model"])
                evaluations = {}
                for split, schedule in (("diagnostic", c["evaluation"]["diagnostic_steps"]),
                                        ("math500", c["evaluation"]["math500_steps"])):
                    for step in schedule:
                        result = read_json(root / "eval" / f"{split}_step-{step:06d}.json")
                        validate_evaluation(result, contract, splits[split], split, step)
                        evaluations[split, step] = result
                evaluation = evaluations["math500", 300]
                all_runs[role, seed, order] = evaluation
                for step in c["evaluation"]["diagnostic_steps"]:
                    result = evaluations["diagnostic", step]
                    for difficulty, m in {"All": result["metrics"], **result["by_difficulty"]}.items():
                        curves.append({"model": role, "seed": seed, "order": order, "step": result["step"],
                                       "difficulty": difficulty, **m})
                steps = [read_json(p) for p in sorted((root / "steps").glob("*.json"))]
                expected_rows = ordered_problems(splits["train"], order, seed_plan(c, seed)["order"])
                validate_steps(c, metadata, steps, expected_rows)
                for r in steps:
                    costs.append({"model": role, "seed": seed, "order": order, "step": r["exposure"],
                                  "reward": r["reward_mean"], "mixed_fraction": r["cumulative_group_fractions"]["mixed"],
                                  "truncation_fraction": sum(x["truncated"] for x in r["responses"]) / 4,
                                  "mean_response_tokens": sum(x["generated_tokens"] for x in r["responses"]) / 4,
                                  "generated_tokens": r["cumulative_generated_tokens"],
                                  "seconds": r["cumulative_training_seconds"], "entropy_nats": r["token_entropy_nats"]})
    if missing and not allow_incomplete:
        raise ValueError("Missing completed runs; use --allow-incomplete only for interim inspection: " + str(missing))
    if len(signatures) > 1:
        raise ValueError("Cannot pool runs with different data/config/code/package identities")
    if any(len(values) > 1 for values in role_models.values()):
        raise ValueError("Original model contents changed between conditions/seeds for the same role")
    if role_models["base"] & role_models["rl"]:
        raise ValueError("Base and RL use the same original model")
    seeds = sorted({s for _, s, _ in all_runs if all((role, s, order) in all_runs for role in ("base", "rl") for order in ("random", "curriculum"))})
    if not seeds:
        raise ValueError("Need at least one completed seed with all four conditions")
    seed_rows, summaries = [], []
    for metric in METRICS:
        deltas = {role: [paired_differences(all_runs[role, s, "random"]["questions"], all_runs[role, s, "curriculum"]["questions"], metric) for s in seeds] for role in ("base", "rl")}
        interaction = []
        for i, seed in enumerate(seeds):
            if deltas["base"][i].keys() != deltas["rl"][i].keys():
                raise ValueError("Base/RL evaluation IDs differ")
            interaction.append({k: deltas["rl"][i][k] - deltas["base"][i][k] for k in deltas["base"][i]})
            seed_rows.append({"metric": metric, "seed": seed,
                              **{f"{role}_{order}": all_runs[role, seed, order]["metrics"][metric] for role in ("base", "rl") for order in ("random", "curriculum")},
                              "delta_base": statistics.mean(deltas["base"][i].values()),
                              "delta_rl": statistics.mean(deltas["rl"][i].values()),
                              "interaction": statistics.mean(interaction[-1].values())})
        for name, maps in {"delta_base": deltas["base"], "delta_rl": deltas["rl"], "interaction": interaction}.items():
            values = [statistics.mean(m.values()) for m in maps]
            lo, hi = bootstrap_question_means(maps, c["seeds"]["bootstrap"], bootstrap_repeats)
            summaries.append({"metric": metric, "contrast": name, "seeds": len(values),
                              "mean": statistics.mean(values), "seed_sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                              "seed_min": min(values), "seed_max": max(values),
                              "question_bootstrap_ci_low": lo, "question_bootstrap_ci_high": hi,
                              "bootstrap_repeats": bootstrap_repeats})
    out.mkdir(parents=True)
    write_csv(out / "seed_contrasts.csv", seed_rows)
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "diagnostic_curves.csv", curves)
    write_csv(out / "training_curves.csv", costs)
    seed_uncertainty = ("Only 1 training seed; seed SD is undefined and training-seed variation cannot be estimated. "
                        if len(seeds) == 1 else
                        f"Seed SD describes training-seed variation ({len(seeds)} observed seeds; unstable variance, no population claim). ")
    atomic_json(out / "analysis.json", {"seed_results": seed_rows, "summary": summaries, "missing_runs": missing,
                "uncertainty": seed_uncertainty + "Bootstrap resamples evaluation questions jointly across conditions/seeds conditional on observed seeds. It excludes training-seed uncertainty. Eight responses and two reward groups are NOT independent questions.",
                "primary_endpoint": "MATH500 step 300; Curriculum - Random and (RL difference) - (Base difference)",
                "diagnostic_claim": "Not used in this additional training; public model historical exposure unknown"})
    if plots:
        plot_curves(curves, costs, out)
    print("metric contrast mean seed_SD question_bootstrap_95% (conditional on observed seeds)")
    for r in summaries:
        print(r["metric"], r["contrast"], f"{r['mean']:.6f}", r["seed_sample_sd"],
              f"[{r['question_bootstrap_ci_low']:.6f}, {r['question_bootstrap_ci_high']:.6f}]")
    return summaries


def plot_curves(curves, costs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def plot(rows, field, title, filename):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        single_seed = len({r["seed"] for r in rows}) <= 1
        for role, color in (("base", "#2672b8"), ("rl", "#d77920")):
            for order, style in (("random", "--"), ("curriculum", "-")):
                selected = [r for r in rows if r["model"] == role and r["order"] == order]
                steps = sorted({r["step"] for r in selected})
                grouped = [[r[field] for r in selected if r["step"] == s and r[field] is not None] for s in steps]
                means = [statistics.mean(v) if v else float("nan") for v in grouped]
                ax.plot(steps, means, style, color=color, label=f"{role} {order}")
                if not single_seed:
                    sds = [statistics.stdev(v) if len(v) > 1 else float("nan") for v in grouped]
                    ax.fill_between(steps, [a-b for a,b in zip(means,sds)], [a+b for a,b in zip(means,sds)], color=color, alpha=0.10)
        variation = "single seed; seed SD unavailable" if single_seed else "mean ± seed SD"
        ax.set(xlabel="Problem exposures / scheduled updates", ylabel=field, title=f"{title} ({variation})")
        ax.legend()
        ax.grid(alpha=.2)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(out / f"{filename}.{suffix}", dpi=180)
        plt.close(fig)
    for difficulty in sorted({r["difficulty"] for r in curves}):
        rows = [r for r in curves if r["difficulty"] == difficulty]
        for field in ("avg_at_8", "pass_at_8", "mixed_fraction", "sampled_truncation_fraction", "sampled_mean_tokens"):
            plot(rows, field, f"Diagnostic {difficulty}", f"diagnostic_{difficulty}_{field}")
    for field in ("mixed_fraction", "truncation_fraction", "mean_response_tokens", "generated_tokens", "seconds", "entropy_nats"):
        plot(costs, field, "Additional training", f"train_{field}")
