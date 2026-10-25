"""Pair conditions by training seed and evaluation question, never by response."""
import csv
from pathlib import Path
import random
import statistics

from .core import atomic_json, digest, read_json
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
    if mode == "smoke":
        raise ValueError("Smoke results are not scientific experiments")
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite analysis {out}")
    out.mkdir(parents=True)
    all_runs, missing, curves, costs = {}, [], [], []
    signatures = set()
    role_models = {"base": set(), "rl": set()}
    selected_seeds = c["seeds"]["training"] if mode == "full" else [s for s in c["seeds"]["training"]
                     if any(run_path(c, mode, role, order, s).exists() for role in ("base", "rl") for order in ("random", "curriculum"))] or [1]
    for seed in selected_seeds:
        for role in ("base", "rl"):
            for order in ("random", "curriculum"):
                root = run_path(c, mode, role, order, seed)
                if not (root / "complete.json").exists():
                    missing.append(str(root))
                    continue
                metadata = read_json(root / "metadata.json")
                completion = read_json(root / "complete.json")
                if completion["skipped_steps"]:
                    raise ValueError(f"{root} has skipped optimizer steps; inspect before drawing protocol conclusions")
                if metadata["contract"]["config"] != digest(c):
                    raise ValueError(f"Analysis config differs from {root}")
                signatures.add((metadata["contract"]["data"], metadata["contract"]["config"], metadata["contract"]["code"], digest(metadata["contract"]["packages"])))
                role_models[role].add(metadata["contract"]["model"])
                evaluation = read_json(root / "eval" / "math500_step-000300.json")
                if evaluation["contract"] != metadata["contract"]:
                    raise ValueError(f"Evaluation contract differs from run: {root}")
                all_runs[role, seed, order] = evaluation
                for path in sorted((root / "eval").glob("diagnostic_step-*.json")):
                    result = read_json(path)
                    for difficulty, m in {"All": result["metrics"], **result["by_difficulty"]}.items():
                        curves.append({"model": role, "seed": seed, "order": order, "step": result["step"],
                                       "difficulty": difficulty, **m})
                for p in sorted((root / "steps").glob("*.json")):
                    r = read_json(p)
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
    write_csv(out / "seed_contrasts.csv", seed_rows)
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "diagnostic_curves.csv", curves)
    write_csv(out / "training_curves.csv", costs)
    atomic_json(out / "analysis.json", {"seed_results": seed_rows, "summary": summaries, "missing_runs": missing,
                "uncertainty": "Seed SD describes training-seed variation (only 3 seeds; unstable variance, no population claim). Bootstrap resamples evaluation questions jointly across conditions/seeds conditional on observed seeds. Eight responses and two reward groups are NOT independent questions.",
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
        for role, color in (("base", "#2672b8"), ("rl", "#d77920")):
            for order, style in (("random", "--"), ("curriculum", "-")):
                selected = [r for r in rows if r["model"] == role and r["order"] == order]
                steps = sorted({r["step"] for r in selected})
                grouped = [[r[field] for r in selected if r["step"] == s and r[field] is not None] for s in steps]
                means = [statistics.mean(v) if v else float("nan") for v in grouped]
                sds = [statistics.stdev(v) if len(v) > 1 else 0 for v in grouped]
                ax.plot(steps, means, style, color=color, label=f"{role} {order}")
                ax.fill_between(steps, [a-b for a,b in zip(means,sds)], [a+b for a,b in zip(means,sds)], color=color, alpha=0.10)
        ax.set(xlabel="Problem exposures / scheduled updates", ylabel=field, title=title + " (mean ± seed SD)")
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
