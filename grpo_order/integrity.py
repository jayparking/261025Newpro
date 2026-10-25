"""Validate the actual observations before accepting a completed experiment."""
import math

from .core import advantages, digest, question_metrics


def validate_completion(metadata, completion, expected_updates):
    if completion.get("contract") != metadata["contract"]:
        raise ValueError("Completion contract differs from run metadata")
    expected = {"cursor": expected_updates, "optimizer_steps": expected_updates,
                "skipped_steps": 0, "generated_responses": 4 * expected_updates,
                "expected_updates": expected_updates, "valid_without_step_skips": True}
    if any(completion.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Run is not a valid completed {expected_updates}-update experiment")


def validate_responses(responses, count):
    if len(responses) != count or [r["sample_index"] for r in responses] != list(range(count)):
        raise ValueError("Missing, duplicated or reordered response samples")
    for r in responses:
        if r["reward"] not in (0, 1):
            raise ValueError("Expected a binary correctness reward")
        if not r["token_ids"] or r["generated_tokens"] != len(r["token_ids"]):
            raise ValueError("Response token count mismatch")
        if r.get("failure_reason", "") and r["failure_reason"].startswith("grader_error:"):
            raise ValueError("Grader failures cannot be accepted as wrong answers")


def validate_steps(c, metadata, steps, expected_rows):
    sequence = [{"id": r["id"], "difficulty": r["difficulty"]} for r in expected_rows]
    if metadata["sequence"] != sequence or metadata["contract"]["sequence"] != digest([r["id"] for r in expected_rows]):
        raise ValueError("Training set/order differs from the specified seeded experiment")
    if len(steps) != len(expected_rows):
        raise ValueError("Missing or extra committed training steps")
    tokens = 0
    for index, (r, row) in enumerate(zip(steps, expected_rows), 1):
        if (r["exposure"] != index or r["problem_id"] != row["id"] or r["difficulty"] != row["difficulty"]
                or r["optimizer_steps"] != index or r["skipped_steps"] != 0 or not r["optimizer_step_performed"]):
            raise ValueError(f"Invalid training step/order at exposure {index}")
        validate_responses(r["responses"], 4)
        mean, std, adv, kind = advantages([x["reward"] for x in r["responses"]], c["train"]["advantage_epsilon"])
        if (r["reward_mean"] != mean or not math.isclose(r["reward_std_population"], std)
                or r["advantages"] != adv or r["group_type"] != kind):
            raise ValueError("Saved rewards/advantages disagree")
        if any(r.get(k) is None or not math.isfinite(r[k]) for k in ("loss", "gradient_norm_pre_clip", "token_entropy_nats")):
            raise ValueError("Non-finite training statistics in completed run")
        tokens += sum(x["generated_tokens"] for x in r["responses"])
        if r["cumulative_generated_tokens"] != tokens:
            raise ValueError("Cumulative training token count mismatch")


def validate_evaluation(result, contract, rows, split, step):
    from .evaluation import summarize
    if result["contract"] != contract or result["split"] != split or result["step"] != step:
        raise ValueError("Evaluation contract/split/step differs from experiment")
    questions = result["questions"]
    if [q["id"] for q in questions] != [r["id"] for r in rows]:
        raise ValueError(f"{split} evaluation has missing, extra or reordered questions")
    for q, row in zip(questions, rows):
        if q["difficulty"] != row["difficulty"]:
            raise ValueError("Evaluation difficulty differs from prepared data")
        validate_responses(q["sampled"], 8)
        validate_responses([q["greedy_response"]], 1)
        derived = question_metrics([r["reward"] for r in q["sampled"]], q["greedy_response"]["reward"])
        derived.update(sampled_truncation_fraction=sum(r["truncated"] for r in q["sampled"]) / 8,
                       sampled_mean_tokens=sum(r["generated_tokens"] for r in q["sampled"]) / 8,
                       greedy_truncation=float(q["greedy_response"]["truncated"]),
                       greedy_tokens=q["greedy_response"]["generated_tokens"])
        if any(q[k] != v for k, v in derived.items()):
            raise ValueError("Evaluation metrics disagree with saved responses")
    if result["metrics"] != summarize(questions):
        raise ValueError("Evaluation aggregate metrics disagree with question records")
    expected_difficulty = {d: summarize([q for q in questions if q["difficulty"] == d])
                           for d in sorted({q["difficulty"] for q in questions})}
    if result["by_difficulty"] != expected_difficulty:
        raise ValueError("Evaluation difficulty metrics disagree with question records")
