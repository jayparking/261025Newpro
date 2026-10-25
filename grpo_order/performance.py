"""Summarize observed full-run quality and cost without projecting GPU throughput."""
from .core import digest, read_json
from .storage import run_path


def experiment_workload(c, splits):
    """Counts only; output length and speed need measurements on the real GPU."""
    runs = len(c['models']) * 2 * len(c['seeds']['training'])
    e = c['evaluation']
    batches = e.get('problem_batch_size', 1)
    initial_groups = (len(c['models']) if e.get('step0_share_across_seeds', False)
                      else len(c['models']) * len(c['seeds']['training'])) if e.get('step0_cache') else runs
    evaluated = witnesses = 0
    for split in ('diagnostic', 'math500'):
        n = len(splits[split])
        schedule = e[split + '_steps']
        evaluated += n * (runs * sum(step != 0 for step in schedule) + initial_groups * int(0 in schedule))
        if e.get('step0_cache') and 0 in schedule:
            witnesses += runs * min(n, batches)
    return {'runs': runs, 'training_updates': runs * len(splits['train']),
            'training_responses': runs * len(splits['train']) * 4,
            'evaluation_problem_visits_excluding_witnesses': evaluated,
            'evaluation_responses_excluding_witnesses': evaluated * 9,
            'step0_witness_responses': witnesses * 9,
            'evaluation_problem_batch_size': batches,
            'assumptions': 'All eligible initial-cache requests hit; excludes retries. No time estimate without GPU measurements.'}


def performance_report(c):
    records = []
    for seed in c['seeds']['training']:
        for role in ('base', 'rl'):
            for order in ('random', 'curriculum'):
                root = run_path(c, 'full', role, order, seed)
                if not (root / 'metadata.json').exists():
                    continue
                contract = read_json(root / 'metadata.json')['contract']
                if contract['config'] != digest(c):
                    raise ValueError(f'Performance report config mismatch: {root}')
                steps = [read_json(p) for p in sorted((root / 'steps').glob('*.json'))]
                seconds = sum(r['seconds'] for r in steps)
                tokens = sum(sum(x['generated_tokens'] for x in r['responses']) for r in steps)
                generation = sum(r.get('generation_seconds', 0) for r in steps)
                evaluations = []
                for p in sorted((root / 'eval').glob('*_step-*.json')):
                    result = read_json(p)
                    if result['contract'] != contract:
                        raise ValueError(f'Performance report evaluation contract mismatch: {p}')
                    evaluations.append({
                        'split': result['split'], 'step': result['step'], 'metrics': result['metrics'],
                        'cache_hit': result.get('cache', {}).get('hit', False),
                        'last_attempt_seconds': result['wall_seconds_this_attempt'],
                        'last_attempt_generated_tokens': result['generated_tokens_this_attempt'],
                    })
                done_path = root / 'complete.json'
                done = read_json(done_path) if done_path.exists() else None
                if done and done['contract'] != contract:
                    raise ValueError(f'Performance report completion contract mismatch: {root}')
                records.append({
                    'model': role, 'order': order, 'seed': seed, 'run': str(root),
                    'complete': done is not None, 'valid_without_step_skips': done.get('valid_without_step_skips') if done else None,
                    'committed_updates': len(steps), 'training_seconds': seconds,
                    'seconds_per_update': seconds / len(steps) if steps else None,
                    'generated_training_tokens': tokens,
                    'training_tokens_per_second': tokens / seconds if seconds else None,
                    'generation_only_tokens_per_second': tokens / generation if generation else None,
                    'generation_seconds': generation,
                    'grading_seconds': sum(r.get('grading_seconds', 0) for r in steps),
                    'optimization_seconds': sum(r.get('optimization_seconds', 0) for r in steps),
                    'peak_training_allocated_gib': max((r['gpu_peak_allocated_bytes'] for r in steps), default=0) / 1024**3,
                    'peak_training_reserved_gib': max((r['gpu_peak_reserved_bytes'] for r in steps), default=0) / 1024**3,
                    'evaluations': evaluations,
                })
    return {'config_hash': digest(c), 'gpu_count': len(c['runtime']['allocated_gpu_slots']),
            'expected_runs': 12, 'completed_runs': sum(r['complete'] for r in records), 'runs': records,
            'timing_notes': [
                'Training rates use committed updates only and exclude checkpoints, evaluation, startup and discarded attempts.',
                'Generation-only rate excludes grading and optimization; it is not end-to-end training throughput.',
                'Evaluation attempt costs are not total costs across resumes. Cache hits are explicitly marked.',
                'Peak CUDA allocated/reserved memory is per training process, not total GPU memory usage.',
            ]}
