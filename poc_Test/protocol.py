"""Small, fixed train/validation samples using the main experiment's settings."""
import copy
import random

from grpo_order.core import DIFFICULTIES, ordered_problems, seed_plan


def validate_config(c):
    spec = c.get('poc', {})
    if spec.get('train_problems') not in (12, 24) or spec.get('validation_problems') != 12:
        raise ValueError('POC requires 12/24 training problems and 12 validation problems')
    if (c['train']['max_new_tokens'] != 2048 or c['evaluation']['max_new_tokens'] != 2048
            or c['train']['responses_per_problem'] != 4
            or c['evaluation']['samples'] != 8 or not c['evaluation']['greedy']
            or c['evaluation']['problem_batch_size'] != 4
            or c['runtime']['allocated_gpu_slots'] != [0]):
        raise ValueError('POC requires the real 2048-token, train-4, eval-8+greedy/batch-4, one-GPU settings')


def make_config(source, output_dir, train_problems=12):
    c = copy.deepcopy(source)
    c['output_dir'] = str(output_dir)
    c['poc'] = {'train_problems': train_problems, 'validation_problems': 12}
    validate_config(c)
    return c


def balanced_sample(rows, count, seed):
    rng = random.Random(seed)
    selected = []
    for difficulty in DIFFICULTIES:
        group = sorted((r for r in rows if r['difficulty'] == difficulty), key=lambda r: r['id'])
        if len(group) < count // 3:
            raise ValueError(f'Not enough {difficulty} problems for POC')
        rng.shuffle(group)
        selected.extend(group[:count // 3])
    rng.shuffle(selected)
    if len({r['id'] for r in selected}) != count:
        raise ValueError('POC contains duplicate question IDs')
    return selected


def select_splits(c, splits):
    validate_config(c)
    selected = {
        'train': balanced_sample(splits['train'], c['poc']['train_problems'], c['data']['selection_seed']),
        'validation': balanced_sample(splits['validation'], 12, c['assets']['validation_selection_seed']),
    }
    if {r['id'] for r in selected['train']} & {r['id'] for r in selected['validation']}:
        raise ValueError('POC training and validation must be disjoint')
    return selected


def selection_record(c, splits):
    selected = select_splits(c, splits)
    selected['train'] = ordered_problems(selected['train'], 'random', seed_plan(c, 1)['order'])
    return {name: [{'id': r['id'], 'difficulty': r['difficulty']} for r in rows]
            for name, rows in selected.items()}
