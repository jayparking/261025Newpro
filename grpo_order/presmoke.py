"""Configuration and observed results for the optional one-update GPU check."""
import copy

from .core import DIFFICULTIES, atomic_json, digest, read_json
from .data import load_prepared
from .integrity import validate_completion, validate_evaluation, validate_steps
from .storage import run_path


def quick_config(source, output_dir=None):
    c = copy.deepcopy(source)
    if output_dir is not None:
        c['output_dir'] = str(output_dir)
    c['train']['max_new_tokens'] = 32
    c['evaluation'].update(max_new_tokens=32, problem_batch_size=1, step0_cache=False)
    return c


def select_one(rows):
    if not rows:
        raise ValueError('presmoke requires a nonempty prepared split')
    # Fixed easiest available question; no search for a successful answer.
    return [min(rows, key=lambda r: (DIFFICULTIES.index(r['difficulty']), r['id']))]


def report(c):
    root = run_path(c, 'presmoke', 'base', 'random', 1)
    metadata = read_json(root / 'metadata.json')
    contract = metadata['contract']
    manifest, splits = load_prepared(c)
    expected = {'config': digest(c), 'data': manifest['hash'], 'mode': 'presmoke',
                'role': 'base', 'order': 'random', 'seed': 1}
    if any(contract[k] != v for k, v in expected.items()):
        raise ValueError('presmoke run identity differs from the setup configuration')
    completion = read_json(root / 'complete.json')
    validate_completion(metadata, completion, 1)
    train_rows, test_rows = select_one(splits['train']), select_one(splits['diagnostic'])
    if train_rows[0]['id'] == test_rows[0]['id']:
        raise ValueError('presmoke train and diagnostic questions must be distinct')
    steps = [read_json(p) for p in sorted((root / 'steps').glob('*.json'))]
    validate_steps(c, metadata, steps, train_rows)
    result = read_json(root / 'eval' / 'diagnostic_step-000001.json')
    validate_evaluation(result, contract, test_rows, 'diagnostic', 1)
    step = steps[0]
    changed = result['adapter_hash'] != metadata['lora']['initial_adapter_hash']
    summary = {
        'status': 'passed', 'scope': 'one-update execution check, not model-quality validation',
        'run': str(root), 'contract': contract, 'optimizer_steps': 1,
        'training_problem': train_rows[0], 'test_problem': test_rows[0],
        'training_responses': step['responses'], 'training_reward_mean': step['reward_mean'],
        'training_group_type': step['group_type'], 'gradient_norm_pre_clip': step['gradient_norm_pre_clip'],
        'adapter_changed': changed, 'training_seconds': step['seconds'],
        'test_metrics': result['metrics'], 'test_responses': result['questions'][0],
        'note': 'Four equal rewards yield zero GRPO advantages; the first optimizer step can leave weights unchanged. '
                '32-token outputs may be truncated. Correctness and weight changes are reported, not required to pass.',
    }
    atomic_json(root / 'summary.json', summary)
    print(f'presmoke 완료: Base, 학습 1문제 / optimizer 1회 / 진단 1문제, 출력 상한 32', flush=True)
    print(f'학습 reward={step["reward_mean"]:.2f}, group={step["group_type"]}, '
          f'가중치 변경={changed}; 진단 avg@8={result["metrics"]["avg_at_8"]:.3f}, '
          f'greedy={result["metrics"]["greedy"]:.0f}', flush=True)
    if not changed:
        print('가중치 변경 없음: 4응답의 보상이 같으면 advantage=0이라 첫 업데이트에서도 정상적으로 발생할 수 있습니다.', flush=True)
    print(f'문제·실제 응답·채점·소요 시간: {root / "summary.json"}', flush=True)
    return summary
