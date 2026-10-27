"""Report observations, never declare scientific success or start the full experiment."""
from collections import Counter
import math
from pathlib import Path
import statistics

from grpo_order.core import atomic_json, code_hash, digest, read_json
from grpo_order.integrity import validate_completion, validate_evaluation, validate_steps
from grpo_order.storage import run_path


def model_report(c, role, selection, manifest_hash):
    root = run_path(c, 'poc', role, 'random', 1)
    report = {'model': role, 'run': str(root), 'complete': False, 'committed_updates': 0}
    if not (root / 'metadata.json').exists():
        return report
    metadata = read_json(root / 'metadata.json')
    contract = metadata['contract']
    expected = {'config': digest(c), 'mode': 'poc', 'role': role, 'order': 'random',
                'seed': 1, 'data': manifest_hash, 'code': code_hash()}
    if any(contract[k] != v for k, v in expected.items()):
        raise ValueError(f'POC identity mismatch: {role}')
    steps = [read_json(p) for p in sorted((root / 'steps').glob('*.json'))]
    responses = [r for step in steps for r in step['responses']]
    report.update(
        committed_updates=len(steps), original_model_hash=contract['model'],
        group_counts=dict(Counter(step['group_type'] for step in steps)),
        training_seconds=sum(step['seconds'] for step in steps),
        seconds_per_update=statistics.mean(step['seconds'] for step in steps) if steps else None,
        training_truncation_fraction=sum(r['truncated'] for r in responses) / len(responses) if responses else None,
        training_mean_tokens=statistics.mean(r['generated_tokens'] for r in responses) if responses else None,
        training_failure_reasons=dict(Counter(r['failure_reason'] for r in responses if r.get('failure_reason'))),
        training_peak_allocated_gib=max((s['gpu_peak_allocated_bytes'] for s in steps), default=0) / 1024**3,
        training_peak_reserved_gib=max((s['gpu_peak_reserved_bytes'] for s in steps), default=0) / 1024**3,
        groups_by_difficulty={d: dict(Counter(s['group_type'] for s in steps if s['difficulty'] == d))
                              for d in ('Easy', 'Medium', 'Hard')},
    )
    if not (root / 'complete.json').exists():
        return report
    validate_completion(metadata, read_json(root / 'complete.json'), c['poc']['train_problems'])
    validate_steps(c, metadata, steps, selection['train'])
    ev = read_json(root / 'eval' / f'validation_step-{len(steps):06d}.json')
    validate_evaluation(ev, contract, selection['validation'], 'validation', len(steps))
    all_eval = [r for q in ev['questions'] for r in q['sampled'] + [q['greedy_response']]]
    report.update(complete=True, adapter_changed=ev['adapter_hash'] != metadata['lora']['initial_adapter_hash'],
                  validation_metrics=ev['metrics'], validation_seconds=ev['wall_seconds_this_attempt'],
                  seconds_per_evaluation_batch=ev['wall_seconds_this_attempt'] / 3,
                  validation_failure_reasons=dict(Counter(r['failure_reason'] for r in all_eval if r.get('failure_reason'))),
                  validation_peak_allocated_gib=ev['poc_gpu_memory']['peak_allocated_bytes'] / 1024**3,
                  validation_peak_reserved_gib=ev['poc_gpu_memory']['peak_reserved_bytes'] / 1024**3)
    return report


def project_full(c, counts, records):
    if len(records) != 2 or not all(r['complete'] for r in records):
        return None
    e = c['evaluation']
    runs_per_role = 2 * len(c['seeds']['training'])
    initial = (1 if e.get('step0_share_across_seeds') else len(c['seeds']['training'])) if e.get('step0_cache') else runs_per_role
    batches = 0
    for split in ('diagnostic', 'math500'):
        schedule = e[split + '_steps']
        batches += math.ceil(counts[split] / e['problem_batch_size']) * (
            initial * int(0 in schedule) + runs_per_role * sum(s != 0 for s in schedule))
        if e.get('step0_cache') and 0 in schedule:
            batches += runs_per_role  # Initial-output witness per split/run.
    training = sum(r['seconds_per_update'] * runs_per_role * counts['train'] for r in records)
    evaluation = sum(r['seconds_per_evaluation_batch'] * batches for r in records)
    return {'training_hours': training / 3600, 'evaluation_hours': evaluation / 3600,
            'compute_hours': (training + evaluation) / 3600,
            'runs': runs_per_role * 2, 'evaluation_batches': batches * 2,
            'scope': 'Excludes setup, loading, checkpoints, final analysis and retries. Assumes initial-cache hits.',
            'uncertainty': 'Only 12/24 training and 12 validation questions per model. Length/difficulty and throughput can change; this is not a completion-time guarantee.'}


def write_report(output, status, elapsed, phases, error=None):
    output = Path(output)
    report = {'status': status, 'elapsed_seconds': elapsed, 'phases': phases, 'error': error,
              'models': [], 'full_experiment_projection': None,
              'meaning': 'Execution feasibility only. No score-improvement requirement; no automatic full run.',
              'billing': 'Stopping POC processes does not stop or release the rented server.'}
    if (output / 'config.json').exists():
        c = read_json(output / 'config.json')
        selected = read_json(output / 'selection.json')
        if selected['hash'] != digest(selected['questions']):
            raise ValueError('POC selection checksum mismatch')
        report['models'] = [model_report(c, role, selected['questions'], selected['manifest_hash']) for role in ('base', 'rl')]
        if status == 'completed' and not all(r['complete'] for r in report['models']):
            raise ValueError('POC worker exited without completing all required observations')
        hashes = [r['original_model_hash'] for r in report['models'] if r.get('original_model_hash')]
        if len(hashes) == 2 and hashes[0] == hashes[1]:
            raise ValueError('Base and SimpleRL resolve to the same original model')
        report['full_experiment_projection'] = project_full(read_json(output / 'source_config.json'),
                                                           read_json(output / 'split_counts.json'), report['models'])
    atomic_json(output / 'summary.json', report)
    lines = [f'POC 상태: {status}', f'총 경과: {elapsed / 60:.1f}분', '']
    for r in report['models']:
        lines.append(f"{r['model']}: 완료={r['complete']}, 학습 {r['committed_updates']}문제")
        if r['committed_updates']:
            lines.append(f"  그룹={r['group_counts']}, {r['seconds_per_update']:.1f}초/update, "
                         f"학습 잘림={r['training_truncation_fraction']:.1%}, "
                         f"peak allocated={r['training_peak_allocated_gib']:.2f} GiB")
            lines.append(f"  학습 답 추출 실패={r['training_failure_reasons'].get('answer_extraction_failed', 0)}응답")
            if not r['group_counts'].get('mixed'):
                lines.append('  관측한 학습 문제에 정답/오답 혼합 그룹이 없습니다. 원인 확인이 필요합니다.')
        if r['complete']:
            lines.append(f"  평가 {r['seconds_per_evaluation_batch']:.1f}초/4문제, "
                         f"peak allocated={r['validation_peak_allocated_gib']:.2f} GiB, 가중치 변경={r['adapter_changed']}")
            lines.append(f"  평가 잘림: 샘플={r['validation_metrics']['sampled_truncation_fraction']:.1%}, "
                         f"greedy={r['validation_metrics']['greedy_truncation']:.1%}; "
                         f"답 추출 실패={r['validation_failure_reasons'].get('answer_extraction_failed', 0)}응답")
    projection = report['full_experiment_projection']
    if projection:
        hours = projection['compute_hours']
        lines += ['', f'본실험 학습+평가 계산량 환산: 약 {hours:.1f}시간 (추정치)',
                  '설치·로딩·체크포인트·분석·재시도 시간은 별도이며, 작은 표본으로 전체 시간을 보장할 수 없습니다.']
        if hours + elapsed / 3600 > 6:
            lines.append('이 추정치와 POC 시간만으로도 총 6시간을 넘습니다. 본실험 시작 전 예산을 재검토하세요.')
        else:
            lines.append('6시간 이하 추정이어도 완료 보장은 아닙니다. setup 시간·추가 비용을 포함해 검토하세요.')
    else:
        lines += ['', '두 모델의 학습·평가가 완료되지 않아 본실험 시간 추정치를 만들지 않았습니다.']
    lines += ['', '학습 혼합 그룹·출력 잘림·답 추출 실패·메모리·예산을 확인하세요. 정답률 향상은 통과 조건이 아닙니다.',
              '본실험은 자동 시작하지 않습니다. 진행할 경우 원래 모델에서 새 LoRA/optimizer로 시작합니다.',
              'POC 종료는 서버 대여 종료가 아닙니다. 서버 비용은 대여 서비스에서 별도로 관리하세요.']
    if error:
        lines.append(f'오류: {error}')
    text = '\n'.join(lines) + '\n'
    (output / 'summary.txt').write_text(text)
    print(text, flush=True)
    return report
