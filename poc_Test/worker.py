"""Budgeted child processes: setup verification, then one fresh model per process."""
import argparse
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grpo_order.core import atomic_json, code_hash, digest, file_hash, load_config, packages, read_json
from grpo_order.data import load_prepared
from grpo_order.performance import experiment_workload
from poc_Test.protocol import make_config, selection_record
from scripts.first_pipeline import verify_import


def prepare(session, output, count):
    state = read_json(session / 'state.json')
    if state.get('schema') != 'single-gpu-stages-v1' or not state.get('setup_complete'):
        raise ValueError('먼저 bash 1.setup.sh를 완료하세요. SMOKE 완료는 POC의 필수 조건이 아닙니다.')
    source = load_config(session / 'configs' / 'session.json')
    if (state['config_hash'] != digest(source) or state['code'] != code_hash()
            or state['packages'] != packages()):
        raise ValueError('setup의 코드/설정/패키지가 달라졌습니다. 현재 코드로 새 setup을 실행하세요.')
    receipts = {role: file_hash(Path(spec['path']) / '.asset.json') for role, spec in source['models'].items()}
    receipts.update({name: file_hash(Path(spec['path']) / '.asset.json')
                     for name, spec in source['assets']['datasets'].items()})
    if receipts != state.get('assets'):
        raise ValueError('setup 이후 모델/데이터 영수증이 변경되었습니다.')
    verify_import(source)
    manifest, splits = load_prepared(source)
    c = make_config(source, output / 'runs', count)
    selected = selection_record(c, splits)
    atomic_json(output / 'source_config.json', source)
    atomic_json(output / 'selection.json', {'manifest_hash': manifest['hash'], 'questions': selected,
                                          'hash': digest(selected)})
    atomic_json(output / 'full_workload.json', experiment_workload(source, splits))
    atomic_json(output / 'split_counts.json', {name: len(rows) for name, rows in splits.items()})
    atomic_json(output / 'config.json', c)
    load_config(output / 'config.json')
    print(f'준비 완료: 모델당 학습 {count}문제, validation 12문제; 원래 모델에서 새로 시작', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'base', 'rl'))
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--train-problems', required=True, type=int, choices=(12, 24))
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.phase == 'prepare':
        prepare(args.session, args.output, args.train_problems)
        return
    from grpo_order.grading import check_grader
    from grpo_order.train import train
    c = load_config(args.output / 'config.json')
    if code_hash() != read_json(args.output / 'attempt.json')['code']:
        raise ValueError('POC 실행 도중 코드가 변경되었습니다.')
    check_grader()
    # New output paths and resume=False prevent taking weights from a previous run.
    train(c, args.phase, 'random', 1, mode='poc', resume=False)


if __name__ == '__main__':
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'POC stopped by signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    main()
