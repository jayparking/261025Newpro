"""Run two isolated POC workers within one shared wall-clock budget."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grpo_order.core import atomic_json, code_hash, read_json
from grpo_order.storage import exclusive_lock
from poc_Test.report import write_report


class BudgetExceeded(RuntimeError):
    pass


def kill_group(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def run_child(command, env, deadline, log_path):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise BudgetExceeded('POC 전체 시간 제한에 도달했습니다.')
    # Reserve at most five seconds within the budget for termination.
    grace = min(5.0, remaining / 10)
    with Path(log_path).open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)

        def drain():
            for line in process.stdout:
                log.write(line)
                log.flush()
                try:
                    print(line, end='', flush=True)
                except BrokenPipeError:
                    pass

        thread = threading.Thread(target=drain, daemon=True)
        thread.start()
        group_killed = False
        try:
            try:
                return process.wait(timeout=max(0, deadline - time.monotonic() - grace))
            except subprocess.TimeoutExpired:
                print('POC 시간 제한: 현재 작업을 종료합니다.', flush=True)
                kill_group(process, signal.SIGTERM)
                try:
                    process.wait(timeout=max(0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    kill_group(process, signal.SIGKILL)
                    group_killed = True
                    process.wait()
                raise BudgetExceeded('POC 시간 제한으로 중단했습니다. 미완료를 성공으로 처리하지 않습니다.')
        finally:
            # Also remove descendants if the direct child has already exited.
            if not group_killed:
                kill_group(process, signal.SIGKILL)
            process.wait()
            thread.join(timeout=5)
            if not thread.is_alive():
                process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, help='완료된 setup 세션. 생략하면 최근 setup')
    parser.add_argument('--gpu', default=os.getenv('CUDA_VISIBLE_DEVICES', '0'))
    parser.add_argument('--train-problems', type=int, choices=(12, 24), default=12)
    parser.add_argument('--minutes', type=float, default=60, help='전체 POC 시간 제한, 최대 60분')
    parser.add_argument('--dry-run', action='store_true', help='설치·GPU·파일 생성 없이 범위 표시')
    args = parser.parse_args()
    if not 0 < args.minutes <= 60:
        parser.error('--minutes는 0보다 크고 60 이하여야 합니다.')
    if not args.gpu or ',' in args.gpu or args.gpu == '-1':
        parser.error('GPU 한 개만 지정하세요.')
    print(f'POC: Base → SimpleRL, 각각 학습 {args.train_problems}문제 + 별도 validation 12문제\n'
          f'2048토큰, 학습 4응답, 평가 8+greedy/4문제 배치, GPU={args.gpu}\n'
          f'전체 제한 {args.minutes:g}분(두 모델 합계, 준비 검사 포함). 본실험 자동 시작 없음.', flush=True)
    if args.dry_run:
        return 0
    if sys.platform != 'linux':
        parser.error('실제 실행은 Linux NVIDIA GPU 서버에서 하세요.')
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        parser.error('setup에서 준비한 Python 3.11/3.12를 사용하세요.')
    os.chdir(ROOT)
    latest = ROOT / 'runs' / 'single_gpu_sessions' / 'latest.json'
    if args.session:
        session = args.session.resolve()
    elif latest.is_file():
        session = Path(read_json(latest)['session']).resolve()
    else:
        parser.error('완료된 setup이 없습니다. 먼저 bash 1.setup.sh를 실행하세요.')
    if not (session / 'state.json').is_file():
        parser.error(f'setup 세션을 찾을 수 없습니다: {session}')
    with exclusive_lock(ROOT / '.first.lock'), exclusive_lock(session / 'pipeline.lock'):
        output = ROOT / 'poc_Test' / 'runs' / str(time.time_ns())
        output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        deadline = started + args.minutes * 60
        attempt = {'setup_session': str(session), 'code': code_hash(), 'limit_seconds': args.minutes * 60,
                   'train_problems': args.train_problems, 'gpu': args.gpu}
        atomic_json(output / 'attempt.json', attempt)
        print(f'결과 폴더: {output}', flush=True)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
               'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONUNBUFFERED': '1',
               'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        common = ['--session', str(session), '--output', str(output), '--train-problems', str(args.train_problems)]
        worker = [sys.executable, str(ROOT / 'poc_Test' / 'worker.py')]
        phases = []
        status, error, code = 'failed', None, 1
        try:
            for phase in ('prepare', 'environment', 'base', 'rl'):
                command = (worker + [phase] + common if phase != 'environment' else
                           [sys.executable, '-m', 'grpo_order', 'preflight', '--environment-only',
                            '--config', str(output / 'config.json'), '--report', str(output / 'environment.json')])
                phase_started = time.monotonic()
                print(f'[{phase}] 시작, 남은 시간 {max(0, deadline-phase_started)/60:.1f}분', flush=True)
                result = run_child(command, env, deadline, output / f'{phase}.log')
                phases.append({'phase': phase, 'seconds': time.monotonic() - phase_started, 'exit_code': result})
                if result:
                    raise RuntimeError(f'{phase} 실패(exit={result}); {output / (phase + ".log")} 확인')
                if phase == 'environment':
                    probe = read_json(output / 'environment.json')
                    if probe['errors']:
                        raise RuntimeError(str(probe['errors']))
                    uuid = probe['hardware']['uuid_mapping'][0]['uuid']
                    env.update(CUDA_VISIBLE_DEVICES=uuid, GRPO_ALLOCATED_SLOT='0', GRPO_ASSIGNED_GPU_UUID=uuid)
            status, code = 'completed', 0
        except BudgetExceeded as exc:
            status, error, code = 'timed_out', str(exc), 124
        except KeyboardInterrupt:
            status, error, code = 'interrupted', '사용자 또는 외부 신호로 중단', 130
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        elapsed = time.monotonic() - started
        try:
            write_report(output, status, elapsed, phases, error)
        except Exception as exc:
            status, code = 'invalid_report', 1
            error = f'{error or ""}; report validation: {type(exc).__name__}: {exc}'
            atomic_json(output / 'summary.json', {'status': status, 'error': error, 'elapsed_seconds': elapsed})
            print(error, file=sys.stderr, flush=True)
        atomic_json(output / 'attempt.json', {**attempt, 'status': status, 'elapsed_seconds': elapsed, 'exit_code': code})
        print(f'요약: {output / "summary.json"}\n본실험은 시작하지 않았습니다.', flush=True)
        return code


if __name__ == '__main__':
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(main())
