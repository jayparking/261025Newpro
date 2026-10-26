"""Independent setup, smoke and real experiment stages on one GPU."""
import argparse
import copy
import os
from pathlib import Path
import shlex
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.first_pipeline import Pipeline, immutable_json, session_config, verify_import, verify_smoke
from grpo_order.core import atomic_json, code_hash, digest, load_config, packages, read_json
from grpo_order.data import load_prepared
from grpo_order.freeze import verify_freeze
from grpo_order.performance import experiment_workload, performance_report
from grpo_order.presmoke import quick_config, report as presmoke_report
from grpo_order.storage import exclusive_lock

SESSIONS = ROOT / 'runs' / 'single_gpu_sessions'
LATEST = SESSIONS / 'latest.json'
ENTRIES = {'setup': '1.setup.sh', 'presmoke': '2.1presmoke.sh', 'smoke': '2.SMOKE.sh', 'runreal': '3.runreal.sh'}


class StagedPipeline(Pipeline):
    def check_ready(self, c):
        self.check_identity()
        if not self.state.get('setup_complete'):
            raise ValueError('setup이 완료되지 않았습니다. 1.setup.sh --session DIR로 재개하세요.')
        self.pin_assets(c)
        verify_import(c)
        load_prepared(c)

    def setup(self, c, config):
        if self.state.get('setup_complete'):
            self.check_ready(c)
            return
        self.run('environment-lock', [sys.executable, '-m', 'pip', 'freeze'], once=True)
        self.cli('environment', 'preflight', config, '--environment-only', '--report',
                 self.session / 'logs' / f'environment-{time.time_ns()}.json')
        self.cli('download', 'download', config, '--only', 'all')
        self.pin_assets(c)
        if not Path(c['assets']['converted_dir']).exists():
            self.cli('import', 'import', config)
        verify_import(c)
        if not Path(c['data']['prepared_dir']).exists():
            self.cli('prepare', 'prepare', config)
        load_prepared(c)
        self.cli('preflight', 'preflight', config, '--report',
                 self.session / 'logs' / f'preflight-{time.time_ns()}.json')
        self.state['setup_complete'] = True
        self.save()

    def presmoke(self, c):
        self.check_ready(c)
        quick = quick_config(c, self.session / 'runs')
        path = immutable_json(self.session / 'configs' / 'presmoke.json', quick)
        load_config(path)
        # Repeated invocation verifies the committed run; interrupted work resumes.
        self.cli('presmoke-one-problem', 'presmoke', path)
        summary = presmoke_report(quick)
        self.state['presmoke_complete'] = True
        self.state['presmoke_report'] = str(Path(summary['run']) / 'summary.json')
        self.save()

    def smoke(self, c, config):
        self.check_ready(c)
        if not self.state.get('smoke_complete'):
            self.cli('smoke-stop', 'smoke', config, '--stop-after', '1', '--resume-failed', once=True)
            self.cli('smoke-resume', 'resume', config, '--mode', 'smoke')
        verify_smoke(c)
        self.state['smoke_complete'] = True
        self.save()

    def real_config(self, c):
        final = copy.deepcopy(c)
        final['output_dir'] = str(self.session / 'runs' / 'real')
        final['evaluation']['length_finalized'] = True
        path = immutable_json(self.session / 'configs' / 'real.json', final)
        load_config(path)
        immutable_json(self.session / 'selection.json', {
            'config': str(path), 'config_hash': digest(final),
            'method': 'Explicit runreal stage uses the setup configuration; no tuning pilots or automatic hyperparameter selection',
            'prompt': final['prompt'], 'train_max_new_tokens': final['train']['max_new_tokens'],
            'evaluation_max_new_tokens': final['evaluation']['max_new_tokens'], 'lr': final['train']['lr'],
        })
        return final, path

    def runreal(self, c):
        self.check_ready(c)
        if not self.state.get('smoke_complete'):
            raise ValueError('먼저 2.SMOKE.sh를 완료하세요. runreal은 SMOKE를 자동 실행하지 않습니다.')
        verify_smoke(c)
        final, path = self.real_config(c)
        _, splits = load_prepared(final)
        workload = experiment_workload(final, splits)
        runs = workload['runs']
        atomic_json(self.session / 'workload.json', workload)
        print(f'실제 실행: GPU 1장, {runs} run × 300 updates; prompt={final["prompt"]["mode"]}, '
              f'학습/평가 출력 상한={final["train"]["max_new_tokens"]}/{final["evaluation"]["max_new_tokens"]}, '
              f'LR={final["train"]["lr"]}', flush=True)
        print('설정 탐색 파일럿 없이 저장한 설정으로 실행합니다. 초기 진단/MATH500 평가부터 진행합니다.', flush=True)
        print(f'평가 시점(update): 진단={final["evaluation"]["diagnostic_steps"]}, '
              f'MATH500={final["evaluation"]["math500_steps"]}; '
              f'체크포인트 보존 시점={final["train"]["checkpoints"]}', flush=True)
        print(f'작업량: 학습 {workload["training_responses"]:,}응답, '
              f'평가 {workload["evaluation_responses_excluding_witnesses"]:,}응답 '
              f'+ 초기 확인 {workload["step0_witness_responses"]:,}응답. '
              f'평가 배치={workload["evaluation_problem_batch_size"]}문제, '
              f'attention={final["runtime"].get("attention_implementation", "eager")}', flush=True)
        frozen = Path(final['output_dir']) / 'protocol-freeze.json'
        if frozen.exists():
            verify_freeze(final)
        else:
            self.cli('freeze', 'freeze', path)
        if self.state.get('real_complete'):
            print(f'이미 완료: {self.state["analysis"]}', flush=True)
            self.write_performance(final)
            return
        started = time.monotonic()
        try:
            self.cli(f'full-{runs}-runs', 'full', path, '--resume-failed')
            report = self.session / 'analysis' / str(time.time_ns())
            self.cli('analyze', 'analyze', path, '--mode', 'full', '--output', report)
            atomic_json(self.session / 'finished.json', {'final_config': str(path), 'full_runs': runs,
                                                        'analysis': str(report)})
            self.state['analysis'] = str(report)
            self.state['real_complete'] = True
            self.save()
            print(f'{runs}회 본실험·최종 평가·분석 완료: {report}', flush=True)
        finally:
            self.state.setdefault('real_attempt_seconds', []).append(time.monotonic() - started)
            self.save()
            self.write_performance(final)

    def write_performance(self, final):
        # Include committed updates even if this attempt was interrupted.
        summary = performance_report(final)
        summary['real_attempt_seconds'] = self.state.get('real_attempt_seconds', [])
        summary['wall_time_scope'] = 'full launcher + analysis attempts; excludes setup, smoke, freeze and time between attempts'
        destination = self.session / 'performance.json'
        atomic_json(destination, summary)
        print(f'성능 보고: {destination} (완료 {summary["completed_runs"]}/{summary["expected_runs"]})', flush=True)


def open_session(args):
    if args.session:
        session = args.session.resolve()
    elif args.stage != 'setup':
        if not LATEST.is_file():
            raise ValueError('완료된 setup이 없습니다. 먼저 bash 1.setup.sh를 실행하세요.')
        session = Path(read_json(LATEST)['session'])
    else:
        session = SESSIONS / str(time.time_ns())
        session.mkdir(parents=True, exist_ok=False)
        source = load_config(args.config or 'configs/experiment.json')
        c = session_config(source, session)
        # One visible physical GPU serves both order conditions sequentially.
        c['runtime']['allocated_gpu_slots'] = [0]
        # Preserve the chosen learning rate instead of the legacy pipeline's reset.
        c['train']['lr'] = source['train']['lr']
        immutable_json(session / 'configs' / 'session.json', c)
        atomic_json(session / 'state.json', {
            'schema': 'single-gpu-stages-v1', 'code': code_hash(), 'packages': packages(),
            'config_hash': digest(c), 'completed': {}, 'choices': {},
        })
    if not (session / 'state.json').is_file():
        raise ValueError(f'세션 파일이 없습니다: {session}/state.json')
    return session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=tuple(ENTRIES))
    parser.add_argument('--session', type=Path)
    parser.add_argument('--config')
    args = parser.parse_args()
    if args.config and (args.stage != 'setup' or args.session):
        parser.error('--config는 새 setup에서만 사용합니다.')
    os.chdir(ROOT)
    session = open_session(args)
    gpu = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    resume = f'bash {ENTRIES[args.stage]} --session {shlex.quote(str(session))} --gpu {shlex.quote(gpu)}'
    print(f'실행 폴더: {session}\n중단 시 재개: {resume}', flush=True)
    with exclusive_lock(session / 'pipeline.lock'):
        state = read_json(session / 'state.json')
        if state.get('schema') != 'single-gpu-stages-v1':
            raise ValueError('이전 first.sh 세션은 새 분리 실행에서 재개할 수 없습니다. 새 setup을 실행하세요.')
        config = session / 'configs' / 'session.json'
        c = load_config(config)
        if state['config_hash'] != digest(c):
            raise ValueError('저장된 세션 설정이 변경되었습니다. 새 setup 세션을 사용하세요.')
        pipeline = StagedPipeline(session, state)
        pipeline.check_identity()
        try:
            if args.stage == 'setup':
                pipeline.setup(c, config)
                atomic_json(LATEST, {'session': str(session)})
                print(f'setup 완료. 다음: bash 2.SMOKE.sh --session {shlex.quote(str(session))} --gpu {shlex.quote(gpu)}', flush=True)
            elif args.stage == 'presmoke':
                pipeline.presmoke(c)
            elif args.stage == 'smoke':
                pipeline.smoke(c, config)
                print(f'SMOKE 완료. 다음: bash 3.runreal.sh --session {shlex.quote(str(session))} --gpu {shlex.quote(gpu)}', flush=True)
            else:
                pipeline.runreal(c)
        except BaseException as exc:
            atomic_json(session / f'failure-{args.stage}-{time.time_ns()}.json', {'error': repr(exc)})
            print(f'중단. 결과를 보존했습니다. 재개: {resume}', flush=True)
            raise


if __name__ == '__main__':
    def stop_on_signal(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    signal.signal(signal.SIGTERM, stop_on_signal)
    main()
