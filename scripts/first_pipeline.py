"""Legacy tuning workflow and helpers shared by staged_pipeline; no CUDA in this parent."""
import argparse
import copy
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grpo_order.core import atomic_json, code_hash, digest, file_hash, load_config, packages, read_json
from grpo_order.data import load_prepared
from grpo_order.storage import exclusive_lock, run_path


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Existing file has different content: {path}; use a new session')
    else:
        atomic_json(path, value)
    return path


def session_config(source, session):
    c = copy.deepcopy(source)
    converted = session / 'data' / 'converted'
    c['assets']['converted_dir'] = str(converted)
    for name in ('pool', 'validation', 'math500'):
        c['data'][name] = str(converted / f'{name}.jsonl')
    c['data']['import_manifest'] = str(converted / 'import_manifest.json')
    c['data']['prepared_dir'] = str(session / 'data' / 'prepared')
    c['output_dir'] = str(session / 'runs' / 'smoke')
    c['evaluation']['length_finalized'] = False
    c['train']['lr'] = 5e-6
    return c


def verify_import(c):
    path = Path(c['data']['import_manifest'])
    report = read_json(path)
    if digest({k:v for k,v in report.items() if k != 'hash'}) != report['hash']:
        raise ValueError('Converted data manifest checksum changed')
    for name, expected in report['files'].items():
        if file_hash(path.parent / name) != expected:
            raise ValueError(f'Converted data changed: {name}')
    for name in ('math', 'math500'):
        receipt = read_json(Path(c['assets']['datasets'][name]['path']) / '.asset.json')
        if receipt != report['sources'][name]:
            raise ValueError(f'Converted data source changed: {name}')


def pilot_summary(c):
    """Only validation enters selection. Never read MATH500 output files here."""
    step = c['pilot']['updates']
    records = []
    for role in ('base', 'rl'):
        for order in ('random', 'curriculum'):
            root = run_path(c, 'pilot', role, order, 1)
            done = read_json(root / 'complete.json')
            meta = read_json(root / 'metadata.json')
            if done['skipped_steps'] or done['cursor'] != step:
                raise ValueError(f'Invalid/incomplete pilot updates: {root}')
            if meta['contract']['config'] != digest(c) or done['contract'] != meta['contract']:
                raise ValueError(f'Pilot contract mismatch: {root}')
            initial = read_json(root / 'eval' / 'validation_step-000000.json')
            final = read_json(root / 'eval' / f'validation_step-{step:06d}.json')
            if initial['contract'] != meta['contract'] or final['contract'] != meta['contract']:
                raise ValueError(f'Validation contract mismatch: {root}')
            samples = [r for q in final['questions'] for r in q['sampled']]
            record = {'model':role, 'order':order, 'step':step, **final['metrics'],
                      'initial_avg_at_8':initial['metrics']['avg_at_8'],
                      'delta_avg_at_8':final['metrics']['avg_at_8']-initial['metrics']['avg_at_8'],
                      'extraction_failure_fraction':sum(r.get('failure_reason') == 'answer_extraction_failed' for r in samples)/len(samples),
                      'group_counts':done['group_counts'], 'run':str(root)}
            if step:
                steps = [read_json(root / 'steps' / f'{i:06d}.json') for i in range(1, step+1)]
                last = steps[-1]
                record.update(initial_adapter_norms=meta['lora']['initial_adapter_norms'],
                              final_adapter_norms=last['adapter_norms'],
                              training_seconds=done['training_seconds'],
                              peak_training_allocated_bytes=max(r['gpu_peak_allocated_bytes'] for r in steps),
                              peak_training_reserved_bytes=max(r['gpu_peak_reserved_bytes'] for r in steps))
            records.append(record)
    keys = ('avg_at_8', 'pass_at_8', 'greedy', 'sampled_truncation_fraction',
            'sampled_mean_tokens', 'extraction_failure_fraction', 'delta_avg_at_8')
    return {'split':'validation', 'config_hash':digest(c), 'records':records,
            'mean_over_four_conditions':{k:sum(r[k] for r in records)/4 for k in keys},
            'note':'Four condition means are descriptive, not independent replicates; prompt step 0 contains paired duplicates.'}


def verify_smoke(c):
    expected = copy.deepcopy(c)
    expected['train']['max_new_tokens'] = expected['evaluation']['max_new_tokens'] = 32
    for role in ('base', 'rl'):
        for order in ('random', 'curriculum'):
            root = run_path(c, 'smoke', role, order, 1)
            done = read_json(root / 'complete.json')
            if (done['contract']['config'] != digest(expected) or done['cursor'] != 3
                or done['optimizer_steps'] != 3 or done['skipped_steps'] != 0
                or done['generated_responses'] != 12 or done.get('checkpoint_reload_evaluation') != 'passed'):
                raise ValueError(f'Smoke did not verify generation/update/checkpoint/resume: {root}')


class Pipeline:
    def __init__(self, session, state):
        self.session = session
        self.state = state

    def save(self):
        atomic_json(self.session / 'state.json', self.state)

    def check_identity(self):
        if self.state.get('code', code_hash()) != code_hash() or self.state.get('packages', packages()) != packages():
            raise ValueError('Code/packages changed during this session; start a new session')

    def pin_assets(self, c):
        paths = {role:Path(spec['path'])/'.asset.json' for role,spec in c['models'].items()}
        paths.update({name:Path(spec['path'])/'.asset.json' for name,spec in c['assets']['datasets'].items()})
        receipts = {name:file_hash(path) for name,path in paths.items()}
        if self.state.get('assets', receipts) != receipts:
            raise ValueError('Model/tokenizer/dataset receipts changed during this session')
        self.state['assets'] = receipts
        self.save()

    def run(self, label, args, *, once=False):
        self.check_identity()
        command = [str(v) for v in args]
        if once and label in self.state['completed']:
            if self.state['completed'][label] != command:
                raise ValueError(f'Completed stage command changed: {label}')
            print(f'[완료 단계 유지] {label}', flush=True)
            return
        log = self.session / 'logs' / f'{label}-{time.time_ns()}.log'
        log.parent.mkdir(exist_ok=True)
        print(f'\n[{label}] {shlex.join(command)}\n로그: {log}', flush=True)
        env = {**os.environ, 'PYTHON':sys.executable, 'PYTHONUNBUFFERED':'1'}
        with log.open('x') as stream:
            stream.write('# ' + shlex.join(command) + '\n'); stream.flush()
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       text=True, bufsize=1)
            try:
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end='', flush=True)
                returncode = process.wait()
            except BaseException:
                # Stop the launcher AND its workers, leaving committed checkpoints intact.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=30)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                print(f'중단된 단계: {label}; {log}', flush=True)
                raise
            finally:
                process.stdout.close()
        if returncode:
            print(log.read_text()[-10000:], flush=True)
            raise RuntimeError(f'{label} failed ({returncode}); inspect {log}')
        print(f'[통과] {label}', flush=True)
        self.state['completed'][label] = command
        self.save()

    def cli(self, label, command, config, *args, once=False):
        self.run(label, ['bash', 'scripts/run.sh', command, '--config', config, *args], once=once)

    def config(self, name, source, **changes):
        c = copy.deepcopy(source)
        c['output_dir'] = str(self.session / 'runs' / name)
        c['prompt']['mode'] = changes.get('prompt', c['prompt']['mode'])
        c['pilot']['updates'] = changes.get('updates', c['pilot']['updates'])
        c['train']['lr'] = changes.get('lr', c['train']['lr'])
        c['train']['max_new_tokens'] = changes.get('length', c['train']['max_new_tokens'])
        c['evaluation']['max_new_tokens'] = changes.get('length', c['evaluation']['max_new_tokens'])
        c['evaluation']['length_finalized'] = changes.get('finalized', False)
        path = immutable_json(self.session / 'configs' / f'{name}.json', c)
        load_config(path)
        return c, path

    def pilot(self, name, c, path):
        self.cli(name, 'pilot', path, '--seed', '1', '--resume-failed')
        summary = pilot_summary(c)
        destination = immutable_json(self.session / 'summaries' / f'{name}.json', summary)
        print(f'\n{name} validation 요약: {destination}', flush=True)
        for row in summary['records']:
            print(f"  {row['model']}/{row['order']}: avg@8={row['avg_at_8']:.4f}, "
                  f"변화={row['delta_avg_at_8']:+.4f}, 잘림={row['sampled_truncation_fraction']:.4f}, "
                  f"추출실패={row['extraction_failure_fraction']:.4f}", flush=True)
            if row['step']:
                print(f"    학습 {row['training_seconds']:.1f}초, peak allocated "
                      f"{row['peak_training_allocated_bytes']/1024**3:.2f} GiB, "
                      f"그룹={row['group_counts']}, LoRA norm={row['final_adapter_norms']}", flush=True)
        return summary

    def choose(self, key, title, choices, evidence, default=None):
        signature = digest(evidence)
        if key in self.state['choices']:
            saved = self.state['choices'][key]
            if saved['evidence'] != signature or saved['value'] not in choices:
                raise ValueError(f'Selection evidence changed: {key}; use a new session')
            return saved['value']
        print(f'\n{title}\n선택: {", ".join(choices)}' + (f' (Enter={default})' if default else ''), flush=True)
        try:
            # Read the terminal directly; prompts go to ordinary stdout. No r+
            # (terminals are not seekable) and no permission to write /dev/tty needed.
            with open('/dev/tty', 'r') as tty_in:
                while True:
                    print('선택 > ', end='', flush=True)
                    answer = tty_in.readline()
                    if not answer:
                        raise EOFError('terminal closed')
                    value = answer.strip() or default
                    if value in choices:
                        break
                    print('표시된 선택값 중 하나를 입력하세요.', flush=True)
        except (OSError, EOFError) as exc:
            raise RuntimeError('설정 선택에는 터미널이 필요합니다. 로그를 확인하고 같은 세션으로 --resume 하세요.') from exc
        self.state['choices'][key] = {'value':value, 'evidence':signature}
        self.save()
        return value

    def execute(self, c, config, pilot_only):
        self.run('environment-lock', [sys.executable, '-m', 'pip', 'freeze'])
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
        self.cli('preflight', 'preflight', config, '--report', self.session / 'logs' / f'preflight-{time.time_ns()}.json')
        self.cli('smoke-stop', 'smoke', config, '--stop-after', '1', '--resume-failed', once=True)
        self.cli('smoke-resume', 'resume', config, '--mode', 'smoke')
        verify_smoke(c)

        prompt_evidence = {}
        prompt_configs = {}
        for prompt in ('plain', 'chat'):
            pc, path = self.config(f'prompt_{prompt}', c, prompt=prompt, updates=0, length=2048, lr=5e-6)
            prompt_evidence[prompt] = self.pilot(f'prompt_{prompt}', pc, path)
            prompt_configs[prompt] = pc
        prompt = self.choose('prompt', '두 모델에 공통으로 사용할 prompt를 validation 결과로 선택하세요.',
                             ['plain','chat'], prompt_evidence)
        length_evidence, length_configs = {}, {}
        for length in (2048, 4096):
            lc, path = self.config(f'length_{length}', prompt_configs[prompt], length=length, updates=30)
            length_evidence[str(length)] = self.pilot(f'length_{length}', lc, path)
            length_configs[str(length)] = lc
        length = self.choose('length', 'validation 정확도·잘림률·실측 자원 사용을 보고 공통 출력 상한을 선택하세요.',
                             ['2048','4096'], length_evidence)
        baseline, path = self.config('lr_5e-6', length_configs[length], updates=300, lr=5e-6)
        lr_evidence = {'5e-6':self.pilot('lr_5e-6', baseline, path)}
        candidate = self.choose('lr_candidate', '기본 LR 5e-6의 300 update 결과를 확인했습니다. 추가 LR 비교가 필요한가요?',
                                ['none','1e-5','5e-5','1e-4'], lr_evidence, default='none')
        selected = baseline
        if candidate != 'none':
            other, path = self.config(f'lr_{candidate}', baseline, lr=float(candidate))
            lr_evidence[candidate] = self.pilot(f'lr_{candidate}', other, path)
            chosen = self.choose('lr', 'validation 변화·혼합 그룹·LoRA norm을 보고 최종 LR을 선택하세요.',
                                 ['5e-6',candidate], lr_evidence, default='5e-6')
            if chosen != '5e-6':
                selected = other
        final, final_path = self.config('final', selected, finalized=True, updates=300)
        print(f'\n확정 설정: prompt={final["prompt"]["mode"]}, length={final["evaluation"]["max_new_tokens"]}, '
              f'lr={final["train"]["lr"]}\n설정 파일: {final_path}', flush=True)
        immutable_json(self.session / 'selection.json', {'choices':self.state['choices'],
                       'config':str(final_path), 'config_hash':digest(final), 'selection_split':'validation'})
        if pilot_only:
            print('파일럿 완료. --pilot-only 요청에 따라 full 전에 종료합니다.', flush=True)
            return
        self.check_identity()
        self.pin_assets(c)
        self.cli('final-preflight', 'preflight', final_path, '--report', self.session / 'logs' / f'final-{time.time_ns()}.json')
        freeze_path = Path(final['output_dir']) / 'protocol-freeze.json'
        if not freeze_path.exists():
            self.cli('freeze', 'freeze', final_path)
        else:
            from grpo_order.freeze import verify_freeze
            verify_freeze(final)
        self.cli('full-12-runs', 'full', final_path, '--resume-failed')
        # Use an attempt-specific analysis destination: interrupted analysis cannot overwrite a report.
        report = self.session / 'analysis' / str(time.time_ns())
        self.cli('analyze', 'analyze', final_path, '--mode', 'full', '--output', report)
        immutable_json(self.session / 'finished.json', {'final_config':str(final_path), 'full_runs':12})
        print(f'\n12회 본실험 및 분석 완료: {report}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--pilot-only', action='store_true')
    args = parser.parse_args()
    if args.resume and args.config:
        parser.error('--resume uses its saved configuration; do not combine with --config')
    os.chdir(ROOT)
    if args.resume:
        session = args.resume.resolve()
        if not (session / 'state.json').is_file():
            raise FileNotFoundError(f'Not a first.sh session: {session}')
    else:
        session = ROOT / 'runs' / 'server_sessions' / str(time.time_ns())
        session.mkdir(parents=True, exist_ok=False)
    print(f'실행 폴더: {session}\n재개 명령: bash first.sh --resume {shlex.quote(str(session))}', flush=True)
    with exclusive_lock(session / 'pipeline.lock'):
        config = session / 'configs' / 'smoke.json'
        if args.resume:
            state = read_json(session / 'state.json')
            if state['code'] != code_hash() or state['packages'] != packages():
                raise ValueError('Code/packages changed since session creation; start a new session')
            c = load_config(config)
            if state['config_hash'] != digest(c):
                raise ValueError('Session config changed')
        else:
            c = session_config(load_config(args.config or 'configs/experiment.json'), session)
            immutable_json(config, c)
            state = {'code':code_hash(), 'packages':packages(), 'config_hash':digest(c), 'completed':{}, 'choices':{}}
            atomic_json(session / 'state.json', state)
        pipeline = Pipeline(session, state)
        try:
            pipeline.execute(c, config, args.pilot_only)
        except BaseException as exc:
            atomic_json(session / f'failure-{time.time_ns()}.json', {'error':repr(exc)})
            print(f'\n중단. 기존 결과는 유지됩니다. 재개: bash first.sh --resume {shlex.quote(str(session))}', flush=True)
            raise


if __name__ == '__main__':
    def stop_on_signal(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    signal.signal(signal.SIGTERM, stop_on_signal)
    main()
