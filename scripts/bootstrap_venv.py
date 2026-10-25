"""Create/repair only the project venv, including Python builds without working ensurepip."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request

GET_PIP_URL = 'https://bootstrap.pypa.io/get-pip.py'


class Bootstrap:
    def __init__(self, target, logs):
        self.target = Path(target).absolute()
        # Do not resolve the python symlink: that would select the base interpreter.
        self.python = self.target / 'bin' / 'python'
        self.logs = Path(logs) / str(time.time_ns())
        self.logs.mkdir(parents=True, exist_ok=False)

    def run(self, label, args):
        command = [str(arg) for arg in args]
        path = self.logs / (label + '.log')
        result = subprocess.run(command, text=True, capture_output=True)
        with path.open('x') as stream:
            stream.write('# '+shlex.join(command)+'\n'+result.stdout+result.stderr)
        if result.returncode:
            print(f'{label} 실패 ({result.returncode}), 상세: {path}', flush=True)
            print((result.stdout+result.stderr)[-3000:], flush=True)
        return result

    def pip_ready(self, label):
        return self.run(label, [self.python, '-I', '-m', 'pip', '--version']).returncode == 0

    def verify_target(self):
        result = self.run('verify-interpreter', [self.python, '-I', '-c',
            'import json,sys; print(json.dumps({"prefix":sys.prefix,"base":sys.base_prefix,"version":list(sys.version_info[:2])}))'])
        if result.returncode:
            raise RuntimeError('Existing venv Python is not executable; inspect the log before changing this directory')
        info = json.loads(result.stdout)
        if (Path(info['prefix']).resolve() != self.target.resolve() or info['prefix'] == info['base']
            or info['version'] not in ([3,11],[3,12])):
            raise RuntimeError('Target is not an isolated Python 3.11/3.12 venv; refusing to install into it')

    def install(self):
        if not (self.target / 'pyvenv.cfg').is_file() or not self.python.exists():
            # Repair incomplete creation in place; never clear/remove the directory.
            if self.target.exists() and not (self.target / 'pyvenv.cfg').is_file() and any(self.target.iterdir()):
                raise RuntimeError('Nonempty target lacks pyvenv.cfg; inspect it before choosing a venv path')
            result = self.run('create-without-pip', [sys.executable, '-I', '-m', 'venv', '--without-pip', self.target])
            if result.returncode:
                raise RuntimeError('venv itself could not be created; install the matching python3.11-venv or python3.12-venv package on Debian/Ubuntu')
        self.verify_target()
        if self.pip_ready('existing-pip'):
            return self.python
        # Show the real error that python -m venv otherwise hides in CalledProcessError.
        result = self.run('ensurepip', [self.python, '-I', '-m', 'ensurepip', '--upgrade', '--default-pip'])
        if result.returncode == 0 and self.pip_ready('pip-after-ensurepip'):
            return self.python
        # pip >=22.3 can manage a different interpreter even when it has no pip.
        result = self.run('base-pip-to-venv', [sys.executable, '-I', '-m', 'pip',
                                            '--python', self.python, 'install', 'pip'])
        if result.returncode == 0 and self.pip_ready('pip-after-base-installer'):
            return self.python
        # Official PyPA recovery, executed ONLY after proving the target is a venv.
        print('ensurepip/base pip 복구 실패. 공식 PyPA get-pip.py를 가상환경 안에서 실행합니다.', flush=True)
        with urllib.request.urlopen(GET_PIP_URL, timeout=60) as response:
            content = response.read(5 * 1024 * 1024 + 1)
        if len(content) > 5 * 1024 * 1024:
            raise RuntimeError('Unexpectedly large get-pip.py download')
        script = self.logs / 'get-pip.py'
        script.write_bytes(content)
        (self.logs / 'download.json').write_text(json.dumps({
            'url':GET_PIP_URL, 'sha256':hashlib.sha256(content).hexdigest()}, indent=2)+'\n')
        result = self.run('get-pip', [self.python, '-I', script])
        if result.returncode or not self.pip_ready('pip-after-get-pip'):
            raise RuntimeError('Could not seed pip; inspect bootstrap logs for missing wheels, permissions, disk space, TLS or network errors')
        return self.python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', default='.venv-gpu')
    parser.add_argument('--logs', default='.bootstrap-logs')
    args = parser.parse_args()
    bootstrap = Bootstrap(args.venv, args.logs)
    print(f'가상환경 준비 로그: {bootstrap.logs}', flush=True)
    try:
        print(f'가상환경 pip 준비 완료: {bootstrap.install()}', flush=True)
    except Exception as exc:
        (bootstrap.logs / 'failure.txt').write_text(repr(exc)+'\n')
        print(f'가상환경 준비 실패: {exc}\n상세 로그: {bootstrap.logs}\n기존 폴더는 삭제하지 않았습니다.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
