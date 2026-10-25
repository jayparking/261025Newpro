#!/usr/bin/env bash
# GPU-server entry point. Run from any directory: bash /path/to/project/first.sh
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

usage() {
  cat <<'EOF'
사용: bash first.sh [--pilot-only] [--resume 실행폴더] [--config JSON] [--skip-install]

  기본 실행       설치 → 테스트 → 다운로드 → 데이터 감사 → preflight → smoke/재개
                  → plain/chat → 2048/4096 → LR 파일럿 → 설정 선택 → freeze → full 12회 → 분석
  --pilot-only    같은 파일럿과 설정 선택까지 수행하고 full 전에 종료
  --resume DIR    기존 실행을 검증하고 이어서 진행 (완료 결과를 덮어쓰지 않음)
  --config JSON   시작 설정 (기본 configs/experiment.json, 새 실행에만 적용)
  --skip-install 현재 PYTHON 또는 .venv-gpu/bin/python 사용; 설치 생략, 검사는 유지
  --dry-run      실행 순서만 표시; 설치·다운로드·학습·파일 생성 없음
  --help         도움말

파일럿 후 공통 prompt/길이/LR 선택을 터미널에서 받습니다. 선택이 끝나면 기본 실행은
본실험으로 이어집니다. 입력할 터미널이 없으면 해당 단계에서 안전하게 종료하고
재개 명령을 표시합니다. CUDA_VISIBLE_DEVICES가 있으면 할당을 그대로 보존합니다.
PYTHON_BOOTSTRAP=python3.11 로 venv 생성용 Python을 지정할 수 있습니다.
EOF
}

skip_install=0
resume_arg=''
config_arg=''
pipeline_args=()
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --dry-run) usage; exit 0 ;;
    --skip-install) skip_install=1; shift ;;
    --pilot-only) pipeline_args+=("$1"); shift ;;
    --config|--resume)
      if (($# < 2)); then echo "$1 뒤에 경로가 필요합니다." >&2; exit 2; fi
      if [[ "$1" == --config ]]; then config_arg="$2"; else resume_arg="$2"; fi
      pipeline_args+=("$1" "$2"); shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -n "$resume_arg" && -n "$config_arg" ]]; then
  echo '--resume과 --config는 함께 사용할 수 없습니다. 저장된 설정으로 재개합니다.' >&2; exit 2
fi
if [[ -n "$resume_arg" && ! -f "$resume_arg/state.json" ]]; then
  echo "실행 폴더가 아닙니다: $resume_arg" >&2; exit 2
fi
if [[ -n "$config_arg" && ! -f "$config_arg" ]]; then
  echo "설정 파일이 없습니다: $config_arg" >&2; exit 2
fi

if [[ "$(uname -s)" != Linux ]]; then
  echo 'first.sh 실제 실행은 NVIDIA GPU가 있는 Linux 서버에서 하세요. --dry-run은 로컬에서도 가능합니다.' >&2
  exit 2
fi
command -v flock >/dev/null || { echo 'util-linux의 flock이 필요합니다.' >&2; exit 2; }
exec 9>.first.lock
flock -n 9 || { echo '다른 first.sh가 실행 중입니다.' >&2; exit 2; }
trap 'echo "first.sh 오류: ${LINENO}행, 종료 코드 $? (기존 결과 보존). 표시된 실행 폴더로 --resume 하세요." >&2' ERR

if ((skip_install)); then
  server_python="${PYTHON:-$PWD/.venv-gpu/bin/python}"
else
  bootstrap="${PYTHON_BOOTSTRAP:-}"
  if [[ -z "$bootstrap" ]]; then
    for candidate in python3.11 python3.12 python3; do
      if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; assert sys.version_info[:2] in ((3,11),(3,12))' 2>/dev/null; then
        bootstrap="$candidate"; break
      fi
    done
  fi
  [[ -n "$bootstrap" ]] || { echo 'Python 3.11 또는 3.12와 venv가 필요합니다.' >&2; exit 2; }
  "$bootstrap" -c 'import sys; assert sys.version_info[:2] in ((3,11),(3,12)), "Python 3.11/3.12 required"'
  if [[ ! -d .venv-gpu ]]; then "$bootstrap" -m venv .venv-gpu; fi
  server_python="$PWD/.venv-gpu/bin/python"
  "$server_python" -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  "$server_python" -m pip install -r requirements.txt -r requirements-assets.txt
fi

"$server_python" -c 'import sys; assert sys.version_info[:2] in ((3,11),(3,12)), "Python 3.11/3.12 required"'
export PYTHON="$server_python" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1}"
# No CUDA import/probe in this parent: probing and training run in child processes.
if ((${#pipeline_args[@]})); then
  exec "$server_python" scripts/first_pipeline.py "${pipeline_args[@]}"
else
  # Bash <4.4 treats an empty array expansion as unbound under set -u.
  exec "$server_python" scripts/first_pipeline.py
fi
