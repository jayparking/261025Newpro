#!/usr/bin/env bash
# Shared shell setup for the numbered entry points.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
stage="${1:?stage required}"
shift
case "$stage" in
  setup) entry=1.setup.sh; plan='설치 → GPU 1장 검사 → 모델/데이터 준비 → 전체 preflight. 여기서 종료.' ;;
  presmoke) entry=2.1presmoke.sh; plan='setup 완료 세션 → Base 모델 하나, 학습 1문제 × 4응답으로 1 update → 별도 진단 1문제 평가(8샘플+greedy). 출력 상한 32. 선택 단계이며 SMOKE를 대체하지 않음.' ;;
  smoke) entry=2.SMOKE.sh; plan='준비된 세션 → 네 조건을 GPU 1장에서 순차 실행 → 1 update 중단/재개 → 조건당 3 updates, 출력 상한 32. 여기서 종료.' ;;
  runreal) entry=3.runreal.sh; plan='준비/SMOKE 완료 확인 → 저장된 설정 확정·freeze → 기본 4 run 순차 실행(설정 seed 수에 따름, 각 300 updates, 설정의 실제 출력 길이) → MATH500 평가·분석·성능 보고.' ;;
  *) echo "알 수 없는 단계: $stage" >&2; exit 2 ;;
esac
usage() {
  cat <<EOF
사용: bash $entry [--gpu GPU] [--session DIR | --resume DIR] [--dry-run]
$plan

  --gpu GPU       사용할 GPU 하나. 미지정 시 CUDA_VISIBLE_DEVICES 유지, 미설정이면 0
  --session DIR   기존 setup 세션 지정. presmoke/smoke/runreal은 생략하면 최근 setup 사용
  --resume DIR    해당 단계의 저장된 진행 상태 재개 (--session과 같은 의미)
  --config JSON   setup 새 세션의 설정 (기본 configs/experiment.json)
  --skip-install  setup에서만 설치 생략. PYTHON 또는 .venv-gpu/bin/python 사용
  --dry-run       실행 범위만 표시. 파일 생성·설치·GPU 실행 없음
  --help          도움말

setup 재개는 설치를 반복하지 않습니다. presmoke/smoke/runreal은 설치·다운로드를 하지 않습니다.
runreal은 설정 탐색 파일럿/선택 질문 없이 setup에 저장한 prompt·길이·LR을 사용합니다.
EOF
}
skip_install=0
session_arg=''
config_arg=''
dry_run=0
gpu_arg="${CUDA_VISIBLE_DEVICES-0}"
pipeline_args=()
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --dry-run) dry_run=1; shift ;;
    --skip-install) skip_install=1; shift ;;
    --config|--session|--resume|--gpu)
      (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || { echo "$1 뒤에 값이 필요합니다." >&2; exit 2; }
      case "$1" in
        --config) config_arg="$2" ;;
        --session|--resume) session_arg="$2" ;;
        --gpu) gpu_arg="$2" ;;
      esac
      shift 2 ;;
    *) echo "알 수 없는 옵션: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ -n "$config_arg" && ( "$stage" != setup || -n "$session_arg" ) ]]; then
  echo '--config는 새 setup에서만 사용합니다. 나머지 단계는 저장된 설정을 사용합니다.' >&2; exit 2
fi
if ((skip_install)) && [[ "$stage" != setup ]]; then
  echo '--skip-install은 setup에서만 사용합니다.' >&2; exit 2
fi
if [[ -z "$gpu_arg" || "$gpu_arg" == *,* || "$gpu_arg" == -1 ]]; then
  echo 'GPU 한 개를 지정하세요. 예: --gpu 0 또는 CUDA_VISIBLE_DEVICES=GPU-실제UUID' >&2; exit 2
fi
if ((dry_run)); then
  echo "[$entry] CUDA_VISIBLE_DEVICES=$gpu_arg"
  echo "$plan"
  echo "세션: ${session_arg:-setup은 새 세션 / 나머지는 최근 setup}"
  exit 0
fi
[[ -z "$config_arg" || -f "$config_arg" ]] || { echo "설정 파일 없음: $config_arg" >&2; exit 2; }
[[ -z "$session_arg" || -f "$session_arg/state.json" ]] || { echo "세션 파일 없음: $session_arg/state.json" >&2; exit 2; }
if [[ "$(uname -s)" != Linux ]]; then
  echo '실제 실행은 NVIDIA GPU가 있는 Linux 서버에서 하세요. --dry-run은 로컬에서도 가능합니다.' >&2; exit 2
fi
command -v flock >/dev/null || { echo 'util-linux의 flock이 필요합니다.' >&2; exit 2; }
exec 9>.first.lock
flock -n 9 || { echo '다른 setup/presmoke/SMOKE/runreal 단계가 실행 중입니다.' >&2; exit 2; }
trap 'echo "단계 준비 실패: $entry, ${LINENO}행. 오류 해결 후 같은 명령으로 다시 실행하세요." >&2' ERR

if [[ "$stage" == setup && -z "$session_arg" ]] && (( !skip_install )); then
  bootstrap="${PYTHON_BOOTSTRAP:-}"
  if [[ -z "$bootstrap" ]]; then
    for candidate in python3.11 python3.12 python3; do
      if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; assert sys.version_info[:2] in ((3,11),(3,12))' 2>/dev/null; then
        bootstrap="$candidate"; break
      fi
    done
  fi
  [[ -n "$bootstrap" ]] || { echo 'Python 3.11 또는 3.12와 venv가 필요합니다.' >&2; exit 2; }
  "$bootstrap" scripts/bootstrap_venv.py --venv .venv-gpu
  server_python="$PWD/.venv-gpu/bin/python"
  "$server_python" -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  "$server_python" -m pip install -r requirements.txt -r requirements-assets.txt
else
  server_python="${PYTHON:-$PWD/.venv-gpu/bin/python}"
fi
[[ -x "$server_python" ]] || command -v "$server_python" >/dev/null || { echo '먼저 bash 1.setup.sh를 실행하세요. 서버 Python이 없습니다.' >&2; exit 2; }
"$server_python" -c 'import sys; assert sys.version_info[:2] in ((3,11),(3,12)), "Python 3.11/3.12 required"'
export PYTHON="$server_python" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="$gpu_arg"
pipeline_args=("$stage")
[[ -z "$session_arg" ]] || pipeline_args+=(--session "$session_arg")
[[ -z "$config_arg" ]] || pipeline_args+=(--config "$config_arg")
exec "$server_python" scripts/staged_pipeline.py "${pipeline_args[@]}"
