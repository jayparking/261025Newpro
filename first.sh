#!/usr/bin/env bash
# Compatibility entry point. Each stage is explicitly selected.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
case "${1:-}" in
  setup|presmoke|smoke|SMOKE|runreal)
    stage="$1"
    shift
    [[ "$stage" != SMOKE ]] || stage=smoke
    exec bash scripts/stage.sh "$stage" "$@"
    ;;
  ''|--help|-h|--dry-run)
    cat <<'EOF'
GPU 한 장에서 단계를 나눠 실행합니다.
  bash 1.setup.sh       # 설치, GPU 검사, 모델/데이터 준비 (학습 없음)
  bash 2.1presmoke.sh   # 선택: Base 하나로 학습 1문제 + 별도 진단 1문제
  bash 2.SMOKE.sh       # 네 조건의 짧은 GPU 학습 및 중단/재개 검사
  bash 3.runreal.sh     # 저장한 설정으로 실제 학습·평가·분석

GPU 선택: CUDA_VISIBLE_DEVICES=0 또는 각 명령의 --gpu 0
각 파일의 --help / --dry-run으로 옵션과 실행 범위를 확인하세요.
first.sh setup|presmoke|smoke|runreal 형식도 지원합니다.
EOF
    ;;
  *) echo '단계를 지정하세요: bash first.sh setup|presmoke|smoke|runreal --help' >&2; exit 2 ;;
esac
