# 실험 결과와 전체 자료

## 바로 확인할 결과

- [본실험 4개 결과](runs/single_gpu_sessions/1793099680777695618/runs/real/full): Base/RL × Random/Curriculum, seed 1, 각 300 step.
- [분석 요약표](runs/single_gpu_sessions/1793099680777695618/analysis/1793362100514805053/summary.csv): 조건 간 차이와 bootstrap 구간.
- [분석 CSV·그래프](runs/single_gpu_sessions/1793099680777695618/analysis/1793362100514805053): 57개 파일.
- [파일럿 결과](poc_Test/runs/1793099972759311946/summary.json).

각 본실험 폴더의 `eval/`에는 학습 전·후 평가와 문항별 결과, `steps/`에는 학습 기록, `complete.json`에는 완료 정보가 있다. 이 결과들은 Git 커밋 `3f4ae781e282164740b5abf4991a4bbb3a8e0e63`에 포함되어 있다.

## 추가로 업로드한 전체 자료

**[GitHub Release: artifacts-2026-10-30](https://github.com/jayparking/261025Newpro/releases/tag/artifacts-2026-10-30)**의 Assets에서 내려받을 수 있다. 아래 압축 파일은 업로드 시점에 Git ignore 규칙으로 제외되어 있던 파일 전체를 담는다. 원본 Git 이력과 `.git/` 내부 정보는 이 압축 파일에 포함하지 않는다.

| 파일 이름 시작 부분 | 포함 내용 |
|---|---|
| `checkpoints` | `state.pt` 28개: 본실험 16개, smoke 8개, 파일럿 4개 |
| `model-base` | `models/Qwen2.5-1.5B/` 전체 |
| `model-simplerl` | `models/Qwen-2.5-1.5B-SimpleRL-Zoo/` 전체 |
| `data` | Git에서 제외되었던 `data/` 파일 전체 |
| `environment` | `.venv-gpu/` 전체, 원래 심볼릭 링크 포함 |
| `runtime-extras` | 실행 잠금 파일, Python 캐시, 설치 로그, 자산 잠금 정보 등 나머지 제외 파일 |

큰 압축 파일은 각각 최대 1 GiB인 `.part-000`, `.part-001`, … 조각으로 나누었다. 작은 파일은 `.tar.zst` 하나로 제공한다. GitHub의 기본 **Source code (zip/tar.gz)** 다운로드에는 이 추가 자료가 포함되지 않는다.

- [전체 파일 목록](artifacts/2026-10-30/files.csv): 원래 경로, 묶음, 크기, SHA-256, 심볼릭 링크 대상.
- [자료 manifest](artifacts/2026-10-30/manifest.json): 기준 커밋, 묶음별 파일 수·크기, 업로드 파일의 SHA-256.
- [다운로드 검증용 SHA256SUMS](artifacts/2026-10-30/SHA256SUMS).

## 다운로드와 복원

GitHub CLI, `zstd`, `tar`, `sha256sum`이 설치된 Linux에서 실행한다. 자료 전체를 다운로드하고 압축을 풀 공간이 필요하다.

```bash
gh repo clone jayparking/261025Newpro
mkdir artifact-downloads
gh release download artifacts-2026-10-30 \
  --repo jayparking/261025Newpro --dir artifact-downloads
cd artifact-downloads
sha256sum -c SHA256SUMS

# 각 분할 압축 파일을 번호 순서대로 합친다.
for first in *.tar.zst.part-000; do
  [ -e "$first" ] || continue
  archive="${first%.part-000}"
  cat "$archive".part-* > "$archive"
done

# 새로 clone한 저장소에 원래 상대 경로로 복원한다.
for archive in *.tar.zst; do
  tar --zstd -xf "$archive" -C ../261025Newpro
done
```

특정 자료만 필요하면 Release Assets에서 해당 이름으로 시작하는 파일을 모두 내려받는다. 예를 들어 체크포인트만 받으려면 `gh release download`에 `--pattern 'checkpoints*'`를 추가한다. 전체 `SHA256SUMS` 검사는 모든 자료를 다운로드한 경우에 사용한다.

본실험 체크포인트는 `runs/single_gpu_sessions/1793099680777695618/runs/real/full/<실험명>/checkpoints/step-000000`, `step-000100`, `step-000200`, `step-000300` 아래의 `state.pt`로 복원된다.

`.venv-gpu/`는 원래 서버 환경의 보관본이다. Python 경로·운영체제·CUDA 환경이 다른 컴퓨터에서는 그대로 실행되지 않을 수 있으므로 실행 환경 재구성에는 `1.setup.sh`를 사용한다. 결과와 모델·체크포인트 파일 확인에는 가상환경 복원이 필요하지 않다.
