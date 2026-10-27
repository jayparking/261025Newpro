# 본실험 전 POC: 실제 길이로 학습·평가 확인

Base와 SimpleRL을 GPU 한 장에서 순서대로 실행한다. 기본은 모델당 학습 12문제(Easy/Medium/Hard 각 4개), 별도 validation 12문제(각 4개)다. 두 모델에 같은 학습 문제와 같은 무작위 순서를 사용하며 seed는 1이다. 문제를 정답 여부로 골라 바꾸지 않는다.

학습/평가 출력 상한 2048, 학습 응답 4개, 평가 8샘플+greedy·4문제 배치를 사용한다. LR·LoRA·precision·prompt·attention 등은 setup의 설정을 그대로 가져온다. 2048/배치 4/한 GPU 설정과 다르면 중단한다. 32토큰 SMOKE를 대신 사용하는 단계가 아니다.

## 실행

프로젝트 전체를 GPU 서버에 복사하고 루트에서 실행한다. 이 폴더만 복사하면 공통 학습 모듈이 없어 실행할 수 없다. Python 3.11/3.12, Linux NVIDIA 환경을 사용한다.

```bash
# 현재 코드로 새 setup을 완료한다. 기존 모델/raw 다운로드는 검증 후 재사용한다.
bash 1.setup.sh

# Base 12문제 → validation 12문제 → SimpleRL 12문제 → validation 12문제
bash poc_Test/run.sh
```

POC는 완료된 setup이 필요하며, SMOKE 완료는 요구하지 않는다. 설치/다운로드를 자동 반복하지 않는다. 새 POC 코드가 코드 identity에 포함되므로 변경 전 setup을 그대로 재개할 수 없다.

```bash
# 설치·파일 생성·GPU 사용 없이 실행 범위 확인 (macOS에서도 가능)
bash poc_Test/run.sh --dry-run

# 특정 setup과 GPU, 더 짧은 전체 시간 제한
bash poc_Test/run.sh --session runs/single_gpu_sessions/실제번호 --gpu 0 --minutes 30

# 문제 수를 늘릴 때: 모델당 24문제, 각 난도 8개. 평가와 총 시간 제한은 동일
bash poc_Test/run.sh --train-problems 24
```

기본 **최대 60분은 두 모델을 합한 POC 전체 제한**이다. 준비 자산 검사·GPU 환경 검사·모델 로딩·학습·평가가 포함된다. 사전에 실행하는 setup 설치·다운로드 시간은 별도다. `--minutes`로 60분 이하로 줄일 수 있다. 마지막 최대 5초는 종료 처리용으로 남긴다. 제한에 도달하면 현재 프로세스와 자식 프로세스를 종료하고 미완료로 기록한다. 파일 요약 등 CPU 종료 처리 시간은 소폭 추가될 수 있다. **POC를 종료해도 임대한 서버 자체가 정지하거나 과금이 중단되는 것은 아니다.**

기본 생성량은 학습 2 × 12 × 4 = 96응답, 평가 2 × 12 × 9 = 216응답이다. 초기 평가와 MATH500 평가는 하지 않고, 각 모델의 학습 종료 시 validation을 한 번 평가한다. 학습 전후 정답률 상승을 측정하는 실험이 아니다.

## 결과와 판단

매 실행은 새 `poc_Test/runs/<번호>/`를 만든다. 터미널과 `prepare.log`, `environment.log`, `base.log`, `rl.log`에 진행 상황을 남긴다.

| 파일 | 내용 |
|---|---|
| `summary.txt`, `summary.json` | 실행 상태, 모델별 학습 신호·잘림·답 추출 실패·처리량·메모리, 본실험 시간 추정 |
| `selection.json` | 두 모델이 공유하는 실제 학습/validation ID·난도·순서 |
| `source_config.json`, `config.json` | setup 원본과 POC 설정 |
| `attempt.json`, `environment.json` | 전체 시간 제한·실행 상태·GPU 환경 |
| `runs/poc/<모델>_seed1_random/` | 문제별 응답·reward·gradient·가중치 초기값·체크포인트·평가 |

`completed`는 두 모델의 예정된 학습/평가가 완료되고 기록 검증을 통과했다는 뜻이다. 정답률이 좋아졌거나 본실험이 성공할 것이라는 뜻은 아니다. 네 답이 모두 같은 reward면 advantage가 0이다. 혼합 그룹이 없거나 가중치가 바뀌지 않아도 결과를 조작하지 않고 그대로 표시한다. 채점기 내부 예외와 비정상 gradient는 기존 trainer처럼 실행을 중단한다.

확인할 것은 정답·오답 혼합 그룹의 유무/비율, 출력 잘림, 답 추출 실패, 학습/평가 메모리와 시간이다. 원문 응답도 확인한다. 한 모델에서만 정상이어도 두 모델 모두 통과한 것으로 간주하지 않는다. 제한 시간에 두 모델이 끝나지 않으면 `timed_out`(종료 코드 124)이며, 실행 오류는 비정상 종료, 사용자 중단은 130이다. 정상 완료는 0이다.

본실험 시간 추정은 두 모델이 모두 완료됐을 때만 표시한다. 모델별 평균 update 시간과 12문제 평가 시간을 실제 본실험 학습/평가 개수에 곱한다. 초기 평가 캐시와 확인 생성량도 반영한다. 작은 validation 표본을 MATH500/진단셋에 대입한 값이며, 학습 진행에 따른 출력 길이 변화·설치·로딩·checkpoint I/O·분석·재시도 비용은 보장하지 않는다. **6시간 이내 추정이어도 실행 허가나 시간 보장이 아니며 본실험을 자동 실행하지 않는다.**

실패/시간 초과 파일은 보존하지만 이 스크립트의 재개 옵션은 없다. 다시 실행하면 별도 POC를 원래 모델에서 시작하며 새 시간 예산을 소비한다. 미완료의 마지막 update는 저장되지 않을 수 있고, 원자적으로 저장된 checkpoint는 유지된다. POC 출력은 Git에서 제외한다.

## 본실험으로 넘어갈 때

POC 완료는 SMOKE 완료 표시를 바꾸지 않는다. 시간·실행 가능성을 확인하고 본실험을 진행하기로 결정한 경우에만 기존 단계를 실행한다.

```bash
bash 2.SMOKE.sh
bash 3.runreal.sh
```

본실험은 네 조건 모두 원래 Base/SimpleRL 체크포인트와 새 LoRA·optimizer에서 시작한다. POC의 가중치나 optimizer를 가져오지 않는다. 본실험 설정·출력 경로는 POC에서 변경하지 않는다.

## 파일 역할

`run.sh`는 setup의 Python 환경을 선택하고, `run.py`는 전체 시간 제한·프로세스 종료·로그·결과 요약을 관리한다. `worker.py`는 준비 검사와 모델별 실행을 맡는다. `protocol.py`는 고정 문항 선택, `report.py`는 실제 기록 검증·요약·시간 추정을 담당한다. 학습·채점·생성·저장은 프로젝트의 `grpo_order` 모듈을 재사용한다.
