# 261025Newpro — Qwen2.5-1.5B Base / SimpleRL-Zoo 추가 GRPO 순서 실험

## 완료된 실험 결과와 전체 자료

평가 결과·학습 기록·분석 그래프는 이 저장소에 포함되어 있다. 체크포인트, 모델, 데이터, 실행 환경 등 대용량 자료는 [전체 자료 Release](https://github.com/jayparking/261025Newpro/releases/tag/artifacts-2026-10-30)에서 다운로드할 수 있다. **[결과 위치·파일 목록·다운로드 및 복원 방법](ARTIFACTS.md)**을 참고한다.

## GPU 한 장에서 3단계 실행

본실험 전에 실제 2048토큰 길이로 실행 가능성과 비용을 확인하려면 setup 후 `bash poc_Test/run.sh`를 실행한다. 기본은 Base/SimpleRL 각각 12문제 학습과 별도 validation 12문제 평가, 전체 최대 60분이며 본실험을 자동 시작하지 않는다. 자세한 사용법과 한계는 [poc_Test/README.md](poc_Test/README.md)에 있다.

기본 본실험은 **Base/RL × Random/Curriculum × seed 1 = 총 4회**다. 반복 seed를 3개에서 1개로 줄였으며 조건당 학습 300문제, MATH500 500문제, 평가 8샘플+greedy, 출력 상한 2048은 유지한다. 학습 seed에 따른 결과 변동성은 이 실행만으로 추정할 수 없다. **4회여도 최대 6시간 내 완료를 보장하지 않으며 시간 제한에 따른 자동 종료는 없다.** 기존 12회 설정으로 setup한 세션은 새 setup으로 교체한다(기존 결과는 보존). 기본 중간평가(100·200 update)는 생략하며 학습 전(0)·후(300) 진단/MATH500 평가만 수행한다. 학습 중 reward 채점과 재개용 체크포인트 저장은 유지한다.

Linux NVIDIA GPU 서버에서 아래 세 파일을 차례로 실행한다. 각 단계는 완료 후 종료하며 다음 단계를 자동 시작하지 않는다. 기본 GPU는 0이고, 이미 지정한 `CUDA_VISIBLE_DEVICES`는 유지한다. GPU를 두 개 이상 지정하면 한 개를 고르라는 오류로 중단한다.

```bash
bash 1.setup.sh
bash 2.SMOKE.sh
bash 3.runreal.sh
```

setup 후 더 작은 확인만 먼저 하려면 **선택 단계**인 `bash 2.1presmoke.sh`를 실행한다. 파일 번호와 관계없이 SMOKE 전에 실행할 수 있다. Base 모델 하나(seed 1)에서 Easy 우선으로 고정 선택한 학습 1문제 × 4응답으로 optimizer를 한 번 실행한 뒤, 별도 진단 1문제를 8샘플+greedy로 평가한다. 출력 상한은 32토큰이며 초기 평가·RL 모델·4조건 반복·중단/재개 검증 절차를 실행하지 않는다. GPU 한 장을 사용하고 설치/다운로드를 반복하지 않는다. 모델 로딩과 파일 검증 시간은 필요하다.

```bash
bash 2.1presmoke.sh
# 실행 없이 범위 확인
bash 2.1presmoke.sh --dry-run
# 특정 setup 세션에서 실행하거나 중단 후 이어가기
bash 2.1presmoke.sh --session runs/single_gpu_sessions/실제실행번호
```

문제·실제 답변·채점·학습 시간·가중치 변경 여부는 세션의 `runs/presmoke/base_seed1_random/summary.json`에 저장된다. 같은 세션에서 다시 실행하면 완료 결과를 검증하고 유지한다. 실패했다면 마지막 checkpoint에서 이어간다. 네 응답의 reward가 모두 같으면 GRPO advantage가 0이므로 첫 optimizer 호출에도 가중치가 바뀌지 않을 수 있다. 짧은 출력이 잘리거나 오답이어도 실행 성공과 구분해서 기록한다. 모델 품질/본실험 메모리 검증이 아니며, presmoke 결과는 본실험에 포함되지 않는다. **presmoke는 `2.SMOKE.sh`를 대체하거나 3번의 SMOKE 완료 조건을 해제하지 않는다.** 코드가 바뀐 이전 세션은 기존 identity 규칙에 따라 새 setup이 필요하다.

| 파일 | 실행 범위 |
|---|---|
| `1.setup.sh` | Python venv·고정 패키지 설치 → GPU 1장 환경 검사 → 모델/데이터 다운로드 → 변환·gold 감사·split 준비 → 전체 preflight |
| `2.1presmoke.sh` | 선택 단계: Base 한 조건, 학습 1문제 1 update → 별도 진단 1문제 평가, 출력 상한 32 |
| `2.SMOKE.sh` | 준비된 자산 사용 → Base/RL × Random/Curriculum 네 조건 순차 실행 → 조건당 1 update에서 중단 → 재개하여 총 3 updates → checkpoint 재로딩 검증. 출력 상한 32 |
| `3.runreal.sh` | setup/SMOKE 완료 확인 → 저장한 설정 확정·freeze → 기본 4 run을 한 GPU에서 순차 실행 → 최종 평가·분석·성능 보고 |

`runreal`은 **실제 본실험**이다. 기본 설정은 run당 학습 300문제·4응답·출력 상한 2048, diagnostic step 0/300, MATH500 step 0/300, 평가 8샘플+greedy를 유지한다. 초기 평가가 먼저 실행된다. 별도의 plain/chat·2048/4096·LR 탐색 파일럿이나 선택 질문은 없다. setup 시 지정한 설정의 prompt·학습/평가 길이·LR을 그대로 확정하므로 다른 설정을 원하면 setup 전에 `--config`로 지정한다. vLLM은 사용하지 않는다.

```bash
# 다른 GPU를 쓰려면 세 명령 모두 같은 GPU를 지정한다.
export CUDA_VISIBLE_DEVICES=2
bash 1.setup.sh
bash 2.SMOKE.sh
bash 3.runreal.sh

# 파일 생성·설치·GPU 실행 없이 범위 확인 (Mac에서도 가능)
bash 3.runreal.sh --dry-run

# 다른 설정으로 새 setup. 새 세션의 GPU 수는 항상 1로 저장한다.
bash 1.setup.sh --config configs/experiment.json --gpu 0

# 설치된 환경 사용: Python 3.11/3.12 및 고정 패키지는 검사한다.
PYTHON=/실제/서버/python bash 1.setup.sh --skip-install
```

필요한 것은 Linux, Python 3.11 또는 3.12, 설치된 NVIDIA 드라이버/nvidia-smi, GPU 한 장, 모델·패키지 다운로드 접근이다. `1.setup.sh`만 `.venv-gpu`를 준비하고 torch 2.7.1 CUDA 12.8 wheel 및 requirements를 설치한다. `PYTHON_BOOTSTRAP=python3.11`로 venv 생성용 Python을 지정할 수 있다. `ensurepip` 복구는 `scripts/bootstrap_venv.py`가 기존 환경을 삭제하지 않고 수행하며 `.bootstrap-logs/`에 기록한다. smoke/runreal 및 setup 재개는 패키지를 설치하지 않는다.

최초 setup은 `runs/single_gpu_sessions/<고유번호>/`를 만들고, 완료하면 `runs/single_gpu_sessions/latest.json`에 위치를 저장한다. 이후 SMOKE/runreal은 최근 완료 setup을 자동 선택한다. 다른 세션을 선택하거나 실패 후 재개하려면 실행 중 출력된 경로를 지정한다.

```bash
bash 1.setup.sh --resume runs/single_gpu_sessions/실제실행번호
bash 2.SMOKE.sh --session runs/single_gpu_sessions/실제실행번호
bash 3.runreal.sh --resume runs/single_gpu_sessions/실제실행번호
```

같은 명령으로 재개하면 완료 run은 건너뛰고, 중단된 run의 checkpoint·문제별 평가를 이어간다. 코드·설정·패키지·자산 identity가 달라지면 재개를 거절한다. 기존 두 GPU용 `server_sessions`는 이 새 흐름에서 재개하지 않으며 기존 결과는 보존한다. `first.sh`는 이제 실행 순서 안내 및 `first.sh setup|smoke|runreal` 연결만 제공한다.

긴 본실험은 서버의 tmux 세션에서 실행하면 터미널 연결을 끊어도 계속 진행할 수 있다(서버에 tmux가 설치되어 있어야 한다).

```bash
tmux new -s grpo
bash 3.runreal.sh
# Ctrl-b, d로 분리. 다시 접속: tmux attach -t grpo
```

학습 진행 시 update당 초, 학습 전체 tok/s, peak allocated GiB를 화면과 로그에 출력한다. 평가도 문항 진행률과 누적 avg@8·pass@8을 표시한다. `performance.json`은 runreal 완료 또는 처리된 중단 시 갱신하며, 실행 중 최신 개별 수치는 worker 로그와 `steps/`에서 확인한다. GPU 한 장에서 Random과 Curriculum은 같은 GPU UUID를 사용하고, 이전 worker가 종료한 뒤 다음 worker를 시작한다.

| 세션 폴더 | 내용 |
|---|---|
| `configs/session.json`, `configs/real.json`, `selection.json` | 준비/SMOKE 설정, 본실험 설정과 선택 방식 |
| `data/converted/`, `data/prepared/` | 변환 원천·고정 split·감사 기록 |
| `logs/` | 단계별 명령·실시간 로그, 환경·preflight 보고서 |
| `runs/smoke/`, `runs/real/` | 네 조건 SMOKE, 기본 4개 본실험의 checkpoint·응답·평가·worker 로그 |
| `runs/presmoke/base_seed1_random/` | 선택 단계의 학습·평가·checkpoint와 `summary.json` |
| `performance.json` | run별 정답률, update당 초, 생성/채점/학습 시간, tok/s, GPU peak 메모리 |
| `workload.json` | 실제 split 개수와 캐시 정책에 따른 예정 학습·평가 응답 수 |
| `analysis/<번호>/` | 설정된 모든 run(기본 4회) 완료 후 조건별 비교·CSV·그래프 |
| `state.json`, `failure-*.json`, `finished.json` | 단계 상태·재개 정보·완료 표시 |

학습 tok/s는 생성·채점·backward/optimizer를 포함하며 checkpoint 저장·평가·초기 로딩을 제외한다. 생성만의 tok/s는 별도로 기록한다. 평가 파일의 시간은 마지막 시도 기준이므로 중단 이전 비용을 모두 포함하지 않는다. 측정값은 로컬에서 추정해 채우지 않는다.

## runreal 성능·결과 재점검 반영

기본 설정은 HF 생성 유지, `runtime.attention_implementation="sdpa"`, `evaluation.problem_batch_size=4`, `evaluation.step0_share_across_seeds=true`다. 평가 한 호출에 4문제 × 8샘플을 처리하고 greedy 4개는 별도 호출한다. 총 실행은 4회이며 run당 300 updates·학습 4응답·평가 8+greedy·2048 출력 상한·LR·난도 순서·데이터 split은 유지한다.

초기 LoRA B 텐서가 실제로 모두 0이고 원본 모델이 동결되어 있는지 확인한 뒤, 초기 평가 중에만 adapter를 비활성화해 같은 원본 모델의 seed별 공통 초기값을 계산한다. Base와 RL은 원본 hash가 달라 캐시를 공유하지 않는다. 각 실행의 학습 adapter A 초기화와 학습 seed는 그대로 독립적이다.

기본 4회에서 학습은 총 1,200 updates·4,800응답이다. 진단이 150문제이고 초기 캐시가 적중하면 평가 응답 수는 35,100개이며 기본 배치 4에서는 초기 확인 288응답이 추가된다. 실제 prepared 개수로 계산한 `workload.json`을 runreal 시작 시 저장한다. GPU 한 장에서 4 run은 순차 실행하며 완료 시간은 서버 실측이 필요하다. 초기 평가 등 공통 비용이 있어 12회 대비 총 시간이 정확히 1/3이 되지는 않는다. 32토큰 SMOKE는 실제 2048토큰 평가의 peak 메모리를 보장하지 않는다.

평가 배치와 SDPA는 수치·난수 소비를 바꾸므로 과거 출력과 동일하다고 가정하지 않는다. 설정은 네 조건 전체에 고정한다. OOM 때 자동으로 배치를 줄여 결과를 섞지 않는다. 서버에서 메모리가 부족하면 새 setup 전에 `evaluation.problem_batch_size`를 2 또는 1로 조정한다. SDPA 환경 문제가 있다면 새 설정에서 `attention_implementation="eager"`를 선택할 수 있다.

채점기 내부 예외는 오답으로 바꾸지 않고 중단한다. NaN/Inf loss·entropy·gradient도 optimizer와 cursor 저장 전에 중단한다. 마지막 정상 checkpoint는 유지된다. 최종 분석은 실제 300개 학습 기록의 순서·응답·advantage·정상 update, Random/Curriculum 초기값, 모든 평가 시점의 문항·8+1응답·집계 정답률을 검사한다. 누락되거나 불일치하는 결과는 정상 분석으로 통과하지 않는다.

이번 변경은 코드·평가 정책·채점기 identity를 바꾸므로 기존 setup/SMOKE/full 세션에 덮어써 재개하지 않는다. 새 `1.setup.sh` → `2.SMOKE.sh` → `3.runreal.sh` 세션을 사용한다. 모델/raw 데이터는 영수증을 검증해 재사용하며 기존 결과를 삭제하지 않는다.

## 어떤 소스가 어떤 일을 하는가

| 파일 | 역할 |
|---|---|
| `1.setup.sh`, `2.1presmoke.sh`, `2.SMOKE.sh`, `3.runreal.sh` | GPU 1장용 독립 실행 단계(presmoke는 선택) |
| `first.sh`, `scripts/stage.sh` | 단계 안내·연결, 공통 환경·GPU 선택·설치 |
| `scripts/bootstrap_venv.py` | 불완전한 venv 복구, 격리된 interpreter 검증, ensurepip/pip 상세 설치 로그 |
| `scripts/staged_pipeline.py` | 단계별 세션·재개·freeze·실험·분석 관리 |
| `scripts/first_pipeline.py` | 기존 설정 탐색 흐름과 새 단계에서 재사용하는 검증·로그 도우미 |
| `grpo_order/performance.py` | 실제 본실험의 품질·처리량·메모리 보고 |
| `grpo_order/presmoke.py` | 한 문제 확인용 설정·고정 문항 선택·실행 결과 검증과 요약 |
| `scripts/run.sh`, `grpo_order/cli.py` | 개별 작업 CLI 연결. 수동 실행/디버깅에도 사용 |
| `grpo_order/assets.py`, `sources.py` | 지정 모델/데이터 다운로드, source ID·revision·receipt/hash 검증 |
| `grpo_order/import_math.py`, `data.py` | MATH 원천 변환, 원래 level 보존, gold 검사·중복 제거·고정 split |
| `grpo_order/grading.py` | math-verify 파싱·정오답 채점, 설치/파서/문항 문제 구분 |
| `grpo_order/preflight.py` | 의존성·UUID·GPU/BF16·모델·tokenizer·context·데이터 검사 |
| `grpo_order/model.py` | 모델/LoRA/EOS 검증, 응답 4/8개 배치 생성, 생성 중 결정론 설정 복구 |
| `grpo_order/objective.py`, `train.py` | 한 문제 4응답 advantage, microbatch backward, AdamW 한 번, 설정된 시점의 평가 |
| `grpo_order/evaluation.py`, `step0.py` | 평가 RNG 격리, 8개 샘플+greedy, 초기 상태 검증 캐시 |
| `grpo_order/storage.py` | 배타 lock, 원자적 checkpoint, optimizer/RNG/cursor 저장·복원 |
| `grpo_order/launcher.py` | 한 GPU에서는 순차 실행, 두 GPU 설정에서는 pair 동시 실행. full은 설정된 seed 수 × 4회 |
| `grpo_order/freeze.py`, `core.py` | 설정 불변 조건·hash·seed, 확정 전 full 차단 |
| `grpo_order/analysis.py` | full 평가의 seed별 성능·조건 차이·곡선·그래프 |
| `grpo_order/integrity.py` | 정상 update·학습 순서·응답·평가 집계 검증 |

이하에는 실험 정의와 개별 CLI 명령을 정리했다. 새 3단계 실행은 위 절을 따른다. 아래 두 GPU 배정·설정 탐색 파일럿 예제는 수동 선택 경로이며 기본 실행이 아니다.


**확정 모델·데이터와 프로젝트 상대 경로** — 설정도 `./configs/experiment.json` 맨 위에서 확인한다.

| 용도 | 정확한 Hugging Face ID | 저장 경로 |
|---|---|---|
| Base | `Qwen/Qwen2.5-1.5B` | `./models/Qwen2.5-1.5B/` |
| 공개 RL | `hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo` | `./models/Qwen-2.5-1.5B-SimpleRL-Zoo/` |
| 학습 원천 | `EleutherAI/hendrycks_math` 전체 7과목 | `./data/raw/hendrycks_math/` |
| 최종 평가 | `HuggingFaceH4/MATH-500` | `./data/raw/MATH-500/` |
| 통합·변환 JSONL | 원본 split·과목·level·안정적인 ID 보존 | `./data/converted/hendrycks_math/` |
| 고정 학습·진단·validation·평가 | 문제 집합과 hash | `./data/prepared/qwen15b_math_order_v2/` |
| 본실험 결과 | 2모델 × 2순서 × 1 seed = **4회** | `./runs/qwen15b_math_order_v2/full/` |
| 다운로드 revision lock | 모델·토크나이저·데이터 commit | `./assets.lock.json` |

| 실험 난도 | 원본 level |
|---|---|
| Easy | Level 1–2 |
| Medium | Level 3 |
| Hard | Level 4–5 |

**난도는 데이터셋 등급의 묶음이며 모델 정답률 기반 난도가 아니다.** Math-1.5B/Instruct 모델로 대체하지 않는다. S0/S1/S2 또는 직접 사전 RL을 만드는 실행은 없다. 다운로드 → 데이터 변환 → split 준비 → 사전 점검 → 학습은 각각 별도 명령이며 다운로드가 GPU 학습을 시작하지 않는다.

Base와 **이미 RL 후학습된 공개 모델** 각각에서 Random/Curriculum을 비교한다. 각 실행은 설정한 원본 체크포인트에서 새 LoRA로 시작하며, 이전 조건이나 seed의 최종 가중치를 이어받지 않는다.

초기 확인 결과 이 프로젝트 폴더는 비어 있었다. 기존 README, AGENTS.md, 채점기, 학습 코드, 모델/데이터 경로, 결과가 없었으므로 독립 패키지를 작성했다. 현재 개발 환경은 macOS이며 NVIDIA GPU가 없다. PRO 6000의 세대·VRAM은 **미확인**이다. 서버에서 검사하기 전 48/96 GB 등으로 가정하지 않는다. 본실험은 자동 실행하지 않았다.

## 서버 설치와 사전 검사

Linux, Python 3.11 또는 3.12, 기본 CUDA GPU 한 장(논리 slot 0)을 대상으로 한다. 수동 두 GPU 실행은 runtime.allocated_gpu_slots=[0,1] 설정을 사용한다. 호스트 물리 번호는 가정하지 않는다. 프로젝트 루트에서 실행한다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -r requirements-assets.txt
python -m pip freeze > environment.lock.txt
bash scripts/run.sh preflight --environment-only
```

CUDA 12.8 wheel은 최초 파일럿 환경의 **명시적 선택**이며 GPU/드라이버 호환성을 보장하는 추정이 아니다. [PyTorch 공식 배포표](https://pytorch.org/get-started/previous-versions/)에 있는 버전을 고정했다. `preflight`는 `nvidia-smi`에서 정확한 GPU 이름·UUID·GPU별 전체/사용/가용 메모리·드라이버·compute process를 수집하고, PyTorch CUDA 버전·compute capability·지원 아키텍처·BF16 지원을 기록한다. 실제 BF16 matmul/backward도 검사한다. 실패하면 학습을 중단한다. 호환 환경이 달라 버전을 변경해야 한다면 requirements와 실험 output 경로를 새로 정하고 CPU/GPU 검증을 다시 수행한다.

사전 검사의 가용 VRAM 하한 8 GiB는 **실행 허용 최소 기준**일 뿐 모델의 메모리 적합성 판정이 아니다. 실제 모델의 smoke 및 2,048/4,096 출력 파일럿으로 확인한다. 다른 GPU compute process는 기본적으로 실행을 막는다. 서버의 process 검사/모델 hashing은 시간이 걸릴 수 있다. 검사 결과는 `reports/preflight-*.json`, 실행기는 `./runs/qwen15b_math_order_v2/preflight/*.json`에 저장한다.

`requirements.txt`의 직접 의존성 버전은 실행 전 대조한다. 전이 의존성까지 재현하려면 첫 정상 서버의 `environment.lock.txt`도 보관하고 이후 같은 lock으로 설치한다. 각 실행에는 실제 관련 패키지 버전이 저장되고 재개 시 비교된다. macOS CPU 검증에 사용한 별도 캐시 환경은 서버 기준 환경이 아니다.

## 모델·토크나이저 준비와 계보

모델·데이터 경로는 `./configs/experiment.json` **맨 위**에 모았다. 다운로드/변환 코드는 `grpo_order/assets.py`, `grpo_order/import_math.py`, 고정 source ID와 난도 매핑은 `grpo_order/sources.py`에서 읽을 수 있다. 모든 `./` 경로는 프로젝트 루트 기준이다. shell 스크립트는 자동으로 프로젝트 루트로 이동한다.

로컬 점검 당시 지정된 두 1.5B 모델과 데이터는 없었다. 전역 Hugging Face 캐시의 다른 7B 모델은 사용하지 않았다. 빈 `./models/`, `./data/` 디렉터리는 실제 가중치/데이터가 준비됐다는 뜻이 아니다.

모델·데이터 준비만 하는 명령:

```bash
# GPU 없이 별도 CPU 환경에서도 가능
python -m pip install -r requirements-assets.txt

# 두 지정 모델만 준비. 학습을 호출하지 않음.
bash scripts/run.sh download --only models

# EleutherAI/hendrycks_math 및 최종 평가 MATH500만 준비
bash scripts/run.sh download --only data

# 기존 Hugging Face 로컬 캐시만 복사하려면 --offline 추가
# bash scripts/run.sh download --only models --offline
```

이미 준비된 `./models/.../.asset.json`과 파일 hash를 먼저 검사한다. 없으면 지정 저장소의 로컬 HF 캐시를 확인하고, 없거나 불완전하면 네트워크에서 받는다. 가중치·토크나이저 모두 해당 모델 저장소의 **같은 40자리 commit SHA**를 사용한다. 최초 설정의 `revision: null`은 준비 시 revision을 고정하라는 뜻이다. 준비 결과는 `./assets.lock.json`과 각 snapshot의 `.asset.json`에 남으며 이후에는 lock revision을 재사용한다. 필요하면 준비 전에 `models.*.revision`, `tokenizer_revision`에 같은 commit SHA를 명시할 수 있다. 다른 모델명으로 자동 대체하지 않는다.

준비가 중단되면 숨김 `.모델명.download-<sha>` 디렉터리에 파일을 보관한다. 재시도는 같은 요청 revision의 임시 디렉터리를 사용한다. 완료된 snapshot은 덮어쓰지 않는다. 다른 모델을 수동으로 같은 폴더명으로 바꿔 넣어도 출처 manifest가 없거나 다르면 학습을 거절한다. 학습 명령은 계속 `local_files_only=True`, `trust_remote_code=False`이며 다운로드를 호출하지 않는다.

Base의 [공식 모델 카드](https://huggingface.co/Qwen/Qwen2.5-1.5B)는 사전학습 Base로 명시한다. [SimpleRL 공식 저장소](https://github.com/hkust-nlp/simpleRL-reason/tree/v1)는 Qwen2.5-1.5B Base에서 출발한 Zero-RL 체크포인트를 공개하며 중간 SFT 없이 학습했다고 설명한다. [논문 §2.1, Appendix B.1](https://arxiv.org/html/2503.18892v3)은 GRPO를 사용했고 1.5B 주 실행에 MATH Level 3–5를 사용했다고 명시한다. 개별 [RL 모델 README](https://huggingface.co/hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo)는 확인 시 비어 있어 공식 프로젝트/논문을 계보 근거로 기록했다. 세부 학습 로그·정확한 문항별 노출·문서에 없는 추가 학습/증류 이력은 확인하지 못했으며 추정하지 않는다. 같은 크기라는 이유로 계보를 판단하지 않는다.

논문은 원본 MATH train과 일부 test를 합쳐 RL 학습했으며 MATH500과 별도 validation을 제외했다고 보고한다. 따라서 아래 진단셋은 “**본 추가 학습에 사용하지 않은 문제**”다. 공개 RL 또는 사전학습에서 전혀 본 적 없다고 주장하지 않는다. Base/RL 효과에는 각 출발 체크포인트의 과거 학습 이력이 포함된다.

실제 저장 dtype은 `.safetensors` header에서 dtype별 parameter 수를 읽어 `.asset.json`에 기록한다. Base config의 BF16 표기나 Hub의 RL F32 표기를 실제 파일 검사로 대체하지 않는다. 학습은 원본을 BF16으로 로딩하고 실제 frozen/trainable dtype을 `metadata.json`의 `lora.frozen_dtypes`, `lora.trainable_dtypes`에 기록한다. `model.resolved_model_revision`, `resolved_tokenizer_revision`, `actual_stored_parameter_dtypes`, `lineage_sources`도 저장한다.

## 7개 과목 통합과 데이터 준비

원천은 **`EleutherAI/hendrycks_math`**이며 다음 7개 subset을 모두 읽는다.

```text
algebra, counting_and_probability, geometry, intermediate_algebra,
number_theory, prealgebra, precalculus
```

[공식 데이터 저장소](https://huggingface.co/datasets/EleutherAI/hendrycks_math/tree/main)의 과목별 Parquet를 직접 읽으므로 remote dataset Python 코드를 실행하지 않는다. 원본 train 7,500개, test 5,000개가 모두 합쳐지는지 검증한다. 과목 또는 split 누락, 예상과 다른 전체 수는 오류다. 통합은 **과목 간 통합**이며 train/test 혼합이 아니다. 원본 test 5,000개는 출처 확인용으로 보관만 하고 추가 학습 후보에는 넣지 않는다.

```bash
# 준비한 ./data/raw/를 오프라인 변환: 다운로드/학습 없음
bash scripts/run.sh import

# 중복 검사와 고정 train/diagnostic/validation/MATH500 manifest 생성
bash scripts/run.sh prepare --config ./configs/experiment.json

# GPU 서버에서 전체 사전 점검
bash scripts/run.sh preflight --config ./configs/experiment.json
```

변환 단계에서 원본 `problem`, `solution`, `type`, `level`을 보존하고 `source`, `source_revision`, `split`, `subject`, `subset`, `difficulty`, `source_file`, `source_row`, `original_id`, 안정적인 `id`를 추가한다. ID는 원천 데이터셋·split·subset·정규화 문제 텍스트의 SHA256에 기반하므로 seed나 원본 행 순서가 바뀌어도 같고, 중복 문제는 같은 ID로 검출된다. 원본에 ID가 없으면 `original_id=null`로 명시한다. 원본 출처는 로컬 파일명으로 덮어쓰지 않는다. 이 필드들은 prepared 문제와 학습·평가 로그의 `origin`에도 남는다.

난도는 **원본 데이터셋 등급을 묶은 기준**이다. 모델의 실제 정답률, 출력 길이, GRPO reward로 정의하지 않는다. 알 수 없는 원본 level은 임의로 매핑하지 않는다. solution의 마지막 `\boxed{...}` 또는 `\fbox{...}`에서 중첩 중괄호를 보존해 gold를 추출하며, 추출 불가/불명 난도는 `conversion_exclusions.json`에 기록한다. 해당 원본 행도 `all_train.jsonl`, `all_test.jsonl`에는 보존한다. 모델에는 solution/gold를 prompt로 주지 않는다. gold의 수학적 해석 가능 여부는 실행 전 math-verify로 별도 검사한다.

| 구분 | 원본 | 수량/선택 |
|---|---|---|
| 학습 | 원본 train, 전체 7과목 후보 | Easy/Medium/Hard 각 100 = 300 |
| 진단 | 같은 train에서 추가 학습과 겹치지 않는 문제 | 각 최대 50 = 목표 150 |
| 개발 validation | 원본 train에서 seed 20261024로 우선 분리 | 각 50 = 초기 150, 중복 제거 후 실제 수 기록 |
| 최종 평가 | `HuggingFaceH4/MATH-500` test | 500 전체 |
| 원본 MATH test | 전체 7과목 | 5,000개 보관만; 추가 학습에는 사용하지 않음 |

최종 평가 원본은 [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)이며 사용자가 확정한 학습 원천과 구분한다. MATH500의 `unique_id`, 과목, 원래 숫자 level도 보존한다.

MATH500 → validation → pool 우선순위로 전역 ID, NFKC/공백/대소문자 정규화 텍스트, 문자 5-gram Jaccard ≥ 0.9 중복을 검사한다. 제외 근거는 prepared의 `exclusions.json`에 기록한다. MATH500 내부 중복은 임의로 benchmark를 축소하지 않고 중단한다. 이 검사가 의미적 동형/번역을 완벽히 잡는다는 주장은 하지 않는다.

고정 `selection_seed=20261023`으로 남은 train 후보에서 난도별 학습 100개와 진단 최대 50개를 선택한다. 학습 seed 1/2/3과 무관하게 문제 집합이 동일하다. 진단 부족분을 중복으로 채우지 않으며 학습 100개 미달 또는 난도별 진단 0개는 오류다. validation도 중복 제거 후 비어 있으면 중단한다. 준비 전에 math-verify 기본 분수/소수 동치 검사를 실행한다. 설치·런타임 오류는 전체 준비를 막으며 후보 제외로 처리하지 않는다. 빈 gold 등 명백한 문항 문제는 `question_problem`, 기본 검사를 통과했지만 symbolic parse가 안 되는 경우는 `parser_limitation`으로 구분한다. 파싱 실패만으로 문항 자체가 잘못되었다고 단정하지 않는다. pool/validation 제외 사유·원문·출처는 `exclusions.json`과 `gold_audit.json`에 기록한다. MATH500 실패는 원본 500개 수와 실패 ID를 `prepared_dir` 옆 고유 이름의 gold audit 보고서에 남기고 준비를 중단한다. 축소 benchmark나 자동 오답은 만들지 않는다. 채점기 환경이 바뀌거나 이전 schema의 prepared 데이터를 쓰면 새 prepared_dir로 다시 준비해야 한다.

원본 revision·파일 hash·과목별 수량은 `import_manifest.json`, 최종 선택과 중복 검사 결과는 prepared `manifest.json`에 기록한다. 기존 변환/manifest 디렉터리는 덮어쓰지 않는다.

Random은 고정 300개를 한 번 shuffle한다. Curriculum은 난도별 내부 shuffle 후 Easy → Medium → Hard 순서다. DataLoader/sampler는 없으며 고정 리스트를 직접 순회한다.

## 실험 프로토콜과 수치 정의

| 항목 | 최초 파일럿 값 |
|---|---|
| 본실험 조합 | Base/RL × Random/Curriculum × seed 1 = 4회 |
| 학습 | LoRA r=16, alpha=32, dropout=0 |
| 초기화 | PEFT `init_lora_weights=True`: A Kaiming uniform, B zero |
| 대상 | 지원 attention/MLP Linear·Conv1D 이름을 탐색, 실제 전체 경로 저장; 필요시 `target_modules`로 명시 |
| Optimizer | AdamW lr=5e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, foreach=false, fused=false |
| Scheduler / warmup | constant / 0; 별도 scheduler state 없음 |
| Precision | 원본 BF16, PEFT가 유지하는 FP32 adapter, softmax/log-prob FP32; GradScaler 없음 |
| Grad norm | 1.0으로 clip, clip 전 norm 기록 |
| 노출 / 생성 / 예정 step | 1 epoch, 문제 300회 / 응답 1,200개 / 300회 |
| 생성 | 문제 1개 × 응답 4개, temperature=1.0, top-p=1.0, top-k=0 |
| 출력 상한 | 2,048; 파일럿용 4,096 변형 제공 |
| Microbatch / reuse | 응답 1개 / 생성 그룹 재사용 1회 |
| KL | beta=0; reference 모델 미로딩 |
| Advantage | `(r - mean(r)) / (population_std(r) + 1e-8)`, std는 ddof=0 |
| PPO clip | 0.2, 아래 단일 on-policy step에서는 ratio=1 |
| Loss | 응답별 생성 토큰 평균 후 4개 응답 평균 |

고정 protocol 밖의 KL·group reuse·epoch·loss 종류·top-p 변경 등은 조용히 무시하지 않고 거절한다. LoRA 대상 목록과 trainable 파라미터 목록을 저장하고 LoRA 외 학습 파라미터가 있으면 중단한다. 미지원 모델 구조에는 명시적 `target_modules`를 설정한 뒤 실제 목록을 확인한다. base model의 모듈 dropout도 0으로 만들어 generation과 학습 log-prob 재계산의 정책을 일치시킨다.

한 문제의 응답 4개 전체를 먼저 생성·채점한다. 정답 1, 오답·추출 실패 0이다. 별도 형식/길이 보상은 없다. [Math-Verify](https://github.com/huggingface/Math-Verify)의 수학적 동치 판정을 사용하고 추출 실패·오답·grader 예외를 구분한다. 숫자만 일치시키는 임의 채점기를 쓰지 않는다. 전부 오답/전부 정답 그룹의 advantage는 정확히 0이며 버리지 않는다. Dynamic Sampling은 없다. **advantage가 모두 0이어도 AdamW step을 호출**한다. 이전 momentum 때문에 파라미터가 변할 수 있으며, 이 정책은 두 조건에 동일하다.

응답 i의 길이를 Tᵢ라 하면 loss는 `-(1/4) Σᵢ (1/Tᵢ) Σₜ min(ratioᵢₜ Aᵢ, clip(ratioᵢₜ) Aᵢ)`다. prompt·padding은 제외하고 생성 EOS는 포함한다. Microbatch마다 전체 그룹 크기 4로 나눈 기여를 누적하며 추가 microbatch 평균을 하지 않는다. 모든 backward가 끝난 뒤 clip과 optimizer.step을 한 번 수행한다. advantage가 모두 0인 그룹도 backward를 실행한다. AdamW의 zero gradient는 기존 momentum/weight decay에 따른 업데이트가 가능하지만 grad=None은 해당 parameter를 건너뛰므로 둘을 같은 최적화로 취급하지 않는다.

단일 use·단일 update이고 생성 후 파라미터 변경이 없으므로 `old_logp = current_logp.detach()`가 이 최초 step의 old policy와 같다. `exp(logp - logp.detach())`의 값은 1이고 gradient는 policy gradient다. 따라서 clipping은 이 프로토콜에서 비활성이다. 그룹을 여러 번 재사용하도록 확장할 경우 이 방식은 사용할 수 없다. 학습 log-prob는 생성과 같은 temperature로 logits를 나눈 값이고 학습 top-p는 1로 고정한다. scalar loss가 거의 0이어도 gradient는 0이 아닐 수 있다. [TRL 공식 문서](https://huggingface.co/docs/trl/v0.25.1/grpo_trainer)에 다양한 정규화 옵션이 있어, 여기서는 TRL 기본값에 의존하지 않고 loss를 직접 고정했다.

길이 제한 도달 시 부분 출력에서 동일한 채점기로 답을 추출하고 **생성된 모든 토큰을 loss에 포함**한다. 자동 0점/그룹 제거를 하지 않는다. EOS 종료, 길이 종료, token budget 도달을 별도 저장한다. 마지막 budget 토큰이 EOS면 `hit_token_budget=true`, `truncated=false`다. token entropy는 생성 위치(EOS 포함)에 대한 `softmax(logits / train_temperature)`의 **전체 어휘 Shannon entropy**, 자연로그 nats이며 모든 응답 토큰을 가중 평균한다. 샘플 log-prob의 음수 평균과 구분한다.

입력을 truncate하지 않는다. 문서/모델 config/tokenizer에서 얻은 유효 한도 중 최소를 사용하고 train·diagnostic·validation·MATH500 전부에 `prompt_tokens + max_new_tokens ≤ context_limit`를 확인한다. 기본은 두 모델에 같은 plain prompt다. 모델별 chat template 차이가 추가 변수가 될 수 있으므로 `prompt.mode=chat` 사용 시 그 영향을 기록한다.

## Seed와 초기 상태

| 용도 | 값 |
|---|---|
| 문제 집합 선택 | 20261023, 학습 seed와 독립 |
| 학습 replicate | 1, 2, 3 |
| 순서 | 10000 + replicate |
| LoRA 초기화 | 20000 + replicate |
| 학습 생성 RNG | 30000 + replicate, 실행 중 상태 저장 |
| 평가 | 40000과 split/문제 ID의 SHA256에서 파생 |
| 문제 bootstrap | 50000 |

같은 모델·seed의 Random/Curriculum은 같은 원본 모델 hash와 초기 LoRA hash를 가져야 한다. pair registry를 원자적 lock 아래 비교한다. 서로 다른 Base/RL 모델의 adapter tensor 자체가 같다는 주장은 하지 않는다. 원본 모델 파일·모델 카드·설정·데이터·순서·코드·패키지 버전이 실행 contract에 들어간다. 이전 조건이나 seed의 최종 가중치를 다음 실행의 출발점으로 사용하지 않는다.

## GPU 할당과 실행

실행기는 부모의 `CUDA_VISIBLE_DEVICES` 할당을 보존한다(없으면 설정에 따라 한 GPU는 `0`, 두 GPU는 `0,1`). `CUDA_DEVICE_ORDER=PCI_BUS_ID`를 설정하고 실제 `torch.cuda.get_device_properties(i).uuid`와 nvidia-smi UUID를 대조한다. 기본 한 GPU 설정에서는 두 조건 모두 slot 0에서 순차 실행한다. 두 GPU 설정에서는 slot 0=Random, slot 1=Curriculum이며 worker에는 검증한 **UUID**를 전달한다. 컨테이너 내부 0·1과 호스트 nvidia-smi 0·1이 같다고 가정하지 않는다. UUID를 얻거나 대조할 수 없으면 실행을 막는다. 같은 모델·seed의 두 실행이 모두 끝나야 다음 조합으로 진행한다. 기본은 seed 1에서 Base Random → Base Curriculum → RL Random → RL Curriculum의 총 4회다. `seeds.training=[1,2,3]`인 기존 설정도 지원하며 이 경우 seed 1→2→3 순서로 총 12회다.

GPU UUID별 lock과 실행별 lock이 학습/별도 평가의 충돌을 막는다. compute process 검사도 UUID로 필터링한다. 컨테이너 PID namespace 때문에 자기 PID를 확인할 수 없으면 보수적으로 중단한다. 서버에서 필요하면 호스트 PID namespace로 실행하여 확인한다. 임의로 process 검사를 끄지 않는다. 두 GPU 모드를 선택하면 두 순서가 서로 다른 GPU 개체에 고정되므로 시간·처리량 비교에 GPU 개체 차이가 포함된다. 기본 한 GPU 모드에서는 같은 GPU를 순차 사용한다.

```bash
# 원본 모델/데이터 준비 뒤; 기존 prepared v1/results는 덮어쓰지 않음
bash scripts/run.sh prepare --config configs/experiment.json
bash scripts/run.sh preflight --config configs/experiment.json

# 생성→채점→update→평가→checkpoint 뒤 실제 프로세스 종료
# 각 조건 3문제, 생성 상한 32, 진단 3문제; 과학적 결과에서 제외
bash scripts/run.sh smoke --config configs/experiment.json --stop-after 1
# 새 프로세스에서 LoRA/AdamW/RNG/cursor 복원, 남은 2문제 진행
bash scripts/run.sh resume --config configs/experiment.json --mode smoke
```

중단 없이 smoke 전체를 실행하려면 `--stop-after 1`을 생략한다. smoke 종료 시 저장한 checkpoint를 다시 읽고 평가까지 수행한다. 위 중단/재개 명령은 프로세스 경계를 넘는 복원도 확인한다. 기본 출력 경로가 이미 있다면 새 variant를 만든다. `--resume`/`--resume-failed`는 같은 contract의 미완료 실행에만 사용한다.

## Validation 파일럿: prompt → 길이 → 필요 시 LR

pilot은 **validation만** 자동 평가한다. MATH500은 full의 step 0/300에서만 생성·채점하며 `evaluate --mode pilot --split math500`도 거절한다. 사전 데이터 gold 감사에는 500개 전체가 필요하지만 모델 성능 튜닝에 쓰지 않는다. LR 기본은 계속 **5e-6**이다.

1. 포함된 두 설정으로 Base/RL × Random/Curriculum을 step 0에서 평가한다. `pilot.updates=0`이라 학습하지 않으며, 공통 prompt를 validation 정확도·잘림률·추출 실패를 보고 선택한다.

```bash
bash scripts/run.sh pilot --config configs/pilot_plain2048.json --seed 1
bash scripts/run.sh pilot --config configs/pilot_chat2048.json --seed 1
```

2. 선택한 prompt 설정을 `PROMPT_CONFIG`에 지정한 뒤 길이만 2048/4096으로 비교한다. 아래 변수는 사용자가 실제 validation 결과로 선택해 설정한다. 각 조건 30 update는 초기 비용/학습 신호 점검용이고 본실험 300 update를 대체하지 않는다.

```bash
# 예: export PROMPT_CONFIG=configs/pilot_plain2048.json (또는 pilot_chat2048.json)
: "${PROMPT_CONFIG:?먼저 validation으로 선택한 prompt 설정 경로를 지정하세요}"
python scripts/make_variant.py --source "$PROMPT_CONFIG" --output configs/pilot_len2048.json --train-max-new-tokens 2048 --eval-max-new-tokens 2048 --pilot-updates 30 --output-dir ./runs/qwen15b_pilot_len2048_v2
python scripts/make_variant.py --source "$PROMPT_CONFIG" --output configs/pilot_len4096.json --train-max-new-tokens 4096 --eval-max-new-tokens 4096 --pilot-updates 30 --output-dir ./runs/qwen15b_pilot_len4096_v2
bash scripts/run.sh pilot --config configs/pilot_len2048.json --seed 1
bash scripts/run.sh pilot --config configs/pilot_len4096.json --seed 1
```

3. 선택한 길이 설정을 `LENGTH_CONFIG`로 지정하고 기본 LR로 300 update를 검증한다. validation은 설정된 diagnostic 시점(기본 0/300)과 마지막 update에서 실행된다. reward·혼합 그룹 비율, LoRA A/B norm, validation 변화를 점검한다. 필요할 때만 동일 조건에서 명시한 LR 후보와 비교한다.

```bash
# LENGTH_CONFIG=configs/pilot_len2048.json 또는 configs/pilot_len4096.json
: "${LENGTH_CONFIG:?validation으로 선택한 길이 설정 경로를 지정하세요}"
python scripts/make_variant.py --source "$LENGTH_CONFIG" --output configs/pilot_train.json --pilot-updates 300 --output-dir ./runs/qwen15b_pilot_train_v2
bash scripts/run.sh pilot --config configs/pilot_train.json --seed 1
# 필요할 때만 실행하는 LR 후보. 기본 LR을 자동 변경하지 않는다.
python scripts/make_variant.py --source configs/pilot_train.json --output configs/pilot_lr1e5.json --lr 1e-5 --output-dir ./runs/qwen15b_pilot_lr1e5_v2
bash scripts/run.sh pilot --config configs/pilot_lr1e5.json --seed 1
```

파일럿 평가는 각 실행의 `eval/validation_step-*.json`, 학습 신호는 `steps/*.json`에 있다. `analyze`의 주 endpoint는 full MATH500이므로 validation-only pilot에 그대로 적용하지 않는다. 각 variant는 독립 설정·output_dir를 사용하고 기존 경로를 거절한다.

## 확정 및 full (자동 시작하지 않음)

검증 완료한 최종 파일럿 설정을 `SELECTED_CONFIG`로 지정한다. 코드·공통 prompt·출력 길이·LR·데이터가 확정된 뒤 실행할 명령이다. `freeze`는 full preflight를 통과한 코드/설정/데이터/모델·tokenizer/패키지 hash를 저장하며 학습하지 않는다. 이후 어느 하나라도 바뀌면 full 및 재개가 거절된다. `length_finalized=true`만으로 full을 시작할 수 없다.

```bash
: "${SELECTED_CONFIG:?검증 완료한 최종 파일럿 설정 경로를 지정하세요}"
python scripts/make_variant.py --source "$SELECTED_CONFIG" --output configs/final.json --pilot-updates 300 --finalize-length --output-dir ./runs/qwen15b_math_final_v2
bash scripts/run.sh preflight --config configs/final.json
bash scripts/run.sh freeze --config configs/final.json
bash scripts/run.sh full --config configs/final.json
# 중단 이후에만: 완료 실행은 identity 확인 후 건너뜀
bash scripts/run.sh resume --config configs/final.json --mode full
```

직접 worker를 실행할 때는 preflight에서 확인한 UUID를 사용한다. 예를 들어 Random은 `CUDA_VISIBLE_DEVICES="$RANDOM_GPU_UUID" GRPO_ALLOCATED_SLOT=0 GRPO_ASSIGNED_GPU_UUID="$RANDOM_GPU_UUID" bash scripts/run.sh train --config configs/experiment.json --model base --order random --seed 1 --mode pilot`이다. 기본 한 GPU 설정의 Curriculum도 slot=0과 같은 UUID를 쓴다. 두 GPU 설정에서만 slot=1과 두 번째 UUID를 쓴다. 두 변수의 실제 UUID는 서버 preflight 결과에서 확인해야 한다. 일반 실행에는 위 launcher 명령을 권장한다.

## 생성 정책과 step 0 공유

한 문제의 학습 응답 4개는 `num_return_sequences=4`, 평가 샘플링 8개는 `num_return_sequences=8`로 각각 **한 번의 generate 호출**에서 생성한다. greedy는 별도의 1개 호출이다. 평가는 기본 4문제씩 합치며 학습 forward/backward microbatch 1/2/4 설정과 평가 문제 배치는 구분한다. 하나의 4응답 그룹으로 advantage를 계산한 뒤 optimizer.step은 한 번 수행한다. 첫 EOS까지(해당 EOS 포함)만 보관하여 후속 padding이 길이·채점 text·loss에 들어가지 않는다.

학습 RNG는 연속된 process RNG를 사용한다. 평가는 `SHA256(배치 정책, 평가 seed, split, 순서 있는 배치 문항 ID들)`로 고정한 seed(배치 1은 기존 문제별 seed)로 8응답 배치와 greedy를 실행하고 바깥 Python/NumPy/CPU/CUDA RNG를 예외 발생 시에도 복구한다. 이 배치 정책은 과거 순차 생성과 RNG 소비가 다르므로 과거 출력과 섞지 않는다. `metadata.json` 및 평가 파일에 정책을 기록한다.

기본 결정론은 strict다. 샘플링 `top_p<1`인 **generate 호출 전체**에서만 `torch.use_deterministic_algorithms(True, warn_only=True)`를 사용하고 `finally`에서 enabled/warn_only 두 상태를 복구한다. 이 구간의 **모든 지원되지 않는 비결정론 연산**이 경고 대상이며 cumsum만 예외가 아니다. CUDA 전체의 비트 단위 재현성을 보장하지 않는다. 실패한 생성에 자동 retry나 silent 설정 변경을 하지 않는다. 학습 backward 구간에는 이 완화가 적용되지 않는다. [PyTorch 공식 동작 정의](https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html).

EOS는 실제 tokenizer EOS·generation_config EOS와, 실제 vocab에 존재하는 `<|im_end|>`의 합집합이다. 모든 EOS가 special token이며 1개 토큰으로 round-trip하는지 검사한다. generation_config가 없으면 model config를 사용했다는 사실도 기록한다. 임의의 numeric token ID는 설정하지 않는다. preflight와 metadata에서 모델별 resolved ID/문자열/출처를 확인한다.

step 0 공유는 원본 model/tokenizer hash, 전체 문항·프롬프트·평가 seed·배치·생성 설정·EOS·채점기·코드·패키지·수치 설정·GPU 종류와 첫 배치의 출력 확인값을 사용한다. 기본 seed 간 공유는 실제 LoRA B=0과 파라미터 유한값을 검증하고 adapter를 일시 비활성화한 동일 원본 모델 경로에서만 수행한다. `step0_share_across_seeds=false`이면 초기 adapter hash까지 같아야 공유한다. 캐시 lock·checksum·atomic receipt를 유지하며, 캐시 원본을 복사할 때 현재 run의 adapter hash와 evaluation ID를 기록한다. 캐시 적중에도 첫 배치 출력 확인 비용이 있다.

## 체크포인트와 평가

step은 **노출/예정 update 번호**다. snapshot 0·100·200·300을 보존하고, 정확한 실패 복구를 위해 매 update의 최신 restart state도 저장한다. 실제 optimizer step은 별도 누적하며, 비정상 loss·entropy·gradient는 해당 노출을 commit하기 전에 중단한다. BF16에는 FP16 GradScaler overflow skip이 없다. 과거 skipped-step checkpoint는 계속 학습하지 않는다. skipped step이 있는 실행은 기본 주 분석에서 거절한다.

checkpoint는 LoRA, AdamW moments/step, Python·NumPy·CPU/CUDA RNG, 데이터 cursor, 누적 비용, 마지막 event를 포함한다. 임시 디렉터리에 쓴 후 원자적으로 commit한다. 재개 시 가장 최신 완료 commit만 사용하고 hash를 검사한다. checkpoint commit과 로그 저장 사이에 실패해도 마지막 event를 복구한다. 다음 문제는 `rows[cursor]`이다. 미완료 update는 RNG를 되돌려 다시 계산할 수 있으므로 장애 전 실제 GPU 사용 비용까지 exactly-once인 것은 아니지만 **완료한 노출/학습 step에는 중복·누락이 없다**. 장애 시 남은 `.pending-*` 디렉터리는 무시되며 디스크 정리가 필요할 수 있다.

full은 기본 0·300에서 diagnostic과 MATH500을 평가한다. 100·200 update의 중간 진단평가는 생략하며, 체크포인트는 기존대로 0·100·200·300을 보존한다. pilot은 validation만 평가한다. 학습은 같은 모델·optimizer·RNG로 300 update까지 이어간다. 기존 설정의 diagnostic 0·100·200·300도 지원한다. 평가는 전체 루프(채점 포함)를 RNG 보존 문맥으로 감싸고 고정 배치별 평가 seed를 사용한다. 문제별 commit을 유지하되 중단된 배치는 원래 전체 문항을 다시 생성한다. 이미 저장된 형제 문항의 token IDs와 일치해야 재개하며, 완전히 저장된 배치는 생성하지 않는다. 평가 전후 LoRA hash도 비교한다. 응답 8개는 temperature=0.6, top-p=0.95, top-k=0이며 별도 greedy 1개를 생성한다. 생성 기본값을 새 GenerationConfig로 명시하여 모델 기본 penalty를 상속하지 않는다.

- avg@8: 문제별 `정답 응답 수 / 8`의 문제 평균.
- pass@8: 문제별 `8개 중 하나 이상 정답`의 문제 평균. 추정식으로 보정하지 않는다.
- 혼합 그룹: 응답 index `[0,1,2,3]`, `[4,5,6,7]`를 사전 고정. 두 그룹의 혼합 indicator를 **한 문제 안에서 평균**한 후 문제 평균.
- 평가 중 optimizer update 없음. 중단 시 이미 완료한 문제별 평가 JSON은 재사용한다.

저장 checkpoint 평가 예시:

```bash
CUDA_VISIBLE_DEVICES="$RANDOM_GPU_UUID" GRPO_ALLOCATED_SLOT=0 GRPO_ASSIGNED_GPU_UUID="$RANDOM_GPU_UUID" bash scripts/run.sh evaluate --config configs/pilot_train.json --model base --order random --seed 1 --mode pilot --step 100 --split validation
CUDA_VISIBLE_DEVICES="$RANDOM_GPU_UUID" GRPO_ALLOCATED_SLOT=0 GRPO_ASSIGNED_GPU_UUID="$RANDOM_GPU_UUID" bash scripts/run.sh evaluate --config configs/final.json --model rl --order curriculum --seed 1 --mode full --step 300 --split math500
```

동일 평가 결과가 이미 있으면 입력 identity 확인 후 재사용한다. 평가 설정을 바꾸어 기존 파일을 덮어쓰거나 변경된 config로 checkpoint를 조용히 로딩하지 않는다. `state.pt`는 이 파이프라인이 생성한 신뢰할 수 있는 파일만 로드한다.

## 로그와 분석

실행 예: `./runs/qwen15b_math_order_v2/full/base_seed1_random/`.

| 파일 | 내용 |
|---|---|
| `metadata.json` | 설정, seed 분리, 원본/revision/hash, 모델 카드, 패키지, 하드웨어, LoRA 대상·초기 hash, 전체 예정 순서 |
| `steps/000001.json` 등 | 실제 ID·난도·순서, 응답 text/token IDs, 추출 답·정오답·실패 이유, 길이·종료·잘림, reward mean/std/advantage, 그룹 비율, loss·norm·LR·entropy, 메모리·시간·누적 tokens |
| `checkpoints/step-000100/` 등 | 복구 가능한 상태와 checksum/contract |
| `eval/*_step-*.json` | 문제별 8개+greedy 원출력, 문제별/전체/난도별 지표, 생성 비용 |
| `eval/*_step-*/<id-hash>.json` | 중단 가능한 평가의 문제 단위 commit |
| `complete.json` | 예정/실제/skip update, 노출·응답·토큰 수, 유효성 |
| `failure.json` | 가장 최근 실행 실패의 원인. 이후 정상 재개해도 이력으로 보존 |

학습 비용은 committed 학습 응답만 누적하고 평가 비용은 평가 파일에 따로 저장한다. wall time은 생성·채점·backward/step을 포함하며 checkpoint I/O와 중간 평가 시간은 학습 시간에서 제외한다. GPU peak 메모리도 update마다 재설정한다. 길이 2,048/4,096에서 logits가 차지하는 메모리가 크므로 smoke 이후 실제 상한 파일럿이 필요하다.

```bash
bash scripts/run.sh analyze --config configs/final.json --mode full --output reports/main_v2
# full에서 완료된 seed의 네 조건만 중간 확인할 때
bash scripts/run.sh analyze --config configs/final.json --mode full --output reports/main_partial_v2 --allow-incomplete
```

주 endpoint는 MATH500 step 300의 avg@8, pass@8, greedy다. `seed_contrasts.csv`에 seed별 네 조건 성능, ΔBase=Base Curriculum−Random, ΔRL=RL Curriculum−Random, interaction=ΔRL−ΔBase를 출력한다. `summary.csv`에는 평균·seed sample SD(ddof=1)·범위를 제공한다.

문제 bootstrap 2,000회는 같은 문제 ID를 조건/seed 전체에 공동으로 resample하여 **관측된 seed에 조건부인 평가 문제 표본 불확실성**을 계산한다. 응답 8개나 reward group 2개를 독립 문제처럼 취급하지 않는다. 기본 seed 1개에서는 학습 seed 변동성을 추정할 수 없어 seed SD를 null(CSV에서는 빈칸)로 기록한다. 문제 bootstrap CI가 좁더라도 다른 seed에서의 재현성을 보장하지 않는다. 기존 3개 seed 설정에서는 seed SD를 함께 제공하지만 분산 추정은 여전히 불안정하다.

`diagnostic_curves.csv`, `training_curves.csv`와 PNG/SVG 그래프를 만든다. diagnostic의 전체/난도별 avg@8·pass@8·혼합률·잘림률·출력 길이, 학습 혼합률·잘림률·출력 길이·누적 생성 token·학습 시간·entropy 변화를 표시한다. 기본 seed 1개에서는 해당 seed의 선만 그리고 SD 음영은 표시하지 않는다. 여러 seed에서는 선은 평균, 음영은 seed SD이며 문제 bootstrap CI와 구분한다. 미완료 조건은 기본 오류이고 `--allow-incomplete` 사용 시 결측 목록을 기록한다. 기존 분석 경로는 덮어쓰지 않는다. plotting 패키지가 없는 CPU 검사에는 `--no-plots`를 쓸 수 있다.

## 검증과 남은 확인

```bash
python -m compileall -q grpo_order scripts
```

요청에 따라 CPU 테스트 파일·실행 단계·test 명령·pytest 의존성을 제거했다. 1.setup.sh는 설치 후 GPU 환경 preflight로 진행한다. 데이터 gold 감사, GPU preflight, smoke 중단·재개 검증은 그대로 수행한다.

삭제 전 검증 기록과 환경 한계는 `VALIDATION.md`에 보존한다. 현재 실행 절차에는 해당 테스트가 없다. 모델 ID·학습 원천·난도 기준은 위와 같이 확정했다. 실제 파일 다운로드와 revision lock 생성, PRO 6000 세대/VRAM/드라이버, 고정 패키지 조합에서의 GPU smoke, 본실험 공통 평가 길이는 서버에서 확인해야 한다. 이 항목들이 비어 있는 상태의 설정을 학습 성공으로 보고하지 않는다.
