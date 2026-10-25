# 구현 및 검증 기록

검증일: 2026-10-23. 작업 디렉터리는 처음에 비어 있었고 기존 결과/설정을 덮어쓰지 않았다.

## 1.5B 모델·EleutherAI 데이터 확정 후 재점검

사용자가 확정한 모델은 `Qwen/Qwen2.5-1.5B`, `hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo`이고 학습 원천은 `EleutherAI/hendrycks_math`다. 설정과 README 맨 위에 정확한 ID와 `./models/`, `./data/` 경로를 반영했다. 로컬에 지정 자산은 없었고 다른 7B 캐시는 사용하지 않았다. 실제 다운로드나 NVIDIA 학습을 수행한 것은 아니다.

확장한 전체 테스트는 **26개 중 25개 통과, math-verify 부재로 1개 skip**이다. 로그: `reports/cpu_tests_1p5b.log`. 아래 18개 검증 표는 최초 구현 당시 기록이며, 이번 재검사에서도 기존 검증을 다시 실행했다.

추가 검증/수정 내용:

- 지정 모델 ID를 검증하고 Math-1.5B/Instruct/7B 대체 설정을 거절한다.
- 원본 출처·split·과목·level이 기존 `read_source`에서 사라지던 문제를 수정했다. 변환 → prepared → 학습/평가 `origin`에 유지한다.
- 7개 과목·train/test 전체 통합을 **실제 PyArrow Parquet 파일을 만든 synthetic fixture**로 검증했다. 7,500 train/5,000 test, validation 150 분리와 test의 학습 후보 유입 차단을 확인했다. 실제 원격 데이터 12,500개를 다운로드·검사했다는 뜻은 아니다.
- Easy=Level 1–2, Medium=3, Hard=4–5 매핑, ID의 원본 행 순서/seed 독립성, 중첩/escaped boxed gold 추출을 검사했다.
- 불명 난도/추출 불가 gold는 원본 행을 남기고 제외 사유를 기록한다. 자동 난도 추정은 없다.
- 다운로드는 GPU 학습 호출이 없는 별도 모듈이다. mocked Hub로 **전송 전 SHA 고정 → 전송 → receipt 생성 → 로컬 재사용** 순서를 검사했다. 재시도 중 main이 바뀌어도 같은 SHA를 사용하도록 lock을 전송 전에 기록한다.
- snapshot 출처·revision·파일 hash·토크나이저 동일 revision 검증과 safetensors의 실제 저장 dtype 읽기를 검사했다. 분할 가중치의 index/파일 누락도 차단한다.
- 계보 근거는 공식 Qwen 카드, SimpleRL 저장소와 논문이며, 비어 있는 RL README로 상세 과거 이력을 추정하지 않는다. 과거 문항 노출은 여전히 보증할 수 없다.

추가 스크립트의 `--help`, Python compileall, shell `bash -n`도 통과했다. 실제 원본 revision/파일 dtype은 자산 준비 후 receipt에서 확정하며, Linux 고정 버전·math-verify·GPU 검증은 서버에서 남아 있다.

## 수행한 검사

| 검사 | 실제 결과 |
|---|---|
| Python 소스 compileall | 통과 |
| 모든 실행 shell의 `bash -n` | 통과 |
| 표준 라이브러리 protocol 테스트 | 통과 |
| CPU tensor/HF 포함 전체 테스트 | 총 18개 중 17개 통과, 1개 skip |
| Random/Curriculum 집합 동일·중복/누락 없음·난도 순서 | 통과 |
| 데이터 선택 seed와 생성/초기화/평가 seed 분리 | 통과 |
| exact/near 중복 제거·진단 부족 시 실제 개수·manifest 변조 차단 | synthetic fixture로 통과 |
| 보상 분산 0 / population std / avg@8·pass@8·혼합률 | 통과 |
| microbatch 1·2·4 gradient 대조 | 실제 CPU backward 통과; prompt/padding 제외한 직접 계산 gradient와도 일치 |
| AdamW·RNG·cursor checkpoint 복구 | 중단 없는 실행과 tensor bitwise 일치 |
| LoRA 초기 hash·LoRA 외 동결·gradient checkpointing | 무작위 작은 GPT-2 CPU fixture에서 통과 |
| 평가 RNG·가중치·train/eval mode 보존 | 실제 HF 생성으로 통과; 이 테스트의 채점은 fixture로 대체 |
| 실제 train 루프 중단 후 재개 | 첫 commit 직후 오류 주입 → 재개; 최종 LoRA hash, 응답, 문제 순서가 연속 실행과 일치 |
| 12회 실행기 | subprocess mock으로 GPU 0/1 지정, 같은 모델·seed의 pair 시작/대기 순서 확인; GPU 실행은 아님 |
| 분석 | synthetic 완료 결과로 3 seed의 ΔBase·ΔRL·상호작용 및 CSV/JSON 계산 검증 |
| 수학 동치 채점 | **미실행**: `math-verify` 패키지 없음; 테스트는 skip |
| matplotlib PNG/SVG 생성 | **미실행**: 로컬에 matplotlib 없음; 구현과 문법만 확인 |
| NVIDIA preflight | 예상한 실패: CUDA GPU 0개, nvidia-smi 없음 |
| 미설정 모델/데이터 검사 | 필요한 값과 누락 패키지를 보고하며 실행 중단 확인 |
| NVIDIA GPU smoke / 실제 모델 파일럿 / 12회 본실험 | **실행하지 않음** |

테스트 원문은 `reports/cpu_tests.log`, 로컬 preflight는 `reports/local_environment.json`, `reports/local_preflight.json`에 있다. reports는 연구 결과와 함께 `.gitignore` 대상이다. 이 문서는 검증 요약을 버전 관리하기 위한 파일이다.

## CPU 검증 환경의 한계

기본 Python 3.14에는 학습 패키지가 없었고 네트워크 DNS 제한으로 requirements 설치가 실패했다. 사용자 로컬 캐시에 있던 패키지를 읽기 전용으로 참조하는 별도 `.venv-cpu` 환경을 구성해 검증했다. 기존 캐시 파일은 변경하지 않았다.

- Python 3.11.11, macOS arm64
- PyTorch 2.14.0, NumPy 2.4.6
- Transformers 5.15.0, PEFT 0.21.0, Accelerate 1.15.0
- CPU 사용, NVIDIA CUDA 사용하지 않음

이 버전들은 **서버용 requirements의 고정 버전과 다르다**. 따라서 CPU 검증 결과를 고정 Linux/CUDA 환경의 실행 성공으로 해석해서는 안 된다. 서버 runtime은 requirements 직접 의존성을 검사하며 위 캐시 버전으로 실제 실험을 시작하지 못하게 한다. 캐시 참조 초기에는 불필요한 DeepSpeed가 MPS를 호출해 프로세스가 종료되는 문제가 있었으며, CPU 전용 환경의 캐시 참조에서 DeepSpeed를 제외한 후 전체 CPU 테스트가 완료됐다. 프로젝트의 실행 의존성에는 DeepSpeed가 없다.

이 머신에서 재실행한 명령:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-cpu/bin/python -m unittest discover -s tests -v
python3 -m compileall -q grpo_order tests scripts
```

`.venv-cpu`는 이 머신의 캐시를 참조하는 로컬 검증 전용이며 서버로 복사할 환경이 아니다. 서버에서는 README대로 새 `.venv`를 만들고 고정 의존성으로 모든 테스트를 수행한다.

## 구현 중 확인해 수정한 사항

- 실행기 본체가 CUDA 사전 검사를 하면 GPU context가 남아 child의 process 점검에 걸릴 수 있어, 별도 검사 프로세스가 종료된 후 학습 worker를 시작한다.
- 생성뿐 아니라 채점까지 평가 RNG 보존 범위에 포함했다.
- 모델 원본 디렉터리 안에 실행 결과를 쓰면 모델 content hash가 바뀌므로 해당 경로 배치를 명시적으로 거절한다.
- model content, config, 데이터, 코드, 패키지, 실제 순서의 identity를 재개와 결과 결합 시 확인한다.
- checkpoint commit 직후 장애에서 마지막 event를 복구하고 다음 cursor에서 이어가는 것을 통합 테스트했다.

## 서버에서 확정할 항목

1. 확정된 Base/공개 RL 원본을 지정 로컬 경로에 준비하고 실제 revision·파일 dtype 기록.
2. 확정된 원천과 난도 기준으로 실제 데이터 변환·중복 검사 후 고정 split 개수 확인.
3. PRO 6000 정확한 모델/세대/VRAM·드라이버, CUDA/BF16 실제 kernel 및 점유 상태.
4. 고정 requirements에서 CPU 테스트(특히 math-verify 포함)와 작은 GPU smoke.
5. 2,048/4,096 파일럿의 메모리·잘림·학습 신호 점검, 본실험 공통 평가 출력 길이 확정.

전체 본실험을 시작하는 명령은 제공했지만 자동 실행하지 않았다.

## 2026-10-25 소스 정리 후 재검증

- 10개 shell 래퍼를 `scripts/run.sh`로 통합, `train.py`/`cli.py` 실행 준비 코드 공통화(`open_run`, `configure_torch`, `check_environment`, `select_rows`), `preflight.resolve_model`의 파일 해시를 `assets.inventory`로 교체, 함수 내부 import 상단 이동, 미사용 import 제거.
- `python -m compileall -q grpo_order tests scripts`, `bash -n scripts/run.sh` 통과.
- `.venv-cpu`에서 `python -m unittest discover -s tests`: 26개 중 25개 통과, 1개 skip(math-verify 미설치). 정리 전과 같다.
- 남은 수정·확정 항목은 `수정.md`에 있다.


## 2026-10-25 v2 수정 후 검증 (최신)

- 구현: Base/RL × Random/Curriculum × seed 1·2·3, 총 12회. 새 v2 output/prepared 경로 사용. 기존 결과 및 reports는 보존했다. 본실험/공개 모델 GPU 학습은 실행하지 않았다.
- 실제 명령: `PYTHON=.venv-cpu/bin/python bash scripts/run.sh test`. **38개 중 37 통과, 1 skip, 실패 0**. 마지막 실행은 13.946초이며 CPU 테스트 시간이다. 실험 처리 시간/VRAM 측정이 아니다.
- 로그: `reports/validation_v2_final_1792912764512029000/cpu_tests.log`; 환경·코드 hash·검사 요약: `reports/validation_v2_final_1792912764512029000/summary.json`.
- `compileall`, `bash -n scripts/run.sh`, 포함된 세 JSON 설정 로딩, preflight/smoke/pilot/full/resume/freeze/evaluate 및 make_variant의 `--help` 검사 통과. CLI 로그는 `reports/validation_v2_1792912588131854000/`에 보존했다.
- 실제 preflight는 exit 1: CUDA 장치 0개, nvidia-smi 없음, 두 원본 모델 없음, grader 없음, 서버 고정 패키지와 로컬 버전 불일치. 보고서: `reports/validation_v2_1792912588131854000/preflight.json`. 이를 성공으로 보고하지 않는다.
- 로컬 Python 3.11.11, torch 2.14.0, transformers 5.15.0, peft 0.21.0, accelerate 1.15.0, numpy 2.4.6. `requirements.txt`의 서버 고정 버전으로 검증한 결과가 아니다.
- `.venv-cpu/bin/python -m pip install 'math-verify[antlr4_13_2]==0.8.0' 'antlr4-python3-runtime==4.13.2'`를 시도했으나 DNS/네트워크 오류로 실패했다. 실제 수식 채점 동치 테스트 1개는 skip이다. 설치/파서/문항 분류 테스트는 오류 주입 fixture로 검증했으며 실제 데이터 gold 감사는 서버에서 해야 한다.
- CPU tiny HF fixture에서 실제 batch generation→fixture 채점→backward/AdamW→평가→checkpoint→오류 후 재개, 명시적 stop-after 재개를 검사했다. 연속 실행 대비 응답·LoRA·optimizer moments/step·RNG·cursor가 일치한다. validation-only step-0 pilot도 검사했다.
- 추가 검증: 4개 응답 한 호출, EOS 이후 pad 제외/마지막 budget EOS, 결정론 플래그와 모델 mode의 예외 복구, microbatch 1/2/4 gradient, zero gradient와 grad=None의 AdamW 차이, UUID body와 GPU- 접두사의 join, 12 worker pair 대기, MATH500 실패 ID 보고/중단, cache identity 변경/동시 접근/checksum, 초기 출력 확인 및 cache hit RNG 보존, freeze 차단.
- 남은 서버 확인: pinned dependencies에서 skip 0, 실제 UUID/PID namespace/BF16/드라이버, 두 tokenizer와 generation_config의 EOS, 전체 데이터 gold audit, GPU smoke 중단/재개, 2048/4096 batch VRAM·실측 시간, validation 기반 prompt·길이·필요 시 LR 확정. `freeze` 전 full은 차단된다.

서버 명령은 README의 현재 CLI 예제를 따른다. 과거 절의 테스트 개수·명령·설계 설명은 당시 기록이며 최신 구현과 구분한다.


## 2026-10-25 first.sh 통합 실행 및 3차 점검 (최신)

`first.sh`와 `scripts/first_pipeline.py`를 추가했다. 서버용 venv 설치, 엄격한 CPU 테스트(skip 허용 안 함), 자산 다운로드, 변환/준비, preflight, smoke 중단/재개, validation prompt/길이/LR 선택, freeze, 12회 full, 분석을 연결한다. 단계별 로그·선택 근거·immutable 설정·실행 상태를 별도 세션에 보관한다. 기본 실행은 선택 후 full까지 진행하며 `--pilot-only`로 제한할 수 있다. 부모 Python은 CUDA 모델을 생성하지 않는다.

1. **설치/셸 점검:** Linux 명령 및 Python/pip 명령을 대역으로 만들어 실제 Bash 진입점을 실행했다. 설치 경로/설치 생략, 인수 없음/옵션 전달, 공백 경로, 기본/기존 GPU 할당 보존을 검사했다. Bash <4.4에서 빈 배열 + set -u로 인수 없는 실행이 실패하는 문제를 재현하고 수정했다. 통합 테스트 11개 통과.
2. **실행 흐름/재개 점검:** immutable 설정, 완료 단계 재사용, 실제 실패 subprocess, 선택 근거 변경 거절, code/package/asset 변경 차단, 자식 process group 종료, smoke 실패 시 파일럿/full 차단, validation-only 요약, 기본/추가 LR 선택과 freeze 전후 순서를 검사했다. 통합 테스트 14개 통과. `/dev/tty` 실제 PTY 입력에서 `chat`을 전달해 선택 저장도 확인했다.
3. **전체 회귀 점검:** 비기본 `CUDA_VISIBLE_DEVICES=GPU-test-seven,GPU-test-three` 환경에서 전체 테스트를 실행했다. **52개 중 51 통과, math-verify 미설치로 1 skip, 실패 0**. CPU 테스트가 서버 GPU 할당을 `0,1`로 가정하던 부분을 고쳤으며 first.sh의 CPU 테스트 child는 CUDA를 비활성화한다. `--require-no-skips` 실제 실행은 의도대로 exit 1로 중단했다. bash -n, compileall, CLI 도움말, --dry-run도 통과했다.

로그와 요약: `reports/first_three_pass_1792914026068237000/`의 `pass1_shell.log`, `pass2_workflow.log`, `pass3_all_tests.log`, `strict_skip_gate.log`, `audit_summary.json`.

macOS CPU 환경에서 검사했다. 설치 명령/실행 순서 검증에 쓰인 대역은 실제 GPU 서버·패키지 다운로드·공개 모델 학습의 성공 증거가 아니다. math-verify는 네트워크 제한으로 설치되지 않아 실제 채점 테스트 1개는 여전히 skip이다. 서버에서는 first.sh가 이를 허용하지 않으므로 설치 오류가 남아 있으면 학습을 시작하지 않는다. GPU 드라이버/UUID/BF16/실제 EOS/gold 데이터/메모리/실측 소요 시간은 서버 preflight와 smoke/pilot에서 확인해야 한다. 전체 12회 본실험은 로컬에서 실행하지 않았다.
