# 실행 가이드

이 문서는 데이터 생성, 모델 학습, 튜닝, OpenEvolve 실행, 시각화 생성 절차를 정리한다. 프로젝트 개요와 최종 결론은 [README](../README.md)를 참고한다.

## PowerShell 인코딩 설정

Windows PowerShell에서 한글 출력이 깨지면 현재 세션을 UTF-8로 맞춘 뒤 명령을 실행한다.

```powershell
chcp 65001
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
```

## 의존성 설치

```bash
pip install -r requirements.txt
```

PowerShell에서 `pip` 명령을 찾지 못하면 가상환경의 Python을 직접 사용한다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 데이터 생성

```bash
python -m src.data.generate_dummy_data
```

출력 파일:

- `data/raw/dummy_request_rate.csv`
- `data/raw/dummy_cpu_utilization.csv`
- `data/raw/dummy_anomaly_events.csv`
- `data/processed/traffic.csv`

## 기본 모델 학습 및 평가

```bash
python -m src.models.prophet.train
python -m src.models.sarima.train
python -m src.models.gru.train
python -m src.models.lstm.train
```

각 모델은 다음 파일을 생성한다.

- `data/predictions/{model}_predictions.csv`
- `experiments/results/{model}_metrics.json`
- `models/{model}.pt` for GRU/LSTM

## 모델 비교

```bash
python -m src.evaluation.compare_models
```

비교 결과 저장 위치:

- `experiments/results/comparison_results.csv`
- `experiments/results/best_model.json`
- `experiments/plots/model_comparison.png`

## Prophet 하이퍼파라미터 튜닝

```bash
python -m src.models.prophet.tune --trials 30 --n-jobs 2
```

튜닝은 Optuna로 수행하며, objective score는 under-provisioning rate를 중심으로 SMAPE와 over-provisioning rate를 보조 penalty로 반영한다.

```text
score = under_provisioning_rate + 0.1 * smape + 0.2 * over_provisioning_rate
```

`--n-jobs`는 병렬 trial 수다. Prophet은 trial마다 CPU를 많이 사용하므로 로컬 환경에서는 `2`부터 확인하는 것을 권장한다.

튜닝 결과 저장 위치:

- `experiments/results/prophet_tuning_trials.csv`
- `experiments/results/prophet_best_params.json`
- `experiments/results/prophet_tuning_summary.json`
- `experiments/results/prophet_tuned_metrics.json`
- `data/predictions/prophet_tuned_predictions.csv`

저장된 best params로 기본 학습 스크립트를 다시 실행할 수도 있다.

```bash
python -m src.models.prophet.train --params-path experiments/results/prophet_best_params.json
```

## OpenEvolve 기반 Prophet 최적화

OpenEvolve 관련 파일:

- `experiments/openevolve/prophet_model/initial_program.py`
- `experiments/openevolve/prophet_model/evaluator.py`
- `experiments/openevolve/prophet_model/config.yaml`
- `experiments/openevolve/prophet_model/README.md`

상세한 LLM endpoint 설정과 row-limit 설정은 [OpenEvolve Prophet README](../experiments/openevolve/prophet_model/README.md)를 참고한다.

OpenEvolve 실행 예시:

```bash
python openevolve-run.py experiments/openevolve/prophet_model/initial_program.py experiments/openevolve/prophet_model/evaluator.py --config experiments/openevolve/prophet_model/config.yaml --iterations 40
```

선정된 OpenEvolve Prophet recipe 실행:

```bash
python -m src.models.prophet.openevolve_train
```

출력 파일:

- `data/predictions/openevolve_prophet_predictions.csv`
- `experiments/results/openevolve_prophet_metrics.json`

## Holdout 시각화

```bash
python -m src.evaluation.plot_holdout_overview
python -m src.evaluation.plot_holdout_comparison --model prophet
python -m src.evaluation.plot_holdout_comparison --model prophet_tuned
python -m src.evaluation.plot_holdout_comparison --model openevolve_prophet
```

문서용 시각화 자료는 `docs/assets/`에 저장된다. 1년 overview는 전체 holdout 추세를 확인하기 위한 그래프이고, 30일 상세 그래프는 고트래픽 구간의 autoscaling decision을 확인하기 위한 그래프다.
