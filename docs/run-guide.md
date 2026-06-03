# 실행 가이드

이 문서는 데이터 생성, 모델 학습, GRU/LSTM 튜닝, 시각화 생성 절차를 정리한다. 프로젝트 개요와 최종 결론은 [README](../README.md)를 참고한다.

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
python -m src.evaluation.compare_models --models prophet sarima gru lstm
```

비교 결과 저장 위치:

- `experiments/results/comparison_results.csv`
- `experiments/results/best_model.json`
- `experiments/plots/model_comparison.png`

GRU/LSTM 튜닝까지 끝난 뒤 최종 비교를 다시 생성할 때는 기본 전체 비교 대상을 사용한다.

```bash
python -m src.evaluation.compare_models
```

## GRU/LSTM 하이퍼파라미터 튜닝

```bash
python -m src.models.sequence_tune --model gru --trials 30
python -m src.models.sequence_tune --model lstm --trials 30
```

1차 모델 비교에서 선정된 GRU와 LSTM만 Optuna로 튜닝한다. objective score는 under-provisioning rate를 중심으로 SMAPE와 over-provisioning rate를 보조 penalty로 반영한다.

```text
score = under_provisioning_rate + 0.1 * smape + 0.2 * over_provisioning_rate
```

`--trials`는 시도할 Optuna trial 수이고, `--n-jobs`는 동시에 실행할 병렬 trial 수다. CPU 코어가 충분하면 `--cpu-threads`와 함께 조정할 수 있다.

예시:

```powershell
.\.venv\Scripts\python.exe -m src.models.sequence_tune --model lstm --trials 30 --n-jobs 1 --cpu-threads 6
```

튜닝 결과 저장 위치:

- `experiments/results/{model}_tuning_trials.csv`
- `experiments/results/{model}_best_params.json`
- `experiments/results/{model}_tuning_summary.json`
- `experiments/results/{model}_tuned_metrics.json`
- `data/predictions/{model}_tuned_predictions.csv`

`{model}`에는 `gru` 또는 `lstm`이 들어간다.

## Holdout 시각화

```bash
python -m src.evaluation.plot_holdout_overview
python -m src.evaluation.plot_sequence_tuning_comparison
python -m src.evaluation.plot_holdout_comparison --model gru
python -m src.evaluation.plot_holdout_comparison --model gru_tuned
python -m src.evaluation.plot_holdout_comparison --model lstm
python -m src.evaluation.plot_holdout_comparison --model lstm_tuned
```

문서용 시각화 자료는 `docs/assets/`에 저장된다. 1년 overview는 전체 holdout 추세를 확인하기 위한 그래프이고, 30일 상세 그래프는 고트래픽 구간의 autoscaling decision을 확인하기 위한 그래프다.
