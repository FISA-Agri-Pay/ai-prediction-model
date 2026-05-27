# AI Prediction Model Selection

이 프로젝트는 Kubernetes predictive autoscaling에 적합한 트래픽 예측 모델을 선정하기 위해 Prophet, SARIMA, GRU, LSTM을 동일 조건에서 비교하는 프로젝트입니다.

## Purpose

reactive autoscaling은 트래픽 증가가 발생한 뒤에 pod 수를 조정하기 때문에 급격한 부하 변화에 늦게 대응할 수 있습니다. 이 레포는 사전에 트래픽을 예측하고 필요한 pod 수를 계산하기 위한 후보 모델을 비교합니다.

## Candidate Models

- Prophet
- SARIMA
- GRU
- LSTM

## Data Generation

Synthetic traffic data can be generated with:

```bash
python src/data/generate_dummy_data.py
```

Outputs are written to:

- `data/raw/dummy_request_rate.csv`
- `data/raw/dummy_cpu_utilization.csv`
- `data/raw/dummy_anomaly_events.csv`
- `data/processed/traffic.csv`

`data/processed/traffic.csv` is the common input dataset for Prophet, SARIMA, GRU, and LSTM experiments. It includes:

- `ds`: timestamp
- `y`: target request rate
- `request_rate`: request rate
- `cpu_utilization`: synthetic CPU utilization
- `is_monsoon`: weather regressor
- `typhoon_index`: weather regressor
- `hour`, `day_of_week`, `month`: calendar features

## Project Structure

```text
ai-prediction-model/
├─ docs/
│  └─ module-spec/
├─ src/
│  ├─ data/
│  ├─ models/
│  │  ├─ prophet/
│  │  ├─ sarima/
│  │  ├─ gru/
│  │  └─ lstm/
│  ├─ evaluation/
│  └─ utils/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ predictions/
├─ experiments/
│  ├─ configs/
│  ├─ results/
│  └─ plots/
├─ models/
└─ notebooks/
```

## Status

현재는 프로젝트 기본 구조와 공통 실험 입력 데이터 생성 기능이 추가된 상태입니다. 모델별 학습 코드, 평가 지표, 전체 비교 파이프라인, 상세 문서화는 후속 이슈에서 진행합니다.
