# AI Prediction Model Selection

이 프로젝트는 Kubernetes predictive autoscaling에 적합한 트래픽 예측 모델을 선정하기 위해 Prophet, SARIMA, GRU, LSTM을 동일 조건에서 비교하는 프로젝트입니다.

## Purpose

reactive autoscaling은 트래픽 증가가 발생한 뒤에 pod 수를 조정하기 때문에 급격한 부하 변화에 늦게 대응할 수 있습니다. 이 레포는 사전에 트래픽을 예측하고 필요한 pod 수를 계산하기 위한 후보 모델을 비교합니다.

## Candidate Models

- Prophet
- SARIMA
- GRU
- LSTM

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

현재는 프로젝트 기본 구조를 세팅하는 단계입니다. 기존 `prophet-autoscaler` 코드 마이그레이션, 경로 정리, 평가 파이프라인, 문서화는 후속 이슈에서 진행합니다.
