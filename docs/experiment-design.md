# 실험 설계

## 데이터 기간

기본 synthetic dataset은 5년 기간을 사용한다.

- 시작: `2020-01-01`
- 종료: `2024-12-31 23:00`
- 단위: 1시간

데이터 생성 명령:

```bash
python src/data/generate_dummy_data.py
```

## Train/Holdout 분리

모든 모델은 동일한 chronological split을 사용한다.

- Train: 앞쪽 80%, 약 4년
- Holdout: 뒤쪽 20%, 약 1년

시계열 데이터이므로 random split을 사용하지 않는다. 과거 4년 데이터로 학습하고 마지막 1년 전체를 평가하는 형태가 실제 운영 상황에 더 가깝고, holdout 구간에 계절성을 한 번 포함할 수 있다.

## 동일 조건 비교 원칙

모델 비교는 다음 조건을 고정한다.

- 동일한 입력 파일: `data/processed/traffic.csv`
- 동일한 holdout ratio: 기본값 `0.2`
- 동일한 target: `y`
- 동일한 pod 산정 정책
- 동일한 metric 계산 함수
- 동일한 결과 저장 위치

## 모델별 입력 Feature

| 모델 | 입력 |
| --- | --- |
| Prophet | `ds`, `y`, `is_monsoon`, `typhoon_index` |
| SARIMA | `y` + calendar/weather exogenous features |
| GRU | sequence window of `y` + features |
| LSTM | sequence window of `y` + features |

GRU와 LSTM은 holdout 예측 시 실제 holdout `y`를 다음 입력 window에 넣지 않도록 recursive forecast 방식을 사용한다.

## 실험 실행 순서

1. 데이터 생성

```bash
python src/data/generate_dummy_data.py
```

2. 모델별 학습 및 metric 생성

```bash
python src/models/prophet/train.py
python src/models/sarima/train.py
python src/models/gru/train.py
python src/models/lstm/train.py
```

3. 전체 모델 비교

```bash
python src/evaluation/compare_models.py
```

4. 결과 확인

- `experiments/results/{model}_metrics.json`
- `experiments/results/comparison_results.csv`
- `experiments/results/best_model.json`
- `experiments/plots/model_comparison.png`
