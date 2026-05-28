# AI Prediction Model Selection

이 프로젝트는 Kubernetes predictive autoscaling에 적합한 트래픽 예측 모델을 선정하기 위해 Prophet, SARIMA, GRU, LSTM을 동일 조건에서 비교한다.

## 프로젝트 목적

Reactive autoscaling은 트래픽 증가가 발생한 뒤에 pod 수를 조정한다. 급격한 트래픽 증가가 발생하면 HPA가 반응하기 전까지 지연이 생기고, 이 구간에서 서비스 응답 지연이나 장애가 발생할 수 있다.

이 레포는 트래픽을 사전에 예측하고 예측값을 required pod 수로 변환하여, Kubernetes autoscaling에 사용할 최종 예측 모델을 선정하는 것을 목표로 한다.

## 문제 정의

- 입력: 시간별 트래픽 데이터와 보조 feature
- 출력: holdout 구간의 트래픽 예측값과 required pod 수
- 목표: 동일한 데이터와 동일한 holdout 조건에서 후보 모델을 비교하고 최종 모델 선정 근거를 문서화
- Primary metric: Under-provisioning rate

자세한 문제 정의는 [docs/problem-definition.md](docs/problem-definition.md)를 참고한다.

## 후보 모델

| 모델 | 역할 |
| --- | --- |
| Prophet | 계절성과 이벤트성 변동을 빠르게 반영하는 baseline |
| SARIMA | 전통적인 통계 기반 시계열 baseline |
| GRU | 순차 패턴을 학습하는 경량 recurrent neural network |
| LSTM | 장기 의존성을 고려하는 recurrent neural network |

## 데이터 생성 방식

Synthetic traffic data는 기존 `prophet-autoscaler`의 더미 데이터 생성 아이디어를 참고하여 새 구조에 맞게 재구성했다.

기본 생성 기간은 5년이다.

- 시작: `2020-01-01`
- 종료: `2024-12-31 23:00`
- 단위: 1시간

반영된 패턴:

- 월별 계절성
- 요일별 패턴
- 시간대별 패턴
- 장마철 변동성
- 태풍 영향
- 이상치 이벤트

데이터 생성:

```bash
python src/data/generate_dummy_data.py
```

출력 파일:

- `data/raw/dummy_request_rate.csv`
- `data/raw/dummy_cpu_utilization.csv`
- `data/raw/dummy_anomaly_events.csv`
- `data/processed/traffic.csv`

`data/processed/traffic.csv`는 모든 모델이 공통으로 사용하는 입력 데이터다.

주요 컬럼:

- `ds`: timestamp
- `y`: 예측 대상 request rate
- `request_rate`: synthetic request rate
- `cpu_utilization`: synthetic CPU utilization
- `is_monsoon`: 장마 여부
- `typhoon_index`: 태풍 영향 지수
- `hour`, `day_of_week`, `month`: calendar features

## 실험 조건

모든 모델은 다음 원칙을 따른다.

- 동일한 입력 파일 사용: `data/processed/traffic.csv`
- 동일한 train/holdout split 사용: 기본 5년 데이터의 마지막 20%, 약 1년을 holdout으로 평가
- 동일한 평가 지표 사용
- 동일한 pod 산정 정책 사용
- 모델별 결과를 동일한 위치에 저장

자세한 실험 설계는 [docs/experiment-design.md](docs/experiment-design.md)를 참고한다.

## 평가 지표

평가는 [src/evaluation/metrics.py](src/evaluation/metrics.py)와 [src/evaluation/pod_policy.py](src/evaluation/pod_policy.py)의 공통 로직을 사용한다.

| 지표 | 의미 | 방향 |
| --- | --- | --- |
| SMAPE | 예측값과 실제값의 symmetric error | 낮을수록 좋음 |
| Pod accuracy | 예측 pod 수와 실제 필요 pod 수가 일치한 비율 | 높을수록 좋음 |
| Under-provisioning rate | 예측 pod 수가 실제 필요 pod 수보다 부족한 비율 | 가장 낮아야 함 |
| Over-provisioning rate | 예측 pod 수가 실제 필요 pod 수보다 많은 비율 | 낮을수록 비용 효율적 |

Autoscaling에서는 pod 부족이 서비스 장애로 이어질 수 있으므로 under-provisioning rate를 primary metric으로 둔다.

자세한 기준은 [docs/model-selection-criteria.md](docs/model-selection-criteria.md)를 참고한다.

## 실행 방법

1. 의존성 설치

```bash
pip install -r requirements.txt
```

2. 데이터 생성

```bash
python src/data/generate_dummy_data.py
```

3. 모델별 학습 및 평가

```bash
python src/models/prophet/train.py
python src/models/sarima/train.py
python src/models/gru/train.py
python src/models/lstm/train.py
```

각 모델은 다음 파일을 생성한다.

- `data/predictions/{model}_predictions.csv`
- `experiments/results/{model}_metrics.json`
- `models/{model}.pt` for GRU/LSTM

4. 전체 모델 비교

```bash
python src/evaluation/compare_models.py
```

비교 결과 저장 위치:

- `experiments/results/comparison_results.csv`
- `experiments/results/best_model.json`
- `experiments/plots/model_comparison.png`

## 최종 모델 선정 방식

모델 ranking은 다음 순서로 판단한다.

1. Under-provisioning rate 낮은 모델
2. SMAPE 낮은 모델
3. Pod accuracy 높은 모델
4. Over-provisioning rate 낮은 모델

최종 선정 문서는 [docs/final-decision.md](docs/final-decision.md)에 기록한다. 현재 metric 값이 아직 확정되지 않은 경우 placeholder로 남긴다.

## 현재 실험 결과

현재 5년치 synthetic data 기준 실험에서는 Prophet이 최종 모델로 선정되었다.

| Rank | 모델 | SMAPE | Pod accuracy | Under-provisioning rate | Over-provisioning rate |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Prophet | 0.6369 | 0.6834 | 0.1268 | 0.1899 |
| 2 | GRU | 0.7730 | 0.4829 | 0.1840 | 0.3331 |
| 3 | LSTM | 0.7652 | 0.5250 | 0.2083 | 0.2667 |
| 4 | SARIMA | 1.8726 | 0.5895 | 0.4105 | 0.0000 |

![Model comparison](docs/assets/model_comparison.png)

Prophet은 primary metric인 under-provisioning rate가 가장 낮고, pod accuracy와 SMAPE도 가장 좋아 최종 모델로 선정했다. 상세 근거는 [docs/final-decision.md](docs/final-decision.md)를 참고한다.

## 디렉터리 구조

```text
ai-prediction-model/
├─ README.md
├─ requirements.txt
├─ docs/
│  ├─ problem-definition.md
│  ├─ experiment-design.md
│  ├─ model-selection-criteria.md
│  ├─ final-decision.md
│  ├─ prompt-log.md
│  └─ module-spec/
├─ src/
│  ├─ data/
│  │  └─ generate_dummy_data.py
│  ├─ models/
│  │  ├─ prophet/
│  │  ├─ sarima/
│  │  ├─ gru/
│  │  └─ lstm/
│  ├─ evaluation/
│  │  ├─ compare_models.py
│  │  ├─ metrics.py
│  │  └─ pod_policy.py
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
├─ notebooks/
└─ tests/
```

## 현재 상태

프로젝트 구조, synthetic data 생성, 모델별 학습 스크립트, 공통 평가 지표, 전체 모델 비교 파이프라인, 문서 템플릿이 준비되어 있다.

실제 최종 모델 선정은 모델별 학습 실행 후 생성되는 metric 값을 기준으로 확정한다.
