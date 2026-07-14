# 실험 설계

이 문서는 데이터 기간, 트래픽 생성 로직, train/holdout 분리 기준, 동일 조건 비교 원칙, 모델별 입력 feature와 실험 실행 순서를 정리합니다.

## 데이터 기간

기본 synthetic dataset은 5년 기간을 사용합니다.

- 시작: `2020-01-01`
- 종료: `2024-12-31 23:00`
- 단위: 1시간

데이터 생성 명령:

```bash
python src/data/generate_dummy_data.py
```

## 트래픽 생성 로직

`src/data/generate_dummy_data.py`는 작물별 계절 수요, 요일/시간대 패턴, 장마/태풍, 이상 이벤트를 결합해 `request_rate`(그리고 같은 방식의 `cpu_utilization`)를 생성합니다.

### 작물별 월별 활동도

벼, 고추, 콩, 마늘, 양파 5개 작물마다 1~12월 활동도(0~1)를 정의하고, 작물별 트래픽 비중(벼 30%, 고추 25%, 콩·마늘·양파 각 15%)으로 가중합해 `crop_activity_score`를 계산합니다. 예를 들어 벼는 5월(모내기)에 활동도 `1.00`으로 가장 높고, 고추는 3월(정식)에 `1.00`으로 가장 높습니다.

### 요일/시간대 패턴

| 요일 | 가중치 |
| --- | --- |
| 월~금 | 1.0 |
| 토요일 | 0.7 |
| 일요일 | 0.4 |

| 시간대 | 가중치 |
| --- | --- |
| 06~09시 | 0.90 |
| 09~12시 | 0.60 |
| 12~14시 | 0.30 |
| 14~18시 | 0.55 |
| 18~21시 | 0.85 |
| 그 외 | 0.10 |

### 장마/태풍

- 장마: 매년 `6/25~7/25` 구간을 `is_monsoon=1`로 표시합니다.
- 태풍: 매년 1~2회, 8~9월 중 임의 날짜에 상륙(landfall)한다고 가정하고, 상륙일 기준 `-1, 0, +1, +2, +3`일에 각각 `0.1, 0.9, 0.7, 0.3, 0.1`의 `typhoon_index`를 부여합니다. 같은 날짜에 태풍 영향이 겹치면 더 큰 값을 유지합니다.

### 이상 이벤트(anomaly)

| 시나리오 | 기간(매년) | 배수 |
| --- | --- | --- |
| `spring_purchase_spike` | 3/10~3/20 | ×1.4 |
| `post_typhoon_recovery_spike` | 9/2~9/4 | ×1.5 |
| `monsoon_volatility` | 7/5~7/15 | ×`uniform(0.5, 1.5)` (시간별 랜덤) |

이 이벤트 목록은 `data/raw/dummy_anomaly_events.csv`로 별도 저장됩니다.

### 최종 계산식

```text
y = base
    * crop_activity_score (월별 계절 가중)
    * 요일 가중치
    * 시간대 가중치
    * typhoon_effect  (typhoon_index >= 0.8 -> 0.4, >= 0.3 -> 1.3, else 1.0)
    * monsoon_effect  (is_monsoon=1 -> uniform(0.8, 1.2), else 1.0)
    * anomaly_boost
    + noise (정규분포, sigma = base * 0.03)
```

값은 `[0, clip_max]`로 clip합니다. `request_rate`는 `base=100, clip_max=500`, `cpu_utilization`은 `base=50, clip_max=100`이며 서로 다른 random seed로 독립 생성합니다.

`typhoon_effect`는 태풍 상륙 당일(`typhoon_index >= 0.8`)에는 실제 서비스 요청이 급감(×0.4)하고, 태풍 전후 주변부(`typhoon_index >= 0.3`)에는 오히려 트래픽이 증가(×1.3, 피해 확인·주문 몰림 등)하는 상황을 가정합니다.

## Train/Holdout 분리

모든 모델은 동일한 chronological split을 사용합니다.

- Train: 앞쪽 80%, 약 4년
- Holdout: 뒤쪽 20%, 약 1년

시계열 데이터이므로 random split을 사용하지 않습니다. 과거 4년 데이터로 학습하고 마지막 1년 전체를 평가하는 형태가 실제 운영 상황에 더 가깝고, holdout 구간에 계절성을 한 번 포함할 수 있습니다.

## 동일 조건 비교 원칙

모델 비교는 다음 조건을 고정합니다.

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

GRU와 LSTM은 holdout 예측 시 실제 holdout `y`를 다음 입력 window에 넣지 않도록 recursive forecast 방식을 사용합니다.

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
