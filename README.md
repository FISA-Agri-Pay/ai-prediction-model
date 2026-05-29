# AI Prediction Model Selection

Kubernetes predictive autoscaling에 사용할 트래픽 예측 모델을 선정하기 위한 실험 레포다. Prophet, SARIMA, GRU, LSTM을 동일한 데이터와 평가 기준으로 비교하고, 선정된 Prophet 계열 모델에 대해 Optuna 튜닝과 OpenEvolve 기반 모델 recipe 최적화를 추가로 수행한다.

## 프로젝트 목적

Reactive autoscaling은 트래픽 증가가 발생한 뒤에 pod 수를 조정한다. 급격한 트래픽 증가가 발생하면 HPA가 반응하기 전까지 지연이 생기고, 이 구간에서 서비스 응답 지연이나 장애가 발생할 수 있다.

이 프로젝트는 트래픽을 사전에 예측하고 예측값을 required pod 수로 변환하여, Kubernetes autoscaling에 사용할 최종 예측 모델을 선정하는 것을 목표로 한다.

## 디렉터리 구조

```text
ai-prediction-model/
├─ README.md
├─ requirements.txt
├─ docs/
│  ├─ run-guide.md
│  ├─ problem-definition.md
│  ├─ experiment-design.md
│  ├─ model-selection-criteria.md
│  ├─ final-decision.md
│  ├─ prompt-log.md
│  ├─ assets/
│  └─ module-spec/
├─ src/
│  ├─ data/
│  │  └─ generate_dummy_data.py
│  ├─ models/
│  │  ├─ prophet/
│  │  │  ├─ train.py
│  │  │  ├─ tune.py
│  │  │  └─ openevolve_train.py
│  │  ├─ sarima/
│  │  ├─ gru/
│  │  └─ lstm/
│  ├─ evaluation/
│  │  ├─ compare_models.py
│  │  ├─ metrics.py
│  │  ├─ pod_policy.py
│  │  ├─ plot_holdout_comparison.py
│  │  └─ plot_holdout_overview.py
│  ├─ optimization/
│  │  └─ autoscaling_score.py
│  └─ utils/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ predictions/
├─ experiments/
│  ├─ configs/
│  ├─ openevolve/
│  ├─ results/
│  └─ plots/
├─ models/
├─ notebooks/
└─ tests/
```

## 전체 파이프라인

```text
Synthetic agriculture traffic data
        |
        v
data/processed/traffic.csv
        |
        +--> Prophet baseline ---------> prophet_metrics.json
        +--> SARIMA baseline ----------> sarima_metrics.json
        +--> GRU baseline -------------> gru_metrics.json
        +--> LSTM baseline ------------> lstm_metrics.json
        |
        +--> Optuna tuned Prophet -----> prophet_tuned_metrics.json
        +--> OpenEvolve Prophet recipe -> openevolve_prophet_metrics.json
        |
        v
experiments/results/comparison_results.csv
        |
        +--> docs/assets/*holdout*.png
        |
        v
experiments/results/best_model.json
```

실행 절차는 [실행 가이드](docs/run-guide.md)를 참고한다.

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
| Tuned Prophet | Optuna로 Prophet 하이퍼파라미터를 튜닝한 모델 |
| OpenEvolve Prophet | OpenEvolve로 Prophet 모델 recipe를 최적화한 모델 |

## 데이터 생성 방식

Synthetic traffic data는 기존 `prophet-autoscaler`의 더미 데이터 생성 아이디어를 참고하여 새 구조에 맞게 재구성했다. 농자재 BNPL 서비스에서 주요 수요가 발생하는 작물은 벼, 고추, 콩, 마늘, 양파로 두고, 각 작물의 파종/정식, 생육 관리, 수확, 상환 조회 시기를 월별 activity로 반영한다.

기본 생성 기간은 `2020-01-01`부터 `2024-12-31 23:00`까지의 5년이며, 단위는 1시간이다.

반영된 패턴:

- 작물별 월별 계절성
- 요일별 패턴
- 시간대별 패턴
- 장마철 변동성
- 태풍 영향
- 이상치 이벤트

### 작물별 월별 Activity 기준

작물별 월별 activity는 내부 생성 로직에서 가중 평균되어 하나의 전체 농업 수요 계절성으로 반영된다. 가중치는 벼 `0.30`, 고추 `0.25`, 콩 `0.15`, 마늘 `0.15`, 양파 `0.15`로 둔다.

| 작물 | 주요 수요 시기 | 트래픽 반영 의도 |
| --- | --- | --- |
| 벼 | 3-5월 파종/모내기 준비, 9-10월 수확·상환 | 봄철 농자재 구매와 가을 상환 조회 반영 |
| 고추 | 3-4월 파종/정식, 7-9월 방제·수확·상환 | 연초 최고 피크와 여름 병충해/상환 수요 반영 |
| 콩 | 5-6월 파종 준비, 10-11월 수확·상환 | 초여름 구매와 가을 상환 조회 반영 |
| 마늘 | 5-6월 수확·상환, 10-11월 파종 | 여름 상환과 가을 파종 수요 반영 |
| 양파 | 5-6월 수확·상환, 9-10월 정식 | 봄/초여름 상환과 가을 정식 수요 반영 |

작물별 activity는 별도 모델 입력 컬럼으로 저장하지 않고, 기존 데이터 스키마를 유지한 채 `request_rate` 생성에만 반영한다. 이후 Prophet/SARIMA/GRU/LSTM 및 OpenEvolve 평가 코드는 기존 컬럼 기준으로 그대로 동작한다.

`data/processed/traffic.csv` 주요 컬럼:

- `ds`: timestamp
- `y`: 예측 대상 request rate
- `request_rate`: synthetic request rate
- `cpu_utilization`: synthetic CPU utilization
- `is_monsoon`: 장마 여부
- `typhoon_index`: 태풍 영향 지수
- `hour`, `day_of_week`, `month`: calendar features

## 평가 기준

평가는 [src/evaluation/metrics.py](src/evaluation/metrics.py)와 [src/evaluation/pod_policy.py](src/evaluation/pod_policy.py)의 공통 로직을 사용한다.

기본 pod 정책:

```text
capacity_per_pod = 21.1
safety_margin = 0.2
min_pods = 1
max_pods = 8
```

| 지표 | 최적화 방향 | autoscaling 관점 |
| --- | --- | --- |
| SMAPE | 낮을수록 좋음 | 트래픽 예측 자체의 정확도 |
| Pod accuracy | 높을수록 좋음 | autoscaling decision 일치도 |
| Under-provisioning rate | 가장 낮아야 함 | 서비스 지연/장애 위험 |
| Over-provisioning rate | 낮을수록 비용 효율적 | 불필요한 pod 비용 |

Autoscaling에서는 pod 부족이 서비스 장애로 이어질 수 있으므로 under-provisioning rate를 primary metric으로 둔다. 자세한 기준은 [docs/model-selection-criteria.md](docs/model-selection-criteria.md)를 참고한다.

## 최적화 방식

### Optuna Tuned Prophet

Optuna는 Prophet 모델 구조를 고정한 상태에서 하이퍼파라미터 공간을 탐색한다. objective score는 under-provisioning rate를 중심으로 SMAPE와 over-provisioning rate를 보조 penalty로 반영한다.

```text
score = under_provisioning_rate + 0.1 * smape + 0.2 * over_provisioning_rate
```

| 파라미터 | 의미 |
| --- | --- |
| `changepoint_prior_scale` | 추세 변화점에 얼마나 민감하게 반응할지 조정한다. |
| `seasonality_prior_scale` | 일/주/연 단위 계절성 패턴을 얼마나 강하게 반영할지 조정한다. |
| `holidays_prior_scale` | 이벤트성 효과를 얼마나 강하게 허용할지 조정한다. |
| `changepoint_range` | 학습 데이터 중 어느 구간까지 changepoint 후보를 둘지 정한다. |
| `seasonality_mode` | 계절성을 `additive` 또는 `multiplicative` 방식으로 반영할지 결정한다. |

### OpenEvolve Prophet

OpenEvolve는 Prophet 라이브러리 자체가 아니라 Prophet 모델 recipe 코드를 최적화한다.

| 최적화 대상 | 의미 |
| --- | --- |
| Prophet 하이퍼파라미터 | 추세 변화 민감도, 계절성 강도, changepoint 범위, seasonality mode 등을 조정한다. |
| Regressor 선택 | Prophet에 외생 변수로 추가할 feature 조합을 선택한다. |
| Feature engineering | 원본 컬럼에서 `is_peak_hour`, `is_weekend`, `monsoon_typhoon` 같은 파생 feature를 만든다. |
| Custom seasonality | 기본 daily/weekly/yearly seasonality 외에 monthly seasonality 같은 주기를 추가한다. |
| Target transform hook | 필요하면 학습 target과 예측값을 변환하는 구조를 탐색할 수 있게 한다. |

선정된 OpenEvolve Prophet recipe는 `is_monsoon`, `typhoon_index`, `is_peak_hour`, `is_weekend`, `monsoon_typhoon`을 regressor로 사용하고, period `30.5`, Fourier order `5`의 monthly seasonality를 추가한다.

## 현재 실험 결과

5년치 synthetic data 기준 기본 후보 모델 비교에서는 Prophet이 가장 좋은 baseline으로 선정되었다.

| Rank | 모델 | SMAPE | Pod accuracy | Under-provisioning rate | Over-provisioning rate |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Prophet | 0.6369 | 0.6834 | 0.1268 | 0.1899 |
| 2 | GRU | 0.7730 | 0.4829 | 0.1840 | 0.3331 |
| 3 | LSTM | 0.7652 | 0.5250 | 0.2083 | 0.2667 |
| 4 | SARIMA | 1.8726 | 0.5895 | 0.4105 | 0.0000 |

![Model comparison](docs/assets/model_comparison.png)

Prophet 계열 추가 최적화 결과:

| 모델 | SMAPE | Pod accuracy | Under-provisioning rate | Over-provisioning rate |
| --- | ---: | ---: | ---: | ---: |
| Prophet | 0.6369 | 0.6834 | 0.1268 | 0.1899 |
| Tuned Prophet | 0.6338 | 0.6796 | 0.1238 | 0.1966 |
| OpenEvolve Prophet | 0.6347 | 0.6861 | 0.1253 | 0.1886 |

Tuned Prophet은 under-provisioning rate를 `0.1268`에서 `0.1238`로 낮췄고, SMAPE도 `0.6369`에서 `0.6338`로 소폭 개선했다. OpenEvolve Prophet은 pod accuracy와 over-provisioning rate에서 가장 좋은 결과를 보였지만, primary metric 기준으로는 Tuned Prophet보다 낮지 않았다.

![Holdout year overview](docs/assets/holdout_year_overview.png)

### Holdout 상세 비교

![Prophet holdout comparison](docs/assets/prophet_holdout_comparison.png)

![Tuned Prophet holdout comparison](docs/assets/prophet_tuned_holdout_comparison.png)

![OpenEvolve Prophet holdout comparison](docs/assets/openevolve_prophet_holdout_comparison.png)

## 최종 사용 모델

최종 사용 모델은 **Tuned Prophet**으로 선정한다.

선정 기준은 Kubernetes predictive autoscaling에서 가장 중요한 지표를 under-provisioning rate로 두었기 때문이다. Tuned Prophet은 비교 대상 중 under-provisioning rate가 `0.1238`로 가장 낮아, 실제 필요한 pod 수보다 적게 예측할 위험이 가장 작다.

OpenEvolve Prophet은 pod accuracy와 over-provisioning rate에서 가장 좋은 결과를 보였지만, primary metric인 under-provisioning rate 기준으로는 Tuned Prophet보다 낮지 않았다. 따라서 최종 운영 후보는 Tuned Prophet으로 두고, OpenEvolve Prophet은 추가 최적화 가능성이 있는 보조 후보로 정리한다.

## 참고 문서

- [실행 가이드](docs/run-guide.md)
- [문제 정의](docs/problem-definition.md)
- [실험 설계](docs/experiment-design.md)
- [모델 선정 기준](docs/model-selection-criteria.md)
- [최종 모델 선정 결과](docs/final-decision.md)
- [프롬프트 기록](docs/prompt-log.md)
