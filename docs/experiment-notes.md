# 실험 기록

이 문서는 README 메인 흐름에 모두 담기 어려운 실험 이력, 해석, 보류된 최적화 기록을 보존하기 위한 문서입니다.

## 현재 README 범위

메인 README는 현재 재현 가능한 기본 모델 비교에 집중합니다.

- Prophet
- SARIMA
- GRU
- LSTM

Optuna Tuned Prophet과 OpenEvolve Prophet은 Prophet 계열 추가 최적화 실험으로 보존하되, 현재 README의 기본 후보 모델 비교 표에서는 제외합니다.

## 실험 흐름 요약

초기 실험에서는 Prophet, SARIMA, GRU, LSTM을 동일한 train/holdout split으로 비교했습니다. 당시 GRU/LSTM/SARIMA는 `hour`, `day_of_week`, `month`를 정수 feature로 그대로 사용했습니다.

초기 결과에서는 Prophet 계열이 상대적으로 안정적이었고, GRU/LSTM은 holdout 구간에서 비정상적인 예측 패턴을 보였습니다. 이후 모델 입력을 점검하면서 시간 feature의 순환성이 깨지는 문제가 확인되었습니다.

## 전처리 개선

`hour`, `day_of_week`, `month`는 순환 feature입니다.

- `23시`와 `0시`는 시간상 이어집니다.
- `일요일`과 `월요일`은 주기상 이어집니다.
- `12월`과 `1월`은 계절상 이어집니다.

하지만 raw integer로 입력하면 모델은 이 값들을 큰 숫자 차이로 해석할 수 있습니다. 이를 보완하기 위해 다음 encoding을 적용했습니다.

| Raw feature | Encoded features |
| --- | --- |
| `hour` | `hour_sin`, `hour_cos` |
| `day_of_week` | `dow_sin`, `dow_cos` |
| `month` | `month_sin`, `month_cos` |

Prophet은 `ds` timestamp를 기반으로 seasonality를 내부적으로 처리하므로 이 cyclic feature를 직접 사용하지 않습니다. SARIMA와 GRU/LSTM은 명시적 feature가 필요하므로 cyclic encoding을 적용했습니다.

## GRU/LSTM 학습 개선

초기 GRU/LSTM 학습은 전체 sequence를 한 번에 넣는 full-batch 방식이었습니다. 기본 epoch가 20이면 optimizer update도 20번에 그쳐 학습이 부족할 수 있었습니다.

개선 후에는 다음 내용을 적용했습니다.

- `torch.utils.data.DataLoader` 기반 mini-batch 학습
- 기본 `batch_size=256`
- 기본 `epochs=50`
- `_TorchSequenceRegressor`를 `torch.nn.Module` 상속 구조로 변경
- 학습 시 `model.train()`, 추론 시 `model.eval()` 사용

## 전처리 개선 전후 성능

| Model | Version | SMAPE | Pod accuracy | Under-provisioning | Over-provisioning |
| --- | --- | ---: | ---: | ---: | ---: |
| GRU | raw integer time | 0.7730 | 0.4829 | 0.1840 | 0.3331 |
| GRU | cyclic encoding + mini-batch | 0.4064 | 0.8442 | 0.0426 | 0.1131 |
| LSTM | raw integer time | 0.7652 | 0.5250 | 0.2083 | 0.2667 |
| LSTM | cyclic encoding + mini-batch | 0.3931 | 0.8806 | 0.0555 | 0.0639 |
| SARIMA | raw integer time | 1.8726 | 0.5895 | 0.4105 | 0.0000 |
| SARIMA | cyclic exog | 0.8197 | 0.6083 | 0.3762 | 0.0155 |

GRU/LSTM은 전처리와 학습 방식 개선 후 성능이 크게 향상됐습니다. SARIMA는 0으로 붕괴하던 현상은 완화됐지만, 수렴 문제와 높은 under-provisioning rate가 남았습니다.

## Prophet 계열 최적화 이력

Prophet 계열에서는 기본 Prophet 외에 다음 실험을 수행했습니다.

- Optuna Tuned Prophet
- OpenEvolve Prophet

다만 현재 모델 선정 흐름은 기본 모델 비교 후 후보 모델을 선정하는 구조로 정리했습니다. 개선 후 기본 모델 비교에서 GRU/LSTM이 Prophet보다 primary metric에서 우수했기 때문에, Prophet 계열 추가 튜닝 결과는 메인 README가 아니라 실험 기록으로 보존합니다.

### Optuna Tuned Prophet

Optuna Tuned Prophet은 Prophet hyperparameter를 objective score 기준으로 탐색했습니다.

```text
score = under_provisioning_rate + 0.1 * smape + 0.2 * over_provisioning_rate
```

현재 저장된 tuned Prophet 결과는 다음과 같습니다.

| Model | SMAPE | Pod accuracy | Under-provisioning | Over-provisioning |
| --- | ---: | ---: | ---: | ---: |
| Tuned Prophet | 0.6342 | 0.6802 | 0.1243 | 0.1956 |

기본 Prophet보다 SMAPE와 under-provisioning rate는 소폭 개선됐지만, 개선된 GRU/LSTM보다는 primary metric에서 뒤처졌습니다.

### OpenEvolve Prophet

OpenEvolve Prophet은 외부 LLM 기반 recipe 탐색을 사용하는 실험입니다. 관련 산출물은 `experiments/openevolve/prophet_model/` 아래에 보관되어 있습니다.

현재 README에서는 OpenEvolve Prophet을 기본 모델 비교 대상에 포함하지 않습니다. 외부 LLM API 비용과 재탐색 제약이 있고, 현재 후보 모델 선정 흐름에서는 기본 모델 비교에서 선정된 GRU/LSTM을 중심으로 후속 튜닝을 진행하는 것이 더 일관적이기 때문입니다.

## SARIMA 트러블슈팅 기록

SARIMA는 cyclic exog 적용 후에도 학습 시간이 매우 길었고, 다음 warning이 발생했습니다.

```text
ConvergenceWarning: Maximum Likelihood optimization failed to converge
```

최종 metric은 다음과 같습니다.

| Model | SMAPE | Pod accuracy | Under-provisioning | Over-provisioning |
| --- | ---: | ---: | ---: | ---: |
| SARIMA | 0.8197 | 0.6083 | 0.3762 | 0.0155 |

SARIMA는 over-provisioning rate는 낮지만 under-provisioning rate가 높습니다. Autoscaling 안정성 기준에서는 pod 부족 위험이 크므로 최종 후보에서 제외했습니다.

## 시각화 자료

실험 결과를 확인하기 위한 그래프는 `docs/assets/`에 보관합니다. README에는 핵심 holdout 비교 이미지를 포함하고, 상세 실험 문서에서는 필요에 따라 같은 asset을 재사용합니다.

| Asset | Description |
| --- | --- |
| `assets/holdout_year_overview.png` | 전체 holdout 구간의 모델별 traffic/pod 예측 overview |
| `assets/gru_holdout_comparison.png` | GRU high-traffic holdout 상세 비교 |
| `assets/lstm_holdout_comparison.png` | LSTM high-traffic holdout 상세 비교 |
| `assets/prophet_holdout_comparison.png` | Prophet high-traffic holdout 상세 비교 |
| `assets/sarima_holdout_comparison.png` | SARIMA high-traffic holdout 상세 비교 |

![Holdout overview](assets/holdout_year_overview.png)

## 후속 과제

- GRU/LSTM hyperparameter tuning
- GRU/LSTM ensemble 실험
- SARIMA 경량 baseline 설정 검토
- 실제 traffic data 기반 검증
- 최신 metric 기준 그래프 재생성 및 문서 이미지 갱신
