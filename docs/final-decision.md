# 최종 모델 선정

이 문서는 실험 결과를 바탕으로 최종 사용 모델을 선정한 근거와 한계를 정리합니다.

## 실험 결과 요약

5년치 synthetic traffic data를 생성하고, 앞쪽 80%를 train, 뒤쪽 20%를 holdout으로 사용했습니다.

- Train rows: 35,078
- Holdout rows: 8,770
- Holdout 기간: 마지막 약 1년
- Primary metric: Under-provisioning rate

Autoscaling에서는 pod 부족이 서비스 지연이나 장애로 이어질 수 있으므로, 최종 선정은 under-provisioning rate를 가장 우선합니다.

## 선정 흐름

이번 실험은 두 단계로 진행했습니다.

1. Prophet, SARIMA, GRU, LSTM 4개 후보 모델을 동일한 holdout 조건에서 1차 비교합니다.
2. 1차 비교에서 sequence model인 GRU와 LSTM의 성능이 좋아 두 모델만 Optuna로 튜닝합니다.
3. 기본 GRU/LSTM과 Tuned GRU/Tuned LSTM을 2차 비교해 최종 모델을 선정합니다.

## 1차 모델 비교: 튜닝 대상 선정

| Rank | 모델 | SMAPE | Pod accuracy | Under-provisioning rate | Over-provisioning rate | 비고 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | GRU | 0.4064 | 0.8442 | 0.0426 | 0.1131 | 튜닝 대상 선정 |
| 2 | LSTM | 0.3931 | 0.8806 | 0.0555 | 0.0639 | 튜닝 대상 선정 |
| 3 | Prophet | 0.6368 | 0.6832 | 0.1268 | 0.1900 | baseline |
| 4 | SARIMA | 0.8197 | 0.6083 | 0.3762 | 0.0155 | statistical baseline |

GRU와 LSTM은 Prophet, SARIMA보다 under-provisioning rate와 pod accuracy 측면에서 더 좋은 결과를 보였습니다. 따라서 2차 실험에서는 GRU와 LSTM만 Optuna 튜닝 대상으로 선정했습니다.

![Baseline model comparison](assets/baseline_model_comparison.png)

## 2차 모델 비교: 튜닝 전후 비교

| Rank | 모델 | SMAPE | Pod accuracy | Under-provisioning rate | Over-provisioning rate | 비고 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | GRU | 0.4064 | 0.8442 | 0.0426 | 0.1131 | 기본 모델 |
| 2 | LSTM | 0.3931 | 0.8806 | 0.0555 | 0.0639 | 기본 모델 |
| 3 | Tuned LSTM | 0.3835 | 0.8819 | 0.0666 | 0.0515 | Optuna tuning |
| 4 | Tuned GRU | 0.4117 | 0.8432 | 0.0667 | 0.0901 | Optuna tuning |

Tuned LSTM은 SMAPE, pod accuracy, over-provisioning rate를 개선했고 Tuned GRU도 over-provisioning rate를 낮췄습니다. 그러나 최종 holdout의 primary metric인 under-provisioning rate는 기본 GRU/LSTM보다 높아졌습니다.

![Sequence tuning metric comparison](assets/sequence_tuning_metric_comparison.png)

## 전체 모델 비교

2차 비교 이후 전체 결과를 다시 정렬하면 기본 GRU가 가장 낮은 under-provisioning rate를 유지합니다.

![Model comparison](assets/model_comparison.png)

## Holdout 상세 비교

![GRU holdout comparison](assets/gru_holdout_comparison.png)

![Tuned GRU holdout comparison](assets/gru_tuned_holdout_comparison.png)

![LSTM holdout comparison](assets/lstm_holdout_comparison.png)

![Tuned LSTM holdout comparison](assets/lstm_tuned_holdout_comparison.png)

## 전체 Holdout Overview

![Holdout year overview](assets/holdout_year_overview.png)

위 그래프는 전체 holdout 약 1년을 일 단위 평균으로 압축해 실제 트래픽/예측 트래픽과 실제 pod/예측 pod 흐름을 비교한 것입니다. 장기 추세와 계절성 추종 여부를 확인하기 위한 보조 자료로 사용합니다.

## 최종 선정 모델

```text
GRU
```

## 선정 근거

GRU는 최종 비교에서 under-provisioning rate가 `0.0426`으로 가장 낮았습니다. 이는 실제 필요한 pod 수보다 적게 예측할 위험이 가장 작다는 의미이므로, Kubernetes predictive autoscaling의 서비스 안정성 기준에 가장 잘 맞습니다.

LSTM과 Tuned LSTM은 SMAPE와 pod accuracy가 GRU보다 좋지만, primary metric인 under-provisioning rate는 GRU보다 높습니다. Tuned GRU/Tuned LSTM은 일부 보조 지표를 개선했지만, 최종 holdout에서 pod 부족 위험을 기본 GRU보다 낮추지는 못했습니다.

따라서 이번 실험의 최종 사용 모델은 기본 GRU로 선정합니다.

## 한계 및 후속 개선

- 튜닝 objective가 validation split에서는 낮은 score를 찾았지만 holdout의 pod 부족 위험을 충분히 낮추지 못했습니다.
- 향후에는 walk-forward validation 또는 여러 holdout window를 사용해 튜닝 일반화 성능을 재검증할 수 있습니다.
- 이번 결과는 synthetic data 기준이므로 실제 운영 traffic과 HPA metric으로 재검증해야 합니다.
