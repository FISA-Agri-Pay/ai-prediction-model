# 최종 모델 선정

## 실험 결과 요약

5년치 synthetic traffic data를 생성하고, 앞쪽 80%를 train, 뒤쪽 20%를 holdout으로 사용했다.

- Train rows: 35,078
- Holdout rows: 8,770
- Holdout 기간: 마지막 약 1년
- Primary metric: Under-provisioning rate

## 모델 비교 결과

| Rank | 모델 | SMAPE | Pod accuracy | Under-provisioning rate | Over-provisioning rate | 비고 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Prophet | 0.6369 | 0.6834 | 0.1268 | 0.1899 | 최종 선정 |
| 2 | GRU | 0.7730 | 0.4829 | 0.1840 | 0.3331 | sequence model |
| 3 | LSTM | 0.7652 | 0.5250 | 0.2083 | 0.2667 | sequence model |
| 4 | SARIMA | 1.8726 | 0.5895 | 0.4105 | 0.0000 | statistical baseline |

## 모델 비교 그래프

![Model comparison](assets/model_comparison.png)

## 선정 기준

최종 모델은 다음 순서로 선정했다.

1. Under-provisioning rate 낮은 모델
2. SMAPE 낮은 모델
3. Pod accuracy 높은 모델
4. Over-provisioning rate 낮은 모델

## 최종 선정 모델

```text
Prophet
```

## 선정 근거

Prophet은 under-provisioning rate가 `0.1268`로 가장 낮았다. Autoscaling 환경에서는 pod 부족이 서비스 지연이나 장애로 이어질 수 있으므로, under-provisioning rate를 primary metric으로 두었고 이 기준에서 Prophet이 가장 안정적이었다.

또한 Prophet은 pod accuracy도 `0.6834`로 후보 모델 중 가장 높았다. SMAPE 역시 `0.6369`로 가장 낮아, traffic value 예측과 pod decision 측면 모두에서 가장 균형이 좋았다.

GRU와 LSTM은 sequence model로서 비선형 패턴을 학습할 가능성이 있지만, 이번 기본 설정에서는 under-provisioning rate와 pod accuracy 모두 Prophet보다 낮았다. SARIMA는 over-provisioning rate가 0으로 추가 pod 비용은 적지만, under-provisioning rate가 높아 autoscaling 안정성 기준에서는 적합하지 않았다.

## 한계 및 후속 개선

- GRU/LSTM은 기본 hyperparameter로만 실행했으므로 튜닝 여지가 있다.
- SARIMA는 5년 hourly 데이터에서 학습 비용이 높고, order 후보 탐색이 필요하다.
- 이번 결과는 synthetic data 기준이므로 실제 운영 metric으로 재검증해야 한다.
- 후속 실험에서는 Optuna 또는 grid search로 under-provisioning rate를 직접 최소화하는 방향의 하이퍼파라미터 튜닝을 수행할 수 있다.
