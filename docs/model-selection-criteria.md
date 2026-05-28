# 모델 선정 기준

## 선정 기준 개요

최종 모델은 Kubernetes predictive autoscaling에 사용할 모델이므로, 단순 예측 정확도뿐 아니라 pod 부족 위험을 함께 평가한다.

모델 ranking 기준은 다음 순서다.

1. Under-provisioning rate 낮은 모델
2. SMAPE 낮은 모델
3. Pod accuracy 높은 모델
4. Over-provisioning rate 낮은 모델

## Primary Metric: Under-Provisioning Rate

Under-provisioning rate는 예측 기반 pod 수가 실제 필요한 pod 수보다 부족한 시점의 비율이다.

```text
predicted_pods < actual_pods
```

이 값이 높으면 실제 운영에서 다음 문제가 발생할 수 있다.

- 요청 처리 지연
- timeout 증가
- queue 적체
- 장애 위험 증가
- HPA가 뒤늦게 scale out하는 동안의 성능 저하

따라서 autoscaling 목적에서는 under-provisioning rate를 primary metric으로 둔다.

## SMAPE

SMAPE는 실제 트래픽과 예측 트래픽의 symmetric error를 나타낸다.

낮을수록 예측값이 실제값에 가깝다. 다만 autoscaling에서는 예측 오차가 조금 낮더라도 pod 부족을 자주 만드는 모델은 최종 모델로 적합하지 않을 수 있다.

## Pod Accuracy

Pod accuracy는 예측 pod 수와 실제 필요 pod 수가 정확히 일치한 비율이다.

```text
predicted_pods == actual_pods
```

이 값이 높으면 예측 모델이 실제 autoscaling decision과 잘 맞는다는 의미다.

## Over-Provisioning Rate

Over-provisioning rate는 예측 기반 pod 수가 실제 필요한 pod 수보다 많은 시점의 비율이다.

```text
predicted_pods > actual_pods
```

과잉 provision은 비용 증가로 이어질 수 있다. 하지만 under-provisioning이 서비스 안정성에 직접적인 위험을 주기 때문에, 본 프로젝트에서는 over-provisioning보다 under-provisioning을 더 중요하게 본다.

## 모델별 Holdout 시각 비교

아래 그래프는 네 후보 모델을 같은 holdout 구간에서 비교하기 위해 실제 트래픽 평균이 가장 높은 30일 구간을 공통으로 사용했다.

- 선택 구간: `2024-03-04 06:00:00` ~ `2024-04-03 05:00:00`
- 상단 그래프: 실제 트래픽과 예측 트래픽 비교
- 하단 그래프: 실제 필요 pod 수와 예측 pod 수 비교
- 붉은 음영: under-provisioning 구간
- 파란 음영: over-provisioning 구간

### Prophet

![Prophet holdout comparison](assets/prophet_holdout_comparison.png)

### SARIMA

![SARIMA holdout comparison](assets/sarima_holdout_comparison.png)

### GRU

![GRU holdout comparison](assets/gru_holdout_comparison.png)

### LSTM

![LSTM holdout comparison](assets/lstm_holdout_comparison.png)

이 시각화는 최종 ranking을 대체하지 않고, metric으로 확인한 결과가 실제 시간축에서 어떤 형태로 나타나는지 검토하기 위한 보조 자료다. 최종 선정은 전체 holdout 기간의 under-provisioning rate를 가장 우선으로 판단한다.
