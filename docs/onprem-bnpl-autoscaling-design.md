# On-prem BNPL Predictive Autoscaling 설계

이 문서는 On-prem BNPL 서비스별 predictive autoscaling 확장 설계를 정리합니다.

## 목표

농업 BNPL 업무 서비스를 서비스별로 예측하고, 제한된 온프레미스 pod budget 안에서 우선순위 기반 pod 비율을 조정합니다.

## 대상 서비스

```text
auth
payment
batch
admin
limit_scoring
```

## 더미 데이터

메인 README의 작물별 월별 activity, 요일/시간대 패턴, 장마/태풍, anomaly 설계를 그대로 사용합니다. 여기에 서비스별 multiplier를 추가해 `request_rate`를 생성합니다.

```text
crop_activity_score
+ repayment_day
+ harvest/purchase season
+ service hour weight
+ monsoon/typhoon/anomaly
-> service request_rate
```

## Scaling

```text
effective_capacity = capacity_per_pod * (1 - safety_margin)
required_pods = ceil(request_rate / effective_capacity)
```

예측 pod 합이 온프레미스 pod budget을 초과하면 `min_pods`를 먼저 보장하고, 남은 pod를 priority와 demand gap 기준으로 재배분합니다.
