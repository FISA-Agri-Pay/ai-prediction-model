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

`src/data/generate_dummy_data.py`의 작물별 계절 수요, 요일/시간대, 장마/태풍, anomaly 로직(자세한 계산식은 [`experiment-design.md`](experiment-design.md)의 "트래픽 생성 로직" 참고)을 그대로 재사용하고, `src/data/generate_service_dummy_data.py`에서 서비스별 특성을 추가로 반영합니다.

### 서비스별 시간대 가중치

| 서비스 | 피크 시간대 | 기본 트래픽 |
| --- | --- | --- |
| `auth` | 07~10시(×1.35), 18~22시(×1.25) | 로그인/가입 피크 |
| `payment` | 13~17시(×1.35), 09~12시(×1.25) | 결제 처리 피크 |
| `batch` | 01~05시(×1.70) | 야간 배치 |
| `admin` | 09~18시(×1.10) | 업무 시간 |
| `limit_scoring` | 09~12시(×1.25), 14~18시(×1.15) | 한도 심사 신청 |

### 서비스별 추가 부스트

- `payment`: 매월 25일 이후 상환일 근접(×1.35), 1~3일(×1.15)
- `batch`: 매월 26일 이후 월말(×1.30), 분기말(3/6/9/12월, ×1.60)
- `limit_scoring`: 파종/수확기(3,4,5,9,10,11월) 한도 신청 증가(×1.25)
- `auth`: 파종/수확기(3,4,5,9,10월) 가입/로그인 증가(×1.12)

### 파생 지표

서비스별 `request_rate`에 서비스마다 고정된 계수를 곱해 `queue_depth`, `cpu_utilization`, `p95_latency`, `error_rate`, `db_connection_usage`를 함께 생성합니다. `batch`는 큐 적체(`queue_factor=1.20`)와 지연(`latency_factor=2.10`)이 가장 크고, `admin`이 가장 완만합니다.

각 레코드에는 `business_event`(예: `repayment_day`, `month_end_batch`, `credit_limit_application_peak`) 라벨이 함께 저장되어 이상 구간의 원인을 해석할 수 있습니다.

```text
crop_activity_score
+ repayment_day / month_end
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
