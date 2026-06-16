# Service-Level Predictive Autoscaling Experiments

단일 트래픽 기준 모델 비교에서 최종 선정된 GRU를 실제 autoscaling 시나리오로 확장하는 후속 실험 모음이다.

메인 README는 모델 선택 결과를 중심으로 유지하고, 이 디렉터리는 서비스별 autoscaling 후속 실험을 분리해 관리한다.

## 실험 구분

| Experiment | 목적 | 대상 workload | Scaling 방향 |
| --- | --- | --- | --- |
| [On-prem BNPL](onprem_bnpl/README.md) | 고정된 온프레미스 pod 8개 안에서 업무 서비스 pod 비율 조정 | `auth`, `payment`, `batch`, `admin`, `limit_scoring` | priority 기반 pod budget 재배분 |

## 설계 원칙

이 실험은 메인 README의 더미 데이터 설계 흐름을 출발점으로 둔다.

```text
작물별 월별 activity
+ 요일/시간대 패턴
+ 장마/태풍
+ 이상치 이벤트
-> 예측 target 생성
-> GRU 예측
-> pod scaling decision
```

- On-prem BNPL은 농업 BNPL 업무 서비스의 `request_rate`를 예측한다.
- On-prem BNPL은 KEDA가 서비스별 Deployment pod 비율을 조정하도록 `adjusted_pods`를 만든다.

## 실행 요약

On-prem BNPL:

```powershell
.\.venv\Scripts\python.exe -m src.data.generate_service_dummy_data
.\.venv\Scripts\python.exe -m src.models.gru.train_service --onprem-pod-budget 8
.\.venv\Scripts\python.exe -m src.evaluation.plot_service_autoscaling
```

## 참고 문서

- [On-prem BNPL autoscaling design](../../docs/onprem-bnpl-autoscaling-design.md)

## aiops-platform 연동

최종 운영 연동은 `mcp-aiops-backend` 기획서의 Prediction Metric / KEDA ERD를 따른다.

```text
model_versions
prediction_runs
prediction_metrics
actual_metrics
prediction_error_metrics
scaling_events
```

이 레포지토리는 학습/실험과 scaling decision 산출을 담당하고, backend는 `prediction-runner`, `prediction-metric-exporter`, `scaling-event-collector`, `prediction-scaling-mcp`를 통해 DB 저장, `/metrics` 노출, KEDA/HPA event 수집, 예측 오차 분석을 담당한다.
