# Prompt Log

이 문서는 프로젝트 진행 중 사용한 주요 프롬프트와 의사결정 흐름을 기록한다.

## 프로젝트 구조 설계

주요 요청:

```text
기존 prophet-autoscaler 프로젝트를 바탕으로 ai-prediction-model 레포를 AI 예측 모델 선정 및 비교 프로젝트로 정리한다.
Prophet, SARIMA, GRU, LSTM을 동일 데이터와 동일 holdout 조건에서 비교할 수 있는 구조로 재구성한다.
```

결과:

- `src/data`, `src/models`, `src/evaluation` 구조 정의
- `data/raw`, `data/processed`, `data/predictions` 분리
- `experiments/results`, `experiments/plots` 결과 저장 위치 정의

## 모델 비교 기준 수립

주요 요청:

```text
Kubernetes predictive autoscaling 프로젝트이므로 SMAPE뿐 아니라 Pod accuracy, Under-provisioning rate, Over-provisioning rate를 평가 지표에 포함한다.
Under-provisioning rate를 가장 중요한 지표로 설명한다.
```

결과:

- `src/evaluation/metrics.py` 작성
- `src/evaluation/pod_policy.py` 작성
- 모델 ranking 기준 정의

## 최종 모델 선정 근거 정리

주요 요청:

```text
GRU를 최종 모델로 선택하는 방향의 근거를 작성하되, 실제 metric 값이 없으면 placeholder로 남긴다.
```

결과:

- `docs/final-decision.md`에 metric placeholder 표 작성
- GRU를 최종 후보로 두는 근거 정리
- 실제 metric 기반 확정 필요성을 명시
