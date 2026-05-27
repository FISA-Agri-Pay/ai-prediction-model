# 최종 모델 선정

## 현재 상태

아직 실제 모델별 metric 값이 확정되지 않았다. 아래 표는 실험 실행 후 채워야 하는 템플릿이다.

## 모델 비교 결과

| 모델 | SMAPE | Pod accuracy | Under-provisioning rate | Over-provisioning rate | 비고 |
| --- | ---: | ---: | ---: | ---: | --- |
| Prophet | TBD | TBD | TBD | TBD | baseline |
| SARIMA | TBD | TBD | TBD | TBD | statistical baseline |
| GRU | TBD | TBD | TBD | TBD | 최종 후보 |
| LSTM | TBD | TBD | TBD | TBD | deep learning baseline |

## 선정 기준

최종 모델은 다음 순서로 선정한다.

1. Under-provisioning rate
2. SMAPE
3. Pod accuracy
4. Over-provisioning rate

## 최종 선정 모델

```text
TBD: GRU 후보
```

## GRU 선택 방향의 근거

실제 metric 값이 확인되기 전까지 GRU를 최종 후보로 둔다. GRU는 LSTM보다 구조가 단순해 학습과 추론 비용이 낮고, 시계열 sequence 패턴을 직접 학습할 수 있어 Prophet/SARIMA보다 비선형 트래픽 변화에 대응할 가능성이 있다.

다만 최종 선정은 반드시 `experiments/results/comparison_results.csv`와 `experiments/results/best_model.json`의 실제 결과를 기준으로 확정한다.

## 확정 후 작성할 내용

- 최종 선택 모델
- 선택 metric 값
- 경쟁 모델 대비 장점
- under-provisioning 감소 여부
- 운영 적용 시 주의사항
