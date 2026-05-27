# 문제 정의

## Predictive Autoscaling이 필요한 이유

Kubernetes HPA 기반 reactive scaling은 CPU, memory, custom metric 같은 현재 상태를 보고 pod 수를 조정한다. 이 방식은 구현이 단순하지만, 트래픽 증가가 이미 발생한 뒤에 반응한다는 한계가 있다.

Predictive autoscaling은 과거 트래픽 패턴을 학습해 가까운 미래의 트래픽을 예측하고, 트래픽이 실제로 증가하기 전에 pod를 미리 준비하는 방식이다. 예측이 충분히 정확하다면 급격한 부하 변화에도 서비스 지연을 줄일 수 있다.

## Reactive Scaling의 한계

Reactive scaling은 다음 구간에서 늦게 반응할 수 있다.

- 갑작스러운 이벤트성 트래픽 증가
- 특정 시간대 반복 피크
- 계절성에 따른 장기 트래픽 변화
- pod 생성 및 readiness까지 걸리는 지연 시간

트래픽 증가 후 metric 수집, HPA 판단, pod 생성, 애플리케이션 준비가 순차적으로 진행되기 때문에 실제 요청 증가 시점과 처리 용량 증가 시점 사이에 공백이 생긴다.

## 트래픽 예측 모델의 역할

트래픽 예측 모델은 미래 request rate를 예측하고, 이 값을 required pod 수로 변환하는 데 사용된다.

이 프로젝트의 모델은 다음 역할을 수행한다.

- 시간별 request rate 예측
- holdout 구간에서 실제 트래픽과 예측 트래픽 비교
- 예측 트래픽 기반 required pod 산정
- pod 부족과 pod 과잉 여부 평가

최종 목표는 단순히 예측 오차가 낮은 모델이 아니라, Kubernetes predictive autoscaling에 사용할 때 under-provisioning risk를 가장 잘 줄이는 모델을 선정하는 것이다.
