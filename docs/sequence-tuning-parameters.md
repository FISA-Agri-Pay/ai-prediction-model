# GRU/LSTM Optuna Tuning Parameters

이 문서는 GRU/LSTM sequence model에 대해 Optuna가 탐색한 하이퍼파라미터와 각 파라미터의 의미를 정리한다.

튜닝 코드는 [src/models/sequence_tune.py](../src/models/sequence_tune.py)에 있으며, 1차 모델 비교에서 선정된 GRU와 LSTM만 대상으로 한다. Prophet 튜닝 코드는 별도로 존재하지만, 이번 최종 모델 선정 흐름에서는 Prophet/SARIMA를 baseline으로만 사용했다.

## 튜닝 목적

튜닝의 목적은 단순히 트래픽 예측 오차를 낮추는 것이 아니라, Kubernetes autoscaling에서 pod 부족 위험을 줄이는 것이다.

Optuna objective score는 다음과 같이 계산한다.

```text
score = under_provisioning_rate + 0.1 * smape + 0.2 * over_provisioning_rate
```

각 trial은 이 score를 최소화하는 방향으로 탐색한다.

| 항목 | 의미 |
| --- | --- |
| `under_provisioning_rate` | 실제 필요한 pod 수보다 적게 예측한 비율 |
| `smape` | 트래픽 예측값과 실제값의 상대 오차 |
| `over_provisioning_rate` | 실제보다 많은 pod를 예측한 비율 |

`under_provisioning_rate`를 score의 중심에 둔 이유는 pod 부족이 서비스 지연이나 장애로 이어질 수 있기 때문이다. SMAPE와 over-provisioning rate는 예측 품질과 비용 효율을 보조적으로 반영한다.

## 탐색 방식

Optuna는 training 구간의 일부를 validation 구간으로 분리해 trial별 성능을 비교한다.

```text
전체 데이터
  ├─ train_full
  │   ├─ train
  │   └─ validation
  └─ holdout
```

- `train`: trial별 모델 학습에 사용
- `validation`: Optuna objective score 계산에 사용
- `holdout`: 최종 tuned 모델 평가에 사용

기본 validation 비율은 `train_full`의 마지막 `20%`다. 최종 평가는 Optuna가 선택한 best params로 `train_full` 전체를 다시 학습한 뒤 holdout에서 수행한다.

## 튜닝 파라미터

| 구분 | 파라미터 | 탐색 범위 | 의미 |
| --- | --- | --- | --- |
| 입력 window | `sequence_length` | `12`, `24`, `48`, `72`, `168` | 과거 몇 시간의 데이터를 보고 다음 시점을 예측할지 결정한다. |
| 모델 크기 | `hidden_size` | `16`, `32`, `64`, `128` | GRU/LSTM hidden state 크기다. 값이 클수록 표현력은 커지지만 과적합 위험과 학습 비용도 증가한다. |
| 모델 깊이 | `num_layers` | `1` - `3` | recurrent layer를 몇 층 쌓을지 결정한다. 깊을수록 복잡한 패턴을 학습할 수 있지만 과적합 위험이 커진다. |
| 정규화 | `dropout` | `0.0` - `0.4` | multi-layer GRU/LSTM에서 일부 연결을 무작위로 비활성화해 과적합을 줄인다. `num_layers=1`이면 `0.0`으로 고정된다. |
| 학습률 | `learning_rate` | `1e-4` - `5e-2`, log scale | Adam optimizer의 step size다. 너무 크면 학습이 불안정하고, 너무 작으면 수렴이 느리다. |
| 학습 반복 | `epochs` | `10` - `60` | 전체 training data를 몇 번 반복해서 학습할지 결정한다. 많을수록 충분히 학습할 수 있지만 과적합 가능성도 커진다. |
| 배치 크기 | `batch_size` | `0`, `256`, `512`, `1024` | 한 번의 gradient update에 사용할 sample 수다. `0`은 full-batch 학습을 의미한다. |
| 가중치 정규화 | `weight_decay` | `1e-6` - `1e-2`, log scale | Adam optimizer의 L2 regularization 강도다. 모델 가중치가 과도하게 커지는 것을 억제한다. |
| 학습 안정화 | `gradient_clip` | `0.0`, `0.5`, `1.0`, `5.0` | gradient norm 상한을 둬 gradient 폭주를 완화한다. `0.0`은 clipping을 사용하지 않는다는 의미다. |

## 파라미터별 해석

### `sequence_length`

입력으로 사용할 과거 window 길이다. 예를 들어 `24`는 과거 24시간을 보고 다음 시점을 예측한다는 의미다.

- 짧은 window: 최근 변화에 빠르게 반응할 수 있지만, 주간/월간 패턴을 충분히 보지 못할 수 있다.
- 긴 window: 장기 패턴을 반영할 수 있지만, 학습 난이도와 계산 비용이 증가한다.

이번 탐색에서는 반나절, 하루, 이틀, 사흘, 일주일 단위의 후보를 비교했다.

### `hidden_size`

GRU/LSTM 내부 hidden state의 차원이다. 모델이 과거 패턴을 얼마나 풍부하게 표현할 수 있는지에 영향을 준다.

- 값이 작으면 모델이 단순하고 빠르지만 복잡한 패턴을 놓칠 수 있다.
- 값이 크면 표현력은 좋아지지만 과적합과 학습 비용이 증가할 수 있다.

### `num_layers`와 `dropout`

`num_layers`는 recurrent layer의 깊이다. `dropout`은 layer가 2개 이상일 때만 적용된다.

깊은 모델은 더 복잡한 패턴을 학습할 수 있지만, 데이터의 특정 validation 구간에 과하게 맞춰질 수 있다. 이 때문에 `dropout`을 함께 탐색해 과적합을 완화하도록 했다.

### `learning_rate`

학습률은 optimizer가 한 번에 얼마나 크게 parameter를 업데이트할지 결정한다.

- 너무 크면 loss가 불안정하게 흔들릴 수 있다.
- 너무 작으면 제한된 epoch 안에서 충분히 수렴하지 못할 수 있다.

탐색 범위는 log scale로 두어 작은 값과 큰 값을 모두 비교할 수 있게 했다.

### `epochs`

학습 반복 횟수다. epoch가 많아지면 training data에는 더 잘 맞을 수 있지만, validation 또는 holdout에 대한 일반화 성능은 오히려 나빠질 수 있다.

이번 실험에서 tuned 모델이 일부 보조 지표를 개선했음에도 holdout under-provisioning rate가 높아진 것은, 선택된 조합이 validation 구간에 과적합되었을 가능성을 보여준다.

### `batch_size`

mini-batch 학습의 batch 크기다.

- `0`: full-batch 학습
- `256`, `512`, `1024`: mini-batch 학습

mini-batch 학습은 gradient update가 더 자주 일어나고, full-batch보다 일반화에 유리할 수 있다. 다만 batch 크기에 따라 학습 안정성과 속도가 달라질 수 있다.

### `weight_decay`

가중치가 과도하게 커지는 것을 막는 L2 regularization이다. 과적합을 줄이는 데 도움이 될 수 있지만, 너무 강하면 모델이 충분히 학습하지 못할 수 있다.

### `gradient_clip`

GRU/LSTM은 recurrent 구조 때문에 gradient가 커지는 문제가 생길 수 있다. `gradient_clip`은 gradient norm에 상한을 두어 학습을 안정화한다.

값이 `0.0`이면 clipping을 사용하지 않는다.

## 결과 해석 시 주의점

Optuna의 best params는 validation objective score 기준으로 가장 좋은 조합이다. 따라서 best params가 항상 최종 holdout에서도 가장 좋은 성능을 보장하지는 않는다.

이번 실험에서도 tuned GRU/LSTM은 SMAPE, pod accuracy, over-provisioning rate 일부를 개선했지만, 최종 기준인 under-provisioning rate에서는 기본 GRU보다 좋지 않았다. 따라서 최종 모델은 tuned 모델이 아니라 기본 GRU로 선정했다.
