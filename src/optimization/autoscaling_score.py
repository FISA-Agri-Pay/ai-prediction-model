"""Scoring helpers for prediction post-processing optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.evaluation.metrics import evaluate_predictions
from src.evaluation.pod_policy import DEFAULT_POD_POLICY, PodPolicy, required_pods


@dataclass(frozen=True)
class ScoreWeights:
    """Penalty weights for autoscaling-oriented optimization."""

    smape: float = 0.1
    over_provisioning_rate: float = 0.2
    severe_under_provisioning_rate: float = 0.1
    pod_change_rate: float = 0.05


DEFAULT_SCORE_WEIGHTS = ScoreWeights()


def pod_change_rate(predicted_pods: Iterable[int] | np.ndarray) -> float:
    """Return how often the predicted pod count changes between timestamps."""
    pods = np.asarray(predicted_pods, dtype=int)
    if pods.size <= 1:
        return 0.0
    return float(np.mean(pods[1:] != pods[:-1]))


def severe_under_provisioning_rate(
    actual_pods: Iterable[int] | np.ndarray,
    predicted_pods: Iterable[int] | np.ndarray,
) -> float:
    """Return the share of timestamps under-provisioned by more than one pod."""
    actual_arr = np.asarray(actual_pods, dtype=int)
    predicted_arr = np.asarray(predicted_pods, dtype=int)
    if actual_arr.shape != predicted_arr.shape:
        raise ValueError("actual_pods and predicted_pods must have the same shape")
    if actual_arr.size == 0:
        raise ValueError("pod arrays must not be empty")
    return float(np.mean((actual_arr - predicted_arr) > 1))


def autoscaling_penalty_score(
    metrics: dict[str, float],
    weights: ScoreWeights = DEFAULT_SCORE_WEIGHTS,
) -> float:
    """Combine metrics into a lower-is-better penalty score."""
    return float(
        metrics["under_provisioning_rate"]
        + weights.smape * metrics["smape"]
        + weights.over_provisioning_rate * metrics["over_provisioning_rate"]
        + weights.severe_under_provisioning_rate * metrics["severe_under_provisioning_rate"]
        + weights.pod_change_rate * metrics["pod_change_rate"]
    )


def fitness_from_penalty(score: float) -> float:
    """Convert a lower-is-better penalty into a higher-is-better fitness."""
    if score < 0:
        raise ValueError("score must be non-negative")
    return float(1.0 / (1.0 + score))


def evaluate_autoscaling_predictions(
    actual: Iterable[float] | np.ndarray,
    predicted: Iterable[float] | np.ndarray,
    policy: PodPolicy = DEFAULT_POD_POLICY,
    weights: ScoreWeights = DEFAULT_SCORE_WEIGHTS,
) -> dict[str, float]:
    """Evaluate adjusted traffic predictions with autoscaling-specific metrics."""
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.maximum(np.asarray(predicted, dtype=float), 0)
    if actual_arr.shape != predicted_arr.shape:
        raise ValueError("actual and predicted must have the same shape")
    if actual_arr.size == 0:
        raise ValueError("actual and predicted must not be empty")

    metrics = evaluate_predictions(actual_arr, predicted_arr, policy)
    actual_pods = required_pods(actual_arr, policy)
    predicted_pods = required_pods(predicted_arr, policy)
    metrics["severe_under_provisioning_rate"] = severe_under_provisioning_rate(
        actual_pods,
        predicted_pods,
    )
    metrics["pod_change_rate"] = pod_change_rate(predicted_pods)
    metrics["penalty_score"] = autoscaling_penalty_score(metrics, weights)
    metrics["combined_score"] = fitness_from_penalty(metrics["penalty_score"])
    return metrics

