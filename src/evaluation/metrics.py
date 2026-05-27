"""Common metrics for traffic forecasting and autoscaling evaluation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from src.evaluation.pod_policy import DEFAULT_POD_POLICY, PodPolicy, required_pods


def _as_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def smape(actual: Iterable[float] | np.ndarray, predicted: Iterable[float] | np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error as a ratio.

    A lower value is better. Positions where both actual and predicted are zero
    contribute 0 instead of NaN.
    """
    actual_arr = _as_array(actual)
    predicted_arr = _as_array(predicted)
    denominator = (np.abs(actual_arr) + np.abs(predicted_arr)) / 2
    diff = np.abs(actual_arr - predicted_arr)
    values = np.divide(diff, denominator, out=np.zeros_like(diff), where=denominator != 0)
    return float(np.mean(values))


def pod_accuracy(
    actual_pods: Iterable[int] | np.ndarray,
    predicted_pods: Iterable[int] | np.ndarray,
) -> float:
    """Return the share of timestamps where predicted pod count matches actual."""
    return float(np.mean(np.asarray(actual_pods) == np.asarray(predicted_pods)))


def under_provisioning_rate(
    actual_pods: Iterable[int] | np.ndarray,
    predicted_pods: Iterable[int] | np.ndarray,
) -> float:
    """Return the share of timestamps where predicted pods are below actual pods."""
    return float(np.mean(np.asarray(predicted_pods) < np.asarray(actual_pods)))


def over_provisioning_rate(
    actual_pods: Iterable[int] | np.ndarray,
    predicted_pods: Iterable[int] | np.ndarray,
) -> float:
    """Return the share of timestamps where predicted pods exceed actual pods."""
    return float(np.mean(np.asarray(predicted_pods) > np.asarray(actual_pods)))


def evaluate_predictions(
    actual: Iterable[float] | np.ndarray,
    predicted: Iterable[float] | np.ndarray,
    policy: PodPolicy = DEFAULT_POD_POLICY,
) -> dict[str, float]:
    """Evaluate forecast values with traffic and pod-level metrics."""
    actual_arr = _as_array(actual)
    predicted_arr = _as_array(predicted)
    actual_pods = required_pods(actual_arr, policy)
    predicted_pods = required_pods(predicted_arr, policy)

    return {
        "smape": smape(actual_arr, predicted_arr),
        "pod_accuracy": pod_accuracy(actual_pods, predicted_pods),
        "under_provisioning_rate": under_provisioning_rate(actual_pods, predicted_pods),
        "over_provisioning_rate": over_provisioning_rate(actual_pods, predicted_pods),
    }

