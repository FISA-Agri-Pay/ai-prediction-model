"""Service-specific pod sizing and scaling decision helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ServicePodPolicy:
    capacity_per_pod: float
    safety_margin: float
    min_pods: int
    max_pods: int
    priority: int
    node_pool: str
    node_pod_capacity: int

    @property
    def effective_capacity(self) -> float:
        if self.capacity_per_pod <= 0:
            raise ValueError("capacity_per_pod must be greater than 0")
        if not 0 <= self.safety_margin < 1:
            raise ValueError("safety_margin must be in the range [0, 1)")
        if self.min_pods < 0:
            raise ValueError("min_pods must be greater than or equal to 0")
        if self.max_pods < self.min_pods:
            raise ValueError("max_pods must be greater than or equal to min_pods")
        if self.priority <= 0:
            raise ValueError("priority must be greater than 0")
        if self.node_pod_capacity <= 0:
            raise ValueError("node_pod_capacity must be greater than 0")
        return self.capacity_per_pod * (1 - self.safety_margin)


DEFAULT_SERVICE_POLICIES: dict[str, ServicePodPolicy] = {
    "payment": ServicePodPolicy(25.0, 0.25, 3, 12, 1, "onprem-critical", 18),
    "auth": ServicePodPolicy(35.0, 0.20, 2, 8, 2, "onprem-critical", 18),
    "limit_scoring": ServicePodPolicy(15.0, 0.25, 2, 9, 2, "onprem-compute", 14),
    "batch": ServicePodPolicy(18.0, 0.15, 1, 7, 4, "onprem-batch", 16),
    "admin": ServicePodPolicy(40.0, 0.10, 1, 3, 5, "onprem-general", 20),
}


def policies_to_json_dict(
    policies: dict[str, ServicePodPolicy] = DEFAULT_SERVICE_POLICIES,
) -> dict[str, dict[str, float | int | str]]:
    return {service: asdict(policy) for service, policy in policies.items()}


def load_service_policies(path: Path | None = None) -> dict[str, ServicePodPolicy]:
    if path is None:
        return DEFAULT_SERVICE_POLICIES.copy()
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    return {service: ServicePodPolicy(**values) for service, values in raw.items()}


def required_service_pods(
    service: str,
    request_rate: float | Iterable[float] | np.ndarray,
    policies: dict[str, ServicePodPolicy] = DEFAULT_SERVICE_POLICIES,
) -> int | np.ndarray:
    if service not in policies:
        raise KeyError(f"missing service pod policy for {service}")
    policy = policies[service]
    values = np.asarray(request_rate, dtype=float)
    pods = np.ceil(np.maximum(values, 0) / policy.effective_capacity).astype(int)
    pods = np.clip(pods, policy.min_pods, policy.max_pods)
    if values.ndim == 0:
        return int(pods)
    return pods


def add_service_pod_columns(
    predictions: pd.DataFrame,
    policies: dict[str, ServicePodPolicy] = DEFAULT_SERVICE_POLICIES,
) -> pd.DataFrame:
    required = {"service", "actual", "predicted"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"prediction frame is missing required columns: {missing}")

    result = predictions.copy()
    result["actual_pods"] = [
        required_service_pods(row.service, row.actual, policies) for row in result.itertuples(index=False)
    ]
    result["predicted_pods"] = [
        required_service_pods(row.service, row.predicted, policies) for row in result.itertuples(index=False)
    ]
    return result


def _allocate_timestamp(
    rows: pd.DataFrame,
    pod_budget: int,
    policies: dict[str, ServicePodPolicy],
) -> tuple[dict[str, int], str]:
    desired = {row.service: int(row.predicted_pods) for row in rows.itertuples(index=False)}
    min_pods = {service: policies[service].min_pods for service in desired}
    min_total = sum(min_pods.values())

    if sum(desired.values()) <= pod_budget:
        return desired, "within_budget"

    if pod_budget < min_total:
        allocated = {service: 0 for service in desired}
        ordered = sorted(desired, key=lambda service: policies[service].priority)
        remaining = pod_budget
        while remaining > 0:
            progressed = False
            for service in ordered:
                if allocated[service] < min_pods[service] and remaining > 0:
                    allocated[service] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break
        return allocated, "budget_below_minimum"

    allocated = min_pods.copy()
    remaining = pod_budget - min_total
    extra_demand = {service: max(desired[service] - allocated[service], 0) for service in desired}
    weighted = {service: extra_demand[service] / policies[service].priority for service in desired}
    weighted_total = sum(weighted.values())
    if weighted_total == 0:
        return allocated, "priority_rebalanced"

    raw_shares = {service: remaining * weighted[service] / weighted_total for service in desired}
    for service, raw_share in raw_shares.items():
        grant = min(int(np.floor(raw_share)), extra_demand[service])
        allocated[service] += grant
        remaining -= grant

    order = sorted(
        desired,
        key=lambda service: (
            raw_shares[service] - np.floor(raw_shares[service]),
            extra_demand[service],
            1 / policies[service].priority,
        ),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for service in order:
            if allocated[service] < desired[service] and remaining > 0:
                allocated[service] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    return allocated, "priority_rebalanced"


def build_onprem_scaling_decisions(
    predictions: pd.DataFrame,
    pod_budget: int,
    policies: dict[str, ServicePodPolicy] = DEFAULT_SERVICE_POLICIES,
) -> pd.DataFrame:
    if pod_budget <= 0:
        raise ValueError("pod_budget must be greater than 0")
    required = {"ds", "service", "predicted_pods"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"prediction frame is missing required columns: {missing}")

    records: list[dict[str, object]] = []
    for ds, rows in predictions.groupby("ds", sort=True):
        allocation, reason = _allocate_timestamp(rows, pod_budget, policies)
        for row in rows.itertuples(index=False):
            policy = policies[row.service]
            records.append(
                {
                    "ds": ds,
                    "service": row.service,
                    "predicted_pods": int(row.predicted_pods),
                    "onprem_adjusted_pods": allocation[row.service],
                    "priority": policy.priority,
                    "pod_budget": pod_budget,
                    "reason": reason,
                }
            )
    return pd.DataFrame.from_records(records)


def build_aws_scaling_decisions(
    predictions: pd.DataFrame,
    policies: dict[str, ServicePodPolicy] = DEFAULT_SERVICE_POLICIES,
) -> pd.DataFrame:
    required = {"ds", "service", "predicted_pods"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"prediction frame is missing required columns: {missing}")

    records: list[dict[str, object]] = []
    for row in predictions.itertuples(index=False):
        policy = policies[row.service]
        predicted_pods = int(row.predicted_pods)
        records.append(
            {
                "ds": row.ds,
                "service": row.service,
                "predicted_pods": predicted_pods,
                "node_pool": policy.node_pool,
                "node_pod_capacity": policy.node_pod_capacity,
            }
        )
    result = pd.DataFrame.from_records(records)
    pool_totals = (
        result.groupby(["ds", "node_pool", "node_pod_capacity"], as_index=False)["predicted_pods"]
        .sum()
        .rename(columns={"predicted_pods": "node_pool_predicted_pods"})
    )
    pool_totals["required_nodes"] = np.ceil(
        pool_totals["node_pool_predicted_pods"] / pool_totals["node_pod_capacity"]
    ).astype(int)
    pool_totals["required_nodes"] = pool_totals["required_nodes"].clip(lower=1)
    result = result.merge(pool_totals, on=["ds", "node_pool", "node_pod_capacity"], how="left")
    result["reason"] = np.where(
        result["required_nodes"] > 1,
        "node_pool_scale_out",
        "baseline_capacity",
    )
    return result
