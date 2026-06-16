"""Generate service-level synthetic traffic for autoscaling follow-up experiments.

The base agricultural seasonality, monsoon, typhoon, and anomaly windows are
shared with ``generate_dummy_data.py``. This module expands the single traffic
series into five service-level series.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.generate_dummy_data import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    apply_seasonal_patterns,
    apply_weather_patterns,
    generate_base_pattern,
    inject_anomalies,
)


SERVICES = ("auth", "payment", "batch", "admin", "limit_scoring")


@dataclass(frozen=True)
class ServiceGeneratedPaths:
    service_traffic: Path
    service_events: Path


def _service_hour_weight(service: str, hours: pd.Series) -> np.ndarray:
    hour_values = hours.to_numpy()
    if service == "auth":
        return np.select(
            [
                (7 <= hour_values) & (hour_values < 10),
                (18 <= hour_values) & (hour_values < 22),
                (10 <= hour_values) & (hour_values < 18),
            ],
            [1.35, 1.25, 0.75],
            default=0.25,
        )
    if service == "payment":
        return np.select(
            [
                (9 <= hour_values) & (hour_values < 12),
                (13 <= hour_values) & (hour_values < 17),
                (18 <= hour_values) & (hour_values < 21),
            ],
            [1.25, 1.35, 0.85],
            default=0.20,
        )
    if service == "batch":
        return np.select(
            [(1 <= hour_values) & (hour_values < 5), (22 <= hour_values) | (hour_values < 1)],
            [1.70, 0.95],
            default=0.18,
        )
    if service == "admin":
        return np.select([(9 <= hour_values) & (hour_values < 18)], [1.10], default=0.08)
    if service == "limit_scoring":
        return np.select(
            [
                (9 <= hour_values) & (hour_values < 12),
                (14 <= hour_values) & (hour_values < 18),
                (19 <= hour_values) & (hour_values < 21),
            ],
            [1.25, 1.15, 0.65],
            default=0.15,
        )
    raise ValueError(f"unknown service: {service}")


def _repayment_boost(ds: pd.Series) -> np.ndarray:
    days = ds.dt.day
    return np.where(days >= 25, 1.35, np.where(days <= 3, 1.15, 1.0))


def _month_end_boost(ds: pd.Series) -> np.ndarray:
    days = ds.dt.day
    months = ds.dt.month
    month_end = days >= 26
    quarter_end = month_end & months.isin([3, 6, 9, 12])
    return np.where(quarter_end, 1.60, np.where(month_end, 1.30, 1.0))


def _business_event_labels(df: pd.DataFrame, service: str) -> list[str]:
    labels = np.full(len(df), "normal", dtype=object)
    ds = df["ds"]
    days = ds.dt.day
    months = ds.dt.month
    hours = ds.dt.hour

    spring_spike = (months == 3) & (days.between(10, 20))
    typhoon_recovery = (months == 9) & (days.between(2, 4))
    repayment = days >= 25
    month_end = days >= 26
    business_hours = (9 <= hours) & (hours < 18)
    purchase_season = months.isin([3, 4, 5, 9, 10, 11])

    if service == "auth":
        labels[spring_spike | purchase_season] = "signup_or_login_peak"
    elif service == "payment":
        labels[repayment] = "repayment_day"
        labels[typhoon_recovery] = "post_typhoon_payment_recovery"
        labels[spring_spike] = "spring_purchase_payment_spike"
    elif service == "batch":
        labels[month_end] = "month_end_batch"
        labels[(1 <= hours) & (hours < 5)] = "nightly_batch"
    elif service == "admin":
        labels[business_hours] = "business_hours"
    elif service == "limit_scoring":
        labels[purchase_season] = "credit_limit_application_peak"
        labels[spring_spike] = "spring_credit_limit_spike"
    else:
        raise ValueError(f"unknown service: {service}")
    return labels.tolist()


def _build_service_target(patterned: pd.DataFrame, service: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    result = patterned[["ds", "is_monsoon", "typhoon_index", "crop_activity_score"]].copy()
    ds = result["ds"]

    service_base = {
        "auth": 75.0,
        "payment": 95.0,
        "batch": 55.0,
        "admin": 22.0,
        "limit_scoring": 80.0,
    }[service]
    seasonal_strength = {
        "auth": 0.35,
        "payment": 0.95,
        "batch": 0.55,
        "admin": 0.15,
        "limit_scoring": 0.80,
    }[service]

    crop_score = result["crop_activity_score"].to_numpy()
    seasonal = 1.0 + seasonal_strength * crop_score
    weekly = patterned["_weekly"].to_numpy()
    hourly = _service_hour_weight(service, ds.dt.hour)
    anomaly = patterned["_anomaly_boost"].to_numpy()
    monsoon = np.where(result["is_monsoon"].to_numpy() == 1, 1.08, 1.0)
    typhoon = np.where(result["typhoon_index"].to_numpy() >= 0.8, 0.85, 1.0)

    repayment = _repayment_boost(ds) if service == "payment" else 1.0
    batch_end = _month_end_boost(ds) if service == "batch" else 1.0
    purchase_season = np.where(ds.dt.month.isin([3, 4, 5, 9, 10, 11]), 1.25, 1.0)
    limit_purchase = purchase_season if service == "limit_scoring" else 1.0
    auth_signup = np.where(ds.dt.month.isin([3, 4, 5, 9, 10]), 1.12, 1.0)
    auth_signup = auth_signup if service == "auth" else 1.0

    noise = rng.normal(0, service_base * 0.035, len(result))
    request_rate = (
        service_base
        * seasonal
        * weekly
        * hourly
        * anomaly
        * monsoon
        * typhoon
        * repayment
        * batch_end
        * limit_purchase
        * auth_signup
        + noise
    )
    result["service"] = service
    result["request_rate"] = np.clip(request_rate, 0, None)

    queue_factor = {
        "auth": 0.01,
        "payment": 0.04,
        "batch": 1.20,
        "admin": 0.00,
        "limit_scoring": 0.22,
    }[service]
    cpu_factor = {
        "auth": 0.42,
        "payment": 0.50,
        "batch": 0.72,
        "admin": 0.30,
        "limit_scoring": 0.68,
    }[service]
    db_factor = {
        "auth": 0.07,
        "payment": 0.34,
        "batch": 0.52,
        "admin": 0.05,
        "limit_scoring": 0.38,
    }[service]
    latency_factor = {
        "auth": 0.75,
        "payment": 1.20,
        "batch": 2.10,
        "admin": 0.65,
        "limit_scoring": 1.55,
    }[service]

    queue_noise = rng.normal(0, 2.0, len(result))
    result["queue_depth"] = np.clip(result["request_rate"] * queue_factor + queue_noise, 0, None)
    result["cpu_utilization"] = np.clip(
        12 + result["request_rate"] * cpu_factor + result["queue_depth"] * 0.03,
        0,
        100,
    )
    result["p95_latency"] = np.clip(
        60 + result["request_rate"] * latency_factor + result["queue_depth"] * 0.35,
        20,
        None,
    )
    result["error_rate"] = np.clip(0.001 + (result["cpu_utilization"] / 100) ** 3 * 0.025, 0, 0.10)
    result["db_connection_usage"] = np.clip(result["request_rate"] * db_factor, 0, 100)
    result["business_event"] = _business_event_labels(result, service)
    result["hour"] = result["ds"].dt.hour
    result["day_of_week"] = result["ds"].dt.dayofweek
    result["month"] = result["ds"].dt.month
    result["y"] = result["request_rate"]

    return result[
        [
            "ds",
            "service",
            "is_monsoon",
            "typhoon_index",
            "crop_activity_score",
            "request_rate",
            "cpu_utilization",
            "p95_latency",
            "queue_depth",
            "error_rate",
            "db_connection_usage",
            "business_event",
            "hour",
            "day_of_week",
            "month",
            "y",
        ]
    ]


def build_service_datasets(start_date: str, end_date: str, freq: str = "1h") -> tuple[pd.DataFrame, pd.DataFrame]:
    base = generate_base_pattern(start_date, end_date, freq=freq)
    patterned = apply_seasonal_patterns(base)
    patterned = apply_weather_patterns(patterned)
    patterned, anomaly_events = inject_anomalies(patterned)

    service_frames = [
        _build_service_target(patterned, service, seed=2024 + index)
        for index, service in enumerate(SERVICES)
    ]
    service_traffic = pd.concat(service_frames, ignore_index=True)
    service_traffic = service_traffic.sort_values(["ds", "service"]).reset_index(drop=True)

    service_events = anomaly_events.copy()
    service_events["applies_to_services"] = "auth,payment,batch,admin,limit_scoring"
    return service_traffic, service_events


def save_service_datasets(
    service_traffic: pd.DataFrame,
    service_events: pd.DataFrame,
    processed_dir: Path = PROCESSED_DATA_DIR,
    raw_dir: Path = RAW_DATA_DIR,
) -> ServiceGeneratedPaths:
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths = ServiceGeneratedPaths(
        service_traffic=processed_dir / "onprem_bnpl_service_traffic.csv",
        service_events=raw_dir / "dummy_service_events.csv",
    )
    service_traffic.to_csv(paths.service_traffic, index=False)
    service_events.to_csv(paths.service_events, index=False)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate service-level synthetic traffic data.")
    parser.add_argument("--start", default="2020-01-01", help="Start date, e.g. 2020-01-01")
    parser.add_argument("--end", default="2024-12-31 23:00", help="End date, e.g. 2024-12-31 23:00")
    parser.add_argument("--freq", default="1h", help="Pandas frequency string. Default: 1h")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service_traffic, service_events = build_service_datasets(args.start, args.end, args.freq)
    paths = save_service_datasets(service_traffic, service_events)

    print("Service-level synthetic traffic data generated.")
    print(f"- service_traffic: {paths.service_traffic} ({len(service_traffic):,} rows)")
    print(f"- service_events: {paths.service_events} ({len(service_events):,} rows)")


if __name__ == "__main__":
    main()
