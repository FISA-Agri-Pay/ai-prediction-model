"""Shared GRU/LSTM training implementation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from src.models.common import (
    MODEL_FEATURE_COLUMNS,
    MODELS_DIR,
    add_common_args,
    add_cyclic_time_features,
    build_prediction_frame,
    load_traffic_data,
    save_model_outputs,
    split_train_holdout,
)


@dataclass
class ScalingParams:
    mean: np.ndarray
    std: np.ndarray


class SequenceRegressor:
    @staticmethod
    def gru(input_size: int, hidden_size: int):
        import torch

        return _TorchSequenceRegressor(torch.nn.GRU, input_size, hidden_size)

    @staticmethod
    def lstm(input_size: int, hidden_size: int):
        import torch

        return _TorchSequenceRegressor(torch.nn.LSTM, input_size, hidden_size)


_TorchModuleBase = torch.nn.Module if torch is not None else object


class _TorchSequenceRegressor(_TorchModuleBase):
    def __init__(self, recurrent_cls, input_size: int, hidden_size: int):
        import torch

        super().__init__()
        self.recurrent = recurrent_cls(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.output = torch.nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        output, _ = self.recurrent(inputs)
        return self.output(output[:, -1, :]).squeeze(-1)


def parse_args(model_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Train {model_name.upper()} model.")
    add_common_args(parser)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    args = parser.parse_args()

    if args.sequence_length <= 0:
        parser.error("--sequence-length must be a positive integer")
    if args.epochs <= 0:
        parser.error("--epochs must be a positive integer")
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")
    if args.hidden_size <= 0:
        parser.error("--hidden-size must be a positive integer")
    if not 0 < args.learning_rate < 1:
        parser.error("--learning-rate must be greater than 0 and less than 1")

    return args


def scale_values(values: np.ndarray, params: ScalingParams | None = None) -> tuple[np.ndarray, ScalingParams]:
    if params is None:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std == 0, 1, std)
        params = ScalingParams(mean=mean, std=std)
    return (values - params.mean) / params.std, params


def make_sequences(values: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= sequence_length:
        raise ValueError("not enough rows to build sequence dataset")

    inputs = []
    targets = []
    for index in range(sequence_length, len(values)):
        inputs.append(values[index - sequence_length : index])
        targets.append(values[index, 0])
    return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def recursive_holdout_forecast(
    model,
    scaled_all: np.ndarray,
    holdout_start: int,
    sequence_length: int,
) -> np.ndarray:
    """Forecast holdout recursively without using true holdout target history."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than 0")
    if len(scaled_all) < sequence_length:
        raise ValueError(
            f"scaled_all must have at least sequence_length rows "
            f"(got {len(scaled_all)} rows and sequence_length={sequence_length})"
        )
    if not isinstance(holdout_start, int):
        raise ValueError("holdout_start must be an integer")
    if holdout_start < sequence_length or holdout_start >= len(scaled_all):
        raise ValueError(
            f"holdout_start must be >= sequence_length and < len(scaled_all) "
            f"(got holdout_start={holdout_start}, sequence_length={sequence_length}, "
            f"len(scaled_all)={len(scaled_all)})"
        )

    import torch

    forecast_values = scaled_all.copy()
    predictions = []
    if hasattr(model, "eval"):
        model.eval()

    for index in range(holdout_start, len(forecast_values)):
        window = forecast_values[index - sequence_length : index]
        with torch.no_grad():
            prediction = model(torch.tensor(window[np.newaxis, :, :], dtype=torch.float32)).item()
        forecast_values[index, 0] = prediction
        predictions.append(prediction)

    return np.asarray(predictions, dtype=np.float32)


def main(model_name: str, model_cls: Callable[[int, int], _TorchSequenceRegressor]) -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    args = parse_args(model_name)
    torch.manual_seed(42)
    np.random.seed(42)

    df = add_cyclic_time_features(load_traffic_data(args.data_path))
    train, holdout = split_train_holdout(df, args.holdout_ratio)
    columns = ["y", *MODEL_FEATURE_COLUMNS]

    train_values = train[columns].to_numpy(dtype=float)
    all_values = df[columns].to_numpy(dtype=float)
    scaled_train, scaling = scale_values(train_values)
    scaled_all, _ = scale_values(all_values, scaling)

    train_x, train_y = make_sequences(scaled_train, args.sequence_length)
    holdout_start = len(train)

    model = model_cls(input_size=len(columns), hidden_size=args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.MSELoss()

    train_x_tensor = torch.tensor(train_x)
    train_y_tensor = torch.tensor(train_y)
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    epoch_losses = []
    for _ in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
        epoch_losses.append(total_loss / len(train_dataset))

    scaled_predictions = recursive_holdout_forecast(
        model,
        scaled_all,
        holdout_start,
        args.sequence_length,
    )

    target_mean = scaling.mean[0]
    target_std = scaling.std[0]
    predictions_values = np.maximum((scaled_predictions * target_std) + target_mean, 0)

    predictions = build_prediction_frame(holdout, predictions_values, model_name)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / f"{model_name}.pt")

    metrics = save_model_outputs(
        model_name,
        predictions,
        {
            "train_rows": len(train),
            "holdout_rows": len(holdout),
            "features": columns,
            "sequence_length": args.sequence_length,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "final_train_loss": epoch_losses[-1],
            "model_path": str((MODELS_DIR / f"{model_name}.pt").relative_to(MODELS_DIR.parent)),
        },
    )
    print(metrics)
