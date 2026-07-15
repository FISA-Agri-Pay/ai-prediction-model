"""Train and evaluate the LSTM traffic forecasting model."""

from __future__ import annotations

from src.models.sequence_model import SequenceRegressor, main


if __name__ == "__main__":
    main(model_name="lstm", model_cls=SequenceRegressor.lstm)

