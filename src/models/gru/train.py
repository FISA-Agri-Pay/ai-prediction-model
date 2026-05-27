"""Train and evaluate the GRU traffic forecasting model."""

from __future__ import annotations

from src.models.sequence_model import SequenceRegressor, main


if __name__ == "__main__":
    main(model_name="gru", model_cls=SequenceRegressor.gru)

