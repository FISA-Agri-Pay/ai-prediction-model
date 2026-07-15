import unittest

from src.data.generate_dummy_data import build_datasets
from src.models.common import FEATURE_COLUMNS, TARGET_COLUMN, TIMESTAMP_COLUMN


class GenerateDummyDataTest(unittest.TestCase):
    def test_processed_traffic_keeps_canonical_model_schema(self):
        _, _, _, processed = build_datasets("2024-01-01", "2024-01-02 23:00")

        self.assertEqual(
            list(processed.columns),
            [
                TIMESTAMP_COLUMN,
                "is_monsoon",
                "typhoon_index",
                "request_rate",
                "cpu_utilization",
                "hour",
                "day_of_week",
                "month",
                TARGET_COLUMN,
            ],
        )
        self.assertTrue({TIMESTAMP_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS}.issubset(processed.columns))


if __name__ == "__main__":
    unittest.main()
