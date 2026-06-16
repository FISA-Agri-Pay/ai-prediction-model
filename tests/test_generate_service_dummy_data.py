import unittest

from src.data.generate_service_dummy_data import SERVICES, build_service_datasets
from src.models.common import FEATURE_COLUMNS, TARGET_COLUMN, TIMESTAMP_COLUMN


class GenerateServiceDummyDataTest(unittest.TestCase):
    def test_service_traffic_keeps_expected_schema_and_services(self):
        service_traffic, service_events = build_service_datasets(
            "2024-01-01",
            "2024-01-02 23:00",
        )

        self.assertEqual(len(service_traffic), 48 * len(SERVICES))
        self.assertEqual(sorted(service_traffic["service"].unique()), sorted(SERVICES))
        self.assertTrue(
            {
                TIMESTAMP_COLUMN,
                "service",
                TARGET_COLUMN,
                *FEATURE_COLUMNS,
                "request_rate",
                "cpu_utilization",
                "p95_latency",
                "queue_depth",
                "error_rate",
                "db_connection_usage",
                "business_event",
            }.issubset(service_traffic.columns)
        )
        self.assertFalse(service_events.empty)

    def test_payment_uses_repayment_business_event(self):
        service_traffic, _ = build_service_datasets(
            "2024-01-25",
            "2024-01-25 23:00",
        )

        payment = service_traffic[service_traffic["service"] == "payment"]

        self.assertIn("repayment_day", payment["business_event"].unique().tolist())


if __name__ == "__main__":
    unittest.main()
