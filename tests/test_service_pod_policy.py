import unittest

import pandas as pd

from src.evaluation.service_pod_policy import (
    DEFAULT_SERVICE_POLICIES,
    add_service_pod_columns,
    build_aws_scaling_decisions,
    build_onprem_scaling_decisions,
    required_service_pods,
)


class ServicePodPolicyTest(unittest.TestCase):
    def test_required_service_pods_uses_service_specific_policy(self):
        payment_pods = required_service_pods("payment", 200)
        admin_pods = required_service_pods("admin", 200)

        self.assertGreater(payment_pods, admin_pods)

    def test_add_service_pod_columns_adds_actual_and_predicted_pods(self):
        predictions = pd.DataFrame(
            {
                "ds": pd.date_range("2024-01-01", periods=2, freq="h"),
                "service": ["payment", "admin"],
                "actual": [100.0, 20.0],
                "predicted": [120.0, 30.0],
            }
        )

        result = add_service_pod_columns(predictions)

        self.assertIn("actual_pods", result.columns)
        self.assertIn("predicted_pods", result.columns)

    def test_onprem_scaling_decisions_respect_pod_budget(self):
        rows = []
        for service, policy in DEFAULT_SERVICE_POLICIES.items():
            rows.append(
                {
                    "ds": pd.Timestamp("2024-01-01 00:00:00"),
                    "service": service,
                    "predicted_pods": policy.max_pods,
                }
            )
        predictions = pd.DataFrame(rows)

        result = build_onprem_scaling_decisions(predictions, pod_budget=12)

        self.assertEqual(result["onprem_adjusted_pods"].sum(), 12)
        payment = result[result["service"] == "payment"]["onprem_adjusted_pods"].iloc[0]
        admin = result[result["service"] == "admin"]["onprem_adjusted_pods"].iloc[0]
        self.assertGreaterEqual(payment, admin)

    def test_aws_scaling_decisions_include_node_pool_and_required_nodes(self):
        predictions = pd.DataFrame(
            {
                "ds": [pd.Timestamp("2024-01-01 00:00:00")],
                "service": ["payment"],
                "predicted_pods": [12],
            }
        )

        result = build_aws_scaling_decisions(predictions)

        self.assertEqual(result.loc[0, "node_pool"], "on-demand-critical")
        self.assertGreaterEqual(result.loc[0, "required_nodes"], 1)


if __name__ == "__main__":
    unittest.main()
