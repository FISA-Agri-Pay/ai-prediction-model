import unittest

from src.evaluation.metrics import evaluate_predictions, smape
from src.evaluation.pod_policy import PodPolicy, required_pods


class PodPolicyTest(unittest.TestCase):
    def test_required_pods_clamps_to_min_and_max(self):
        policy = PodPolicy(capacity_per_pod=10, safety_margin=0.0, min_pods=1, max_pods=3)

        self.assertEqual(required_pods(-5, policy), 1)
        self.assertEqual(required_pods(35, policy), 3)
        self.assertEqual(required_pods(11, policy), 2)


class MetricsTest(unittest.TestCase):
    def test_smape_handles_zero_denominator(self):
        self.assertEqual(smape([0, 10], [0, 20]), 1 / 3)

    def test_metrics_reject_empty_inputs(self):
        with self.assertRaisesRegex(ValueError, "smape: inputs must not be empty"):
            smape([], [])

    def test_metrics_reject_misaligned_inputs(self):
        with self.assertRaisesRegex(ValueError, "evaluate_predictions: actual and predicted"):
            evaluate_predictions([10, 20], [10])

    def test_evaluate_predictions_returns_autoscaling_metrics(self):
        policy = PodPolicy(capacity_per_pod=10, safety_margin=0.0, min_pods=1, max_pods=5)
        metrics = evaluate_predictions([10, 20, 30, 40], [10, 10, 40, 40], policy)

        self.assertEqual(metrics["pod_accuracy"], 0.5)
        self.assertEqual(metrics["under_provisioning_rate"], 0.25)
        self.assertEqual(metrics["over_provisioning_rate"], 0.25)


if __name__ == "__main__":
    unittest.main()
