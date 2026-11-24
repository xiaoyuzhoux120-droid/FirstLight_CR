import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("policy_diagnostics", ROOT / "tools/research/policy_diagnostics.py")
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


class PolicyDiagnosticsTest(unittest.TestCase):
    def test_temperature_keeps_even_odds_and_sharpens_minority(self):
        self.assertAlmostEqual(diagnostics.tempered_binary_probability(0.5, 0.2), 0.5)
        self.assertAlmostEqual(diagnostics.tempered_binary_probability(0.3, 1.0), 0.3)
        self.assertLess(diagnostics.tempered_binary_probability(0.2, 0.2), 0.001)

    def test_class_weight_multiplies_odds(self):
        fitted = diagnostics.class_weighted_probability(0.05, 8.0)
        self.assertAlmostEqual(fitted / (1 - fitted), 8.0 * 0.05 / 0.95)

    def test_card_id_reliance_on_fresh_model(self):
        from native_runner.training.v4.factory import build_production_model_v4

        stats = diagnostics.card_id_reliance(build_production_model_v4(device="cpu"))
        # Untrained ID gate starts at sigmoid(-2) for every card.
        self.assertAlmostEqual(stats["id_gate_mean"], 0.1192, places=3)
        self.assertEqual(stats["duplicate_feature_rows"], 0.0)


if __name__ == "__main__":
    unittest.main()
