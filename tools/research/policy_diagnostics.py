"""Offline diagnostics for released V4 checkpoints (no engine required).

Run from the repository root:

    python tools/research/policy_diagnostics.py          # all released checkpoints
    python tools/research/policy_diagnostics.py --math   # gate math only

Reports, per checkpoint: parameter count, stored PPO temperatures, the learned
card-ID residual gate in ``GlobalCardEncoder`` (how much the card representation
relies on memorised IDs versus semantic features), and the relative L2 weight
change versus the first checkpoint (IL).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

RELEASED_CHECKPOINTS = {
    "IL": "checkpoints/IL/checkpoint-step-00029396.pt",
    "ActiveIL": "checkpoints/active IL/checkpoint-step-00000030.pt",
    "General": "checkpoints/General/checkpoint-step-00000460.pt",
    "Hog1": "checkpoints/2_6hog_expert/hog26-specialist1.pt",
    "Hog2": "checkpoints/2_6hog_expert/hog26-specialist2.pt",
}


def tempered_binary_probability(probability: float, temperature: float) -> float:
    """P(ACT) after dividing both gate logits by ``temperature``."""

    if not 0.0 < probability < 1.0 or temperature <= 0.0:
        raise ValueError("probability must be in (0,1) and temperature positive")
    act = math.log(probability) / temperature
    wait = math.log(1.0 - probability) / temperature
    return 1.0 / (1.0 + math.exp(wait - act))


def class_weighted_probability(human_rate: float, act_weight: float) -> float:
    """Calibrated P(ACT) of a gate fitted with an ACT class weight."""

    odds = act_weight * human_rate / (1.0 - human_rate)
    return odds / (1.0 + odds)


def probability_of_any_act(per_step: float, steps: int) -> float:
    return 1.0 - (1.0 - per_step) ** steps


def print_gate_math() -> None:
    print("PPO gate temperature 0.2 (4 decisions/s, 40 steps = 10 s)")
    for p in (0.1, 0.2, 0.3, 0.4, 0.5):
        q = tempered_binary_probability(p, 0.2)
        print(f"  P(ACT) {p:.2f} -> {q:.5f}; P(any ACT in 10 s) {probability_of_any_act(q, 40):.3f}")
    print("IL gate ACT weight 8: human per-frame ACT rate -> fitted P(ACT)")
    for h in (0.03, 0.05, 0.08, 0.11, 0.2):
        print(f"  {h:.2f} -> {class_weighted_probability(h, 8.0):.3f}")


def card_id_reliance(model) -> dict[str, float]:
    """Summarise the ID residual gate over every non-padding card."""

    import torch

    encoder = model.card_encoder if hasattr(model, "card_encoder") else next(
        module for module in model.modules() if type(module).__name__ == "GlobalCardEncoder"
    )
    with torch.no_grad():
        ids = torch.arange(1, encoder.id_embedding.num_embeddings)
        static = encoder.static_encoder(encoder.catalog_static[ids])
        mechanics = encoder.mechanic_encoder(encoder.catalog_mechanics[ids])
        gate = torch.sigmoid(encoder.id_gate(torch.cat((static, mechanics), dim=-1))).squeeze(-1)
        ratio = (gate[:, None] * encoder.id_embedding(ids)).norm(dim=-1) / (static + mechanics).norm(dim=-1)
        features = torch.cat((encoder.catalog_static[ids], encoder.catalog_mechanics[ids]), dim=-1)
        distance = torch.cdist(features, features)
        distance.fill_diagonal_(float("inf"))
        nearest = distance.min(dim=1).values
    return {
        "cards": float(ids.numel()),
        "id_gate_mean": float(gate.mean()),
        "id_gate_min": float(gate.min()),
        "id_to_semantic_norm_median": float(ratio.median()),
        "duplicate_feature_rows": float((nearest < 1e-6).sum()),
        "nearest_feature_distance_median": float(nearest.median()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--math", action="store_true", help="only print the gate math")
    args = parser.parse_args()
    print_gate_math()
    if args.math:
        return 0

    sys.path.insert(0, str(args.root))
    from native_runner.training.v4.checkpoint import load_actor_critic_checkpoint
    from native_runner.training.v4.factory import build_production_model_v4

    reference = None
    for name, relative in RELEASED_CHECKPOINTS.items():
        path = args.root / relative
        if not path.is_file():
            print(f"{name}: missing {path}")
            continue
        model = build_production_model_v4(device="cpu").eval()
        payload = load_actor_critic_checkpoint(path, model, map_location="cpu")
        stats = card_id_reliance(model)
        state = {key: value for key, value in model.state_dict().items() if value.is_floating_point()}
        change = ""
        if reference is None:
            reference = state
        else:
            numerator = sum(float(((state[key] - reference[key]) ** 2).sum()) for key in state)
            denominator = sum(float((reference[key] ** 2).sum()) for key in state)
            change = f" weight_change_vs_first={math.sqrt(numerator / denominator):.4f}"
        print(
            f"{name}: params={model.parameter_count()} gate_temp={payload['ppo_gate_temperature']}"
            f" id_gate_mean={stats['id_gate_mean']:.3f} id/semantic={stats['id_to_semantic_norm_median']:.2f}"
            f"{change}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
