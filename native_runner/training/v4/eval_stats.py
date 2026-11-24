"""Statistics and schedules for offline model-versus-model evaluation.

This module is deliberately free of PyTorch and native-engine imports so that
schedules and reports can be built, tested and re-computed on any machine from
``matches.jsonl`` files alone.

Conventions
-----------
* A game result is scored from model A's point of view: win = 1, draw = 0.5,
  loss = 0.  Truncated or failed games are *not* scored; they are reported
  separately because they indicate execution problems, not playing strength.
* Elo differences use the logistic model ``score = 1 / (1 + 10^(-d/400))``.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVAL_SCHEDULE_SCHEMA = "v4-offline-eval-schedule.v1"
EVAL_REPORT_SCHEMA = "v4-offline-eval-report.v1"
Z_95 = 1.959963984540054


# ---------------------------------------------------------------------------
# Scalar statistics


def wilson_interval(successes: float, trials: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval; ``successes`` may be fractional (draws = 0.5)."""

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("wilson interval requires 0 <= successes <= trials")
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    centre = (p + z2 / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4 * trials * trials)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def elo_from_score(score: float) -> float:
    """Elo difference implied by an expected score; clipped to +-800 at 0/1."""

    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be in [0, 1]")
    eps = 10 ** (-800 / 400) / (1 + 10 ** (-800 / 400))
    score = min(max(score, eps), 1.0 - eps)
    return -400.0 * math.log10(1.0 / score - 1.0)


def score_from_elo(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def points(self) -> float:
        return self.wins + 0.5 * self.draws

    @property
    def score(self) -> float | None:
        return None if self.games == 0 else self.points / self.games

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"wins": self.wins, "draws": self.draws, "losses": self.losses, "games": self.games}
        if self.games:
            low, high = wilson_interval(self.points, self.games)
            score = self.points / self.games
            result.update(
                score=score,
                score_ci95=[low, high],
                elo=elo_from_score(score),
                elo_ci95=[elo_from_score(low), elo_from_score(high)],
            )
        return result


def summarize_scores(results: Iterable[float]) -> ScoreSummary:
    wins = draws = losses = 0
    for value in results:
        if value == 1.0:
            wins += 1
        elif value == 0.5:
            draws += 1
        elif value == 0.0:
            losses += 1
        else:
            raise ValueError(f"game score must be 0, 0.5 or 1, got {value!r}")
    return ScoreSummary(wins, draws, losses)


# ---------------------------------------------------------------------------
# Sequential probability ratio test (trinomial, Fishtest-style approximation)


@dataclass(frozen=True, slots=True)
class SprtResult:
    llr: float
    lower: float
    upper: float
    decision: str  # "H1" (A is at least elo1 stronger), "H0", or "continue"

    def to_dict(self) -> dict[str, Any]:
        return {"llr": self.llr, "lower": self.lower, "upper": self.upper, "decision": self.decision}


def sprt(
    summary: ScoreSummary, *, elo0: float = 0.0, elo1: float = 30.0, alpha: float = 0.05, beta: float = 0.05
) -> SprtResult:
    """Generalized SPRT on game scores with a normal approximation.

    H0: true Elo difference = ``elo0``; H1: true Elo difference = ``elo1``.
    Uses the variance of observed per-game scores (draw aware).  Suitable as a
    promotion gate: stop as soon as the decision is not ``continue``.
    """

    if not (0 < alpha < 1 and 0 < beta < 1) or elo1 <= elo0:
        raise ValueError("invalid SPRT parameters")
    lower = math.log(beta / (1 - alpha))
    upper = math.log((1 - beta) / alpha)
    n = summary.games
    if n == 0:
        return SprtResult(0.0, lower, upper, "continue")
    mean = summary.points / n
    second = (summary.wins + 0.25 * summary.draws) / n
    variance = second - mean * mean
    if variance <= 1e-12:
        # Degenerate (all identical results): use a conservative floor.
        variance = 0.25 / max(n, 1)
    s0, s1 = score_from_elo(elo0), score_from_elo(elo1)
    llr = n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * variance)
    decision = "H1" if llr >= upper else "H0" if llr <= lower else "continue"
    return SprtResult(llr, lower, upper, decision)


# ---------------------------------------------------------------------------
# Multi-model ratings (Bradley-Terry with draws counted as half wins)


def bradley_terry(
    pair_points: Mapping[tuple[str, str], tuple[float, int]],
    *,
    iterations: int = 2000,
    tolerance: float = 1e-10,
    prior_games: float = 1.0,
) -> dict[str, float]:
    """Fit Bradley-Terry strengths and return Elo ratings centred on 0.

    ``pair_points[(x, y)] = (points_of_x_vs_y, games)``.  A weak prior of
    ``prior_games`` virtual draws between every observed pair keeps the fit
    finite when a model wins or loses every game.
    """

    players = sorted({name for pair in pair_points for name in pair})
    if not players:
        return {}
    wins: dict[str, float] = {name: 0.0 for name in players}
    games: dict[tuple[str, str], float] = {}
    for (x, y), (points, count) in pair_points.items():
        if x == y or count <= 0:
            continue
        if points < 0 or points > count:
            raise ValueError("pair points out of range")
        wins[x] += points + 0.5 * prior_games
        wins[y] += count - points + 0.5 * prior_games
        key = (x, y) if x < y else (y, x)
        games[key] = games.get(key, 0.0) + count + prior_games
    strength = {name: 1.0 for name in players}
    for _ in range(iterations):
        updated = {}
        for name in players:
            denominator = 0.0
            for (x, y), count in games.items():
                if name == x or name == y:
                    denominator += count / (strength[x] + strength[y])
            updated[name] = wins[name] / denominator if denominator > 0 else strength[name]
        geometric = math.exp(sum(math.log(v) for v in updated.values()) / len(updated))
        updated = {name: value / geometric for name, value in updated.items()}
        delta = max(abs(math.log(updated[name]) - math.log(strength[name])) for name in players)
        strength = updated
        if delta < tolerance:
            break
    return {name: 400.0 * math.log10(value) for name, value in strength.items()}


# ---------------------------------------------------------------------------
# Schedules


@dataclass(frozen=True, slots=True)
class DeckSpec:
    """One playable deck: eight card IDs, their form masks and a tower troop."""

    name: str
    cards: tuple[int, ...]
    form_availability: tuple[int, ...] = (0,) * 8
    tower_troop_id: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("deck name cannot be empty")
        if len(self.cards) != 8 or len(set(self.cards)) != 8:
            raise ValueError(f"deck {self.name!r} must contain eight distinct cards")
        if len(self.form_availability) != 8:
            raise ValueError(f"deck {self.name!r} needs eight form masks")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DeckSpec:
        return cls(
            name=str(value["name"]),
            cards=tuple(int(item) for item in value["cards"]),
            form_availability=tuple(int(item) for item in value.get("form_availability", (0,) * 8)),
            tower_troop_id=None if value.get("tower_troop_id") is None else int(value["tower_troop_id"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cards": list(self.cards),
            "form_availability": list(self.form_availability),
            "tower_troop_id": self.tower_troop_id,
        }


def load_deck_pool(path: str | Path) -> tuple[DeckSpec, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = raw["decks"] if isinstance(raw, Mapping) else raw
    decks = tuple(DeckSpec.from_mapping(item) for item in rows)
    if len({deck.name for deck in decks}) != len(decks):
        raise ValueError("deck pool contains duplicate names")
    if not decks:
        raise ValueError("deck pool is empty")
    return decks


@dataclass(frozen=True, slots=True)
class ScheduledGame:
    """One game of a schedule.

    ``model_a``/``model_b`` are labels.  ``a_owner`` is the engine owner (0/1)
    that model A controls.  ``deck_a``/``deck_b`` are the decks the models play.
    Games of the same ``block`` share one seed and differ only by side and deck
    assignment, so side and deck advantages cancel within a complete block.
    """

    game_id: str
    block: int
    model_a: str
    model_b: str
    a_owner: int
    deck_a: str
    deck_b: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "block": self.block,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "a_owner": self.a_owner,
            "deck_a": self.deck_a,
            "deck_b": self.deck_b,
            "seed": self.seed,
        }


def _game_id(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()[:16]


def mirrored_schedule(
    models: Sequence[str],
    decks: Sequence[DeckSpec],
    *,
    blocks_per_pair: int,
    base_seed: int = 1,
    fixed_deck_a: str | None = None,
    rng_seed: int = 0,
) -> tuple[ScheduledGame, ...]:
    """Round-robin schedule with mirrored side/deck blocks.

    For each unordered model pair and each block, two decks X and Y are drawn
    from the pool (X == Y allowed only when the pool has one deck).  The block
    plays four games with one seed: (A:X vs B:Y) and (A:Y vs B:X), each from
    both sides.  With ``fixed_deck_a`` model A always plays that deck (specialist
    evaluation) and a block contains two games (both sides) instead.
    """

    if len(models) < 2 or len(set(models)) != len(models):
        raise ValueError("a schedule needs at least two distinct model labels")
    if blocks_per_pair < 1:
        raise ValueError("blocks_per_pair must be positive")
    names = [deck.name for deck in decks]
    if not names:
        raise ValueError("deck pool is empty")
    if fixed_deck_a is not None and fixed_deck_a not in names:
        raise ValueError(f"unknown fixed deck {fixed_deck_a!r}")
    rng = random.Random(rng_seed)
    games: list[ScheduledGame] = []
    block_index = 0
    for i, model_a in enumerate(models):
        for model_b in models[i + 1 :]:
            for _ in range(blocks_per_pair):
                seed = base_seed + block_index
                if fixed_deck_a is not None:
                    pairs = [(fixed_deck_a, rng.choice(names))]
                else:
                    x, y = rng.sample(names, 2) if len(names) > 1 else (names[0], names[0])
                    pairs = [(x, y)] if x == y else [(x, y), (y, x)]
                for deck_a, deck_b in pairs:
                    for a_owner in (0, 1):
                        games.append(
                            ScheduledGame(
                                game_id=_game_id(model_a, model_b, block_index, deck_a, deck_b, a_owner, seed),
                                block=block_index,
                                model_a=model_a,
                                model_b=model_b,
                                a_owner=a_owner,
                                deck_a=deck_a,
                                deck_b=deck_b,
                                seed=seed,
                            )
                        )
                block_index += 1
    return tuple(games)


# ---------------------------------------------------------------------------
# Reports


def game_score_for_a(row: Mapping[str, Any]) -> float | None:
    """Return A's score for a normally terminated game, else ``None``."""

    if row.get("status", "ok") != "ok" or row.get("truncated") or not row.get("terminated", True):
        return None
    winner = row.get("winner")
    if winner == "a":
        return 1.0
    if winner == "b":
        return 0.0
    if winner is None:
        return 0.5
    raise ValueError(f"unknown winner label {winner!r}")


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            # A crash can leave a partial last line; ignore only that case.
            if number == len(lines):
                break
            raise ValueError(f"{path}:{number}: invalid JSON") from error
    return rows


def build_report(rows: Sequence[Mapping[str, Any]], *, sprt_elo: tuple[float, float] | None = None) -> dict[str, Any]:
    """Aggregate per-game rows into pairwise, per-deck, side and rating tables.

    Rows from ``evaluate.py`` (no ``model_a`` field) are labelled ``a``/``b``.
    """

    pairs: dict[tuple[str, str], list[float]] = {}
    by_deck: dict[tuple[str, str], list[float]] = {}
    by_matchup: dict[tuple[str, str, str], list[float]] = {}
    side_scores: dict[int, list[float]] = {0: [], 1: []}
    status_counts: dict[str, int] = {}
    behaviour: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        model_a = str(row.get("model_a", "a"))
        model_b = str(row.get("model_b", "b"))
        status = "truncated" if row.get("truncated") else str(row.get("status", "ok"))
        status_counts[status] = status_counts.get(status, 0) + 1
        score = game_score_for_a(row)
        if score is None:
            continue
        pairs.setdefault((model_a, model_b), []).append(score)
        a_owner = int(row.get("a_owner", 0))
        # Owner-0 score: how often the bottom/owner-0 seat wins, model-agnostic.
        side_scores[0].append(score if a_owner == 0 else 1.0 - score)
        if "deck_a" in row:
            by_deck.setdefault((model_a, str(row["deck_a"])), []).append(score)
            by_deck.setdefault((model_b, str(row["deck_b"])), []).append(1.0 - score)
            by_matchup.setdefault((model_a, model_b, str(row["deck_a"])), []).append(score)
        actions = row.get("actions_by_owner")
        first = row.get("first_action_tick_by_owner")
        for label, owner in ((model_a, a_owner), (model_b, 1 - a_owner)):
            stats = behaviour.setdefault(label, {"actions": [], "first_action_tick": [], "full_elixir_ticks": []})
            if isinstance(actions, Sequence) and len(actions) == 2:
                stats["actions"].append(float(actions[owner]))
            if isinstance(first, Sequence) and len(first) == 2 and first[owner] is not None:
                stats["first_action_tick"].append(float(first[owner]))
            leak = row.get("full_elixir_ticks_by_owner")
            if isinstance(leak, Sequence) and len(leak) == 2:
                stats["full_elixir_ticks"].append(float(leak[owner]))
    pair_table = []
    pair_points: dict[tuple[str, str], tuple[float, int]] = {}
    for (model_a, model_b), scores in sorted(pairs.items()):
        summary = summarize_scores(scores)
        entry = {"model_a": model_a, "model_b": model_b, **summary.to_dict()}
        if sprt_elo is not None:
            entry["sprt"] = sprt(summary, elo0=sprt_elo[0], elo1=sprt_elo[1]).to_dict()
        pair_table.append(entry)
        previous = pair_points.get((model_a, model_b), (0.0, 0))
        pair_points[(model_a, model_b)] = (previous[0] + summary.points, previous[1] + summary.games)
    side = summarize_scores(side_scores[0])
    report: dict[str, Any] = {
        "schema": EVAL_REPORT_SCHEMA,
        "games": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "pairs": pair_table,
        "owner0_seat": side.to_dict(),
        "decks": [
            {"model": model, "deck": deck, **summarize_scores(scores).to_dict()}
            for (model, deck), scores in sorted(by_deck.items())
        ],
        "matchups": [
            {"model_a": a, "model_b": b, "deck_a": deck, **summarize_scores(scores).to_dict()}
            for (a, b, deck), scores in sorted(by_matchup.items())
        ],
        "behaviour": {
            label: {
                key: {"mean": sum(values) / len(values), "n": len(values)} if values else None
                for key, values in stats.items()
            }
            for label, stats in sorted(behaviour.items())
        },
    }
    if len({name for pair in pair_points for name in pair}) >= 2:
        report["ratings_elo"] = dict(sorted(bradley_terry(pair_points).items(), key=lambda kv: -kv[1]))
    return report


def format_report(report: Mapping[str, Any]) -> str:
    """Human-readable plain-text rendering of :func:`build_report`."""

    lines = [f"games: {report['games']}  status: {report['status_counts']}"]
    for entry in report["pairs"]:
        if not entry["games"]:
            continue
        low, high = entry["score_ci95"]
        lines.append(
            f"{entry['model_a']} vs {entry['model_b']}: {entry['wins']}W {entry['draws']}D {entry['losses']}L "
            f"score {entry['score']:.3f} [{low:.3f}, {high:.3f}]  Elo {entry['elo']:+.0f} "
            f"[{entry['elo_ci95'][0]:+.0f}, {entry['elo_ci95'][1]:+.0f}]"
            + (f"  SPRT {entry['sprt']['decision']} (LLR {entry['sprt']['llr']:.2f})" if "sprt" in entry else "")
        )
    seat = report["owner0_seat"]
    if seat["games"]:
        lines.append(
            f"owner-0 seat score: {seat['score']:.3f} [{seat['score_ci95'][0]:.3f}, {seat['score_ci95'][1]:.3f}]"
        )
    if "ratings_elo" in report:
        lines.append(
            "ratings (Bradley-Terry Elo): " + ", ".join(f"{k} {v:+.0f}" for k, v in report["ratings_elo"].items())
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Summarize offline evaluation matches.jsonl files.")
    parser.add_argument("matches", nargs="+", type=Path)
    parser.add_argument("--sprt", nargs=2, type=float, metavar=("ELO0", "ELO1"))
    parser.add_argument("--json", type=Path, help="write the full report here")
    args = parser.parse_args(argv)
    rows = [row for path in args.matches for row in read_rows(path)]
    report = build_report(rows, sprt_elo=tuple(args.sprt) if args.sprt else None)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
