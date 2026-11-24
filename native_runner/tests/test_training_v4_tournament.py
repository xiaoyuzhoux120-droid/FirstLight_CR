"""Integration tests for the offline tournament driver with a simulated engine."""

import json
import random
from types import SimpleNamespace as NS

import pytest

from native_runner.contracts import ActionKind, ActionV1, TerminalV1
from native_runner.match_factory import MatchConfig
from native_runner.training.v4 import evaluate, tournament
from native_runner.training.v4.eval_stats import DeckSpec, elo_from_score, mirrored_schedule, score_from_elo
from native_runner.training.v4.expert import FIRST_POLICY_DECISION_TICK

DECKS = (
    DeckSpec("strong", (26000000, 26000001, 26000005, 28000001, 28000000, 26000003, 26000014, 26000018)),
    DeckSpec(
        "weak",
        (26000024, 26000044, 26000084, 28000011, 28000000, 26000061, 26000002, 26000050),
        tower_troop_id=159000001,
    ),
)


def _simulated_world(strength, *, seat_bonus=0.0, deck_bonus=None, rng_seed=0):
    """Logistic match model with model strength, seat advantage and deck strength."""

    deck_bonus = deck_bonus or {}
    rng = random.Random(rng_seed)
    calls = []

    def run(game, match):
        calls.append(game.game_id)
        elo = strength[game.model_a] - strength[game.model_b]
        elo += seat_bonus if game.a_owner == 0 else -seat_bonus
        elo += deck_bonus.get(game.deck_a, 0.0) - deck_bonus.get(game.deck_b, 0.0)
        winner = "a" if rng.random() < score_from_elo(elo) else "b"
        return {"winner": winner, "terminated": True, "truncated": False, "actions_by_owner": [1, 1]}

    return run, calls


def test_resolve_match_puts_each_models_deck_on_its_owner():
    base = MatchConfig(seed=1)
    (game,) = [
        g for g in mirrored_schedule(["a", "b"], DECKS, blocks_per_pair=1) if g.a_owner == 1 and g.deck_a == "weak"
    ]
    match = tournament.resolve_match(base, game, {d.name: d for d in DECKS})
    assert match.deck1 == DECKS[1].cards and match.deck0 == DECKS[0].cards
    assert match.tower_troop1_id == 159000001 and match.tower_troop0_id == base.tower_troop0_id
    assert match.seed == game.seed


def test_mirrored_tournament_separates_model_strength_from_seat_and_deck_bias(tmp_path):
    strength = {"good": 120.0, "mid": 0.0, "bad": -120.0}
    run, _ = _simulated_world(strength, seat_bonus=150.0, deck_bonus={"strong": 200.0}, rng_seed=5)
    schedule = mirrored_schedule(list(strength), DECKS, blocks_per_pair=150)
    report = tournament.run_schedule(schedule, DECKS, MatchConfig(), run, tmp_path, log=lambda _m: None)
    ratings = report["ratings_elo"]
    assert ratings["good"] > ratings["mid"] > ratings["bad"]
    # Seat/deck noise averages out, but through the logistic curve it compresses the
    # observable gap: the expected good-vs-bad score maps to ~169 Elo, not 240.
    expected = elo_from_score(
        sum(score_from_elo(240 + seat + deck) for seat in (150, -150) for deck in (200, -200)) / 4
    )
    assert ratings["good"] - ratings["bad"] == pytest.approx(expected, abs=40)
    # The seat advantage is visible, but cancelled out of model comparisons.
    assert report["owner0_seat"]["score"] > 0.6
    by_deck = {(d["model"], d["deck"]): d["score"] for d in report["decks"]}
    assert by_deck[("mid", "strong")] > by_deck[("mid", "weak")]
    assert (tmp_path / "report.txt").read_text().startswith("games:")
    saved = json.loads((tmp_path / "schedule.json").read_text())
    assert len(saved["games"]) == len(schedule) and saved["decks"][1]["tower_troop_id"] == 159000001


def test_tournament_resumes_without_replaying_finished_games(tmp_path):
    schedule = mirrored_schedule(["a", "b"], DECKS, blocks_per_pair=3)
    run, calls = _simulated_world({"a": 0.0, "b": 0.0})
    first = schedule[:5]
    tournament.run_schedule(first, DECKS, MatchConfig(), run, tmp_path, log=lambda _m: None)
    # Simulate a crash that left half a JSON line behind.
    with (tmp_path / "matches.jsonl").open("a") as handle:
        handle.write('{"game_id": "partial')
    calls.clear()
    report = tournament.run_schedule(schedule, DECKS, MatchConfig(), run, tmp_path, log=lambda _m: None)
    assert calls == [g.game_id for g in schedule[5:]]
    assert report["games"] == len(schedule) and report["status_counts"] == {"ok": len(schedule)}
    lines = (tmp_path / "matches.jsonl").read_text().splitlines()
    assert len(lines) == len(schedule) and all(json.loads(line) for line in lines)


def test_engine_failures_are_recorded_retried_on_resume_and_capped(tmp_path):
    schedule = mirrored_schedule(["a", "b"], DECKS, blocks_per_pair=2)
    failing = {schedule[1].game_id}

    def flaky(game, match):
        if game.game_id in failing:
            raise ConnectionError("engine dropped")
        return {"winner": "a", "terminated": True, "truncated": False}

    report = tournament.run_schedule(schedule, DECKS, MatchConfig(), flaky, tmp_path, log=lambda _m: None)
    assert report["status_counts"] == {"error": 1, "ok": len(schedule) - 1}
    error_row = next(json.loads(l) for l in (tmp_path / "matches.jsonl").read_text().splitlines() if "error" in l)
    assert error_row["error"] == "ConnectionError: engine dropped"
    failing.clear()
    report = tournament.run_schedule(schedule, DECKS, MatchConfig(), flaky, tmp_path, log=lambda _m: None)
    assert report["pairs"][0]["games"] == len(schedule)

    def broken(game, match):
        raise RuntimeError("no engine")

    report = tournament.run_schedule(
        schedule, DECKS, MatchConfig(), broken, tmp_path / "broken", max_errors=2, log=lambda _m: None
    )
    assert report["status_counts"] == {"error": 3}


def test_sprt_gate_stops_early_on_a_block_boundary(tmp_path):
    schedule = mirrored_schedule(["new", "old"], DECKS, blocks_per_pair=200)
    run, calls = _simulated_world({"new": 250.0, "old": 0.0}, rng_seed=2)
    report = tournament.run_schedule(
        schedule, DECKS, MatchConfig(), run, tmp_path, sprt_elo=(0.0, 30.0), log=lambda _m: None
    )
    assert report["stopped_by_sprt"]["decision"] == "H1"
    assert len(calls) < len(schedule) and len(calls) % 4 == 0
    assert report["pairs"][0]["sprt"]["decision"] == "H1"


def test_focus_deck_file_resolves_to_valid_match_configs():
    decks = tournament.deck_pool_from_focus_file(tournament.Path(evaluate.__file__).with_name("focus_decks_20.json"))
    assert len(decks) == 20 and len({d.name for d in decks}) == 20
    schedule = mirrored_schedule(["a", "b"], decks, blocks_per_pair=10)
    by_name = {d.name: d for d in decks}
    for game in schedule:
        tournament.resolve_match(MatchConfig(), game, by_name)  # MatchConfig validates decks/forms/towers


def test_cli_dry_run_writes_schedule_without_an_engine(tmp_path, capsys):
    assert (
        tournament.main(
            [
                "--checkpoint",
                "a=x.pt",
                "--checkpoint",
                "b=y.pt",
                "--blocks-per-pair",
                "2",
                "--output",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 0
    )
    assert "8 games scheduled" in capsys.readouterr().out
    assert len(json.loads((tmp_path / "schedule.json").read_text())["games"]) == 8
    with pytest.raises(SystemExit):
        tournament.main(["--checkpoint", "a=x.pt", "--checkpoint", "a=y.pt", "--output", str(tmp_path), "--dry-run"])


# --- play_match behaviour metrics -------------------------------------------------


class _ScriptedEnvironment:
    """Three-player-free stand-in that ends the game after ``length`` steps."""

    def __init__(self, length, elixir):
        self.length, self.elixir, self.steps = length, elixir, 0

    def _observations(self, ended):
        terminal = (
            TerminalV1(ended=True, winner=1, result_by_owner=(-1.0, 1.0), terminal_tick=999) if ended else TerminalV1()
        )
        tick = FIRST_POLICY_DECISION_TICK + 5 * self.steps
        return {
            owner: NS(
                tick=tick,
                owner=owner,
                players=[NS(owner=owner, elixir_exact=self.elixir[owner])],
                events=[],
                terminal=terminal,
            )
            for owner in (0, 1)
        }

    def reset(self, *, match_config, options):
        self.steps = 0
        return self._observations(False), {}

    def step(self, commands, **kwargs):
        self.steps += 1
        ended = self.steps >= self.length
        return self._observations(ended), {}, {"__all__": ended}, {"__all__": False}, {}


def test_play_match_reports_first_action_forced_opening_and_full_elixir(monkeypatch):
    forced_calls = []

    def session(environment, model, *, actor_owner, **kwargs):
        def decide(observation, force_act=False):
            forced_calls.append((actor_owner, force_act))
            # Model "active" plays from its third decision; "passive" only when forced.
            play = force_act or (model == "active" and observation.tick >= FIRST_POLICY_DECISION_TICK + 10)
            action = ActionV1.wait(actor_owner, ticks=5)
            if play:
                action = NS(kind=ActionKind.PLAY_CARD)
            return NS(decoded=NS(actions=(action,)))

        return NS(
            start_episode=lambda *a, **k: None,
            decide=decide,
            end_episode=lambda: None,
            tensorizer=NS(tensorize=lambda *a, **k: None),
        )

    monkeypatch.setattr(evaluate, "build_policy_session_v4", session)
    env = _ScriptedEnvironment(length=8, elixir={0: 10.0, 1: 4.0})
    outcome = evaluate.play_match(
        env, {0: "active", 1: "passive"}, MatchConfig(seed=3), force_first_action_tick=FIRST_POLICY_DECISION_TICK + 20
    )
    assert outcome.terminated and outcome.terminal.winner == 1
    assert outcome.first_action_tick_by_owner == [FIRST_POLICY_DECISION_TICK + 10, FIRST_POLICY_DECISION_TICK + 20]
    assert outcome.forced_actions_by_owner == [0, 1]
    assert outcome.actions_by_owner == [6, 1]
    assert outcome.full_elixir_ticks_by_owner == [40, 0]
    assert sum(1 for owner, forced in forced_calls if forced) == 1


def test_evaluate_writes_statistical_report(monkeypatch, tmp_path):
    terminal = TerminalV1(ended=True, winner=0, result_by_owner=(1.0, -1.0), terminal_tick=100)
    observations = {
        o: NS(tick=100, owner=o, players=[NS(owner=o, elixir_exact=5.0)], events=[], terminal=terminal) for o in (0, 1)
    }
    monkeypatch.setattr(
        evaluate,
        "load_policy_v4",
        lambda p, **k: NS(model=p, checkpoint_path=tournament.Path(p), checkpoint_sha256="h", checkpoint_id=p),
    )
    monkeypatch.setattr(
        evaluate,
        "production_semantic_bundle",
        lambda: NS(native_card_catalog=None, semantic_subset_contract=None, native_effect_catalog=None),
    )
    monkeypatch.setattr(evaluate.RuntimeEffectCatalog, "from_catalog", lambda c: None)
    monkeypatch.setattr(evaluate, "NativeClashEnv", lambda *a, **k: NS(close_transport=lambda: None))
    monkeypatch.setattr(
        evaluate,
        "BattleEnvV1",
        lambda **k: NS(
            reset=lambda **kw: (observations, {}),
            step=lambda c, **kw: (observations, {}, {"__all__": True}, {"__all__": False}, {}),
        ),
    )
    monkeypatch.setattr(
        evaluate,
        "build_policy_session_v4",
        lambda *a, actor_owner, **k: NS(
            start_episode=lambda *x, **y: None,
            end_episode=lambda: None,
            decide=lambda o: NS(decoded=NS(actions=(ActionV1.wait(actor_owner, ticks=5),))),
        ),
    )
    evaluate.evaluate("a.pt", "b.pt", games=4, match=MatchConfig(), output=tmp_path)
    report = json.loads((tmp_path / "report.json").read_text())
    # Owner 0 always wins: the two models split 2-2 and the seat bias is exposed.
    assert report["pairs"][0]["score"] == 0.5 and report["owner0_seat"]["score"] == 1.0
