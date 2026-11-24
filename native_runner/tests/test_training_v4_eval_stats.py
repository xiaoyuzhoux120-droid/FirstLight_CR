import json
import math
import random

import pytest

from native_runner.training.v4 import eval_stats as es


def test_wilson_interval_matches_reference_values_and_bounds():
    low, high = es.wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-4) and high == pytest.approx(0.5962, abs=1e-4)
    assert es.wilson_interval(0, 0) == (0.0, 1.0)
    low, high = es.wilson_interval(10, 10)
    assert high == pytest.approx(1.0) and 0.69 < low < 0.73
    with pytest.raises(ValueError):
        es.wilson_interval(3, 2)


def test_elo_conversion_is_symmetric_and_invertible():
    assert es.elo_from_score(0.5) == pytest.approx(0.0)
    assert es.elo_from_score(0.75) == pytest.approx(190.85, abs=0.01)
    assert es.elo_from_score(0.25) == pytest.approx(-es.elo_from_score(0.75))
    for elo in (-300.0, -40.0, 0.0, 15.0, 250.0):
        assert es.elo_from_score(es.score_from_elo(elo)) == pytest.approx(elo)
    assert math.isfinite(es.elo_from_score(1.0)) and es.elo_from_score(1.0) == pytest.approx(800, abs=1)


def test_score_summary_counts_draws_as_half_points():
    summary = es.summarize_scores([1.0, 0.5, 0.0, 1.0])
    assert (summary.wins, summary.draws, summary.losses, summary.points) == (2, 1, 1, 2.5)
    as_dict = summary.to_dict()
    assert as_dict["score"] == 0.625 and as_dict["score_ci95"][0] < 0.625 < as_dict["score_ci95"][1]
    with pytest.raises(ValueError):
        es.summarize_scores([0.7])


def test_sprt_accepts_clear_improvement_rejects_regression_and_waits_on_noise():
    strong = es.ScoreSummary(wins=140, draws=0, losses=60)
    weak = es.ScoreSummary(wins=60, draws=0, losses=140)
    few = es.ScoreSummary(wins=3, draws=0, losses=2)
    assert es.sprt(strong, elo0=0, elo1=30).decision == "H1"
    assert es.sprt(weak, elo0=0, elo1=30).decision == "H0"
    assert es.sprt(few, elo0=0, elo1=30).decision == "continue"
    assert es.sprt(es.ScoreSummary(0, 0, 0)).decision == "continue"
    with pytest.raises(ValueError):
        es.sprt(strong, elo0=10, elo1=10)


def test_bradley_terry_recovers_two_player_elo_and_transitive_order():
    ratings = es.bradley_terry({("a", "b"): (75.0, 100)}, prior_games=0.0)
    assert ratings["a"] - ratings["b"] == pytest.approx(es.elo_from_score(0.75), abs=0.5)
    rng = random.Random(3)
    truth = {"x": 200.0, "y": 0.0, "z": -150.0}
    table = {}
    for a, b in (("x", "y"), ("y", "z"), ("x", "z")):
        p = es.score_from_elo(truth[a] - truth[b])
        n = 4000
        table[(a, b)] = (float(sum(rng.random() < p for _ in range(n))), n)
    fitted = es.bradley_terry(table)
    assert fitted["x"] > fitted["y"] > fitted["z"]
    assert fitted["x"] - fitted["z"] == pytest.approx(350, abs=40)
    # A clean sweep stays finite thanks to the prior.
    swept = es.bradley_terry({("a", "b"): (10.0, 10)})
    assert all(math.isfinite(v) for v in swept.values()) and swept["a"] > swept["b"]


def _decks(count):
    return tuple(es.DeckSpec(f"d{i}", tuple(range(100 * i + 1, 100 * i + 9))) for i in range(count))


def test_mirrored_schedule_balances_sides_and_decks_within_every_block():
    schedule = es.mirrored_schedule(["m1", "m2", "m3"], _decks(5), blocks_per_pair=6, base_seed=10)
    assert len(schedule) == 3 * 6 * 4
    assert len({g.game_id for g in schedule}) == len(schedule)
    blocks = {}
    for game in schedule:
        blocks.setdefault(game.block, []).append(game)
    for games in blocks.values():
        assert len({g.seed for g in games}) == 1
        assert len({(g.model_a, g.model_b) for g in games}) == 1
        assert sorted(g.a_owner for g in games) == [0, 0, 1, 1]
        decks_a = sorted(g.deck_a for g in games)
        decks_b = sorted(g.deck_b for g in games)
        assert decks_a == decks_b and len(set(decks_a)) == 2  # each model plays both decks
    # Deterministic for a given RNG seed.
    again = es.mirrored_schedule(["m1", "m2", "m3"], _decks(5), blocks_per_pair=6, base_seed=10)
    assert again == schedule


def test_fixed_deck_schedule_keeps_candidate_deck_and_validates_inputs():
    schedule = es.mirrored_schedule(["hog", "general"], _decks(3), blocks_per_pair=5, fixed_deck_a="d1")
    assert len(schedule) == 10 and {g.deck_a for g in schedule} == {"d1"}
    single = es.mirrored_schedule(["a", "b"], _decks(1), blocks_per_pair=2)
    assert len(single) == 4 and {g.deck_a for g in single} == {"d0"}
    with pytest.raises(ValueError):
        es.mirrored_schedule(["a"], _decks(2), blocks_per_pair=1)
    with pytest.raises(ValueError):
        es.mirrored_schedule(["a", "b"], _decks(2), blocks_per_pair=1, fixed_deck_a="missing")
    with pytest.raises(ValueError):
        es.DeckSpec("bad", (1, 1, 2, 3, 4, 5, 6, 7))


def test_deck_pool_round_trip(tmp_path):
    decks = _decks(2)
    path = tmp_path / "pool.json"
    path.write_text(json.dumps({"decks": [d.to_dict() for d in decks]}))
    assert es.load_deck_pool(path) == decks
    path.write_text(json.dumps([decks[0].to_dict(), decks[0].to_dict()]))
    with pytest.raises(ValueError, match="duplicate"):
        es.load_deck_pool(path)


def test_report_excludes_truncated_and_failed_games_and_builds_tables():
    rows = [
        {
            "model_a": "x",
            "model_b": "y",
            "deck_a": "d0",
            "deck_b": "d1",
            "a_owner": 0,
            "winner": "a",
            "terminated": True,
            "truncated": False,
            "status": "ok",
            "actions_by_owner": [10, 4],
            "first_action_tick_by_owner": [60, 300],
            "full_elixir_ticks_by_owner": [0, 50],
        },
        {
            "model_a": "x",
            "model_b": "y",
            "deck_a": "d1",
            "deck_b": "d0",
            "a_owner": 1,
            "winner": None,
            "terminated": True,
            "truncated": False,
            "status": "ok",
        },
        {
            "model_a": "x",
            "model_b": "y",
            "deck_a": "d1",
            "deck_b": "d0",
            "a_owner": 1,
            "winner": None,
            "terminated": False,
            "truncated": True,
            "status": "ok",
        },
        {"model_a": "x", "model_b": "y", "deck_a": "d0", "deck_b": "d1", "a_owner": 0, "status": "error"},
    ]
    report = es.build_report(rows, sprt_elo=(0, 30))
    assert report["status_counts"] == {"error": 1, "ok": 2, "truncated": 1}
    (pair,) = report["pairs"]
    assert (pair["wins"], pair["draws"], pair["losses"]) == (1, 1, 0) and pair["score"] == 0.75
    assert pair["sprt"]["decision"] == "continue"
    decks = {(d["model"], d["deck"]): d["score"] for d in report["decks"]}
    assert decks == {("x", "d0"): 1.0, ("x", "d1"): 0.5, ("y", "d0"): 0.5, ("y", "d1"): 0.0}
    # x won from seat 0 and drew from seat 1 -> seat 0 scored 1 and 0.5.
    assert report["owner0_seat"]["score"] == 0.75
    assert report["behaviour"]["x"]["actions"] == {"mean": 10.0, "n": 1}
    assert report["behaviour"]["y"]["first_action_tick"] == {"mean": 300.0, "n": 1}
    assert report["behaviour"]["y"]["full_elixir_ticks"] == {"mean": 50.0, "n": 1}
    assert set(report["ratings_elo"]) == {"x", "y"}
    assert "x vs y" in es.format_report(report)


def test_report_accepts_legacy_evaluate_rows():
    rows = [
        {"game": 1, "a_owner": 0, "winner": "a", "terminated": True, "truncated": False},
        {"game": 2, "a_owner": 1, "winner": "b", "terminated": True, "truncated": False},
    ]
    report = es.build_report(rows)
    assert report["pairs"][0]["model_a"] == "a" and report["pairs"][0]["score"] == 0.5


def test_read_rows_tolerates_only_a_truncated_final_line(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n{"a": ')
    assert es.read_rows(path) == [{"a": 1}, {"a": 2}]
    path.write_text('{"a": \n{"a": 2}\n')
    with pytest.raises(ValueError):
        es.read_rows(path)


def test_cli_prints_report(tmp_path, capsys):
    path = tmp_path / "m.jsonl"
    path.write_text(json.dumps({"a_owner": 0, "winner": "a", "terminated": True, "truncated": False}) + "\n")
    assert es.main([str(path), "--json", str(tmp_path / "r.json")]) == 0
    assert "a vs b" in capsys.readouterr().out
    assert json.loads((tmp_path / "r.json").read_text())["schema"] == es.EVAL_REPORT_SCHEMA
