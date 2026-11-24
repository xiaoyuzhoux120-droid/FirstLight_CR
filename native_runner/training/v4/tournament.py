"""Resumable offline round-robin tournament between V4 checkpoints.

Compared with ``evaluate.py`` (two models, one deck pair, alternating seeds),
the tournament

* evaluates any number of checkpoints against each other;
* draws deck pairs from a pool and plays *mirrored blocks*: each block uses one
  seed and lets every model play every drawn deck from both sides, so side and
  deck advantages cancel instead of being confounded with model strength;
* writes one JSON line per game keyed by a stable ``game_id`` and resumes from an
  interrupted ``matches.jsonl`` without replaying finished games;
* records engine failures as ``status: "error"`` rows instead of aborting the run;
* can stop a two-model comparison early with an SPRT promotion gate;
* writes ``report.json`` with Wilson intervals, Elo, Bradley-Terry ratings,
  per-deck generalisation tables, seat bias and behaviour metrics.

Only offline, local engine instances are used.  Nothing here connects to game
servers.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .eval_stats import (
    EVAL_SCHEDULE_SCHEMA,
    DeckSpec,
    ScheduledGame,
    build_report,
    format_report,
    game_score_for_a,
    load_deck_pool,
    mirrored_schedule,
    read_rows,
    sprt,
    summarize_scores,
)

# A game runner receives one scheduled game and the concrete owner-0/1 match
# assignment, and returns the outcome fields of a matches.jsonl row.
GameRunner = Callable[[ScheduledGame, Any], Mapping[str, Any]]


def resolve_match(base_match: Any, game: ScheduledGame, decks: Mapping[str, DeckSpec]) -> Any:
    """Place model A's and B's decks on the correct engine owners for ``game``."""

    deck_a, deck_b = decks[game.deck_a], decks[game.deck_b]
    owner_deck = {game.a_owner: deck_a, 1 - game.a_owner: deck_b}
    changes: dict[str, Any] = {"seed": game.seed}
    for owner in (0, 1):
        deck = owner_deck[owner]
        changes[f"deck{owner}"] = deck.cards
        changes[f"deck{owner}_form_availability"] = deck.form_availability
        if deck.tower_troop_id is not None:
            changes[f"tower_troop{owner}_id"] = deck.tower_troop_id
    return replace(base_match, **changes)


def run_schedule(
    schedule: Sequence[ScheduledGame],
    decks: Sequence[DeckSpec],
    base_match: Any,
    runner: GameRunner,
    output: str | Path,
    *,
    sprt_elo: tuple[float, float] | None = None,
    max_errors: int = 10,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Play the schedule, resuming from ``output/matches.jsonl``; return the report."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    by_name = {deck.name: deck for deck in decks}
    write_schedule(schedule, decks, output / "schedule.json")
    matches = output / "matches.jsonl"
    rows = read_rows(matches) if matches.exists() else []
    # Rewrite without a possibly truncated last line before appending.
    matches.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    finished = {row["game_id"] for row in rows if row.get("status") == "ok"}
    errors = sum(row.get("status") == "error" for row in rows)
    models = {name for game in schedule for name in (game.model_a, game.model_b)}
    gate = sprt_elo is not None and len(models) == 2
    stopped_by_sprt = None
    previous_block = None
    with matches.open("a", encoding="utf-8", buffering=1) as handle:
        for game in schedule:
            if game.game_id in finished:
                continue
            block_starts, previous_block = game.block != previous_block, game.block
            if gate and block_starts:
                scores = [
                    s for s in map(game_score_for_a, (r for r in rows if r.get("status") == "ok")) if s is not None
                ]
                decision = sprt(summarize_scores(scores), elo0=sprt_elo[0], elo1=sprt_elo[1])
                # Only stop on block boundaries so side/deck balance is kept.
                if decision.decision != "continue" and _block_complete(rows, schedule, game.block - 1):
                    stopped_by_sprt = decision.to_dict()
                    log(f"SPRT stop: {decision.decision} (LLR {decision.llr:.2f})")
                    break
            match = resolve_match(base_match, game, by_name)
            started = time.monotonic()
            try:
                outcome = dict(runner(game, match))
                status = "ok"
            except KeyboardInterrupt:
                raise
            except Exception as error:  # noqa: BLE001 - isolate engine failures per game
                outcome = {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(limit=5)}
                status = "error"
                errors += 1
            row = {**game.to_dict(), **outcome, "status": status, "wall_seconds": round(time.monotonic() - started, 3)}
            rows.append(row)
            handle.write(json.dumps(row) + "\n")
            log(
                json.dumps(
                    {
                        k: row.get(k)
                        for k in ("game_id", "model_a", "model_b", "deck_a", "deck_b", "a_owner", "winner", "status")
                    }
                )
            )
            if errors > max_errors:
                log(f"stopping: {errors} failed games exceed --max-errors {max_errors}")
                break
    report = build_report(rows, sprt_elo=sprt_elo)
    report["scheduled_games"] = len(schedule)
    report["stopped_by_sprt"] = stopped_by_sprt
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "report.txt").write_text(format_report(report) + "\n", encoding="utf-8")
    return report


def write_schedule(schedule: Sequence[ScheduledGame], decks: Sequence[DeckSpec], path: Path) -> None:
    payload = {
        "schema": EVAL_SCHEDULE_SCHEMA,
        "decks": [deck.to_dict() for deck in decks],
        "games": [game.to_dict() for game in schedule],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _block_complete(rows: Sequence[Mapping[str, Any]], schedule: Sequence[ScheduledGame], block: int) -> bool:
    if block < 0:
        return True
    wanted = {game.game_id for game in schedule if game.block == block}
    done = {row["game_id"] for row in rows if row.get("status") == "ok"}
    return wanted <= done


def deck_pool_from_focus_file(path: str | Path) -> tuple[DeckSpec, ...]:
    """Convert a ``v4-ppo-focus-decks.v1`` file (RoyaleAPI card keys) to deck specs."""

    from .deck_pool import deck_identity_from_keys

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    decks = []
    for index, item in enumerate(raw["decks"], start=1):
        identity = deck_identity_from_keys(item["card_keys"], item.get("tower_key", "tower-princess"), uses=1)
        decks.append(
            DeckSpec(
                name=f"rank{item.get('rank', index):02d}",
                cards=identity.card_ids,
                form_availability=identity.form_availability,
                tower_troop_id=identity.tower_troop_id,
            )
        )
    return tuple(decks)


def _load_decks(path: Path | None) -> tuple[DeckSpec, ...]:
    from ...match_factory import ROYAL_GIANT_DECK, STANDARD_DECK

    if path is None:
        return (DeckSpec("standard", STANDARD_DECK), DeckSpec("royal-giant", ROYAL_GIANT_DECK))
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping) and raw.get("schema") == "v4-ppo-focus-decks.v1":
        return deck_pool_from_focus_file(path)
    return load_deck_pool(path)


def native_runner_factory(checkpoints: Mapping[str, str], *, host, port, device, sample, force_first_action_tick):
    """Build a GameRunner that plays real headless matches on a local engine."""

    import torch

    from ...battle_env import BattleEnvV1
    from ...cr_native_env import NativeClashEnv
    from ...rich_telemetry_adapter import RuntimeEffectCatalog
    from .evaluate import play_match
    from .factory import production_semantic_bundle
    from .policy_session import load_policy_v4

    torch.set_num_threads(1)
    policies = {label: load_policy_v4(path, device=device) for label, path in checkpoints.items()}
    bundle = production_semantic_bundle()
    native = NativeClashEnv(host, port, timeout=30.0)
    native.public_card_play_events_from_combat_ring = True
    environment = BattleEnvV1(
        native=native,
        card_catalog=bundle.native_card_catalog,
        semantic_subset_contract=bundle.semantic_subset_contract,
        effect_catalog=RuntimeEffectCatalog.from_catalog(bundle.native_effect_catalog),
    )

    def run(game: ScheduledGame, match: Any) -> dict[str, Any]:
        models = {game.a_owner: policies[game.model_a].model, 1 - game.a_owner: policies[game.model_b].model}
        sessions: list = []
        try:
            outcome = play_match(
                environment,
                models,
                match,
                device=device,
                sample=sample,
                force_first_action_tick=force_first_action_tick,
                on_sessions=sessions.extend,
            )
        except BaseException:
            for session in sessions:
                with contextlib.suppress(Exception):
                    session.end_episode()
            raise
        terminal = outcome.terminal
        winner = None if terminal.winner is None else ("a" if terminal.winner == game.a_owner else "b")
        return {
            "winner": winner,
            "terminated": outcome.terminated,
            "truncated": outcome.truncated,
            "tick": outcome.tick,
            "terminal": terminal.to_dict(),
            "actions_by_owner": outcome.actions_by_owner,
            "rejected_actions": outcome.rejected_actions,
            "first_action_tick_by_owner": outcome.first_action_tick_by_owner,
            "full_elixir_ticks_by_owner": outcome.full_elixir_ticks_by_owner,
            "forced_actions_by_owner": outcome.forced_actions_by_owner,
        }

    identity = {
        label: {"path": str(p.checkpoint_path), "sha256": p.checkpoint_sha256, "checkpoint_id": p.checkpoint_id}
        for label, p in policies.items()
    }
    return run, identity, native.close_transport


def _parse_checkpoints(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        label, sep, path = value.partition("=")
        if not sep:
            label, path = Path(value).stem, value
        if not label or label in result:
            raise SystemExit(f"duplicate or empty checkpoint label: {value!r}")
        result[label] = path
    return result


def main(argv: Sequence[str] | None = None) -> int:
    from ...cr_native_env import DEFAULT_HOST, DEFAULT_PORT
    from ...match_factory import MatchConfig

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", action="append", required=True, help="LABEL=PATH (repeat, at least two)")
    parser.add_argument("--decks", type=Path, help="deck pool JSON (DeckSpec list or v4-ppo-focus-decks.v1)")
    parser.add_argument("--fixed-deck-a", help="model A always plays this deck (specialist evaluation)")
    parser.add_argument("--blocks-per-pair", type=int, default=8)
    parser.add_argument("--match-config", type=Path, help="base MatchConfig JSON (levels, mode, arena)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--schedule-seed", type=int, default=0, help="RNG seed for deck-pair sampling")
    parser.add_argument("--sprt", nargs=2, type=float, metavar=("ELO0", "ELO1"))
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--force-first-action-tick", type=int)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="write schedule.json and exit")
    args = parser.parse_args(argv)
    checkpoints = _parse_checkpoints(args.checkpoint)
    if len(checkpoints) < 2:
        parser.error("at least two --checkpoint values are required")
    decks = _load_decks(args.decks)
    config = json.loads(args.match_config.read_text(encoding="utf-8")) if args.match_config else {}
    for key in ("owner0_account_id", "owner1_account_id"):
        config.pop(key, None)
    base_match = MatchConfig(**config)
    schedule = mirrored_schedule(
        list(checkpoints),
        decks,
        blocks_per_pair=args.blocks_per_pair,
        base_seed=args.seed,
        fixed_deck_a=args.fixed_deck_a,
        rng_seed=args.schedule_seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for game in schedule:  # validate every deck assignment without an engine
            resolve_match(base_match, game, {deck.name: deck for deck in decks})
        write_schedule(schedule, decks, args.output / "schedule.json")
        print(f"{len(schedule)} games scheduled; see {args.output / 'schedule.json'}")
        return 0
    runner, identity, close = native_runner_factory(
        checkpoints,
        host=args.host,
        port=args.port,
        device=args.device,
        sample=args.sample,
        force_first_action_tick=args.force_first_action_tick,
    )
    (args.output / "identity.json").write_text(
        json.dumps(
            {
                "checkpoints": identity,
                "sample": args.sample,
                "force_first_action_tick": args.force_first_action_tick,
                "base_match": {k: v for k, v in config.items()},
                "seed": args.seed,
                "schedule_seed": args.schedule_seed,
                "blocks_per_pair": args.blocks_per_pair,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        report = run_schedule(
            schedule,
            decks,
            base_match,
            runner,
            args.output,
            sprt_elo=tuple(args.sprt) if args.sprt else None,
            max_errors=args.max_errors,
        )
    finally:
        close()
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
