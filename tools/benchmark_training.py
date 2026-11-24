"""Synthetic IL forward/backward benchmark for the V4 policy.

The benchmark needs neither the game engine nor an IL cache.  It reuses the
model-test fixture observation and pads every fixed-capacity collection to the
production capacities (48 groups, 160 children, 32 events, 6 candidates), so the
dense compute per frame matches real training.  Use it to size ``--shards-per-rank``,
``--time-steps`` and ``--micro-batches`` for a given GPU before a real run.

Examples::

    python tools/benchmark_training.py --device cpu --batch 4 --time-steps 8
    python tools/benchmark_training.py --device cuda --batch 64 --time-steps 32 --autocast bfloat16
    python tools/benchmark_training.py --device cuda --sweep 16,32,64,128 --json runs/bench.jsonl
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from native_runner.training.v4.config import ModelConfigV4  # noqa: E402
from native_runner.training.v4.imitation import ILSequenceV4, imitation_loss  # noqa: E402
from native_runner.training.v4.learning import evaluate_recurrent_sequence  # noqa: E402
from native_runner.training.v4.tensors import GATE_ACT, GATE_WAIT, ActionSequenceV4  # noqa: E402


def _pad_record(record: object, old: int, new: int) -> object:
    """Zero/False-pad every tensor field whose second axis equals ``old``."""

    if not is_dataclass(record):
        raise TypeError(type(record))
    values = {}
    for item in fields(record):
        value = getattr(record, item.name)
        if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == old and new > old:
            shape = list(value.shape)
            shape[1] = new - old
            fill = torch.zeros(shape, dtype=value.dtype)
            value = torch.cat((value, fill), dim=1)
        values[item.name] = value
    return type(record)(**values)


def _pad_children(groups: object, old: int, new: int) -> object:
    values = {}
    for item in fields(groups):
        value = getattr(groups, item.name)
        if item.name.startswith("child_") and isinstance(value, torch.Tensor) and value.ndim >= 2:
            shape = list(value.shape)
            shape[1] = new - old
            value = torch.cat((value, torch.zeros(shape, dtype=value.dtype)), dim=1)
        values[item.name] = value
    return type(groups)(**values)


def production_capacity_batch(batch_size: int, *, active_fraction: float = 0.5):
    """Fixture batch padded to production capacities, ~``active_fraction`` rows live."""

    from native_runner.tests.test_training_v4_model import _batch

    config = ModelConfigV4()
    batch = _batch(batch_size)
    groups = batch.groups
    group_rows = groups.mask.shape[1]
    child_rows = groups.child_mask.shape[1]
    groups = _pad_children(groups, child_rows, config.max_children_total)
    group_values = {}
    for item in fields(groups):
        value = getattr(groups, item.name)
        if not item.name.startswith("child_") and value.ndim >= 2 and value.shape[1] == group_rows:
            shape = list(value.shape)
            shape[1] = config.max_battle_groups - group_rows
            value = torch.cat((value, torch.zeros(shape, dtype=value.dtype)), dim=1)
        group_values[item.name] = value
    groups = type(groups)(**group_values)
    live_groups = max(group_rows, int(config.max_battle_groups * active_fraction))
    live_children = max(child_rows, int(config.max_children_total * active_fraction))
    groups.mask[:, :live_groups] = True
    groups.child_mask[:, :live_children] = True
    groups.child_group_index[:, child_rows:live_children] = torch.arange(live_children - child_rows) % live_groups
    groups.child_position[:, child_rows:] = torch.tensor([9.0, 16.0])
    groups.child_extent[:, child_rows:] = torch.tensor([9.0, 16.0, 9.0, 16.0])
    groups.position[:, group_rows:] = torch.tensor([9.0, 16.0])
    groups.extent[:, group_rows:] = torch.tensor([8.0, 15.0, 10.0, 17.0])
    events = _pad_record(batch.events, batch.events.mask.shape[1], config.max_recent_events)
    events.mask[:, : int(config.max_recent_events * active_fraction)] = True
    candidates = _pad_record(batch.candidates, batch.candidates.mask.shape[1], config.max_action_candidates)
    return replace(batch, groups=groups, events=events, candidates=candidates)


def _actions(batch_size: int, step: int) -> ActionSequenceV4:
    gate = torch.full((batch_size,), GATE_WAIT, dtype=torch.long)
    count = torch.zeros(batch_size, dtype=torch.long)
    index = torch.full((batch_size, 2), -1, dtype=torch.long)
    uid = torch.full((batch_size, 2), -1, dtype=torch.long)
    cell = torch.full((batch_size, 2), -1, dtype=torch.long)
    delay = torch.full((batch_size, 2), -1, dtype=torch.long)
    if step % 4 == 0:  # the fixture's first row has legal candidates
        gate[0], count[0], index[0, 0], uid[0, 0], cell[0, 0], delay[0, 0] = GATE_ACT, 1, 0, 101, 0, 0
    return ActionSequenceV4(
        gate=gate, micro_action_count=count, candidate_index=index, candidate_uid=uid,
        target_cell=cell, delay_offset_bin=delay,
    )


def synthetic_sequence(batch_size: int, time_steps: int, device: torch.device) -> ILSequenceV4:
    observation = production_capacity_batch(batch_size)
    start = torch.zeros(time_steps, batch_size, dtype=torch.bool)
    start[0] = True
    sequence = ILSequenceV4(
        observations=tuple(observation for _ in range(time_steps)),
        actions=tuple(_actions(batch_size, step) for step in range(time_steps)),
        episode_start=start,
        valid_mask=torch.ones(time_steps, batch_size, dtype=torch.bool),
        returns=torch.zeros(time_steps, batch_size),
    )
    return sequence.to(device)


def _peak_memory_mb(device: torch.device) -> dict[str, float]:
    if device.type == "cuda":
        return {
            "cuda_peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
            "cuda_peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
        }
    return {"host_max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}


def run_case(
    *, device: torch.device, batch: int, time_steps: int, micro_batches: int, autocast: str,
    preencode: bool, warmup: int, iterations: int,
) -> dict[str, object]:
    from native_runner.tests.test_training_v4_model import _model

    if batch % micro_batches:
        raise ValueError("batch must be divisible by micro_batches")
    torch.manual_seed(0)
    policy = _model().to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5, fused=device.type == "cuda")
    dtype = {"none": None, "bfloat16": torch.bfloat16, "float16": torch.float16}[autocast]
    scaler = torch.amp.GradScaler(device.type, enabled=dtype is torch.float16)
    micro = batch // micro_batches
    sequences = [synthetic_sequence(micro, time_steps, device) for _ in range(micro_batches)]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        for sequence in sequences:
            state = policy.initial_state(micro, device=device)
            with torch.autocast(device_type=device.type, dtype=dtype or torch.bfloat16, enabled=dtype is not None):
                evaluation = evaluate_recurrent_sequence(
                    policy, sequence.observations, sequence.actions, sequence.episode_start,
                    initial_state=state, gate_temperature=1.0, action_temperature=1.0,
                    continue_temperature=1.0, validate=False, preencode_observations=preencode,
                )
                loss = imitation_loss(sequence, evaluation).total / micro_batches
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    frames = batch * time_steps * iterations
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "cpu",
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "batch": batch,
        "time_steps": time_steps,
        "micro_batches": micro_batches,
        "autocast": autocast,
        "preencode": preencode,
        "iterations": iterations,
        "seconds_per_update": round(elapsed / iterations, 4),
        "owner_frames_s": round(frames / elapsed, 2),
        **_peak_memory_mb(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=16, help="owner sequences per optimizer update")
    parser.add_argument("--sweep", help="comma-separated batch sizes; overrides --batch")
    parser.add_argument("--time-steps", type=int, default=32)
    parser.add_argument("--micro-batches", type=int, default=1)
    parser.add_argument("--autocast", choices=("none", "bfloat16", "float16"), default="none")
    parser.add_argument("--preencode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--json", type=Path, help="append one JSON line per case")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    batches = [int(item) for item in args.sweep.split(",")] if args.sweep else [args.batch]
    for batch in batches:
        try:
            row = run_case(
                device=device, batch=batch, time_steps=args.time_steps, micro_batches=args.micro_batches,
                autocast=args.autocast, preencode=args.preencode, warmup=args.warmup, iterations=args.iterations,
            )
        except torch.OutOfMemoryError as error:
            row = {"batch": batch, "error": "out_of_memory", "detail": str(error).splitlines()[0]}
            torch.cuda.empty_cache()
        print(json.dumps(row, sort_keys=True), flush=True)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            with args.json.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
