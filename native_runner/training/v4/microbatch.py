"""Memory-bounded IL updates: micro-batching, group prefetch and backend choice.

A 12M-parameter V4 policy is small, but TBPTT activations scale with
``owner sequences x time steps`` (about 12 MB per frame under bf16 autocast).
Splitting one optimizer update into micro-batches keeps the effective batch of a
multi-GPU run while bounding the peak activation memory of an 8-12 GB consumer
GPU.  Every action head is normalized by its *global* population, which only
depends on expert labels, so the accumulated gradient equals the full-batch
gradient (up to floating-point summation order).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Generic, TypeVar

import torch
from torch import Tensor, nn

from .imitation import IL_LOSS_HEADS, ILLossWeightsV4, ILSequenceV4, il_loss_counts, imitation_loss
from .learning import RecurrentEvaluationV4
from .tensors import RecurrentPolicyStateV4

_T = TypeVar("_T")


def narrow_il_sequence(sequence: ILSequenceV4, start: int, length: int) -> ILSequenceV4:
    """Zero-copy batch slice ``[:, start:start+length]`` of a chronological IL batch."""

    if start < 0 or length <= 0 or start + length > sequence.batch_size:
        raise ValueError("IL batch slice is outside the sequence")
    return replace(
        sequence,
        observations=tuple(item.narrow_batch(start, length) for item in sequence.observations),
        actions=tuple(item.narrow_batch(start, length) for item in sequence.actions),
        episode_start=sequence.episode_start[:, start : start + length],
        valid_mask=sequence.valid_mask[:, start : start + length],
        returns=sequence.returns[:, start : start + length],
        gate_loss_mask=None if sequence.gate_loss_mask is None else sequence.gate_loss_mask[:, start : start + length],
        value_loss_mask=(
            None if sequence.value_loss_mask is None else sequence.value_loss_mask[:, start : start + length]
        ),
    )


def micro_batch_bounds(batch_size: int, micro_batches: int) -> tuple[tuple[int, int], ...]:
    """Contiguous, nearly equal ``(start, length)`` slices; never empty."""

    if batch_size <= 0 or micro_batches <= 0:
        raise ValueError("batch size and micro-batch count must be positive")
    count = min(batch_size, micro_batches)
    base, extra = divmod(batch_size, count)
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(count):
        length = base + (1 if index < extra else 0)
        bounds.append((start, length))
        start += length
    return tuple(bounds)


@dataclass(slots=True)
class MicroBatchStepV4:
    final_state: RecurrentPolicyStateV4
    metrics: dict[str, float]
    finite: bool


def micro_batched_il_backward(
    forward: Callable[[ILSequenceV4, RecurrentPolicyStateV4], RecurrentEvaluationV4],
    sequence: ILSequenceV4,
    state: RecurrentPolicyStateV4,
    *,
    micro_batches: int,
    weights: ILLossWeightsV4,
    world_size: int = 1,
    all_reduce_sum: Callable[[Tensor], None] | None = None,
    no_sync: Callable[[], contextlib.AbstractContextManager[object]] | None = None,
    autocast_dtype: torch.dtype | None = None,
    scaler: torch.amp.GradScaler | None = None,
    gate_act_weight: float,
    delay_neighbor_weight: float,
) -> MicroBatchStepV4:
    """Run forward/backward over micro-batches; the caller clips and steps.

    ``all_reduce_sum`` sums a tensor across data-parallel ranks in place (omit
    for a single process).  ``no_sync`` is ``DistributedDataParallel.no_sync``
    so that only the final micro-batch triggers the gradient all-reduce.
    The loss of each micro-batch is ``sum_h w_h * mean_h * n_h * world / N_h``
    where ``n_h`` is its local and ``N_h`` the global population of head ``h``;
    DDP's gradient averaging divides ``world`` back out.
    """

    device = sequence.valid_mask.device
    bounds = micro_batch_bounds(sequence.batch_size, micro_batches)
    slices = [narrow_il_sequence(sequence, start, length) for start, length in bounds]
    local_counts = torch.stack([il_loss_counts(item) for item in slices])  # [M,6]
    global_counts = local_counts.sum(dim=0)
    if all_reduce_sum is not None:
        all_reduce_sum(global_counts)
    safe_counts = global_counts.clamp_min(1.0)
    head_weights = torch.tensor(
        [getattr(weights, loss_name) for loss_name, _count, _mask in IL_LOSS_HEADS], device=device
    )
    loss_sums = torch.zeros(len(IL_LOSS_HEADS), device=device, dtype=torch.float32)
    finite = torch.ones((), device=device, dtype=torch.bool)
    final_hidden: list[Tensor] = []
    final_cell: list[Tensor] = []
    for index, ((start, length), item) in enumerate(zip(bounds, slices, strict=True)):
        last = index + 1 == len(slices)
        context = contextlib.nullcontext() if last or no_sync is None else no_sync()
        with context:
            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype or torch.bfloat16, enabled=autocast_dtype is not None
            ):
                evaluation = forward(item, state.narrow_batch(start, length))
                losses = imitation_loss(
                    item,
                    evaluation,
                    weights=weights,
                    gate_act_weight=gate_act_weight,
                    delay_neighbor_weight=delay_neighbor_weight,
                )
            means = torch.stack([getattr(losses, loss_name).float() for loss_name, _c, _m in IL_LOSS_HEADS])
            scale = local_counts[index] * float(world_size) / safe_counts
            present = (global_counts > 0).to(means.dtype)
            loss = (head_weights * means * scale * present).sum()
            finite &= torch.isfinite(loss)
            (loss if scaler is None else scaler.scale(loss)).backward()
        loss_sums += means.detach() * local_counts[index]
        final_hidden.append(evaluation.final_state.hidden.detach())
        final_cell.append(evaluation.final_state.cell.detach())
    if all_reduce_sum is not None:
        all_reduce_sum(loss_sums)
    metrics: dict[str, float] = {}
    for head_index, (loss_name, count_name, _mask) in enumerate(IL_LOSS_HEADS):
        count = float(global_counts[head_index].item())
        metrics[loss_name] = float((loss_sums[head_index] / global_counts[head_index]).item()) if count else 0.0
        metrics[count_name] = count
    final_state = RecurrentPolicyStateV4(hidden=torch.cat(final_hidden), cell=torch.cat(final_cell))
    return MicroBatchStepV4(final_state=final_state, metrics=metrics, finite=bool(finite.item()))


def grad_scaler_for(autocast_dtype: torch.dtype | None, device: torch.device) -> torch.amp.GradScaler | None:
    """FP16 needs dynamic loss scaling; BF16 and FP32 do not."""

    if autocast_dtype is torch.float16:
        return torch.amp.GradScaler(device.type)
    return None


def resolve_dist_backend(requested: str, device: torch.device) -> str:
    """``auto`` picks NCCL when built (Linux CUDA), otherwise Gloo (e.g. Windows)."""

    if requested not in {"auto", "nccl", "gloo"}:
        raise ValueError(f"unknown distributed backend {requested!r}")
    if requested != "auto":
        return requested
    if device.type == "cuda" and torch.distributed.is_nccl_available():
        return "nccl"
    return "gloo"


class PrefetchIterator(Generic[_T]):
    """Load item ``i+depth`` on a worker thread while the caller uses item ``i``.

    Cache shard loading is disk/zstd/numpy work that releases the GIL, so it
    overlaps the GPU-bound training of the current group.  ``depth=0`` loads
    synchronously (the original behaviour).  At most ``depth`` extra groups are
    resident in host memory.
    """

    def __init__(self, keys: Sequence[object], load: Callable[[object], _T], *, depth: int = 1) -> None:
        if depth < 0:
            raise ValueError("prefetch depth must be non-negative")
        self._keys = tuple(keys)
        self._load = load
        self._depth = depth
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="il-prefetch") if depth else None
        self._pending: dict[int, Future[_T]] = {}

    def _submit(self, index: int) -> None:
        if self._executor is not None and index < len(self._keys) and index not in self._pending:
            self._pending[index] = self._executor.submit(self._load, self._keys[index])

    def __iter__(self) -> Iterator[_T]:
        try:
            for index in range(len(self._keys)):
                for ahead in range(index, index + self._depth + 1):
                    self._submit(ahead)
                future = self._pending.pop(index, None)
                yield self._load(self._keys[index]) if future is None else future.result()
        finally:
            self.close()

    def close(self) -> None:
        for future in self._pending.values():
            future.cancel()
        self._pending.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None


def module_no_sync(module: nn.Module) -> Callable[[], contextlib.AbstractContextManager[object]] | None:
    method = getattr(module, "no_sync", None)
    return method if callable(method) else None
