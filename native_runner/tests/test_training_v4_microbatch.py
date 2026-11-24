import threading
import time

import pytest
import torch

from native_runner.tests.test_training_v4_learning import _one_action, _wait_actions
from native_runner.tests.test_training_v4_model import _batch, _model
from native_runner.training.v4.imitation import (
    IL_LOSS_HEADS,
    ILLossWeightsV4,
    ILSequenceV4,
    collate_il_sequences,
    il_loss_counts,
    imitation_loss,
)
from native_runner.training.v4.learning import evaluate_recurrent_sequence
from native_runner.training.v4.microbatch import (
    PrefetchIterator,
    grad_scaler_for,
    micro_batch_bounds,
    micro_batched_il_backward,
    narrow_il_sequence,
    resolve_dist_backend,
)


def _sequence(time_steps: int = 3) -> ILSequenceV4:
    """Four owner rows with different label populations per row."""

    rows = []
    for row in range(4):
        actions = tuple(
            _one_action(2) if (step + row) % 2 == 0 else _wait_actions(2) for step in range(time_steps)
        )
        valid = torch.ones(time_steps, 2, dtype=torch.bool)
        if row == 3:
            valid[-1] = False  # a shorter replay tail
        start = torch.zeros(time_steps, 2, dtype=torch.bool)
        start[0] = True
        rows.append(
            ILSequenceV4(
                observations=tuple(_batch(2) for _ in range(time_steps)),
                actions=actions,
                episode_start=start,
                valid_mask=valid,
                returns=torch.linspace(-1.0, 1.0, time_steps)[:, None].repeat(1, 2) * (row + 1) / 4,
            )
        )
    return collate_il_sequences(rows)


def _forward(policy):
    def run(item, state):
        return evaluate_recurrent_sequence(
            policy,
            item.observations,
            item.actions,
            item.episode_start,
            initial_state=state,
            gate_temperature=1.0,
            action_temperature=1.0,
            continue_temperature=1.0,
            validate=False,
            preencode_observations=True,
        )

    return run


def _grads(policy) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in policy.named_parameters()
        if parameter.grad is not None
    }


def test_loss_counts_match_imitation_loss_populations() -> None:
    torch.manual_seed(0)
    sequence = _sequence()
    policy = _model()
    evaluation = _forward(policy)(sequence, policy.initial_state(sequence.batch_size))
    losses = imitation_loss(sequence, evaluation)
    expected = torch.stack([getattr(losses, count).float() for _loss, count, _mask in IL_LOSS_HEADS])
    torch.testing.assert_close(il_loss_counts(sequence), expected)


def test_micro_batch_bounds_cover_the_batch_once() -> None:
    assert micro_batch_bounds(8, 3) == ((0, 3), (3, 3), (6, 2))
    assert micro_batch_bounds(2, 5) == ((0, 1), (1, 1))
    with pytest.raises(ValueError):
        micro_batch_bounds(0, 1)


def test_narrow_il_sequence_slices_every_time_field() -> None:
    sequence = _sequence()
    part = narrow_il_sequence(sequence, 2, 3)
    part.validate()
    assert part.batch_size == 3
    assert torch.equal(part.valid_mask, sequence.valid_mask[:, 2:5])
    assert torch.equal(part.actions[0].gate, sequence.actions[0].gate[2:5])


@pytest.mark.parametrize("micro_batches", [2, 3, 8])
def test_micro_batched_gradient_equals_full_batch(micro_batches: int) -> None:
    torch.manual_seed(1)
    sequence = _sequence()
    policy = _model()
    initial = policy.initial_state(sequence.batch_size)
    initial.hidden.normal_(std=0.1)
    weights = ILLossWeightsV4()

    full = micro_batched_il_backward(
        _forward(policy), sequence, initial, micro_batches=1, weights=weights,
        gate_act_weight=2.0, delay_neighbor_weight=0.1,
    )
    # The single-pass path must equal the historical one-call loss.
    reference = imitation_loss(
        sequence, _forward(policy)(sequence, initial), weights=weights, gate_act_weight=2.0, delay_neighbor_weight=0.1
    )
    for loss_name, _count, _mask in IL_LOSS_HEADS:
        assert full.metrics[loss_name] == pytest.approx(float(getattr(reference, loss_name)), rel=1e-5, abs=1e-6)
    full_grads = _grads(policy)
    policy.zero_grad(set_to_none=True)

    split = micro_batched_il_backward(
        _forward(policy), sequence, initial, micro_batches=micro_batches, weights=weights,
        gate_act_weight=2.0, delay_neighbor_weight=0.1,
    )
    split_grads = _grads(policy)

    assert full.finite and split.finite
    assert full_grads.keys() == split_grads.keys()
    for name, gradient in full_grads.items():
        torch.testing.assert_close(split_grads[name], gradient, rtol=1e-4, atol=1e-6, msg=name)
    torch.testing.assert_close(split.final_state.hidden, full.final_state.hidden, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(split.final_state.cell, full.final_state.cell, rtol=1e-5, atol=1e-6)
    for key, value in full.metrics.items():
        assert split.metrics[key] == pytest.approx(value, rel=1e-5, abs=1e-6)


def test_simulated_ranks_reduce_counts_globally() -> None:
    """Two fake ranks each holding half the rows reproduce the one-rank gradient."""

    torch.manual_seed(2)
    sequence = _sequence()
    policy = _model()
    initial = policy.initial_state(sequence.batch_size)
    weights = ILLossWeightsV4()
    micro_batched_il_backward(
        _forward(policy), sequence, initial, micro_batches=1, weights=weights,
        gate_act_weight=1.0, delay_neighbor_weight=0.0,
    )
    expected = _grads(policy)

    halves = [narrow_il_sequence(sequence, 0, 4), narrow_il_sequence(sequence, 4, 4)]
    total_counts = il_loss_counts(halves[0]) + il_loss_counts(halves[1])
    rank_grads = []
    for rank, half in enumerate(halves):
        policy.zero_grad(set_to_none=True)
        reductions = iter([total_counts, None])

        def fake_all_reduce(value: torch.Tensor) -> None:
            replacement = next(reductions, None)
            if replacement is not None:
                value.copy_(replacement)

        micro_batched_il_backward(
            _forward(policy), half, initial.narrow_batch(rank * 4, 4), micro_batches=2, weights=weights,
            world_size=2, all_reduce_sum=fake_all_reduce, gate_act_weight=1.0, delay_neighbor_weight=0.0,
        )
        rank_grads.append(_grads(policy))
    for name, gradient in expected.items():  # DDP averages rank gradients
        averaged = (rank_grads[0][name] + rank_grads[1][name]) / 2
        torch.testing.assert_close(averaged, gradient, rtol=1e-4, atol=1e-6, msg=name)


def test_no_sync_wraps_every_micro_batch_but_the_last() -> None:
    sequence = _sequence(time_steps=2)
    policy = _model()
    entered: list[int] = []

    class Recorder:
        def __enter__(self):
            entered.append(1)

        def __exit__(self, *_args):
            return False

    micro_batched_il_backward(
        _forward(policy), sequence, policy.initial_state(sequence.batch_size), micro_batches=4,
        weights=ILLossWeightsV4(), no_sync=Recorder, gate_act_weight=1.0, delay_neighbor_weight=0.0,
    )
    assert len(entered) == 3


def test_prefetch_iterator_preserves_order_and_overlaps() -> None:
    started: list[int] = []
    lock = threading.Lock()

    def load(key):
        with lock:
            started.append(key)
        time.sleep(0.05)
        return key * 10

    iterator = iter(PrefetchIterator(list(range(5)), load, depth=1))
    assert next(iterator) == 0
    time.sleep(0.08)  # "training" on group 0 while group 1 loads
    with lock:
        assert 1 in started
    assert list(iterator) == [10, 20, 30, 40]
    assert list(PrefetchIterator([1, 2], load, depth=0)) == [10, 20]


def test_prefetch_iterator_propagates_loader_errors() -> None:
    def load(key):
        if key == 1:
            raise OSError("bad shard")
        return key

    iterator = iter(PrefetchIterator([0, 1, 2], load, depth=2))
    assert next(iterator) == 0
    with pytest.raises(OSError, match="bad shard"):
        next(iterator)


def test_backend_and_scaler_selection() -> None:
    cpu = torch.device("cpu")
    assert resolve_dist_backend("auto", cpu) == "gloo"
    assert resolve_dist_backend("nccl", cpu) == "nccl"
    with pytest.raises(ValueError):
        resolve_dist_backend("mpi", cpu)
    assert grad_scaler_for(None, cpu) is None
    assert grad_scaler_for(torch.bfloat16, cpu) is None
    assert grad_scaler_for(torch.float16, cpu) is not None


def _ddp_worker(rank: int, world_size: int, port: int, output: str) -> None:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from native_runner.training.v4.microbatch import module_no_sync
    from native_runner.training.v4.train_imitation_cache import _DistributedImitationModule

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(3)
        policy = _model()
        # The fixture's empty effect/mechanic catalogs leave those encoders out of
        # the graph; without find_unused_parameters their bucket would never reduce.
        ddp = DistributedDataParallel(
            _DistributedImitationModule(policy), broadcast_buffers=False, find_unused_parameters=True
        )
        sequence = narrow_il_sequence(_sequence(), rank * 4, 4)
        state = policy.initial_state(4)
        micro_batched_il_backward(
            lambda item, item_state: ddp(item, item_state, validate=False, preencode_observations=True),
            sequence, state, micro_batches=2, weights=ILLossWeightsV4(), world_size=world_size,
            all_reduce_sum=lambda value: dist.all_reduce(value, op=dist.ReduceOp.SUM),
            no_sync=module_no_sync(ddp), gate_act_weight=1.0, delay_neighbor_weight=0.0,
        )
        if rank == 0:
            torch.save(_grads(policy), output)
    finally:
        dist.destroy_process_group()


def test_gloo_ddp_micro_batches_match_single_process(tmp_path) -> None:
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    output = tmp_path / "rank0-grads.pt"
    mp.start_processes(_ddp_worker, args=(2, port, str(output)), nprocs=2, join=True, start_method="spawn")
    distributed = torch.load(output)

    torch.manual_seed(3)
    policy = _model()
    sequence = _sequence()
    micro_batched_il_backward(
        _forward(policy), sequence, policy.initial_state(sequence.batch_size), micro_batches=1,
        weights=ILLossWeightsV4(), gate_act_weight=1.0, delay_neighbor_weight=0.0,
    )
    for name, gradient in _grads(policy).items():
        torch.testing.assert_close(distributed[name], gradient, rtol=1e-4, atol=1e-6, msg=name)
