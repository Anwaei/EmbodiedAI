import pytest

from embodied_ai.evaluation.scheduler import RecedingHorizonScheduler


def test_scheduler_executes_five_then_replans() -> None:
    scheduler = RecedingHorizonScheduler()
    scheduler.accept([[0.0] * 7 for _ in range(50)])
    assert [scheduler.pop().chunk_offset for _ in range(5)] == [0, 1, 2, 3, 4]
    assert scheduler.needs_prediction
    assert scheduler.discarded_prediction_count == 45


def test_scheduler_clips_contract_bounds_but_rejects_hard_limit() -> None:
    scheduler = RecedingHorizonScheduler()
    chunk = [[0.0] * 7 for _ in range(50)]
    chunk[0][0] = 1.2
    scheduler.accept(chunk)
    assert scheduler.pop().executed[0] == 1.0
    scheduler = RecedingHorizonScheduler()
    chunk[0][0] = 1.6
    with pytest.raises(ValueError, match="hard action limit"):
        scheduler.accept(chunk)
