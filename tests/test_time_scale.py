"""Duration scaling preserves poses and committed wall-clock anchors."""
# ruff: noqa: D103

from unittest.mock import AsyncMock

import numpy as np
import pyarrow as pa
import pytest

from dora_openarm_actions_executor import main as executor
from dora_openarm_actions_executor.trajectory import Chunk, TrajectoryScheduler, scale_interval_ns


def event(values=(0, 10, 20), **metadata):
    return {"value": pa.array([[float(v)] * 16 for v in values], type=pa.list_(pa.float32())),
            "metadata": {"interval": 100, "chunk_id": "test", **metadata}}


@pytest.mark.parametrize("scale,interval", [(0.5, 50), (1.0, 100), (2.0, 200)])
def test_untimed_duration_changes_without_moving_or_cropping_knots(scale, interval):
    incoming = event(timestamp=1, observation_timestamp_ns=1)
    chunk = Chunk.from_event(incoming, time_scale=scale)
    assert chunk.interval_ns == interval and chunk.start_ns == chunk.takeover_ns == 0
    assert chunk.metadata["interval"] == incoming["metadata"]["interval"] == 100
    scheduler = TrajectoryScheduler()
    scheduler.adopt(chunk, 10_000, 25, "switch", None, True)
    positions = scheduler.active.positions[:, 0]
    assert len(positions) == 2 * interval // 25 + 1
    np.testing.assert_array_equal(positions[::interval // 25], [0, 10, 20])


def test_first_target_and_takeover_are_not_multiplied():
    epoch = 1_700_000_000_000_000_000
    incoming = event(action_origin_timestamp_ns=epoch, action_window_start=3,
                     takeover_timestamp_ns=epoch + 350, max_lateness_ns=90)
    chunk = Chunk.from_event(incoming, time_scale=2)
    assert chunk.start_ns == epoch + 300
    assert chunk.takeover_ns == epoch + 350
    assert chunk.interval_ns == 200 and chunk.max_lateness_ns == 90
    scheduler = TrajectoryScheduler()
    assert scheduler.submit(chunk) is None
    assert scheduler.take(epoch + 349) == (None, None)
    assert scheduler.take(epoch + 400) == (chunk, None)
    scheduler.adopt(chunk, epoch + 400, 100, "switch", None, True)
    np.testing.assert_array_equal(scheduler.active.positions[:, 0], [5, 10, 15, 20])


def test_expiry_uses_scaled_horizon_and_lateness_keeps_wall_clock_units():
    incoming = event(takeover_timestamp_ns=1000)
    slow = Chunk.from_event(incoming, time_scale=2)
    fast = Chunk.from_event(incoming, time_scale=.5)
    scheduler = TrajectoryScheduler()
    scheduler.submit(slow)
    assert scheduler.take(1250) == (slow, None)
    scheduler.submit(fast)
    assert scheduler.take(1250) == (fast, "selected window expired")
    late = Chunk.from_event(event(takeover_timestamp_ns=1000, max_lateness_ns=50), time_scale=2)
    scheduler.submit(late)
    assert scheduler.take(1051) == (late, "late beyond tolerance")


def test_blend_does_not_scale_old_remaining_trajectory_twice():
    scheduler = TrajectoryScheduler()
    old = Chunk.from_event(event((0, 2, 4), chunk_id="old"), time_scale=2)
    scheduler.adopt(old, 1000, 100, "blend", None, True)
    scheduler.cursor = 1
    new = Chunk.from_event(event((10, 20, 30), chunk_id="new"), time_scale=2)
    scheduler.adopt(new, 1100, 100, "blend", None, True)
    assert scheduler.blend_steps == 4
    np.testing.assert_allclose(scheduler.active.positions[:, 0], [1, 19/3, 43/3, 25, 30], atol=1e-6)
    assert scheduler.active.interval_ns == 100


def test_fixed_blend_duration_stays_in_wall_clock_milliseconds():
    scheduler = TrajectoryScheduler()
    scheduler.adopt(Chunk.from_event(event((0, 2, 4)), time_scale=2), 1000, 100, "blend", 100, True)
    scheduler.cursor = 1
    scheduler.adopt(Chunk.from_event(event((10, 20, 30)), time_scale=2), 1100, 100, "blend", 100, True)
    assert scheduler.blend_steps == 2
    np.testing.assert_array_equal(scheduler.active.positions[:, 0], [1, 15, 20, 25, 30])


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_invalid_scales_fail_before_playback(scale):
    with pytest.raises(ValueError, match="positive and finite"):
        Chunk.from_event(event(), time_scale=scale)


def test_nanosecond_rounding_and_identity_precision():
    assert scale_interval_ns(33_333_333, 1.5) == 50_000_000
    assert scale_interval_ns(2**53 + 1, 1.0) == 2**53 + 1
    with pytest.raises(ValueError, match="one nanosecond"):
        scale_interval_ns(100, 1e-9)
    with pytest.raises(ValueError, match="int64"):
        scale_interval_ns(100, 1e300)


@pytest.mark.parametrize("env,cli,expected", [(None, [], 1.0), ("2", [], 2.0), ("2", ["--time-scale", ".5"], .5)])
def test_cli_overrides_environment(monkeypatch, env, cli, expected):
    monkeypatch.delenv("ACTION_TIME_SCALE", raising=False)
    if env is not None:
        monkeypatch.setenv("ACTION_TIME_SCALE", env)
    run = AsyncMock()
    monkeypatch.setattr(executor, "_main_async", run)
    monkeypatch.setattr("sys.argv", ["executor", *cli])
    executor.main()
    assert run.await_args.kwargs["time_scale"] == expected


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_cli_scale_does_not_start_node(monkeypatch, value):
    run = AsyncMock()
    monkeypatch.setattr(executor, "_main_async", run)
    monkeypatch.setattr("sys.argv", ["executor", "--time-scale", value])
    with pytest.raises(SystemExit):
        executor.main()
    run.assert_not_called()
