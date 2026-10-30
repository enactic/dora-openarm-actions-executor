"""Completion follows both final arm commands and is emitted exactly once."""

# ruff: noqa: D103

import asyncio

import numpy as np
import pyarrow as pa
import pytest

from dora_openarm_actions_executor import main as executor


@pytest.mark.parametrize("points", [1, 3])
@pytest.mark.parametrize("upsample", [False, True])
@pytest.mark.parametrize("mode", ["async", "rtc", "stop-and-go"])
def test_completion_order_and_legacy_feedback(monkeypatch, points, upsample, mode):
    outputs = run_executor(monkeypatch, points=points, upsample=upsample, mode=mode)
    plans = [(i, values[0]) for i, (name, values, _) in enumerate(outputs) if name == "execution_plan"]
    if mode == "async":
        assert plans == []
    elif mode == "rtc":
        assert len(plans) == 1 and "execution_status" not in plans[0][1]
    else:
        assert [p["execution_status"] for _, p in plans] == ["adopted", "completed"]
        done_index, done = plans[-1]
        assert done_index == len(outputs) - 1
        assert [v[0] for v in outputs[-3:-1]] == ["move_position_right", "move_position_left"]
        assert done["chunk_id"] == done["sample_chunk_id"] == "test-chunk"
        assert done["completed_timestamp_ns"] >= outputs[-2][2]["timestamp"]
        assert outputs[-1][2]["episode_attempt_id"] == "test-attempt"
    right = [v[1][0]["qpos"] for v in outputs if v[0] == "move_position_right"]
    left = [v[1][0]["qpos"] for v in outputs if v[0] == "move_position_left"]
    assert len(right) == len(left) and len(right) >= points
    np.testing.assert_allclose(right[-1], [points - 1] * 8)
    np.testing.assert_allclose(left[-1], [points - 1] * 8)


def test_expired_sync_chunk_rejects_instead_of_deadlocking(monkeypatch):
    outputs = run_executor(monkeypatch, expired=True)
    assert len(outputs) == 1 and outputs[0][0] == "execution_plan"
    plan = outputs[0][1][0]
    assert plan["execution_status"] == "rejected"
    assert plan["sample_chunk_id"] == "test-chunk"
    assert plan["rejection_reason"] == "selected window expired"


@pytest.mark.parametrize("command", ["stop", "intervene", "quit"])
def test_cancel_mid_chunk_does_not_report_completion(monkeypatch, command):
    outputs = run_executor(monkeypatch, interrupt=command)
    plans = [v[1][0] for v in outputs if v[0] == "execution_plan"]
    assert plans[0]["execution_status"] == "adopted" and plans[-1] is None
    assert len([v for v in outputs if v[0] == "move_position_right"]) == 1


def run_executor(monkeypatch, *, points=3, upsample=False, mode="stop-and-go", expired=False, interrupt=None, time_scale=1.0):
    now, outputs = [10_000_000_000], []
    metadata = dict(chunk_id="test-chunk", episode_attempt_id="test-attempt", interval=10_000_000)
    if mode == "stop-and-go":
        metadata["inference_mode"] = mode
    elif mode == "rtc":
        metadata["based_on_chunk_id"] = ""
    if expired:
        metadata["takeover_timestamp_ns"] = 1
    events = iter([
        ("command", {"value": pa.array(["start"]), "metadata": {"episode_attempt_id": "test-attempt"}}),
        ("actions", {"value": pa.array([[float(i)] * 16 for i in range(points)], type=pa.list_(pa.float32())),
                     "metadata": metadata}),
    ])
    action_queue, command_queue = asyncio.Queue(), asyncio.Queue()

    async def next_input(*_):
        if not command_queue.empty():
            return "command", command_queue.get_nowait()
        try:
            return next(events)
        except StopIteration:
            raise asyncio.CancelledError

    async def sleep(seconds):
        now[0] += max(1, round(seconds * 1e9))
        if interrupt and not command_queue.qsize():
            command_queue.put_nowait({"value": pa.array([interrupt]), "metadata": metadata})

    class Node:
        def send_output(self, name, value, metadata):
            outputs.append((name, value.to_pylist(), dict(metadata)))

    monkeypatch.setattr(executor, "_next_input", next_input)
    monkeypatch.setattr(executor.time, "time_ns", lambda: now[0])
    monkeypatch.setattr(executor.asyncio, "sleep", sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(executor._main_executor(Node(), action_queue, command_queue, {"left", "right"},
                                           upsample, False, 250, None, "switch", time_scale=time_scale))
    return outputs


@pytest.mark.parametrize("scale", [.5, 1., 2.])
@pytest.mark.parametrize("upsample", [False, True])
@pytest.mark.parametrize("mode", ["rtc", "stop-and-go"])
def test_scaled_dispatch_completion_and_feedback_use_actual_time(monkeypatch, scale, upsample, mode):
    outputs = run_executor(monkeypatch, upsample=upsample, mode=mode, time_scale=scale)
    right = [(value[0]["qpos"], metadata) for name, value, metadata in outputs if name == "move_position_right"]
    elapsed = right[-1][1]["timestamp"] - right[0][1]["timestamp"]
    effective_interval = round(10_000_000 * scale)
    period = 4_000_000 if upsample else effective_interval
    expected_elapsed = ((2 * effective_interval + period - 1) // period) * period
    assert elapsed == expected_elapsed
    np.testing.assert_array_equal(right[0][0], [0] * 8)
    np.testing.assert_array_equal(right[-1][0], [2] * 8)
    plans = [values[0] for name, values, _ in outputs if name == "execution_plan"]
    assert plans[0]["interval_ns"] == period
    assert len(plans[0]["positions"]) == len(right)
    for plan in plans:
        if scale != 1:
            assert plan["action_time_scale"] == scale
            assert plan["action_input_interval_ns"] == 10_000_000
            assert plan["action_effective_interval_ns"] == effective_interval
    if scale != 1:
        assert right[-1][1]["action_effective_interval_ns"] == effective_interval
    else:
        assert "action_time_scale" not in right[-1][1]
    if mode == "stop-and-go":
        assert [p["execution_status"] for p in plans] == ["adopted", "completed"]
        assert plans[-1]["completed_timestamp_ns"] == right[-1][1]["timestamp"]
