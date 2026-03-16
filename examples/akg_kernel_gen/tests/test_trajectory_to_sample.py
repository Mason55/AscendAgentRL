"""Unit tests for trajectory_to_sample — no NPU required."""
import pytest
from unittest.mock import MagicMock

from ascend_agent_rl.agent_trajectory_engine.datatypes import (
    EpisodeResult,
    EpisodeTrajectory,
    TurnData,
)

from examples.akg_kernel_gen.ascendrl_glue.trajectory_to_sample import (
    episode_to_sample_last_turn,
)


def _make_turn(
    role: str = "kernel_gen",
    turn_index: int = 0,
    response_text: str = "def foo(): pass",
    token_ids: list[int] | None = None,
    messages: list[dict] | None = None,
) -> TurnData:
    return TurnData(
        agent_role=role,
        turn_index=turn_index,
        messages=messages or [{"role": "user", "content": "Write code"}],
        response_text=response_text,
        token_ids=token_ids or [10, 20, 30],
        logprobs=None,
        finish_reason="stop",
        timestamp=0.0,
    )


def _make_episode(
    turns: list[TurnData] | None = None,
    final_reward: float = 1.0,
    status: str = "success",
) -> EpisodeResult:
    if turns is None:
        turns = [_make_turn()]
    trajectory = EpisodeTrajectory(
        episode_id="ep-001",
        agent_trajectories={"kernel_gen": turns},
        metadata={"op_name": "1_relu"},
    )
    return EpisodeResult(
        trajectory=trajectory,
        rewards={"kernel_gen": final_reward},
        final_reward=final_reward,
        metadata={"exit_code": 0},
        status=status,
    )


# ── basic last_turn conversion ────────────────────────────────────────────────

def test_basic_conversion_produces_required_fields():
    episode = _make_episode()
    result = episode_to_sample_last_turn(episode)
    assert "prompt" in result
    assert "response" in result
    assert "tokens" in result
    assert "response_length" in result
    assert "reward" in result
    assert "loss_mask" in result
    assert "status" in result
    assert "metadata" in result


def test_response_matches_last_turn():
    turn = _make_turn(response_text="class ModelNew: pass")
    episode = _make_episode(turns=[turn])
    result = episode_to_sample_last_turn(episode)
    assert result["response"] == "class ModelNew: pass"


def test_reward_from_episode():
    episode = _make_episode(final_reward=0.75)
    result = episode_to_sample_last_turn(episode)
    assert result["reward"] == pytest.approx(0.75)


def test_loss_mask_length_matches_response_length():
    turn = _make_turn(token_ids=[1, 2, 3, 4, 5])
    episode = _make_episode(turns=[turn])
    result = episode_to_sample_last_turn(episode)
    assert len(result["loss_mask"]) == result["response_length"]
    assert all(m == 1 for m in result["loss_mask"])


# ── status mapping ────────────────────────────────────────────────────────────

def test_success_maps_to_completed():
    episode = _make_episode(status="success")
    result = episode_to_sample_last_turn(episode)
    assert result["status"] == "completed"


def test_failed_maps_to_failed():
    episode = _make_episode(status="failed")
    result = episode_to_sample_last_turn(episode)
    assert result["status"] == "failed"


def test_timeout_maps_to_truncated():
    episode = _make_episode(status="timeout")
    result = episode_to_sample_last_turn(episode)
    assert result["status"] == "truncated"


# ── empty trajectory ──────────────────────────────────────────────────────────

def test_no_turns_returns_failed_sample():
    trajectory = EpisodeTrajectory(
        episode_id="ep-empty",
        agent_trajectories={"kernel_gen": []},
    )
    episode = EpisodeResult(
        trajectory=trajectory,
        rewards={},
        final_reward=0.0,
        status="success",
    )
    result = episode_to_sample_last_turn(episode)
    assert result["status"] == "failed"
    assert result["reward"] == 0.0
    assert result["response"] == ""


def test_wrong_role_returns_failed_sample():
    trajectory = EpisodeTrajectory(
        episode_id="ep-wrong",
        agent_trajectories={"kernel_designer": [_make_turn(role="kernel_designer")]},
    )
    episode = EpisodeResult(
        trajectory=trajectory,
        rewards={},
        final_reward=0.0,
        status="success",
    )
    result = episode_to_sample_last_turn(episode, target_role="kernel_gen")
    assert result["status"] == "failed"


# ── multi-turn picks last ────────────────────────────────────────────────────

def test_multi_turn_uses_last():
    turns = [
        _make_turn(turn_index=0, response_text="attempt 1"),
        _make_turn(turn_index=1, response_text="attempt 2"),
        _make_turn(turn_index=2, response_text="final answer"),
    ]
    episode = _make_episode(turns=turns)
    result = episode_to_sample_last_turn(episode)
    assert result["response"] == "final answer"


# ── metadata preservation ─────────────────────────────────────────────────────

def test_metadata_contains_episode_info():
    episode = _make_episode()
    result = episode_to_sample_last_turn(episode)
    meta = result["metadata"]
    assert meta["episode_id"] == "ep-001"
    assert meta["op_name"] == "1_relu"
    assert meta["exit_code"] == 0
    assert meta["target_role"] == "kernel_gen"
    assert meta["turn_count"] == 1


# ── tokenizer integration ────────────────────────────────────────────────────

def test_with_mock_tokenizer():
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda text, **kwargs: list(range(len(text)))

    turn = _make_turn(
        messages=[{"role": "user", "content": "hi"}],
        response_text="code",
    )
    episode = _make_episode(turns=[turn])
    result = episode_to_sample_last_turn(episode, tokenizer=mock_tokenizer)

    assert mock_tokenizer.encode.call_count == 2
    assert result["response_length"] == 4  # len("code")
    assert len(result["loss_mask"]) == 4


def test_max_response_tokens_truncates_tokenized_response():
    mock_tokenizer = MagicMock()

    def _encode(text, **kwargs):
        return list(range(len(text)))

    def _decode(token_ids, **kwargs):
        return "x" * len(token_ids)

    mock_tokenizer.encode.side_effect = _encode
    mock_tokenizer.decode.side_effect = _decode

    turn = _make_turn(
        messages=[{"role": "user", "content": "hi"}],
        response_text="abcdef",
    )
    episode = _make_episode(turns=[turn])
    result = episode_to_sample_last_turn(
        episode,
        tokenizer=mock_tokenizer,
        max_response_tokens=3,
    )

    assert result["response"] == "xxx"
    assert result["response_length"] == 3
    assert len(result["loss_mask"]) == 3
