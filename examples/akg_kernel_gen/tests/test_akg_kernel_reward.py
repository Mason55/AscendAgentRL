"""Unit tests for AKGKernelRewardProvider — no NPU required (verifier is mocked)."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from ascend_agent_rl.agent_trajectory_engine.datatypes import EpisodeTrajectory, TurnData

from examples.akg_kernel_gen.ascendrl_glue.akg_kernel_reward import AKGKernelRewardProvider


def _make_turn(response_text: str, turn_index: int = 0) -> TurnData:
    return TurnData(
        agent_role="kernel_gen",
        turn_index=turn_index,
        messages=[],
        response_text=response_text,
        token_ids=[1, 2, 3],
        logprobs=None,
        finish_reason="stop",
        timestamp=0.0,
    )


def _make_trajectory(
    kernel_gen_turns: list[TurnData],
    extra_roles: list[str] | None = None,
) -> EpisodeTrajectory:
    agent_trajectories: dict = {"kernel_gen": kernel_gen_turns}
    for role in (extra_roles or []):
        agent_trajectories[role] = [_make_turn("some output", 0)]
    return EpisodeTrajectory(
        episode_id="test-episode",
        agent_trajectories=agent_trajectories,
    )


def _mock_verifier_factory(success: bool):
    verifier = MagicMock()
    verifier.run = AsyncMock(return_value=(success, "mock log"))
    def factory():
        return verifier
    return factory


# ── correctness reward ────────────────────────────────────────────────────────

def test_returns_required_keys():
    provider = AKGKernelRewardProvider(
        verifier_factory=_mock_verifier_factory(True)
    )
    code_block = "```python\ndef foo(): pass\n```"
    traj = _make_trajectory([_make_turn(code_block)])
    result = provider.compute(traj)
    assert "agent_rewards" in result
    assert "final_reward" in result


def test_correct_code_gives_nonzero_reward():
    provider = AKGKernelRewardProvider(
        alpha=1.0, beta=0.0, gamma=0.0,
        verifier_factory=_mock_verifier_factory(True),
    )
    traj = _make_trajectory([_make_turn("```python\npass\n```")])
    result = provider.compute(traj)
    assert result["final_reward"] == pytest.approx(1.0)


def test_incorrect_code_gives_zero_reward():
    provider = AKGKernelRewardProvider(
        alpha=1.0, beta=0.0, gamma=0.0,
        verifier_factory=_mock_verifier_factory(False),
    )
    traj = _make_trajectory([_make_turn("```python\npass\n```")])
    result = provider.compute(traj)
    assert result["final_reward"] == pytest.approx(0.0)


def test_empty_trajectory_gives_zero_reward():
    provider = AKGKernelRewardProvider(
        verifier_factory=_mock_verifier_factory(True)
    )
    traj = _make_trajectory([])
    result = provider.compute(traj)
    assert result["final_reward"] == pytest.approx(0.0)


# ── iter efficiency ───────────────────────────────────────────────────────────

def test_single_turn_maximises_iter_reward():
    """1 turn out of max_turns=5 -> r_iter = 0.8."""
    provider = AKGKernelRewardProvider(
        alpha=0.0, beta=0.0, gamma=1.0, max_turns=5,
        verifier_factory=_mock_verifier_factory(False),
    )
    traj = _make_trajectory([_make_turn("code")])
    result = provider.compute(traj)
    assert result["final_reward"] == pytest.approx(0.8)


def test_max_turns_gives_zero_iter_reward():
    provider = AKGKernelRewardProvider(
        alpha=0.0, beta=0.0, gamma=1.0, max_turns=3,
        verifier_factory=_mock_verifier_factory(False),
    )
    turns = [_make_turn("code", i) for i in range(3)]
    traj = _make_trajectory(turns)
    result = provider.compute(traj)
    assert result["final_reward"] == pytest.approx(0.0)


# ── multi-role reward ────────────────────────────────────────────────────────

def test_all_roles_receive_same_final_reward():
    provider = AKGKernelRewardProvider(
        alpha=1.0, beta=0.0, gamma=0.0,
        verifier_factory=_mock_verifier_factory(True),
    )
    traj = _make_trajectory(
        [_make_turn("```python\npass\n```")],
        extra_roles=["kernel_designer"],
    )
    result = provider.compute(traj)
    rewards = result["agent_rewards"]
    assert set(rewards.keys()) == {"kernel_gen", "kernel_designer"}
    assert rewards["kernel_gen"] == rewards["kernel_designer"]


# ── code extraction ───────────────────────────────────────────────────────────

def test_extracts_code_from_markdown_block():
    provider = AKGKernelRewardProvider(
        verifier_factory=_mock_verifier_factory(True)
    )
    response = "Here is the code:\n```python\nclass ModelNew: pass\n```\nDone."
    traj = _make_trajectory([_make_turn(response)])
    result = provider.compute(traj)
    assert "final_reward" in result


def test_falls_back_to_raw_text_when_no_code_block():
    provider = AKGKernelRewardProvider(
        verifier_factory=_mock_verifier_factory(True)
    )
    traj = _make_trajectory([_make_turn("class ModelNew: pass")])
    result = provider.compute(traj)
    assert "final_reward" in result
