from __future__ import annotations

import math
from typing import Any, Callable, Protocol

from .datatypes import EpisodeResult, EpisodeTrajectory


class RewardProvider(Protocol):
    def compute(self, trajectory: EpisodeTrajectory) -> dict[str, Any]: ...


class FunctionRewardProvider:
    def __init__(self, func: Callable[[EpisodeTrajectory], dict[str, Any]]):
        self._func = func

    def compute(self, trajectory: EpisodeTrajectory) -> dict[str, Any]:
        return self._func(trajectory)


class RewardWorker:
    def compute(self, trajectory: EpisodeTrajectory, provider: RewardProvider) -> EpisodeResult:
        try:
            result = provider.compute(trajectory)
        except Exception as exc:
            raise RuntimeError(
                f"Reward provider failed for episode '{trajectory.episode_id}'",
            ) from exc

        if not isinstance(result, dict):
            raise TypeError(
                f"Reward payload must be a dict, got {type(result).__name__}",
            )
        if "agent_rewards" not in result:
            raise ValueError("Reward payload missing required key: 'agent_rewards'")
        if "final_reward" not in result:
            raise ValueError("Reward payload missing required key: 'final_reward'")

        agent_rewards = result["agent_rewards"]
        final_reward = result["final_reward"]

        if not isinstance(agent_rewards, dict):
            raise TypeError(
                f"Reward payload key 'agent_rewards' must be a dict, got {type(agent_rewards).__name__}",
            )
        self._validate_agent_rewards(agent_rewards)

        if final_reward is not None and not self._is_finite_number(final_reward):
            raise TypeError(
                "Reward payload key 'final_reward' must be a finite int, finite float, or None",
            )

        return EpisodeResult(
            trajectory=trajectory,
            rewards=agent_rewards,
            final_reward=final_reward,
            metadata={},
        )

    @staticmethod
    def _validate_agent_rewards(agent_rewards: dict[str, Any]) -> None:
        for role, value in agent_rewards.items():
            if RewardWorker._is_finite_number(value):
                continue
            if isinstance(value, list):
                if all(RewardWorker._is_finite_number(item) for item in value):
                    continue
                raise TypeError(
                    f"Reward payload agent_rewards['{role}'] must contain only finite int/float values",
                )
            raise TypeError(
                f"Reward payload agent_rewards['{role}'] must be int, float, or list[int|float]",
            )

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        return math.isfinite(float(value))
