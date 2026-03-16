"""AKG Kernel Reward Provider — computes multi-objective reward from an EpisodeTrajectory.

Reward formula:
    reward = α · r_correct + β · r_perf · r_correct + γ · r_iter

Where:
  r_correct  — 1.0 if KernelVerifier passes, 0.0 otherwise
  r_perf     — speedup vs reference (Phase 2, disabled by default)
  r_iter     — 1.0 - n_turns / max_turns  (efficiency bonus)

The verifier is injected via ``verifier_factory`` to allow mocking in tests.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Callable

from ascend_agent_rl.agent_trajectory_engine.datatypes import EpisodeTrajectory

logger = logging.getLogger(__name__)

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def _extract_code(response_text: str) -> str:
    """Extract code from a Markdown code block, or return the raw text."""
    match = _CODE_BLOCK_RE.search(response_text)
    if match:
        return match.group(1).strip()
    return response_text.strip()


class AKGKernelRewardProvider:
    """Multi-objective reward provider for AKG kernel generation.

    Args:
        alpha:            Weight for correctness reward (default 1.0).
        beta:             Weight for performance reward (default 0.3).
        gamma:            Weight for iteration-efficiency reward (default 0.1).
        enable_profiling: Whether to run the profiler for r_perf (default False).
        max_turns:        Maximum expected turns per episode (default 5).
        verifier_factory: Callable[[] -> KernelVerifier]. Required in production;
                          can be replaced with a mock in tests.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.3,
        gamma: float = 0.1,
        enable_profiling: bool = False,
        max_turns: int = 5,
        verifier_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.enable_profiling = enable_profiling
        self.max_turns = max_turns
        self._verifier_factory = verifier_factory

    def compute(self, trajectory: EpisodeTrajectory) -> dict[str, Any]:
        """Compute reward for one episode.

        Returns:
            {"agent_rewards": {role: float, ...}, "final_reward": float}
        """
        kernel_gen_turns = trajectory.agent_trajectories.get("kernel_gen", [])

        if not kernel_gen_turns:
            logger.warning(
                "[AKGKernelRewardProvider] No kernel_gen turns in trajectory %s",
                trajectory.episode_id,
            )
            final_reward = 0.0
            return {
                "agent_rewards": {
                    role: final_reward
                    for role in trajectory.agent_trajectories
                },
                "final_reward": final_reward,
            }

        last_response = kernel_gen_turns[-1].response_text
        code = _extract_code(last_response)

        r_correct = self._compute_correctness(code)

        r_perf = 0.0
        if r_correct and self.enable_profiling:
            r_perf = self._compute_performance(code)

        n_turns = len(kernel_gen_turns)
        r_iter = max(0.0, 1.0 - n_turns / self.max_turns)

        final_reward = (
            self.alpha * r_correct
            + self.beta * r_perf * r_correct
            + self.gamma * r_iter
        )

        logger.debug(
            "[AKGKernelRewardProvider] episode=%s r_correct=%.2f r_perf=%.2f "
            "r_iter=%.2f final=%.3f",
            trajectory.episode_id, r_correct, r_perf, r_iter, final_reward,
        )

        return {
            "agent_rewards": {
                role: final_reward for role in trajectory.agent_trajectories
            },
            "final_reward": final_reward,
        }

    def _compute_correctness(self, code: str) -> float:
        if not code or self._verifier_factory is None:
            logger.warning(
                "[AKGKernelRewardProvider] No verifier_factory configured; "
                "returning r_correct=0.0"
            )
            return 0.0
        try:
            verifier = self._verifier_factory()
            loop = asyncio.new_event_loop()
            try:
                success, log = loop.run_until_complete(verifier.run(code))
            finally:
                loop.close()
            return 1.0 if success else 0.0
        except Exception as exc:
            logger.warning(
                "[AKGKernelRewardProvider] Verifier raised %s: %s", type(exc).__name__, exc
            )
            return 0.0

    def _compute_performance(self, code: str) -> float:
        """Phase 2: compute speedup ratio. Not implemented yet."""
        return 0.0
