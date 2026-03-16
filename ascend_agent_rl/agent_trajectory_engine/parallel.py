from __future__ import annotations

import asyncio
import logging

from .backend import InferenceBackend
from .datatypes import EpisodeResult
from .pipe import AgentPipe, AgentPipeConfig
from .reward import RewardProvider

_LOGGER = logging.getLogger(__name__)


async def parallel_rollout(
    prompts: list[str],
    reward_provider: RewardProvider,
    config: AgentPipeConfig,
    backend: InferenceBackend,
    n_samples_per_prompt: int = 1,
    max_concurrent: int | None = None,
) -> list[EpisodeResult]:
    """
    对每个 prompt 并行采样 n_samples_per_prompt 条 episode。
    max_concurrent 限制同时运行的 AgentPipe 数量（None = 不限制）。
    """
    if n_samples_per_prompt < 1:
        raise ValueError("n_samples_per_prompt must be >= 1")
    if max_concurrent is not None and max_concurrent < 1:
        raise ValueError("max_concurrent must be >= 1 when provided")
    if not prompts:
        return []

    semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent is not None else None

    async def run_one(prompt: str) -> EpisodeResult:
        pipe = AgentPipe(config=config, backend=backend)
        if semaphore is None:
            return await pipe.run(prompt=prompt, reward_provider=reward_provider)
        async with semaphore:
            return await pipe.run(prompt=prompt, reward_provider=reward_provider)

    tasks = [
        run_one(prompt)
        for prompt in prompts
        for _ in range(n_samples_per_prompt)
    ]
    gathered = await asyncio.gather(*tasks, return_exceptions=True)

    results: list[EpisodeResult] = []
    for item in gathered:
        if isinstance(item, Exception):
            _LOGGER.warning("parallel_rollout dropped failed episode: %s", item)
            continue
        if isinstance(item, BaseException):
            raise item
        results.append(item)
    return results
