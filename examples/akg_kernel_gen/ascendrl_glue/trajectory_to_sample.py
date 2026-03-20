"""Convert an EpisodeResult into one or more slime Sample objects.

Phase 1 strategy: ``last_turn``
  - Extract only the final turn of ``kernel_gen``
  - Build prompt from that turn's messages, response from its output
  - Tokenize prompt + response to fill ``sample.tokens``
  - Set ``loss_mask = [1] * response_length`` (train on response only)
  - Copy episode reward as sample reward

Future strategies (Phase 2/3):
  - ``aggregate_role``:  multi-turn loss mask for one role
  - ``multi_role``:      separate samples per role
"""
from __future__ import annotations

import logging
from typing import Any

from ascend_agent_rl.agent_trajectory_engine.datatypes import EpisodeResult, TurnData

logger = logging.getLogger(__name__)


def episode_to_sample_last_turn(
    episode: EpisodeResult,
    target_role: str = "kernel_gen",
    tokenizer: Any | None = None,
    max_response_tokens: int | None = None,
) -> dict[str, Any]:
    """Convert an EpisodeResult into a dict that can populate a slime Sample.

    This implements the ``last_turn`` strategy: only the final turn of
    *target_role* is used as the trainable response.

    Args:
        episode:     The completed episode with trajectory and reward.
        target_role: Which agent role to extract the training sample from.
        tokenizer:   HuggingFace tokenizer; if provided, ``tokens`` and
                     ``response_length`` are computed from actual tokenization.
                     Otherwise, ``token_ids`` from the trajectory are used.
        max_response_tokens:
                     Optional hard cap for the response portion of the sample.
                     When used with a tokenizer, ``response`` is also truncated
                     to match the retained token ids.

    Returns:
        A dict with keys matching slime ``Sample`` fields:
        ``prompt``, ``response``, ``tokens``, ``response_length``,
        ``reward``, ``loss_mask``, ``status``, ``metadata``.
    """
    trajectory = episode.trajectory
    role_turns: list[TurnData] = trajectory.agent_trajectories.get(target_role, [])

    if not role_turns:
        logger.warning(
            "[trajectory_to_sample] No turns for role=%s in episode=%s",
            target_role, trajectory.episode_id,
        )
        return _make_failed_sample(episode, target_role)

    last_turn = role_turns[-1]

    prompt_text = _build_prompt_text(last_turn.messages)
    response_text = last_turn.response_text

    if tokenizer is not None:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
        response_ids = tokenizer.encode(response_text, add_special_tokens=False)
        if max_response_tokens is not None and len(response_ids) > max_response_tokens:
            response_ids = response_ids[:max_response_tokens]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        tokens = prompt_ids + response_ids
        response_length = len(response_ids)
    elif last_turn.token_ids is not None:
        tokens = last_turn.token_ids
        if max_response_tokens is not None and len(tokens) > max_response_tokens:
            tokens = tokens[:max_response_tokens]
        response_length = len(tokens)
    else:
        tokens = []
        response_length = 0

    loss_mask = [1] * response_length

    status = _map_status(episode.status)

    final_reward = episode.final_reward if episode.final_reward is not None else 0.0

    metadata = {
        "episode_id": trajectory.episode_id,
        "op_name": trajectory.metadata.get("op_name", ""),
        "turn_count": len(role_turns),
        "exit_code": episode.metadata.get("exit_code"),
        "target_role": target_role,
    }
    if isinstance(episode.rewards, dict):
        metadata["agent_rewards"] = episode.rewards
    if episode.failure_info is not None:
        metadata["failure_info"] = episode.failure_info

    return {
        "prompt": prompt_text,
        "response": response_text,
        "tokens": tokens,
        "response_length": response_length,
        "reward": final_reward,
        "loss_mask": loss_mask,
        "status": status,
        "metadata": metadata,
    }


def _build_prompt_text(messages: list[dict[str, Any]]) -> str:
    """Reconstruct a single prompt string from OpenAI-style messages."""
    if not messages:
        return ""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    return "\n".join(parts)


def _map_status(episode_status: str) -> str:
    """Map EpisodeResult.status to slime Sample.Status value strings."""
    mapping = {
        "success": "completed",
        "failed": "failed",
        "timeout": "truncated",
        "partial": "truncated",
    }
    return mapping.get(episode_status, "completed")


def _make_failed_sample(episode: EpisodeResult, target_role: str) -> dict[str, Any]:
    """Produce a minimal sample dict for an episode with no usable turns."""
    return {
        "prompt": "",
        "response": "",
        "tokens": [],
        "response_length": 0,
        "reward": 0.0,
        "loss_mask": [],
        "status": "failed",
        "metadata": {
            "episode_id": episode.trajectory.episode_id,
            "target_role": target_role,
            "failure_info": {"reason": f"No turns found for role {target_role}"},
        },
    }
