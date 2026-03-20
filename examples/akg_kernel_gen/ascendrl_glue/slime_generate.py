"""Slime custom_generate entry point for AKG kernel generation.

This module is registered via slime's ``--custom-generate-function-path``
and is called once per sample during rollout.  Each call runs one full
AKG multi-agent episode **asynchronously** via ``AgentPipe``:

  1. Parse the KernelBench task from ``sample.prompt``
  2. Create a ``SlimeSglangBackend`` pointed at slime's SGLang router
  3. Run ``AgentPipe.run()`` → ModelMonitor → MAS subprocess (akg_rl_entry)
  4. Compute reward via ``AKGKernelRewardProvider``
  5. Map the ``kernel_gen`` last-turn trajectory back onto the slime ``Sample``

slime's rollout loop calls ``generate_akg_episode`` concurrently via
``asyncio`` — each sample gets its own ``AgentPipe`` instance, its own
``ModelMonitor`` (ephemeral port), and its own MAS subprocess, so
multiple episodes execute in parallel without blocking each other.

Only the ``kernel_gen`` role's last turn is used as the trainable response.
``kernel_designer`` is frozen (its calls are proxied by ``ModelMonitor``
to the same SGLang router, but its tokens are not included in the loss).

Usage in slime:
    --custom-generate-function-path \\
        examples.akg_kernel_gen.ascendrl_glue.slime_generate.generate_akg_episode
"""
from __future__ import annotations

import json
import logging
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PIPE_STATE: dict[str, Any] = {}


def _get_project_root() -> Path:
    """Resolve AscendAgentRL project root from this file's location."""
    return Path(__file__).resolve().parent.parent.parent.parent


def _get_mas_entry_path() -> str:
    """Return absolute path to the akg_rl_entry.py MAS subprocess."""
    return str(
        _get_project_root()
        / "examples"
        / "akg_kernel_gen"
        / "mas_entry"
        / "akg_rl_entry.py"
    )


def _get_config_template() -> dict[str, Any]:
    """Load the AKG config template YAML."""
    import yaml

    template_path = (
        _get_project_root()
        / "examples"
        / "akg_kernel_gen"
        / "configs"
        / "akg_config_template.yaml"
    )
    with open(template_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_pipe_state(args: Namespace) -> dict[str, Any]:
    """Lazily initialize the shared AgentPipe configuration and backend.

    Everything here is created once and reused across all rollout calls
    within the same worker process.
    """
    global _PIPE_STATE
    if _PIPE_STATE:
        return _PIPE_STATE

    from transformers import AutoTokenizer

    from ascend_agent_rl.agent_trajectory_engine import (
        AgentPipeConfig,
        ModelMappingEntry,
    )
    from examples.akg_kernel_gen.ascendrl_glue.slime_sglang_backend import (
        SlimeSglangBackend,
    )

    hf_path = getattr(args, "hf_checkpoint", None) or getattr(args, "model_path", None)
    if not hf_path:
        raise ValueError("Cannot determine HF model path from args")

    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)

    router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    backend = SlimeSglangBackend(
        router_url=router_url,
        tokenizer=tokenizer,
        timeout=120.0,
    )

    config_template = _get_config_template()
    mas_entry = _get_mas_entry_path()

    mas_command = (
        f"{sys.executable} {mas_entry}"
        " --config {config_path}"
        " --task {prompt}"
    )

    model_mapping = {
        "kernel_gen": ModelMappingEntry(),
        "kernel_designer": ModelMappingEntry(),
    }

    pipe_config = AgentPipeConfig(
        mas_command_template=mas_command,
        config_template=config_template,
        model_mapping=model_mapping,
        timeout=300.0,
        monitor_host="127.0.0.1",
        monitor_port=0,
    )

    _PIPE_STATE = {
        "pipe_config": pipe_config,
        "backend": backend,
        "tokenizer": tokenizer,
        "reward_provider": _make_reward_provider(),
    }
    logger.info(
        "[slime_generate] Initialized AgentPipe state: router=%s hf=%s",
        router_url,
        hf_path,
    )
    return _PIPE_STATE


def _make_reward_provider() -> Any:
    """Create the AKG kernel reward provider (code-heuristic for now)."""
    from examples.akg_kernel_gen.ascendrl_glue.akg_kernel_reward import (
        AKGKernelRewardProvider,
    )

    return AKGKernelRewardProvider(
        alpha=1.0,
        beta=0.0,
        gamma=0.1,
        enable_profiling=False,
        max_turns=5,
        verifier_factory=None,
    )


def _parse_task_prompt(prompt: str) -> dict[str, str]:
    """Extract op_name and task_desc from a JSON-encoded prompt."""
    try:
        data = json.loads(prompt)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Prompt must be valid JSON with op_name and task_desc") from exc

    if "op_name" not in data or "task_desc" not in data:
        raise ValueError("Prompt JSON must contain op_name and task_desc")

    return {"op_name": data["op_name"], "task_desc": data["task_desc"]}


def _apply_episode_sample_to_slime_sample(
    sample: Any,
    sample_data: dict[str, Any],
    rollout_log_probs: list[float] | None = None,
) -> Any:
    """Copy shared trajectory_to_sample output onto a runtime slime Sample."""
    from slime.utils.types import Sample as SlimeSample

    sample.prompt = sample_data["prompt"]
    sample.response = sample_data["response"]
    sample.tokens = sample_data["tokens"]
    sample.response_length = sample_data["response_length"]
    sample.reward = sample_data["reward"]
    sample.loss_mask = sample_data["loss_mask"]
    sample.status = SlimeSample.Status(sample_data["status"])

    merged_metadata = dict(sample.metadata or {})
    merged_metadata.update(sample_data["metadata"])
    sample.metadata = merged_metadata

    if rollout_log_probs:
        sample.rollout_log_probs = rollout_log_probs[: sample.response_length]
    else:
        sample.rollout_log_probs = [0.0] * sample.response_length

    return sample


async def generate_akg_episode(
    args: Namespace,
    sample: Any,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Any:
    """Slime custom_generate function for AKG kernel generation.

    Runs a full agent episode via AgentPipe:
    1. Parse task from sample.prompt
    2. Run AgentPipe → ModelMonitor → MAS subprocess → trajectory
    3. Extract last turn of kernel_gen for training
    4. Map trajectory data back onto the slime Sample
    """
    from ascend_agent_rl.agent_trajectory_engine import AgentPipe
    from examples.akg_kernel_gen.ascendrl_glue.trajectory_to_sample import (
        episode_to_sample_last_turn,
    )

    state = _ensure_pipe_state(args)
    pipe = AgentPipe(
        config=state["pipe_config"],
        backend=state["backend"],
    )
    tokenizer = state["tokenizer"]
    reward_provider = state["reward_provider"]

    prompt_text = sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt)
    task_info = _parse_task_prompt(prompt_text)
    task_json = json.dumps(task_info, ensure_ascii=False)

    try:
        episode = await pipe.run(
            prompt=task_json,
            reward_provider=reward_provider,
            allow_partial=True,
        )

        trajectory = episode.trajectory
        kernel_gen_turns = trajectory.agent_trajectories.get("kernel_gen", [])

        max_response_tokens = getattr(args, "rollout_max_response_len", 1024)

        if kernel_gen_turns:
            last_turn = kernel_gen_turns[-1]
            sample_data = episode_to_sample_last_turn(
                episode,
                target_role="kernel_gen",
                tokenizer=tokenizer,
                max_response_tokens=max_response_tokens,
            )
            sample = _apply_episode_sample_to_slime_sample(
                sample,
                sample_data,
                rollout_log_probs=last_turn.logprobs,
            )
            sample.metadata["reward_type"] = "akg_kernel_reward"
            sample.metadata["episode_status"] = episode.status
            sample.metadata["op_name"] = task_info.get("op_name", "")

            logger.info(
                "[slime_generate] episode=%s op=%s turns=%d reward=%.3f status=%s response_len=%d",
                trajectory.episode_id,
                task_info.get("op_name", "?"),
                len(kernel_gen_turns),
                sample.reward,
                episode.status,
                len(sample.response),
            )
        else:
            logger.warning(
                "[slime_generate] No kernel_gen turns in episode %s, falling back to direct generation",
                trajectory.episode_id,
            )
            sample = await _fallback_generate(args, sample, sampling_params)

    except Exception as exc:
        logger.error(
            "[slime_generate] AgentPipe failed: %s: %s — falling back to direct generation",
            type(exc).__name__,
            exc,
        )
        sample = await _fallback_generate(args, sample, sampling_params)

    return sample


async def _fallback_generate(
    args: Namespace,
    sample: Any,
    sampling_params: dict[str, Any],
) -> Any:
    """Fallback: use slime's default SGLang generation with heuristic reward.

    This ensures the training loop never breaks even if the agent pipeline
    encounters an error.
    """
    from slime.rollout.sglang_rollout import generate

    sample = await generate(args, sample, sampling_params)
    response_text = sample.response if isinstance(sample.response, str) else ""
    sample.reward = _compute_simple_reward(response_text)
    sample.metadata = sample.metadata or {}
    sample.metadata["reward_type"] = "fallback_heuristic"
    sample.metadata["response_len"] = len(response_text)
    return sample


def _compute_simple_reward(code: str) -> float:
    """Heuristic reward for generated kernel code (fallback placeholder)."""
    if not code or not code.strip():
        return 0.0

    score = 0.1
    code_lower = code.lower()
    if "def " in code_lower or "class " in code_lower:
        score += 0.2
    if "import " in code_lower:
        score += 0.1
    if "triton" in code_lower or "torch" in code_lower or "cuda" in code_lower:
        score += 0.2
    if "@triton.jit" in code or "@torch.compile" in code:
        score += 0.2
    if "return " in code_lower:
        score += 0.1
    if code.count("\n") > 5:
        score += 0.1

    return min(score, 1.0)
