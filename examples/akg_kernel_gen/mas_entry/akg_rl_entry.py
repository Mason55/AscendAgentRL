"""AKG RL Entry Point — launched by MASLauncher as a subprocess.

Usage:
    python akg_rl_entry.py --config /tmp/trajectory_mas_xxx.yaml \
                           --task '{"op_name":"1_relu","task_desc":"..."}'

IMPORTANT: This script always exits 0.
slime drops any episode with a non-zero exit code (before reward
computation), which would remove negative examples needed for GRPO.
Correctness is determined by AKGKernelRewardProvider, not by this script.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _inject_agent_env_vars(config: dict[str, Any]) -> None:
    """Set AKG_AGENTS_* env vars so all agents route through ModelMonitor.

    akg_agents uses a level-based model config (complex/standard/fast).
    We configure two levels so each agent sends a distinct ``model`` name
    in its OpenAI requests, allowing ModelMonitor to attribute turns to
    the correct agent role:

      - ``complex`` level → model_name=kernel_designer (for designer agent)
      - ``standard`` level → model_name=kernel_gen     (for coder agent)

    Both levels point to the same ModelMonitor URL.
    """
    agents_cfg = config.get("agents", {})

    gen_cfg = agents_cfg.get("kernel_gen", {})
    gen_llm = gen_cfg.get("llm", {})
    monitor_url = gen_llm.get("base_url", "")

    if not monitor_url:
        designer_cfg = agents_cfg.get("kernel_designer", {})
        designer_llm = designer_cfg.get("llm", {})
        monitor_url = designer_llm.get("base_url", "")

    if not monitor_url:
        logger.warning("[akg_rl_entry] No monitor base_url found in config")
        return

    gen_model = agents_cfg.get("kernel_gen", {}).get("model", "kernel_gen")
    designer_model = agents_cfg.get("kernel_designer", {}).get("model", "kernel_designer")

    os.environ["AKG_AGENTS_STANDARD_BASE_URL"] = monitor_url
    os.environ["AKG_AGENTS_STANDARD_API_KEY"] = "dummy"
    os.environ["AKG_AGENTS_STANDARD_MODEL_NAME"] = gen_model

    os.environ["AKG_AGENTS_COMPLEX_BASE_URL"] = monitor_url
    os.environ["AKG_AGENTS_COMPLEX_API_KEY"] = "dummy"
    os.environ["AKG_AGENTS_COMPLEX_MODEL_NAME"] = designer_model

    logger.info(
        "[akg_rl_entry] ModelMonitor: %s (coder→standard:%s, designer→complex:%s)",
        monitor_url, gen_model, designer_model,
    )


def _build_task(
    op_name: str,
    task_desc: str,
    task_cfg: dict[str, Any],
) -> Any:
    """Construct a LangGraphTask instance. Extracted for testability."""
    from akg_agents.op.langgraph_op.task import LangGraphTask
    from akg_agents.op.config.config_validator import load_config

    task_id = uuid.uuid4().hex[:8]
    dsl = task_cfg.get("dsl", "triton_cuda")
    backend = task_cfg.get("backend", "cuda")
    config = load_config(dsl=dsl, backend=backend)
    config["log_dir"] = config.get("log_dir", "/tmp/akg_rl_logs")
    config["agent_model_config"] = {
        "designer": "complex",
        "coder": "standard",
    }

    return LangGraphTask(
        op_name=op_name,
        task_desc=task_desc,
        task_id=task_id,
        backend=task_cfg.get("backend", "cuda"),
        arch=task_cfg.get("arch", "a100"),
        dsl=task_cfg.get("dsl", "triton_cuda"),
        framework=task_cfg.get("framework", "torch"),
        config=config,
    )


def run(config_path: str, task_json: str) -> int:
    """Core logic; separated from main() for testability. Always returns 0."""
    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        _inject_agent_env_vars(config)

        task_info = json.loads(task_json)
        op_name = task_info["op_name"]
        task_desc = task_info["task_desc"]
        task_cfg = config.get("task", {})

        task = _build_task(op_name=op_name, task_desc=task_desc, task_cfg=task_cfg)
        op_name_out, success, final_state = asyncio.run(task.run())
        logger.info("[akg_rl_entry] op=%s success=%s", op_name_out, success)
    except Exception as exc:
        logger.error("[akg_rl_entry] episode failed with %s: %s", type(exc).__name__, exc)

    return 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="AKG RL subprocess entry point")
    parser.add_argument("--config", required=True, help="Path to YAML config written by prepare_config()")
    parser.add_argument("--task", required=True, help="JSON string: {op_name, task_desc}")
    args = parser.parse_args()
    sys.exit(run(config_path=args.config, task_json=args.task))


if __name__ == "__main__":
    main()
