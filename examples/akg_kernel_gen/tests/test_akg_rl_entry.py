"""Integration tests for akg_rl_entry.py.

LangGraphTask is mocked so tests run without NPU or LLM server.
"""
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml


ENTRY_SCRIPT = str(
    Path(__file__).parent.parent / "mas_entry" / "akg_rl_entry.py"
)


def _write_config(tmp_path: Path, base_url: str = "http://127.0.0.1:9999/v1") -> Path:
    config = {
        "agents": {
            "kernel_gen": {
                "model": "kernel_gen",
                "llm": {"base_url": base_url, "api_key": "dummy"},
            },
            "kernel_designer": {
                "model": "kernel_designer",
                "llm": {"base_url": base_url, "api_key": "dummy"},
            },
        },
        "task": {
            "framework": "torch",
            "backend": "cuda",
            "arch": "a100",
            "dsl": "triton_cuda",
            "max_iterations": 2,
        },
    }
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _task_arg(op_name: str = "1_relu", task_desc: str = "class Model: pass") -> str:
    return json.dumps({"op_name": op_name, "task_desc": task_desc})


# ── always exits 0 ────────────────────────────────────────────────────────────

def test_exits_zero_on_successful_task(tmp_path):
    config_path = _write_config(tmp_path)
    mock_task = MagicMock()
    mock_task.run = AsyncMock(return_value=("1_relu", True, {}))

    with patch(
        "examples.akg_kernel_gen.mas_entry.akg_rl_entry._build_task",
        return_value=mock_task,
    ):
        from examples.akg_kernel_gen.mas_entry import akg_rl_entry
        code = akg_rl_entry.run(
            config_path=str(config_path),
            task_json=_task_arg(),
        )
    assert code == 0


def test_exits_zero_even_when_task_fails(tmp_path):
    config_path = _write_config(tmp_path)
    mock_task = MagicMock()
    mock_task.run = AsyncMock(return_value=("1_relu", False, {}))

    with patch(
        "examples.akg_kernel_gen.mas_entry.akg_rl_entry._build_task",
        return_value=mock_task,
    ):
        from examples.akg_kernel_gen.mas_entry import akg_rl_entry
        code = akg_rl_entry.run(
            config_path=str(config_path),
            task_json=_task_arg(),
        )
    assert code == 0


def test_exits_zero_even_when_task_raises(tmp_path):
    config_path = _write_config(tmp_path)
    mock_task = MagicMock()
    mock_task.run = AsyncMock(side_effect=RuntimeError("NPU exploded"))

    with patch(
        "examples.akg_kernel_gen.mas_entry.akg_rl_entry._build_task",
        return_value=mock_task,
    ):
        from examples.akg_kernel_gen.mas_entry import akg_rl_entry
        code = akg_rl_entry.run(
            config_path=str(config_path),
            task_json=_task_arg(),
        )
    assert code == 0


# ── env var injection ─────────────────────────────────────────────────────────

def test_sets_akg_env_vars_from_config(tmp_path, monkeypatch):
    config_path = _write_config(tmp_path, base_url="http://monitor:1234/v1")
    captured_env = {}
    mock_task = MagicMock()
    mock_task.run = AsyncMock(return_value=("1_relu", True, {}))

    def capture_and_build(*args, **kwargs):
        captured_env["designer_base_url"] = os.environ.get("AKG_AGENTS_DESIGNER_BASE_URL")
        captured_env["designer_model"] = os.environ.get("AKG_AGENTS_DESIGNER_MODEL_NAME")
        captured_env["coder_base_url"] = os.environ.get("AKG_AGENTS_CODER_BASE_URL")
        captured_env["coder_model"] = os.environ.get("AKG_AGENTS_CODER_MODEL_NAME")
        return mock_task

    with patch(
        "examples.akg_kernel_gen.mas_entry.akg_rl_entry._build_task",
        side_effect=capture_and_build,
    ):
        from examples.akg_kernel_gen.mas_entry import akg_rl_entry
        akg_rl_entry.run(
            config_path=str(config_path),
            task_json=_task_arg(),
        )
    assert captured_env["designer_base_url"] == "http://monitor:1234/v1"
    assert captured_env["designer_model"] == "kernel_designer"
    assert captured_env["coder_base_url"] == "http://monitor:1234/v1"
    assert captured_env["coder_model"] == "kernel_gen"


# ── task JSON parsing ─────────────────────────────────────────────────────────

def test_parses_task_json_and_passes_to_build_task(tmp_path):
    config_path = _write_config(tmp_path)
    received_kwargs: dict = {}
    mock_task = MagicMock()
    mock_task.run = AsyncMock(return_value=("custom_op", True, {}))

    def capture_build(**kwargs):
        received_kwargs.update(kwargs)
        return mock_task

    with patch(
        "examples.akg_kernel_gen.mas_entry.akg_rl_entry._build_task",
        side_effect=capture_build,
    ):
        from examples.akg_kernel_gen.mas_entry import akg_rl_entry
        akg_rl_entry.run(
            config_path=str(config_path),
            task_json=json.dumps({"op_name": "my_relu", "task_desc": "class M: pass"}),
        )
    assert received_kwargs.get("op_name") == "my_relu"
    assert received_kwargs.get("task_desc") == "class M: pass"


# ── failure paths ─────────────────────────────────────────────────────────────

def test_exits_zero_when_config_file_missing(tmp_path):
    from examples.akg_kernel_gen.mas_entry import akg_rl_entry
    code = akg_rl_entry.run(
        config_path=str(tmp_path / "nonexistent.yaml"),
        task_json=_task_arg(),
    )
    assert code == 0


def test_exits_zero_when_task_json_malformed(tmp_path):
    config_path = _write_config(tmp_path)
    from examples.akg_kernel_gen.mas_entry import akg_rl_entry
    code = akg_rl_entry.run(
        config_path=str(config_path),
        task_json="not-valid-json{{{",
    )
    assert code == 0


def test_exits_zero_when_task_json_missing_keys(tmp_path):
    config_path = _write_config(tmp_path)
    from examples.akg_kernel_gen.mas_entry import akg_rl_entry
    code = akg_rl_entry.run(
        config_path=str(config_path),
        task_json=json.dumps({"wrong_key": "value"}),
    )
    assert code == 0


# ── Ascend defaults ───────────────────────────────────────────────────────────

def test_default_backend_is_cuda(tmp_path):
    """Verify that _build_task defaults to CUDA when task config is empty."""
    received_kwargs: dict = {}

    class MockTask:
        def __init__(self, **kwargs):
            received_kwargs.update(kwargs)

    with patch("akg_agents.op.config.config_validator.load_config", return_value={"log_dir": "/tmp/test"}):
        with patch("akg_agents.op.langgraph_op.task.LangGraphTask", MockTask):
            from examples.akg_kernel_gen.mas_entry.akg_rl_entry import _build_task
            _build_task("test_op", "test_desc", {})

    assert received_kwargs["backend"] == "cuda"
    assert received_kwargs["arch"] == "a100"
    assert received_kwargs["dsl"] == "triton_cuda"
