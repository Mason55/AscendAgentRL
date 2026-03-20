"""Unit tests for KernelBenchLoader — no NPU, no AKG imports required."""
import json
import textwrap
import pytest
from pathlib import Path

from examples.akg_kernel_gen.ascendrl_glue.kernelbench_jsonl import KernelBenchLoader


@pytest.fixture()
def dataset_dir(tmp_path):
    """Minimal fake KernelBench level1 directory with 5 tasks."""
    level_dir = tmp_path / "level1"
    level_dir.mkdir()
    for i in range(1, 6):
        task_file = level_dir / f"{i}_op_{i}.py"
        task_file.write_text(
            textwrap.dedent(f"""\
                import torch
                import torch.nn as nn

                class Model(nn.Module):
                    def forward(self, x):
                        return x + {i}

                def get_inputs():
                    return [torch.randn(4, 4)]

                def get_init_inputs():
                    return []
            """)
        )
    return str(tmp_path)


def test_get_step_batch_returns_list_of_dicts(dataset_dir):
    loader = KernelBenchLoader(dataset_dir, level="level1", shuffle=False)
    batch = loader.get_step_batch(step_idx=0, batch_size=3)
    assert isinstance(batch, list)
    assert len(batch) == 3
    for item in batch:
        assert "prompt" in item
        assert "raw" in item


def test_prompt_is_valid_json_with_required_keys(dataset_dir):
    loader = KernelBenchLoader(dataset_dir, level="level1", shuffle=False)
    batch = loader.get_step_batch(step_idx=0, batch_size=1)
    parsed = json.loads(batch[0]["prompt"])
    assert "op_name" in parsed
    assert "task_desc" in parsed
    assert isinstance(parsed["op_name"], str)
    assert isinstance(parsed["task_desc"], str)
    assert "class Model" in parsed["task_desc"]


def test_raw_contains_op_name_and_level(dataset_dir):
    loader = KernelBenchLoader(dataset_dir, level="level1", shuffle=False)
    batch = loader.get_step_batch(step_idx=0, batch_size=1)
    raw = batch[0]["raw"]
    assert "op_name" in raw
    assert "level" in raw
    assert raw["level"] == "level1"


def test_batch_size_respected(dataset_dir):
    loader = KernelBenchLoader(dataset_dir, level="level1", shuffle=False)
    for batch_size in [1, 3, 5]:
        batch = loader.get_step_batch(step_idx=0, batch_size=batch_size)
        assert len(batch) == batch_size


def test_batch_cycles_when_step_exceeds_dataset(dataset_dir):
    loader = KernelBenchLoader(dataset_dir, level="level1", shuffle=False)
    batch0 = loader.get_step_batch(step_idx=0, batch_size=3)
    batch1 = loader.get_step_batch(step_idx=1, batch_size=3)
    assert json.loads(batch1[2]["prompt"])["op_name"] == \
           json.loads(batch0[0]["prompt"])["op_name"]


def test_batch_size_larger_than_dataset(dataset_dir):
    loader = KernelBenchLoader(dataset_dir, level="level1", shuffle=False)
    batch = loader.get_step_batch(step_idx=0, batch_size=8)
    assert len(batch) == 8
    for item in batch:
        json.loads(item["prompt"])


def test_reproducible_order_with_same_seed(dataset_dir):
    loader_a = KernelBenchLoader(dataset_dir, level="level1", shuffle=True, seed=42)
    loader_b = KernelBenchLoader(dataset_dir, level="level1", shuffle=True, seed=42)
    batch_a = loader_a.get_step_batch(step_idx=0, batch_size=5)
    batch_b = loader_b.get_step_batch(step_idx=0, batch_size=5)
    assert [json.loads(x["prompt"])["op_name"] for x in batch_a] == \
           [json.loads(x["prompt"])["op_name"] for x in batch_b]


def test_different_seeds_give_different_order(dataset_dir):
    loader_a = KernelBenchLoader(dataset_dir, level="level1", shuffle=True, seed=42)
    loader_b = KernelBenchLoader(dataset_dir, level="level1", shuffle=True, seed=0)
    names_a = [json.loads(x["prompt"])["op_name"]
               for x in loader_a.get_step_batch(0, 5)]
    names_b = [json.loads(x["prompt"])["op_name"]
               for x in loader_b.get_step_batch(0, 5)]
    assert sorted(names_a) == sorted(names_b)
    assert any(a != b for a, b in zip(names_a, names_b))


def test_raises_if_dataset_dir_missing():
    with pytest.raises(FileNotFoundError):
        KernelBenchLoader("/nonexistent/path/dataset", level="level1")


def test_raises_if_level_dir_empty(tmp_path):
    level_dir = tmp_path / "level1"
    level_dir.mkdir()
    with pytest.raises(ValueError):
        KernelBenchLoader(str(tmp_path), level="level1")
