"""KernelBench JSONL loader for slime datasets.

Reads a JSONL file where each line contains a KernelBench task, and exposes
them as rolling batches compatible with the prepare_kernelbench_data script.

Each line is expected to have at minimum:
    {"prompt": "{\"op_name\": ..., \"task_desc\": ...}"}
"""
from __future__ import annotations

import json
import random
from pathlib import Path


class KernelBenchLoader:
    """Loads KernelBench tasks from a directory and serves them as rolling batches.

    Args:
        dataset_dir: Root directory containing ``level1/``, ``level2/`` etc.
        level:       Sub-directory to load (default ``"level1"``).
        shuffle:     Whether to shuffle tasks before serving (default ``True``).
        seed:        Random seed for reproducible shuffling (default ``42``).
    """

    def __init__(
        self,
        dataset_dir: str,
        level: str = "level1",
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        level_path = Path(dataset_dir) / level
        if not level_path.exists():
            raise FileNotFoundError(
                f"KernelBench level directory not found: {level_path}"
            )

        task_files = sorted(level_path.glob("*.py"))
        if not task_files:
            raise ValueError(f"No .py task files found in {level_path}")

        self._tasks: list[dict] = []
        for f in task_files:
            op_name = f.stem
            task_desc = f.read_text(encoding="utf-8")
            self._tasks.append(
                {"op_name": op_name, "task_desc": task_desc, "level": level}
            )

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self._tasks)

    def get_step_batch(self, step_idx: int, batch_size: int) -> list[dict]:
        """Return *batch_size* tasks for the given training step.

        Cycles through the dataset using ``step_idx * batch_size`` as the
        starting offset, wrapping around as needed.
        """
        n = len(self._tasks)
        start = (step_idx * batch_size) % n
        indices = [(start + i) % n for i in range(batch_size)]
        result = []
        for idx in indices:
            task = self._tasks[idx]
            result.append(
                {
                    "prompt": json.dumps(
                        {
                            "op_name": task["op_name"],
                            "task_desc": task["task_desc"],
                        }
                    ),
                    "raw": {
                        "op_name": task["op_name"],
                        "task_desc": task["task_desc"],
                        "level": task["level"],
                    },
                }
            )
        return result
