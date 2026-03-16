"""Convert KernelBench level .py files to JSONL for slime training.

Usage:
    python examples/akg_kernel_gen/scripts/prepare_kernelbench_data.py \
        --kernelbench-dir data/KernelBench/KernelBench \
        --level level1 \
        --output data/kernelbench_level1.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare KernelBench data for slime training")
    parser.add_argument("--kernelbench-dir", required=True)
    parser.add_argument("--level", default="level1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    level_path = Path(args.kernelbench_dir) / args.level
    if not level_path.exists():
        raise FileNotFoundError(f"Not found: {level_path}")

    task_files = sorted(level_path.glob("*.py"))
    if not task_files:
        raise ValueError(f"No .py files in {level_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for tf in task_files:
            op_name = tf.stem
            task_desc = tf.read_text(encoding="utf-8")
            prompt_obj = {"op_name": op_name, "task_desc": task_desc}
            row = {
                "prompt": json.dumps(prompt_obj),
                "expected_answer": "",
                "op_name": op_name,
                "level": args.level,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(task_files)} tasks to {output_path}")


if __name__ == "__main__":
    main()
