#!/usr/bin/env python3
"""Export mimic-video normalization statistics from a LeRobot v3 dataset.

Reads `observation.state` and `action` columns from all parquet files under
``<dataset_dir>/data/chunk-*/file-*.parquet``, computes the per-dimension
statistics that the mimic-video normalizer expects, and writes them to a
single small ``stats.json`` file.

The resulting JSON has the exact shape that
``cosmos_predict2.module.normalizer.StaticBatchNormalizer.build_from_stats``
expects::

    {
        "obs/lowdim_concat":    {"min": [...], "max": [...], "mean": [...],
                                 "std": [...], "percentiles": [[...], ...],
                                 "clamp_min": [...], "clamp_max": [...],
                                 "mean_clamped": [...], "std_clamped": [...]},
        "action/lowdim_concat": {... same keys ...},
    }

This script does NOT import the LeRobot dataset adapter (it only depends on
``pyarrow`` for I/O), so it works regardless of which lerobot version (0.4 vs
0.5) is installed in the active environment. It calls the same
``array_to_stats`` helper that the mimic-video trainer uses, so the
resulting stats are identical to the ones cached during training. The
mimic-video venv (``mimic-video/model/.venv``) MUST be active when running
this script because ``array_to_stats`` transitively imports torch.

Usage::

    python scripts/export_dataset_stats.py \\
        --dataset-dir dataset/task_1_v3 \\
        --output stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _add_mimic_model_to_path() -> Path:
    """Make the mimic-video model importable so we can reuse array_to_stats."""
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "mimic-video" / "model",
        here / "mimic-video" / "model",
    ]
    for candidate in candidates:
        if (candidate / "cosmos_predict2").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError(
        "Could not locate mimic-video/model. Looked in:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="LeRobot v3 dataset root (contains data/, meta/, videos/).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write stats.json.",
    )
    p.add_argument(
        "--state-col",
        default="observation.state",
        help="Parquet column for the proprioceptive state vector.",
    )
    p.add_argument(
        "--action-col",
        default="action",
        help="Parquet column for the action vector.",
    )
    return p.parse_args()


def _collect_column(parquet_files: list[Path], col_name: str) -> np.ndarray:
    """Concatenate one column across all parquet files into an (N, D) array."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to read LeRobot v3 parquet files. "
            "Install with `uv pip install pyarrow`."
        ) from exc

    chunks: list[np.ndarray] = []
    for path in parquet_files:
        table = pq.read_table(path, columns=[col_name])
        # LeRobot v3 stores each row as a list of floats; convert to (N, D).
        pylist = table[col_name].to_pylist()
        arr = np.asarray(pylist, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"Column '{col_name}' in {path} produced shape {arr.shape}; "
                "expected (N, D)."
            )
        chunks.append(arr)
    return np.concatenate(chunks, axis=0)


def main() -> None:
    args = parse_args()

    if not args.dataset_dir.is_dir():
        sys.exit(f"--dataset-dir does not exist or is not a directory: {args.dataset_dir}")

    _add_mimic_model_to_path()
    from cosmos_predict2.module.normalizer import array_to_stats

    parquet_files = sorted((args.dataset_dir / "data").rglob("*.parquet"))
    if not parquet_files:
        sys.exit(
            f"No parquet files found under {args.dataset_dir / 'data'}. "
            "Make sure --dataset-dir points to a LeRobot v3 dataset root."
        )
    print(f"Found {len(parquet_files)} parquet file(s) under {args.dataset_dir / 'data'}.")

    print(f"Loading column '{args.state_col}' ...")
    states = _collect_column(parquet_files, args.state_col)
    print(f"  state shape: {states.shape}, dtype: {states.dtype}")

    print(f"Loading column '{args.action_col}' ...")
    actions = _collect_column(parquet_files, args.action_col)
    print(f"  action shape: {actions.shape}, dtype: {actions.dtype}")

    if states.shape[0] != actions.shape[0]:
        print(
            f"WARNING: state count ({states.shape[0]}) != action count ({actions.shape[0]}). "
            "This is unusual for LeRobot v3 but not fatal — stats are computed independently."
        )
    if states.shape[-1] != actions.shape[-1]:
        print(
            f"WARNING: state dim ({states.shape[-1]}) != action dim ({actions.shape[-1]})."
        )

    stats = {
        "obs/lowdim_concat": array_to_stats(states),
        "action/lowdim_concat": array_to_stats(actions),
    }

    serializable = {
        group: {key: np.asarray(value).tolist() for key, value in fields.items()}
        for group, fields in stats.items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(serializable, f)

    n_bytes = args.output.stat().st_size
    print(f"\nWrote stats to {args.output} ({n_bytes / 1024:.1f} KB).")
    print("Per-dim mean (state) :", stats["obs/lowdim_concat"]["mean"])
    print("Per-dim mean (action):", stats["action/lowdim_concat"]["mean"])
    print("Per-dim std  (state) :", stats["obs/lowdim_concat"]["std"])
    print("Per-dim std  (action):", stats["action/lowdim_concat"]["std"])


if __name__ == "__main__":
    main()
