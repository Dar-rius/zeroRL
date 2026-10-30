"""Raw-result schema for the scalability benchmark.

One JSON object per run, appended to a JSONL file (one file per experiment
is created by the runner). Every row carries a mandatory ``framework``
column (zeroRL | SB3 | TorchRL), the full configuration, timing of the three
phases, budget accounting and status.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1

FRAMEWORKS = ("zeroRL", "SB3", "TorchRL")
EXPERIMENTS = ("A", "B", "C", "D", "R")   # R = resource consumption
MODES = ("sync", "mp")
STATUSES = ("ok", "unsupported", "failed")

REQUIRED_FIELDS = (
    "schema_version",
    "framework",
    "experiment",
    "mode",
    "num_envs",
    "num_workers",
    "protocol",
    "rep",
    "seed",
    "status",
    "phases",
    "transitions",
    "metrics",
    "config",
    "environment",
    "timestamp_utc",
)


def environment_info() -> dict[str, Any]:
    """Capture versions/hardware metadata recorded into every row."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": "cpu",
        "hostname": platform.node(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_num_threads"] = torch.get_num_threads()
        info["cuda_available"] = torch.cuda.is_available()
    except Exception as exc:  # pragma: no cover - torch is a hard dep
        info["torch"] = f"unavailable: {exc}"
    try:
        import gymnasium

        info["gymnasium"] = gymnasium.__version__
    except Exception as exc:  # pragma: no cover
        info["gymnasium"] = f"unavailable: {exc}"
    try:
        import stable_baselines3

        info["sb3"] = stable_baselines3.__version__
    except ImportError:
        info["sb3"] = "not installed"
    try:
        import torchrl

        info["torchrl"] = torchrl.__version__
    except ImportError:
        info["torchrl"] = "not installed"
    try:
        import tensordict

        info["tensordict"] = tensordict.__version__
    except ImportError:
        info["tensordict"] = "not installed"
    info["git_sha"] = git_sha()
    return info


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # pragma: no cover
        return "unknown"


def make_row(
    *,
    framework: str,
    experiment: str,
    mode: str,
    num_envs: int,
    num_workers: int | None,
    protocol: str | None,
    rep: int,
    seed: int,
    status: str,
    phases: dict[str, float],
    transitions: dict[str, Any],
    metrics: dict[str, Any],
    config: dict[str, Any],
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate one result row."""
    if framework not in FRAMEWORKS:
        raise ValueError(f"framework must be one of {FRAMEWORKS}, got {framework!r}")
    if experiment not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {EXPERIMENTS}, got {experiment!r}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "framework": framework,
        "experiment": experiment,
        "mode": mode,
        "num_envs": num_envs,
        "num_workers": num_workers,
        "protocol": protocol,
        "rep": rep,
        "seed": seed,
        "status": status,
        "phases": phases,
        "transitions": transitions,
        "metrics": metrics,
        "config": config,
        "error": error,
        "environment": environment_info(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        row["extra"] = extra
    validate_row(row)
    return row


def validate_row(row: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_FIELDS if k not in row]
    if missing:
        raise ValueError(f"result row missing fields: {missing}")
    if row["status"] == "failed" and not row.get("error"):
        raise ValueError("failed rows must carry an error message")
    if row["status"] == "ok":
        t = row["transitions"]
        if t.get("measured", 0) <= 0 and row["experiment"] in ("A", "B"):
            raise ValueError("ok collection rows must record measured transitions")
        if row["phases"].get("measure_seconds", 0) <= 0:
            raise ValueError("ok rows must have a positive measure_seconds")


def append_row(path: Path, row: dict[str, Any]) -> None:
    validate_row(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def read_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return list(read_rows(path))
