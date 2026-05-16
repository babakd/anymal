"""Pinned Hugging Face dataset revision helpers for checkpoint evaluators."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_DATASET_REVISIONS: dict[str, dict[str, Any]] = {
    "Mineru/GQA": {
        "revision": "55fbe98d3474e07e0d1fe0078ba2d48c9ea7712e",
        "pinned_on": "2026-05-15",
    },
    "anhdang000/ChartQA-V2": {
        "revision": "93b2f1f6bd69516c1be21faeefa05540768e0537",
        "pinned_on": "2026-05-15",
        "license": "unknown-community-mirror",
        "license_note": (
            "V17 must verify equivalence to a primary ChartQA source before "
            "deployed-model use."
        ),
    },
    "lmms-lab/textvqa": {
        "revision": "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        "pinned_on": "2026-05-15",
    },
}


def _project_dir() -> Path:
    path = Path(__file__).resolve()
    if len(path.parents) >= 3:
        return path.parents[2]
    return Path.cwd()


def dataset_revisions_path() -> Path:
    return _project_dir() / "configs" / "dataset_revisions.json"


def _candidate_revision_paths(path: str | Path | None = None) -> list[Path]:
    if path is not None:
        return [Path(path)]
    candidates = [dataset_revisions_path()]
    env_path = os.environ.get("ANYMAL_DATASET_REVISIONS")
    if env_path:
        candidates.insert(0, Path(env_path))
    for root in (Path("/root/anymal"), Path.cwd()):
        candidates.append(root / "configs" / "dataset_revisions.json")
    seen = set()
    unique = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def load_dataset_revisions(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    revisions_path = None
    for candidate in _candidate_revision_paths(path):
        if candidate.exists():
            revisions_path = candidate
            break
    if revisions_path is None:
        if path is not None:
            raise FileNotFoundError(f"Dataset revision file not found: {path}")
        return dict(DEFAULT_DATASET_REVISIONS)
    with open(revisions_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset revision file must contain an object: {revisions_path}")
    return payload


def pinned_revision(dataset_id: str, *, required: bool = True) -> str | None:
    revisions = load_dataset_revisions()
    entry = revisions.get(str(dataset_id))
    revision = str((entry or {}).get("revision") or "").strip()
    if revision:
        return revision
    if required:
        raise KeyError(f"No pinned revision found for HF dataset {dataset_id!r}")
    return None


def slice_fingerprint(
    *,
    dataset_id: str,
    revision: str,
    split: str,
    seed: int,
    offset: int = 0,
    max_samples: int = 0,
) -> str:
    payload = {
        "dataset_id": str(dataset_id),
        "revision": str(revision),
        "split": str(split),
        "seed": int(seed),
        "offset": int(offset),
        "max_samples": int(max_samples),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
