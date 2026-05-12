"""Lightweight repository hygiene checks.

This script intentionally avoids importing project modules so it can run in a
minimal Python environment. It checks layout rules that matter for humans and
agents before heavier ML dependencies are installed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_ROOT_PREFIXES = ("V", "v")
FORBIDDEN_ROOT_NAMES = {"EXPERIMENTS.md"}
ALLOWED_ROOT_REDIRECTS = {
    "EXPERIMENTS.md",
    "V8_CORE_LLM_SWAP_RESULTS.md",
    "V8_QWEN3_plan.md",
    "V8_experiment.md",
    "v7_experiments.md",
    "v9_qwen_experiment_results.md",
    "v9_qwen_plan.md",
}
FORBIDDEN_PATH_PARTS = ("Users", "babakd", "anymal")
FORBIDDEN_TEXT = "/" + "/".join(FORBIDDEN_PATH_PARTS)


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [REPO_ROOT / line for line in result.stdout.splitlines() if line]


def _check_root_experiment_docs(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        rel = path.relative_to(REPO_ROOT)
        if len(rel.parts) != 1 or path.suffix != ".md":
            continue
        if rel.name.startswith(FORBIDDEN_ROOT_PREFIXES) and rel.name not in {
            "README.md",
            "CONTRIBUTING.md",
            *ALLOWED_ROOT_REDIRECTS,
        }:
            errors.append(f"versioned experiment doc still lives at repo root: {rel}")
            continue
        if rel.name in FORBIDDEN_ROOT_NAMES and rel.name not in ALLOWED_ROOT_REDIRECTS:
            errors.append(f"historical experiment doc still lives at repo root: {rel}")
            continue
        if rel.name in ALLOWED_ROOT_REDIRECTS:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "Moved:" not in text:
                errors.append(f"root compatibility doc is not a redirect stub: {rel}")
    return errors


def _check_hardcoded_local_paths(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        rel = path.relative_to(REPO_ROOT)
        if path.suffix not in {".py", ".sh", ".yaml", ".yml", ".toml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if FORBIDDEN_TEXT in text:
            errors.append(f"hardcoded local checkout path in tracked source: {rel}")
    return errors


def main() -> int:
    files = _tracked_files()
    errors = []
    errors.extend(_check_root_experiment_docs(files))
    errors.extend(_check_hardcoded_local_paths(files))

    if errors:
        print("Repository health check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Repository health check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
