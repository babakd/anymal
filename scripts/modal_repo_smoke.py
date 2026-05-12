"""Modal-hosted lightweight repository smoke tests.

This gives agents a dependency-controlled Python 3.10 environment for checks
that do not require full model weights or a GPU.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal


app = modal.App("anymal-repo-smoke")
PROJECT_DIR = Path(__file__).resolve().parents[1]
REMOTE_PROJECT_DIR = "/root/anymal"

base_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("pytest>=7.4.0")
)

image = base_image.add_local_dir(PROJECT_DIR, remote_path=REMOTE_PROJECT_DIR, copy=False)

torch_image = base_image.pip_install(
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "transformers>=4.53.0,<5.0.0",
    "Pillow>=10.0.0",
    "tqdm>=4.66.0",
).add_local_dir(PROJECT_DIR, remote_path=REMOTE_PROJECT_DIR, copy=False)


def _run(cmd: list[str]) -> dict:
    result = subprocess.run(
        cmd,
        cwd=REMOTE_PROJECT_DIR,
        capture_output=True,
        text=True,
    )
    return {
        "cmd": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


@app.function(image=image, timeout=10 * 60)
def run_smoke() -> list[dict]:
    checks = [
        ["python", "scripts/repo_health_check.py"],
        [
            "python",
            "-m",
            "py_compile",
            "training/__init__.py",
            "evaluation/__init__.py",
            "vqa_checkpoint_eval.py",
            "gqa_checkpoint_eval.py",
            "pope_checkpoint_eval.py",
            "scripts/repo_health_check.py",
            "tests/test_health_monitor.py",
        ],
        ["python", "-m", "pytest", "tests/test_health_monitor.py", "-q"],
    ]
    results = [_run(cmd) for cmd in checks]
    failed = [result for result in results if result["returncode"] != 0]
    if failed:
        details = "\n\n".join(
            f"$ {' '.join(item['cmd'])}\n{item['stdout']}{item['stderr']}" for item in failed
        )
        raise RuntimeError(f"Repo smoke failed:\n{details}")
    return results


@app.function(image=torch_image, timeout=20 * 60)
def run_torch_smoke() -> list[dict]:
    checks = [
        ["python", "-m", "pytest", "tests/test_evaluation.py", "-q"],
        ["python", "-m", "pytest", "tests/test_training.py", "-q"],
    ]
    results = [_run(cmd) for cmd in checks]
    failed = [result for result in results if result["returncode"] != 0]
    if failed:
        details = "\n\n".join(
            f"$ {' '.join(item['cmd'])}\n{item['stdout']}{item['stderr']}" for item in failed
        )
        raise RuntimeError(f"Torch repo smoke failed:\n{details}")
    return results


@app.local_entrypoint()
def main(include_torch: bool = True):
    results = run_smoke.remote()
    for result in results:
        print(f"$ {' '.join(result['cmd'])}")
        print(result["stdout"], end="")
        print(result["stderr"], end="")

    if include_torch:
        torch_results = run_torch_smoke.remote()
        for result in torch_results:
            print(f"$ {' '.join(result['cmd'])}")
            print(result["stdout"], end="")
            print(result["stderr"], end="")
