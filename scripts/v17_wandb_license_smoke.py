"""Create a tiny W&B-only smoke run with the V17 dataset license summary."""

from __future__ import annotations

import json
from typing import Any, Dict

import modal


app = modal.App("anymal-v17-wandb-license-smoke")
image = modal.Image.debian_slim(python_version="3.10").pip_install("wandb>=0.16.0")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("wandb")],
    timeout=5 * 60,
)
def run_smoke(
    dataset_license_summary: Dict[str, Any],
    project: str = "anymal-pretrain",
    run_name: str = "v17-license-summary-smoke-20260515",
) -> Dict[str, Any]:
    import wandb

    run = wandb.init(
        project=project,
        name=run_name,
        config={
            "purpose": "v17_dataset_license_summary_config_smoke",
            "dataset_license_summary": dataset_license_summary,
        },
    )
    run_url = run.get_url()
    run_path = f"{run.entity}/{run.project}/{run.id}"
    run.finish()

    api = wandb.Api()
    public_run = api.run(run_path)
    public_config = dict(public_run.config or {})
    return {
        "run_path": run_path,
        "url": run_url,
        "state": public_run.state,
        "dataset_license_summary": public_config.get("dataset_license_summary"),
    }


@app.local_entrypoint()
def main(
    dataset_license_summary_path: str,
    project: str = "anymal-pretrain",
    run_name: str = "v17-license-summary-smoke-20260515",
) -> None:
    with open(dataset_license_summary_path) as f:
        dataset_license_summary = json.load(f)
    result = run_smoke.remote(
        dataset_license_summary=dataset_license_summary,
        project=project,
        run_name=run_name,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
