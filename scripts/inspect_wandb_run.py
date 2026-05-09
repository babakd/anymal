"""Inspect a W&B training run from Modal using the shared wandb secret.

This is intended for babysitting expensive Modal training jobs when local Python
does not have W&B credentials installed.

Example:
    modal run scripts/inspect_wandb_run.py \
      --run-path babakdam/anymal-pretrain/6ffd2qwa \
      --recent-window 100
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional

import modal


app = modal.App("anymal-wandb-inspector")

image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "wandb>=0.16.0",
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _mean(values: Iterable[float]) -> Optional[float]:
    values = [float(v) for v in values if _is_number(v)]
    return sum(values) / len(values) if values else None


def _last_number(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    for row in reversed(rows):
        value = row.get(key)
        if _is_number(value):
            return float(value)
    return None


def _last_step(rows: List[Dict[str, Any]]) -> Optional[int]:
    for row in reversed(rows):
        value = row.get("train/step", row.get("_step"))
        if _is_number(value):
            return int(value)
    return None


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("wandb")],
    timeout=10 * 60,
)
def inspect_wandb_run_remote(
    run_path: str,
    recent_window: int = 100,
    spike_multiplier: float = 2.0,
    grad_spike_multiplier: float = 5.0,
) -> Dict[str, Any]:
    import wandb

    api = wandb.Api()
    run = api.run(run_path)

    # Do not pass a metric key list here. W&B's public API returns only rows
    # containing all requested keys, while our trainer logs loss, eval, health,
    # and throughput on different cadences.
    rows = list(run.scan_history(page_size=1000))
    train_rows = [row for row in rows if _is_number(row.get("train/loss"))]
    recent = train_rows[-max(1, int(recent_window)) :]

    loss_spikes = []
    grad_spikes = []
    for row in recent:
        loss = row.get("train/loss")
        loss_ema = row.get("health/loss_ema")
        if _is_number(loss) and _is_number(loss_ema) and loss_ema > 0:
            ratio = float(loss) / float(loss_ema)
            if ratio >= spike_multiplier:
                loss_spikes.append(
                    {
                        "step": int(row.get("train/step", row.get("_step", -1))),
                        "loss": float(loss),
                        "loss_ema": float(loss_ema),
                        "ratio": ratio,
                    }
                )

        grad = row.get("train/grad_norm")
        grad_ema = row.get("health/grad_norm_ema")
        if _is_number(grad) and _is_number(grad_ema) and grad_ema > 0:
            ratio = float(grad) / float(grad_ema)
            if ratio >= grad_spike_multiplier:
                grad_spikes.append(
                    {
                        "step": int(row.get("train/step", row.get("_step", -1))),
                        "grad_norm": float(grad),
                        "grad_norm_ema": float(grad_ema),
                        "ratio": ratio,
                    }
                )

    eval_rows = [row for row in rows if _is_number(row.get("eval/loss"))]
    eval_losses = [float(row["eval/loss"]) for row in eval_rows[-5:]]

    recent_losses = [float(row["train/loss"]) for row in recent]
    first_half = recent_losses[: len(recent_losses) // 2]
    second_half = recent_losses[len(recent_losses) // 2 :]

    clip_fraction = _last_number(rows, "health/grad_clip_fraction")
    alerts = []
    if loss_spikes:
        alerts.append("recent_loss_spikes")
    if grad_spikes:
        alerts.append("recent_grad_spikes")
    if clip_fraction is not None and clip_fraction >= 0.2:
        alerts.append("high_grad_clip_fraction")
    if len(eval_losses) >= 3 and eval_losses[-3] < eval_losses[-2] < eval_losses[-1]:
        alerts.append("eval_loss_increasing_3_points")
    if (
        _mean(first_half) is not None
        and _mean(second_half) is not None
        and _mean(second_half) > 1.25 * _mean(first_half)
    ):
        alerts.append("recent_loss_window_mean_up_gt_25pct")

    return {
        "run_path": run_path,
        "url": run.url,
        "state": run.state,
        "name": run.name,
        "history_rows": len(rows),
        "train_rows": len(train_rows),
        "last_step": _last_step(rows),
        "latest": {
            "train_loss": _last_number(rows, "train/loss"),
            "raw_loss": _last_number(rows, "train/raw_loss"),
            "objective_loss": _last_number(rows, "train/objective_loss"),
            "backward_loss": _last_number(rows, "train/backward_loss"),
            "supervised_tokens": _last_number(rows, "train/supervised_tokens"),
            "loss_normalization_multiplier": _last_number(
                rows,
                "train/loss_normalization_multiplier",
            ),
            "accumulation_micro_batches": _last_number(
                rows,
                "train/accumulation_micro_batches",
            ),
            "loss_ema": _last_number(rows, "health/loss_ema"),
            "grad_norm": _last_number(rows, "train/grad_norm"),
            "grad_norm_ema": _last_number(rows, "health/grad_norm_ema"),
            "grad_clip_fraction": clip_fraction,
            "lr": _last_number(rows, "train/lr"),
            "eval_loss": _last_number(rows, "eval/loss"),
            "samples_per_second": _last_number(rows, "perf/samples_per_second"),
            "tokens_per_second": _last_number(rows, "perf/tokens_per_second"),
        },
        "recent_window": {
            "n": len(recent_losses),
            "loss_min": min(recent_losses) if recent_losses else None,
            "loss_max": max(recent_losses) if recent_losses else None,
            "loss_mean": _mean(recent_losses),
            "first_half_loss_mean": _mean(first_half),
            "second_half_loss_mean": _mean(second_half),
        },
        "eval_losses_last5": eval_losses,
        "loss_spikes": loss_spikes[-10:],
        "grad_spikes": grad_spikes[-10:],
        "alerts": alerts,
    }


@app.local_entrypoint()
def main(
    run_path: str,
    recent_window: int = 100,
    spike_multiplier: float = 2.0,
    grad_spike_multiplier: float = 5.0,
):
    result = inspect_wandb_run_remote.remote(
        run_path=run_path,
        recent_window=recent_window,
        spike_multiplier=spike_multiplier,
        grad_spike_multiplier=grad_spike_multiplier,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
