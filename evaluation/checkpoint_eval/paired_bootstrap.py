"""Shared confidence-interval helpers for checkpoint evaluators."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any


def confidence_z(confidence: float) -> float:
    """Return a normal critical value for common confidence levels."""
    confidence = float(confidence)
    known = {
        0.80: 1.2815515655446004,
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.98: 2.3263478740408408,
        0.99: 2.5758293035489004,
    }
    for key, value in known.items():
        if abs(confidence - key) < 1e-9:
            return value
    return known[0.95]


def ci_label(confidence: float) -> str:
    return f"ci{int(round(float(confidence) * 100))}"


def wilson_ci(successes: float, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval on a binomial proportion in [0, 1]."""
    if total <= 0:
        return 0.0, 0.0
    z = confidence_z(confidence)
    phat = float(successes) / float(total)
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    margin = (
        z
        * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
        / denom
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_mean_ci(
    values: Sequence[float | int | bool],
    seed: int = 12345,
    n_resamples: int = 10000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Non-parametric bootstrap CI for the mean of observed item scores."""
    values = [float(value) for value in values]
    if not values:
        return 0.0, 0.0
    n_resamples = int(n_resamples)
    if n_resamples <= 0:
        return 0.0, 0.0
    alpha = (1.0 - float(confidence)) / 2.0
    total = len(values)
    try:
        import numpy as np

        array = np.asarray(values, dtype=float)
        rng = np.random.default_rng(int(seed))
        indices = rng.integers(0, total, size=(n_resamples, total))
        means = array[indices].mean(axis=1)
        low, high = np.quantile(means, [alpha, 1.0 - alpha])
        return float(low), float(high)
    except Exception:
        rng = random.Random(int(seed))
        means = []
        for _ in range(n_resamples):
            means.append(
                sum(values[rng.randrange(total)] for _ in range(total)) / float(total)
            )
        means.sort()
        low_idx = min(max(int(math.floor(alpha * (n_resamples - 1))), 0), n_resamples - 1)
        high_idx = min(
            max(int(math.ceil((1.0 - alpha) * (n_resamples - 1))), 0),
            n_resamples - 1,
        )
        return float(means[low_idx]), float(means[high_idx])


def _as_bool_list(values: Sequence[Any], name: str) -> list[bool]:
    result = []
    for idx, value in enumerate(values):
        if isinstance(value, bool):
            result.append(value)
        elif isinstance(value, (int, float)) and value in {0, 1}:
            result.append(bool(value))
        else:
            raise TypeError(f"{name}[{idx}] must be bool-like, got {value!r}")
    return result


def paired_bootstrap_ci(
    candidate_correct: Sequence[bool | int],
    baseline_correct: Sequence[bool | int],
    seed: int = 12345,
    n_resamples: int = 10000,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Compare paired candidate-vs-baseline binary outcomes by item index.

    Callers are responsible for aligning rows by question_id before passing the
    vectors. The bootstrap resamples paired item indices with replacement and
    computes candidate accuracy minus baseline accuracy for each resample.
    """
    candidate = _as_bool_list(candidate_correct, "candidate_correct")
    baseline = _as_bool_list(baseline_correct, "baseline_correct")
    return paired_bootstrap_mean_ci(
        candidate,
        baseline,
        seed=seed,
        n_resamples=n_resamples,
        confidence=confidence,
    )


def paired_bootstrap_mean_ci(
    candidate_scores: Sequence[float | int | bool],
    baseline_scores: Sequence[float | int | bool],
    seed: int = 12345,
    n_resamples: int = 10000,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Compare paired candidate-vs-baseline item scores by item index.

    Scores may be binary correctness values or continuous item accuracies in
    [0, 1], such as VQAv2/TextVQA soft scores.
    """
    candidate = [float(value) for value in candidate_scores]
    baseline = [float(value) for value in baseline_scores]
    if len(candidate) != len(baseline):
        raise ValueError(
            "candidate_scores and baseline_scores must have the same length; "
            f"got {len(candidate)} and {len(baseline)}"
        )
    if not candidate:
        return {
            "observed_delta": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value_two_sided": 1.0,
        }

    deltas = [
        float(cand) - float(base)
        for cand, base in zip(candidate, baseline)
    ]
    observed_delta = sum(deltas) / float(len(deltas))
    n_resamples = int(n_resamples)
    if n_resamples <= 0:
        return {
            "observed_delta": float(observed_delta),
            "ci_low": float(observed_delta),
            "ci_high": float(observed_delta),
            "p_value_two_sided": 1.0 if observed_delta == 0.0 else 0.0,
        }
    alpha = (1.0 - float(confidence)) / 2.0

    try:
        import numpy as np

        array = np.asarray(deltas, dtype=float)
        rng = np.random.default_rng(int(seed))
        indices = rng.integers(0, len(deltas), size=(n_resamples, len(deltas)))
        resampled = array[indices].mean(axis=1)
        ci_low, ci_high = np.quantile(resampled, [alpha, 1.0 - alpha])
        if observed_delta > 0:
            opposite = float(np.mean(resampled <= 0.0))
        elif observed_delta < 0:
            opposite = float(np.mean(resampled >= 0.0))
        else:
            opposite = 1.0
        return {
            "observed_delta": float(observed_delta),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            # Historical name from the V17 spec; value is the bootstrap
            # opposite-sign tail proportion.
            "p_value_two_sided": float(opposite),
        }
    except Exception:
        rng = random.Random(int(seed))
        resampled = []
        for _ in range(n_resamples):
            resampled.append(
                sum(deltas[rng.randrange(len(deltas))] for _ in range(len(deltas)))
                / float(len(deltas))
            )
        resampled.sort()
        ci_low = resampled[
            min(max(int(math.floor(alpha * (n_resamples - 1))), 0), n_resamples - 1)
        ]
        ci_high = resampled[
            min(max(int(math.ceil((1.0 - alpha) * (n_resamples - 1))), 0), n_resamples - 1)
        ]
        if observed_delta > 0:
            opposite = sum(1 for value in resampled if value <= 0.0) / float(n_resamples)
        elif observed_delta < 0:
            opposite = sum(1 for value in resampled if value >= 0.0) / float(n_resamples)
        else:
            opposite = 1.0
        return {
            "observed_delta": float(observed_delta),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "p_value_two_sided": float(opposite),
        }


def binary_ci_metrics(
    prefix: str,
    correct_values: Sequence[bool | int],
    seed: int = 12345,
    n_resamples: int = 10000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Return standard percent-scale CI fields for binary accuracy metrics."""
    values = [int(bool(value)) for value in correct_values]
    total = len(values)
    correct = int(sum(values))
    label = ci_label(confidence)
    wilson_low, wilson_high = wilson_ci(correct, total, confidence)
    boot_low, boot_high = bootstrap_mean_ci(
        values,
        seed=seed,
        n_resamples=n_resamples,
        confidence=confidence,
    )
    return {
        f"{prefix}_accuracy": 100.0 * correct / total if total else 0.0,
        f"{prefix}_correct": correct,
        f"{prefix}_total": total,
        f"{prefix}_ci_confidence": float(confidence),
        f"{prefix}_accuracy_{label}_binomial_low": 100.0 * wilson_low,
        f"{prefix}_accuracy_{label}_binomial_high": 100.0 * wilson_high,
        f"{prefix}_accuracy_{label}_bootstrap_low": 100.0 * boot_low,
        f"{prefix}_accuracy_{label}_bootstrap_high": 100.0 * boot_high,
        f"{prefix}_bootstrap_seed": int(seed),
        f"{prefix}_bootstrap_resamples": int(n_resamples),
    }


def mean_ci_metrics(
    prefix: str,
    values: Sequence[float | int | bool],
    seed: int = 12345,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    include_binomial_when_binary: bool = False,
) -> dict[str, Any]:
    """Return percent-scale bootstrap CI fields for possibly non-binary scores."""
    scores = [float(value) for value in values]
    total = len(scores)
    total_score = float(sum(scores))
    label = ci_label(confidence)
    boot_low, boot_high = bootstrap_mean_ci(
        scores,
        seed=seed,
        n_resamples=n_resamples,
        confidence=confidence,
    )
    metrics: dict[str, Any] = {
        f"{prefix}_accuracy": 100.0 * total_score / total if total else 0.0,
        f"{prefix}_total_score": total_score,
        f"{prefix}_total": total,
        f"{prefix}_ci_confidence": float(confidence),
        f"{prefix}_accuracy_{label}_bootstrap_low": 100.0 * boot_low,
        f"{prefix}_accuracy_{label}_bootstrap_high": 100.0 * boot_high,
        f"{prefix}_bootstrap_seed": int(seed),
        f"{prefix}_bootstrap_resamples": int(n_resamples),
    }
    is_binary = all(value in {0.0, 1.0} for value in scores)
    if include_binomial_when_binary and is_binary:
        wilson_low, wilson_high = wilson_ci(total_score, total, confidence)
        metrics[f"{prefix}_accuracy_{label}_binomial_low"] = 100.0 * wilson_low
        metrics[f"{prefix}_accuracy_{label}_binomial_high"] = 100.0 * wilson_high
    else:
        metrics[f"{prefix}_accuracy_{label}_binomial_low"] = None
        metrics[f"{prefix}_accuracy_{label}_binomial_high"] = None
        metrics[f"{prefix}_ci_note"] = "Wilson CI is not reported for non-binary item scores."
    return metrics
