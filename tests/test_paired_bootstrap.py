from evaluation.checkpoint_eval.paired_bootstrap import paired_bootstrap_ci


def test_paired_bootstrap_identical_models_contains_zero():
    candidate = [True, False, True, False] * 25
    baseline = list(candidate)

    result = paired_bootstrap_ci(candidate, baseline, seed=1, n_resamples=1000)

    assert result["observed_delta"] == 0.0
    assert result["ci_low"] <= 0.0 <= result["ci_high"]


def test_paired_bootstrap_strictly_better_excludes_zero():
    candidate = [True] * 100
    baseline = [False] * 100

    result = paired_bootstrap_ci(candidate, baseline, seed=2, n_resamples=1000)

    assert result["observed_delta"] == 1.0
    assert result["ci_low"] > 0.0
    assert result["ci_high"] > 0.0


def test_paired_bootstrap_known_delta_coverage_across_seeds():
    baseline = [True] * 500 + [False] * 500
    candidate = [True] * 550 + [False] * 450

    hits = 0
    for seed in range(100):
        result = paired_bootstrap_ci(
            candidate,
            baseline,
            seed=seed,
            n_resamples=1000,
        )
        hits += int(result["ci_low"] <= 0.05 <= result["ci_high"])

    assert hits >= 95
