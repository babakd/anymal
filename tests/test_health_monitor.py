"""Tests for TrainingHealthMonitor and ThroughputTracker.

All tests are CPU-only and require no GPU or model weights.
"""

import math
import time
from unittest.mock import patch

import pytest

import sys
sys.path.insert(0, "/Users/babakd/anymal")

from training.health_monitor import HealthMonitorConfig, TrainingHealthMonitor
from training.throughput_tracker import ThroughputTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_config(**overrides) -> HealthMonitorConfig:
    """Return a config with small windows for fast tests."""
    defaults = dict(
        divergence_window=5,
        plateau_window=10,
        grad_vanish_window=3,
        alert_cooldown=10,
    )
    defaults.update(overrides)
    return HealthMonitorConfig(**defaults)


def _make_monitor(**overrides) -> TrainingHealthMonitor:
    """Create a monitor with small-window config and optional overrides."""
    cfg = _small_config(**overrides)
    return TrainingHealthMonitor(cfg)


def _alert_types(monitor: TrainingHealthMonitor) -> list[str]:
    """Return the list of alert types fired so far (preserves duplicates)."""
    return [a["type"] for a in monitor._alerts]


def _alert_types_set(monitor: TrainingHealthMonitor) -> set[str]:
    """Return the set of unique alert types fired so far."""
    return {a["type"] for a in monitor._alerts}


# ===========================================================================
# HealthMonitorConfig defaults
# ===========================================================================

class TestHealthMonitorConfig:
    def test_default_values(self):
        cfg = HealthMonitorConfig()
        assert cfg.vocab_size == 128256
        assert cfg.initial_loss_tolerance == 0.20
        assert cfg.loss_spike_multiplier == 2.0
        assert cfg.loss_ema_alpha == 0.05
        assert cfg.divergence_window == 50
        assert cfg.plateau_window == 200
        assert cfg.plateau_min_improvement == 0.01
        assert cfg.grad_spike_multiplier == 5.0
        assert cfg.grad_vanish_threshold == 0.01
        assert cfg.grad_vanish_window == 10
        assert cfg.alert_cooldown == 100
        assert cfg.max_grad_norm == 1.0

    def test_override(self):
        cfg = HealthMonitorConfig(vocab_size=50000, alert_cooldown=5)
        assert cfg.vocab_size == 50000
        assert cfg.alert_cooldown == 5


# ===========================================================================
# Initial loss checks
# ===========================================================================

class TestInitialLoss:
    def test_initial_loss_within_tolerance(self):
        monitor = _make_monitor()
        expected = math.log(128256)  # ~11.76
        monitor.on_step(step=0, loss=expected, grad_norm=1.0)
        assert "initial_loss" not in _alert_types(monitor)

    def test_within_15_pct_no_alert(self):
        monitor = _make_monitor(initial_loss_tolerance=0.20)
        expected = math.log(128256)
        loss = expected * 1.15  # 15% deviation, within 20% tolerance
        monitor.on_step(step=0, loss=loss, grad_norm=1.0)
        assert "initial_loss" not in _alert_types(monitor)

    def test_initial_loss_too_low(self):
        monitor = _make_monitor()
        monitor.on_step(step=0, loss=5.0, grad_norm=1.0)
        assert "initial_loss" in _alert_types(monitor)

    def test_initial_loss_too_high(self):
        monitor = _make_monitor()
        monitor.on_step(step=0, loss=20.0, grad_norm=1.0)
        assert "initial_loss" in _alert_types(monitor)

    def test_25_pct_above_expected_triggers_alert(self):
        monitor = _make_monitor(initial_loss_tolerance=0.20)
        expected = math.log(128256)
        monitor.on_step(step=0, loss=expected * 1.25, grad_norm=1.0)
        assert "initial_loss" in _alert_types(monitor)

    def test_only_checked_on_first_step(self):
        """Initial loss check fires only on the very first on_step call."""
        monitor = _make_monitor(initial_loss_tolerance=0.20)
        expected = math.log(128256)
        monitor.on_step(step=0, loss=expected, grad_norm=0.5)
        # Second step with bad loss -- initial check already done
        monitor.on_step(step=1, loss=1.0, grad_norm=0.5)
        assert "initial_loss" not in _alert_types(monitor)


# ===========================================================================
# Loss spike
# ===========================================================================

class TestLossSpike:
    def test_loss_spike_detection(self):
        monitor = _make_monitor(loss_spike_multiplier=2.0)
        # Feed a few normal losses to establish EMA
        for i in range(20):
            monitor.on_step(step=i, loss=10.0, grad_norm=1.0)
        # Now inject a spike (> 2x EMA)
        monitor.on_step(step=20, loss=25.0, grad_norm=1.0)
        assert "loss_spike" in _alert_types(monitor)

    def test_spike_detected_after_one_baseline(self):
        monitor = _make_monitor(loss_spike_multiplier=2.0, alert_cooldown=0)
        monitor.on_step(step=0, loss=2.0, grad_norm=0.5)
        # Step 1: EMA ~ 2.0, spike at 5.0 > 2 * 2.0
        monitor.on_step(step=1, loss=5.0, grad_norm=0.5)
        assert "loss_spike" in _alert_types(monitor)

    def test_no_spike_when_normal(self):
        monitor = _make_monitor(loss_spike_multiplier=2.0)
        monitor.on_step(step=0, loss=2.0, grad_norm=0.5)
        monitor.on_step(step=1, loss=2.1, grad_norm=0.5)
        assert "loss_spike" not in _alert_types(monitor)

    def test_spike_not_on_first_step(self):
        """First step cannot spike because _total_steps == 1."""
        monitor = _make_monitor(loss_spike_multiplier=2.0)
        monitor.on_step(step=0, loss=100.0, grad_norm=0.5)
        assert "loss_spike" not in _alert_types(monitor)


# ===========================================================================
# Loss divergence
# ===========================================================================

class TestLossDivergence:
    def test_monotonically_increasing_triggers_alert(self):
        monitor = _make_monitor(divergence_window=5, alert_cooldown=0)
        for i in range(5):
            monitor.on_step(step=i, loss=1.0 + i * 0.5, grad_norm=0.5)
        assert "loss_divergence" in _alert_types(monitor)

    def test_non_monotonic_no_alert(self):
        monitor = _make_monitor(divergence_window=5)
        losses = [1.0, 1.5, 1.3, 1.8, 2.0]  # dip at index 2
        for i, loss in enumerate(losses):
            monitor.on_step(step=i, loss=loss, grad_norm=0.5)
        assert "loss_divergence" not in _alert_types(monitor)

    def test_window_not_full_no_alert(self):
        monitor = _make_monitor(divergence_window=5)
        for i in range(4):  # only 4 of 5
            monitor.on_step(step=i, loss=1.0 + i, grad_norm=0.5)
        assert "loss_divergence" not in _alert_types(monitor)

    def test_divergence_with_default_window(self):
        monitor = TrainingHealthMonitor(
            HealthMonitorConfig(divergence_window=50, alert_cooldown=0)
        )
        for i in range(50):
            monitor.on_step(step=i, loss=10.0 + i * 0.1, grad_norm=1.0)
        assert "loss_divergence" in _alert_types_set(monitor)


# ===========================================================================
# Loss plateau
# ===========================================================================

class TestLossPlateau:
    def test_plateau_detected(self):
        monitor = _make_monitor(
            plateau_window=10,
            plateau_min_improvement=0.01,
            alert_cooldown=0,
        )
        for i in range(10):
            monitor.on_step(step=i, loss=5.0, grad_norm=0.5)
        assert "loss_plateau" in _alert_types(monitor)

    def test_improving_loss_no_plateau(self):
        monitor = _make_monitor(
            plateau_window=10,
            plateau_min_improvement=0.01,
            alert_cooldown=0,
        )
        for i in range(10):
            monitor.on_step(step=i, loss=5.0 - i * 0.1, grad_norm=0.5)
        assert "loss_plateau" not in _alert_types(monitor)

    def test_window_not_full_no_alert(self):
        monitor = _make_monitor(plateau_window=10)
        for i in range(9):
            monitor.on_step(step=i, loss=5.0, grad_norm=0.5)
        assert "loss_plateau" not in _alert_types(monitor)

    def test_plateau_with_large_window(self):
        monitor = TrainingHealthMonitor(
            HealthMonitorConfig(plateau_window=200, plateau_min_improvement=0.01, alert_cooldown=0)
        )
        for i in range(200):
            monitor.on_step(step=i, loss=10.0, grad_norm=1.0)
        assert "loss_plateau" in _alert_types_set(monitor)


# ===========================================================================
# Gradient spike
# ===========================================================================

class TestGradSpike:
    def test_grad_spike_detected(self):
        monitor = _make_monitor(grad_spike_multiplier=5.0)
        # Establish stable baseline
        for i in range(10):
            monitor.on_step(step=i, loss=5.0, grad_norm=1.0)
        # EMA is ~1.0, spike at 10x > 5x threshold
        monitor.on_step(step=10, loss=5.0, grad_norm=10.0)
        assert "grad_spike" in _alert_types(monitor)

    def test_no_grad_spike_when_normal(self):
        monitor = _make_monitor(grad_spike_multiplier=5.0)
        monitor.on_step(step=0, loss=5.0, grad_norm=1.0)
        monitor.on_step(step=1, loss=5.0, grad_norm=1.5)
        assert "grad_spike" not in _alert_types(monitor)

    def test_grad_spike_not_on_first_step(self):
        monitor = _make_monitor(grad_spike_multiplier=5.0)
        monitor.on_step(step=0, loss=5.0, grad_norm=100.0)
        assert "grad_spike" not in _alert_types(monitor)

    def test_grad_spike_after_warmup(self):
        monitor = _make_monitor(grad_spike_multiplier=5.0, alert_cooldown=0)
        for i in range(20):
            monitor.on_step(step=i, loss=10.0, grad_norm=1.0)
        monitor.on_step(step=20, loss=10.0, grad_norm=10.0)
        assert "grad_spike" in _alert_types_set(monitor)


# ===========================================================================
# Gradient vanishing
# ===========================================================================

class TestGradVanishing:
    def test_vanishing_detected(self):
        monitor = _make_monitor(
            grad_vanish_threshold=0.01,
            grad_vanish_window=3,
            alert_cooldown=0,
        )
        for i in range(3):
            monitor.on_step(step=i, loss=5.0, grad_norm=0.001)
        assert "grad_vanishing" in _alert_types(monitor)

    def test_no_vanish_if_interrupted(self):
        monitor = _make_monitor(
            grad_vanish_threshold=0.01,
            grad_vanish_window=3,
        )
        monitor.on_step(step=0, loss=5.0, grad_norm=0.001)
        monitor.on_step(step=1, loss=5.0, grad_norm=0.001)
        # Normal step breaks the streak
        monitor.on_step(step=2, loss=5.0, grad_norm=0.5)
        monitor.on_step(step=3, loss=5.0, grad_norm=0.001)
        assert "grad_vanishing" not in _alert_types(monitor)

    def test_vanish_counter_resets_on_normal_grad(self):
        monitor = _make_monitor(grad_vanish_threshold=0.01, grad_vanish_window=3)
        monitor.on_step(step=0, loss=5.0, grad_norm=0.001)
        monitor.on_step(step=1, loss=5.0, grad_norm=0.5)
        assert monitor._consecutive_vanish == 0

    def test_zero_grad_norm_counted(self):
        monitor = _make_monitor(
            grad_vanish_threshold=0.01,
            grad_vanish_window=3,
            alert_cooldown=0,
        )
        for i in range(3):
            monitor.on_step(step=i, loss=5.0, grad_norm=0.0)
        assert "grad_vanishing" in _alert_types(monitor)

    def test_vanish_with_default_window(self):
        monitor = TrainingHealthMonitor(
            HealthMonitorConfig(grad_vanish_threshold=0.01, grad_vanish_window=10, alert_cooldown=0)
        )
        for i in range(10):
            monitor.on_step(step=i, loss=10.0, grad_norm=0.001)
        assert "grad_vanishing" in _alert_types_set(monitor)


# ===========================================================================
# Gradient clipping tracking
# ===========================================================================

class TestGradClipping:
    def test_clipped_steps_tracked(self):
        monitor = _make_monitor(max_grad_norm=1.0)
        monitor.on_step(step=0, loss=5.0, grad_norm=1.0, grad_norm_before_clip=2.0)
        monitor.on_step(step=1, loss=5.0, grad_norm=1.0, grad_norm_before_clip=0.5)
        monitor.on_step(step=2, loss=5.0, grad_norm=1.0, grad_norm_before_clip=3.0)
        assert monitor._clipped_steps == 2
        assert monitor._total_steps == 3

    def test_no_clipping_when_none(self):
        monitor = _make_monitor()
        monitor.on_step(step=0, loss=5.0, grad_norm=0.5)
        assert monitor._clipped_steps == 0

    def test_clip_fraction_in_summary(self):
        monitor = _make_monitor(max_grad_norm=1.0)
        monitor.on_step(step=0, loss=10.0, grad_norm=0.5, grad_norm_before_clip=0.5)
        monitor.on_step(step=1, loss=10.0, grad_norm=1.0, grad_norm_before_clip=2.0)
        monitor.on_step(step=2, loss=10.0, grad_norm=0.8, grad_norm_before_clip=0.8)
        monitor.on_step(step=3, loss=10.0, grad_norm=1.0, grad_norm_before_clip=3.0)
        summary = monitor.get_summary()
        assert summary["grad_clip_fraction"] == pytest.approx(0.5)


# ===========================================================================
# Validation monitoring
# ===========================================================================

class TestValidation:
    def test_val_loss_increasing_alert(self):
        monitor = _make_monitor(alert_cooldown=0)
        monitor.on_step(step=0, loss=2.0, grad_norm=0.5)
        monitor.on_eval(step=100, train_loss_ema=2.0, val_loss=2.5)
        monitor.on_eval(step=200, train_loss_ema=2.0, val_loss=2.7)
        monitor.on_eval(step=300, train_loss_ema=2.0, val_loss=3.0)
        assert "val_loss_increasing" in _alert_types(monitor)

    def test_val_loss_not_increasing_no_alert(self):
        monitor = _make_monitor(alert_cooldown=0)
        monitor.on_step(step=0, loss=2.0, grad_norm=0.5)
        monitor.on_eval(step=100, train_loss_ema=2.0, val_loss=2.5)
        monitor.on_eval(step=200, train_loss_ema=2.0, val_loss=2.3)  # decreases
        monitor.on_eval(step=300, train_loss_ema=2.0, val_loss=2.4)
        assert "val_loss_increasing" not in _alert_types(monitor)

    def test_val_gap_alert(self):
        monitor = _make_monitor(alert_cooldown=0)
        for i in range(10):
            monitor.on_step(step=i, loss=5.0, grad_norm=1.0)
        # 3 eval points with increasing train-val gap
        monitor.on_eval(step=10, train_loss_ema=5.0, val_loss=5.5)
        monitor.on_eval(step=20, train_loss_ema=5.0, val_loss=6.0)
        monitor.on_eval(step=30, train_loss_ema=5.0, val_loss=6.5)
        assert "overfitting" in _alert_types(monitor)

    def test_no_overfitting_when_gap_shrinks(self):
        monitor = _make_monitor(alert_cooldown=0)
        monitor.on_step(step=0, loss=2.0, grad_norm=0.5)
        monitor.on_eval(step=100, train_loss_ema=2.0, val_loss=3.0)
        monitor.on_eval(step=200, train_loss_ema=2.0, val_loss=2.5)  # gap shrinks
        monitor.on_eval(step=300, train_loss_ema=2.0, val_loss=2.8)
        assert "overfitting" not in _alert_types(monitor)

    def test_needs_at_least_3_evals(self):
        monitor = _make_monitor(alert_cooldown=0)
        monitor.on_step(step=0, loss=2.0, grad_norm=0.5)
        monitor.on_eval(step=100, train_loss_ema=2.0, val_loss=2.5)
        monitor.on_eval(step=200, train_loss_ema=2.0, val_loss=3.0)
        assert "val_loss_increasing" not in _alert_types(monitor)
        assert "overfitting" not in _alert_types(monitor)


# ===========================================================================
# Alert cooldown
# ===========================================================================

class TestCooldown:
    def test_cooldown_suppresses_duplicate(self):
        monitor = _make_monitor(
            alert_cooldown=10,
            grad_vanish_threshold=0.01,
            grad_vanish_window=3,
        )
        # Trigger vanishing alert at step 2
        for i in range(3):
            monitor.on_step(step=i, loss=5.0, grad_norm=0.001)
        vanish_count = _alert_types(monitor).count("grad_vanishing")
        assert vanish_count == 1

        # Steps 3-5 still vanishing but within cooldown (step 2 + 10 = 12)
        for i in range(3, 6):
            monitor.on_step(step=i, loss=5.0, grad_norm=0.001)
        vanish_count = _alert_types(monitor).count("grad_vanishing")
        assert vanish_count == 1, "Alert should not repeat within cooldown"

    def test_cooldown_allows_after_expiry(self):
        monitor = _make_monitor(
            alert_cooldown=5,
            grad_vanish_threshold=0.01,
            grad_vanish_window=3,
        )
        # First alert at step 2
        for i in range(3):
            monitor.on_step(step=i, loss=5.0, grad_norm=0.001)
        assert _alert_types(monitor).count("grad_vanishing") == 1

        # Continue vanishing, step 7 is past cooldown (2 + 5 = 7)
        for i in range(3, 8):
            monitor.on_step(step=i, loss=5.0, grad_norm=0.001)
        assert _alert_types(monitor).count("grad_vanishing") == 2

    def test_different_alert_types_independent_cooldown(self):
        """Cooldown for one alert type does not affect another."""
        monitor = _make_monitor(
            alert_cooldown=1000,
            grad_vanish_threshold=0.01,
            grad_vanish_window=3,
            loss_spike_multiplier=2.0,
        )
        # Trigger vanishing at step 2
        for i in range(3):
            monitor.on_step(step=i, loss=5.0, grad_norm=0.001)
        assert "grad_vanishing" in _alert_types(monitor)

        # Now trigger loss spike at step 3 -- different type, not blocked
        monitor.on_step(step=3, loss=50.0, grad_norm=0.001)
        assert "loss_spike" in _alert_types(monitor)

    def test_cooldown_with_default_value(self):
        monitor = TrainingHealthMonitor(
            HealthMonitorConfig(
                alert_cooldown=100,
                grad_vanish_threshold=0.01,
                grad_vanish_window=10,
            )
        )
        for i in range(10):
            monitor.on_step(step=i, loss=10.0, grad_norm=0.001)
        assert _alert_types(monitor).count("grad_vanishing") == 1

        # Continue within cooldown window
        for i in range(10, 50):
            monitor.on_step(step=i, loss=10.0, grad_norm=0.001)
        vanish_alerts = [a for a in monitor._alerts if a["type"] == "grad_vanishing"]
        assert len(vanish_alerts) == 1


# ===========================================================================
# get_summary
# ===========================================================================

class TestGetSummary:
    def test_summary_keys(self):
        monitor = _make_monitor()
        monitor.on_step(step=0, loss=5.0, grad_norm=0.5)
        summary = monitor.get_summary()
        assert "loss_ema" in summary
        assert "grad_norm_ema" in summary
        assert "grad_clip_fraction" in summary
        assert "steps_since_improvement" in summary

    def test_summary_zero_steps(self):
        monitor = _make_monitor()
        summary = monitor.get_summary()
        assert summary["grad_clip_fraction"] == 0.0
        assert summary["loss_ema"] is None

    def test_summary_values_after_steps(self):
        monitor = _make_monitor()
        monitor.on_step(step=0, loss=10.0, grad_norm=2.0)
        summary = monitor.get_summary()
        assert summary["loss_ema"] == pytest.approx(10.0)
        assert summary["grad_norm_ema"] == pytest.approx(2.0)
        assert summary["steps_since_improvement"] == 0


# ===========================================================================
# State serialization roundtrip
# ===========================================================================

class TestSerialization:
    def test_state_roundtrip(self):
        monitor = _make_monitor(
            divergence_window=5,
            plateau_window=10,
            alert_cooldown=0,
            grad_vanish_window=3,
        )
        for i in range(8):
            monitor.on_step(step=i, loss=5.0 - i * 0.1, grad_norm=0.5 + i * 0.01)
        monitor.on_eval(step=100, train_loss_ema=4.0, val_loss=4.5)

        state = monitor.get_state()

        restored = _make_monitor(
            divergence_window=5,
            plateau_window=10,
            alert_cooldown=0,
            grad_vanish_window=3,
        )
        restored.load_state(state)

        assert restored.loss_ema == pytest.approx(monitor.loss_ema)
        assert restored.grad_norm_ema == pytest.approx(monitor.grad_norm_ema)
        assert restored.loss_ema_min == pytest.approx(monitor.loss_ema_min)
        assert restored.steps_since_improvement == monitor.steps_since_improvement
        assert list(restored._recent_losses) == list(monitor._recent_losses)
        assert list(restored._ema_history) == list(monitor._ema_history)
        assert restored._consecutive_vanish == monitor._consecutive_vanish
        assert restored._total_steps == monitor._total_steps
        assert restored._clipped_steps == monitor._clipped_steps
        assert restored._val_history == monitor._val_history
        assert restored._train_val_gaps == monitor._train_val_gaps
        assert restored._alerts == monitor._alerts
        assert restored._last_alert_step == monitor._last_alert_step
        assert restored._first_step == monitor._first_step

    def test_restored_monitor_continues_correctly(self):
        """After restoring, the monitor should behave as if it never stopped."""
        monitor = _make_monitor(divergence_window=5, alert_cooldown=0)
        for i in range(3):
            monitor.on_step(step=i, loss=1.0 + i, grad_norm=0.5)

        state = monitor.get_state()
        restored = _make_monitor(divergence_window=5, alert_cooldown=0)
        restored.load_state(state)

        # Feed 2 more increasing losses to complete the window of 5
        restored.on_step(step=3, loss=4.0, grad_norm=0.5)
        restored.on_step(step=4, loss=5.0, grad_norm=0.5)
        assert "loss_divergence" in _alert_types(restored)

    def test_state_with_default_config(self):
        monitor = TrainingHealthMonitor(HealthMonitorConfig())
        for i in range(25):
            monitor.on_step(step=i, loss=10.0 + i * 0.01, grad_norm=1.0)
        monitor.on_eval(step=25, train_loss_ema=monitor.loss_ema, val_loss=11.0)

        state = monitor.get_state()
        monitor2 = TrainingHealthMonitor(HealthMonitorConfig())
        monitor2.load_state(state)

        assert monitor2.loss_ema == pytest.approx(monitor.loss_ema)
        assert monitor2.grad_norm_ema == pytest.approx(monitor.grad_norm_ema)
        assert monitor2._total_steps == monitor._total_steps
        assert monitor2._clipped_steps == monitor._clipped_steps
        assert monitor2.steps_since_improvement == monitor.steps_since_improvement
        assert list(monitor2._recent_losses) == list(monitor._recent_losses)
        assert list(monitor2._ema_history) == list(monitor._ema_history)
        assert monitor2._val_history == monitor._val_history
        assert monitor2._first_step == monitor._first_step


# ===========================================================================
# EMA behavior
# ===========================================================================

class TestEMABehavior:
    def test_ema_initialised_to_first_value(self):
        monitor = _make_monitor()
        monitor.on_step(step=0, loss=10.0, grad_norm=2.0)
        assert monitor.loss_ema == pytest.approx(10.0)
        assert monitor.grad_norm_ema == pytest.approx(2.0)

    def test_ema_update(self):
        monitor = _make_monitor(loss_ema_alpha=0.1)
        monitor.on_step(step=0, loss=10.0, grad_norm=1.0)
        monitor.on_step(step=1, loss=5.0, grad_norm=1.0)
        # EMA = 0.1 * 5.0 + 0.9 * 10.0 = 9.5
        assert monitor.loss_ema == pytest.approx(9.5)

    def test_steps_since_improvement_increases(self):
        monitor = _make_monitor()
        monitor.on_step(step=0, loss=5.0, grad_norm=0.5)
        monitor.on_step(step=1, loss=5.0, grad_norm=0.5)
        monitor.on_step(step=2, loss=5.0, grad_norm=0.5)
        # EMA stays at 5.0 after first step, no improvement
        assert monitor.steps_since_improvement >= 1

    def test_steps_since_improvement_resets(self):
        monitor = _make_monitor()
        monitor.on_step(step=0, loss=5.0, grad_norm=0.5)
        monitor.on_step(step=1, loss=5.0, grad_norm=0.5)
        # Drop loss so EMA decreases below previous min
        monitor.on_step(step=2, loss=1.0, grad_norm=0.5)
        # With alpha=0.05: EMA = 0.05 * 1.0 + 0.95 * 5.0 = 4.8 < 5.0
        assert monitor.steps_since_improvement == 0


# ===========================================================================
# Edge cases and integration
# ===========================================================================

class TestEdgeCases:
    def test_many_steps_no_crash(self):
        """Smoke test: run many steps without error."""
        monitor = _make_monitor()
        for i in range(200):
            monitor.on_step(step=i, loss=5.0 + (i % 3) * 0.1, grad_norm=0.5)
        assert monitor._total_steps == 200

    def test_alert_record_fields(self):
        monitor = _make_monitor(alert_cooldown=0)
        monitor.on_step(step=0, loss=1.0, grad_norm=0.5)  # triggers initial_loss
        alert = monitor._alerts[0]
        assert "type" in alert
        assert "message" in alert
        assert "severity" in alert
        assert "step" in alert
        assert "time" in alert
        assert alert["type"] == "initial_loss"
        assert alert["severity"] == "warning"
        assert alert["step"] == 0

    def test_wandb_logger_called(self):
        class FakeWandb:
            def __init__(self):
                self.calls = []
            def alert(self, title, text, level):
                self.calls.append((title, text, level))

        fake_wandb = FakeWandb()
        cfg = _small_config(alert_cooldown=0)
        monitor = TrainingHealthMonitor(cfg, wandb_logger=fake_wandb)
        monitor.on_step(step=0, loss=1.0, grad_norm=0.5)  # triggers initial_loss
        assert len(fake_wandb.calls) >= 1
        assert fake_wandb.calls[0][0] == "initial_loss"

    def test_wandb_logger_exception_ignored(self):
        """wandb errors should not propagate."""
        class BadWandb:
            def alert(self, **kwargs):
                raise RuntimeError("W&B is down")

        cfg = _small_config(alert_cooldown=0)
        monitor = TrainingHealthMonitor(cfg, wandb_logger=BadWandb())
        # Should not raise
        monitor.on_step(step=0, loss=1.0, grad_norm=0.5)
        assert "initial_loss" in _alert_types(monitor)


# ===========================================================================
# ThroughputTracker
# ===========================================================================

class TestThroughputTracker:
    def test_initial_metrics_zero(self):
        tracker = ThroughputTracker(window_size=5)
        metrics = tracker.get_metrics()
        assert metrics["steps_per_sec"] == 0.0
        assert metrics["tokens_per_sec"] == 0.0
        assert metrics["samples_per_sec"] == 0.0

    def test_single_step_returns_zero(self):
        """Need at least 2 timestamps for a rate."""
        tracker = ThroughputTracker(window_size=5)
        tracker.step(batch_size=4, seq_len=512)
        metrics = tracker.get_metrics()
        assert metrics["steps_per_sec"] == 0.0

    def test_total_metrics_accumulate(self):
        tracker = ThroughputTracker(window_size=5)
        tracker.step(batch_size=4, seq_len=512)
        tracker.step(batch_size=4, seq_len=512)
        totals = tracker.get_total_metrics()
        assert totals["total_steps"] == 2
        assert totals["total_tokens"] == 2 * 4 * 512
        assert totals["total_samples"] == 8
        assert totals["elapsed_sec"] > 0

    def test_window_metrics_positive_with_delay(self):
        """With a small sleep between steps, rates should be positive."""
        tracker = ThroughputTracker(window_size=5)
        tracker.step(batch_size=4, seq_len=128)
        time.sleep(0.05)
        tracker.step(batch_size=4, seq_len=128)
        metrics = tracker.get_metrics()
        assert metrics["steps_per_sec"] > 0
        assert metrics["tokens_per_sec"] > 0
        assert metrics["samples_per_sec"] > 0

    def test_window_metrics_with_mock_time(self):
        """Use mocked time to get deterministic throughput values."""
        tracker = ThroughputTracker(window_size=10)

        with patch("training.throughput_tracker.time") as mock_time:
            mock_time.time.return_value = 1000.0
            tracker._start_time = 1000.0

            # Step 1 at t=1000
            tracker.step(batch_size=4, seq_len=256)

            # Step 2 at t=1001
            mock_time.time.return_value = 1001.0
            tracker.step(batch_size=4, seq_len=256)

            # Step 3 at t=1002
            mock_time.time.return_value = 1002.0
            tracker.step(batch_size=4, seq_len=256)

            metrics = tracker.get_metrics()
            # dt = 1002 - 1000 = 2 seconds, 2 steps (excl first)
            assert metrics["steps_per_sec"] == pytest.approx(1.0)
            # tokens per step = 4 * 256 = 1024, 2 steps / 2 sec
            assert metrics["tokens_per_sec"] == pytest.approx(1024.0)
            # samples per step = 4, 2 steps / 2 sec
            assert metrics["samples_per_sec"] == pytest.approx(4.0)

    def test_window_slides(self):
        """When window is full, oldest entries are dropped."""
        tracker = ThroughputTracker(window_size=3)
        with patch("training.throughput_tracker.time") as mock_time:
            tracker._start_time = 0.0

            mock_time.time.return_value = 0.0
            tracker.step(batch_size=1, seq_len=100)

            mock_time.time.return_value = 1.0
            tracker.step(batch_size=1, seq_len=100)

            mock_time.time.return_value = 2.0
            tracker.step(batch_size=1, seq_len=100)

            assert len(tracker._timestamps) == 3

            mock_time.time.return_value = 3.0
            tracker.step(batch_size=2, seq_len=100)

            # Window is now [1, 2, 3] -- oldest dropped
            assert len(tracker._timestamps) == 3
            assert tracker._timestamps[0] == 1.0

        # Total should still track everything
        assert tracker._total_steps == 4

    def test_total_metrics_elapsed(self):
        tracker = ThroughputTracker(window_size=5)
        with patch("training.throughput_tracker.time") as mock_time:
            mock_time.time.return_value = 100.0
            tracker._start_time = 100.0

            mock_time.time.return_value = 110.0
            totals = tracker.get_total_metrics()
            assert totals["elapsed_sec"] == pytest.approx(10.0)

    def test_throughput_deterministic(self):
        """Verify exact throughput with controlled timing."""
        tracker = ThroughputTracker(window_size=10)
        base_time = 1000.0
        times = [base_time + i * 0.1 for i in range(6)]  # 0.1s apart
        call_idx = 0
        original_time = time.time

        def mock_time():
            nonlocal call_idx
            if call_idx < len(times):
                t = times[call_idx]
                call_idx += 1
                return t
            return original_time()

        with patch("time.time", side_effect=mock_time):
            tracker._start_time = times[0]
            for _ in range(5):
                tracker.step(batch_size=4, seq_len=512)

        metrics = tracker.get_metrics()
        # dt = times[4] - times[0] = 0.4s, n_steps = 4
        assert metrics["steps_per_sec"] == pytest.approx(10.0, rel=0.01)
        # tokens_per_sec: 4 * (4*512) / 0.4 = 20480
        assert metrics["tokens_per_sec"] == pytest.approx(20480.0, rel=0.01)
        # samples_per_sec: 4 * 4 / 0.4 = 40.0
        assert metrics["samples_per_sec"] == pytest.approx(40.0, rel=0.01)

        total = tracker.get_total_metrics()
        assert total["total_steps"] == 5
        assert total["total_tokens"] == 5 * 4 * 512
        assert total["total_samples"] == 5 * 4
