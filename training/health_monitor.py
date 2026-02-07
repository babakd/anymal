"""Training health monitor for AnyMAL.

Standalone module that receives scalar values (loss, grad norm) each step
and emits alerts when training looks unhealthy.  Zero dependencies on
torch / trainer / model -- only standard-library imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from collections import deque


@dataclass
class HealthMonitorConfig:
    vocab_size: int = 128256  # LLaMA-3 vocab size
    initial_loss_tolerance: float = 0.20  # 20% deviation from ln(V) triggers alert
    loss_spike_multiplier: float = 2.0  # loss > 2x EMA triggers spike alert
    loss_ema_alpha: float = 0.05  # EMA smoothing factor
    divergence_window: int = 50  # alert if loss increases over this many steps
    plateau_window: int = 200  # alert if loss EMA doesn't decrease over this many steps
    plateau_min_improvement: float = 0.01  # minimum EMA decrease to not be plateau
    grad_spike_multiplier: float = 5.0  # grad norm > 5x running avg triggers alert
    grad_vanish_threshold: float = 0.01  # threshold for vanishing
    grad_vanish_window: int = 10  # consecutive steps below threshold
    alert_cooldown: int = 100  # don't repeat same alert type within this many steps
    max_grad_norm: float = 1.0  # for tracking clipping frequency


class TrainingHealthMonitor:
    """Monitors training scalars and emits alerts for anomalies.

    Call ``on_step`` after every optimizer step and ``on_eval`` after each
    validation run.  The monitor keeps lightweight running statistics and
    fires alerts (via print + optional W&B) when things look wrong.
    """

    def __init__(self, config: HealthMonitorConfig, wandb_logger=None):
        self.config = config
        self.wandb_logger = wandb_logger

        # --- running statistics ---
        self.loss_ema: float | None = None
        self.grad_norm_ema: float | None = None
        self.loss_ema_min: float | None = None  # best (lowest) EMA so far
        self.steps_since_improvement: int = 0

        # loss divergence tracking -- keep last `divergence_window` losses
        self._recent_losses: deque = deque(maxlen=config.divergence_window)

        # loss plateau tracking -- keep last `plateau_window` EMA snapshots
        self._ema_history: deque = deque(maxlen=config.plateau_window)

        # gradient vanishing tracking
        self._consecutive_vanish: int = 0

        # gradient clipping tracking
        self._total_steps: int = 0
        self._clipped_steps: int = 0

        # validation tracking
        self._val_history: list[tuple[int, float]] = []  # (step, val_loss)
        self._train_val_gaps: list[tuple[int, float]] = []  # (step, gap)

        # alert bookkeeping
        self._alerts: list[dict] = []
        self._last_alert_step: dict[str, int] = {}

        # flag to know if on_step has been called yet
        self._first_step = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_step(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        grad_norm_before_clip: float | None = None,
    ) -> None:
        """Called after each optimizer step with scalar metrics."""

        self._total_steps += 1

        # --- first-step sanity check ---
        if self._first_step:
            self._first_step = False
            expected = math.log(self.config.vocab_size)
            deviation = abs(loss - expected) / expected
            if deviation > self.config.initial_loss_tolerance:
                self._alert(
                    "initial_loss",
                    f"Initial loss {loss:.4f} deviates {deviation:.1%} from "
                    f"expected ln(vocab_size)={expected:.4f}",
                    severity="warning",
                    step=step,
                )

        # --- update EMAs ---
        alpha = self.config.loss_ema_alpha
        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            self.loss_ema = alpha * loss + (1 - alpha) * self.loss_ema

        if self.grad_norm_ema is None:
            self.grad_norm_ema = grad_norm
        else:
            self.grad_norm_ema = alpha * grad_norm + (1 - alpha) * self.grad_norm_ema

        # --- track best EMA / steps since improvement ---
        if self.loss_ema_min is None or self.loss_ema < self.loss_ema_min:
            self.loss_ema_min = self.loss_ema
            self.steps_since_improvement = 0
        else:
            self.steps_since_improvement += 1

        # --- loss spike ---
        # (skip on very first step since EMA == loss)
        if self._total_steps > 1 and loss > self.loss_ema * self.config.loss_spike_multiplier:
            self._alert(
                "loss_spike",
                f"Loss spike at step {step}: {loss:.4f} > "
                f"{self.config.loss_spike_multiplier}x EMA ({self.loss_ema:.4f})",
                severity="warning",
                step=step,
            )

        # --- loss divergence (monotonically increasing over window) ---
        self._recent_losses.append(loss)
        if len(self._recent_losses) == self.config.divergence_window:
            losses = list(self._recent_losses)
            if all(losses[i] < losses[i + 1] for i in range(len(losses) - 1)):
                self._alert(
                    "loss_divergence",
                    f"Loss has increased monotonically over the last "
                    f"{self.config.divergence_window} steps (step {step})",
                    severity="error",
                    step=step,
                )

        # --- loss plateau ---
        self._ema_history.append(self.loss_ema)
        if len(self._ema_history) == self.config.plateau_window:
            oldest = self._ema_history[0]
            newest = self._ema_history[-1]
            if oldest - newest < self.config.plateau_min_improvement:
                self._alert(
                    "loss_plateau",
                    f"Loss EMA has not decreased by >= {self.config.plateau_min_improvement} "
                    f"over the last {self.config.plateau_window} steps (step {step}). "
                    f"EMA: {oldest:.4f} -> {newest:.4f}",
                    severity="warning",
                    step=step,
                )

        # --- gradient spike ---
        if self._total_steps > 1 and grad_norm > self.grad_norm_ema * self.config.grad_spike_multiplier:
            self._alert(
                "grad_spike",
                f"Gradient spike at step {step}: grad_norm={grad_norm:.4f} > "
                f"{self.config.grad_spike_multiplier}x EMA ({self.grad_norm_ema:.4f})",
                severity="warning",
                step=step,
            )

        # --- gradient vanishing ---
        if grad_norm < self.config.grad_vanish_threshold:
            self._consecutive_vanish += 1
        else:
            self._consecutive_vanish = 0

        if self._consecutive_vanish >= self.config.grad_vanish_window:
            self._alert(
                "grad_vanishing",
                f"Gradient norm below {self.config.grad_vanish_threshold} for "
                f"{self._consecutive_vanish} consecutive steps (step {step})",
                severity="warning",
                step=step,
            )

        # --- grad clipping frequency ---
        if grad_norm_before_clip is not None and grad_norm_before_clip > self.config.max_grad_norm:
            self._clipped_steps += 1

    def on_eval(self, step: int, train_loss_ema: float, val_loss: float) -> None:
        """Called after each validation run."""

        self._val_history.append((step, val_loss))
        self._train_val_gaps.append((step, val_loss - train_loss_ema))

        # --- val loss increasing for 3 consecutive eval points ---
        if len(self._val_history) >= 3:
            last3 = [v for _, v in self._val_history[-3:]]
            if last3[0] < last3[1] < last3[2]:
                self._alert(
                    "val_loss_increasing",
                    f"Validation loss has increased for 3 consecutive evals: "
                    f"{last3[0]:.4f} -> {last3[1]:.4f} -> {last3[2]:.4f} (step {step})",
                    severity="warning",
                    step=step,
                )

        # --- train/val gap growing for 3 consecutive eval points ---
        if len(self._train_val_gaps) >= 3:
            last3_gaps = [g for _, g in self._train_val_gaps[-3:]]
            if last3_gaps[0] < last3_gaps[1] < last3_gaps[2]:
                self._alert(
                    "overfitting",
                    f"Train/val gap has grown for 3 consecutive evals: "
                    f"{last3_gaps[0]:.4f} -> {last3_gaps[1]:.4f} -> {last3_gaps[2]:.4f} "
                    f"(step {step})",
                    severity="warning",
                    step=step,
                )

    def get_summary(self) -> dict:
        """Return a dict of current metrics suitable for W&B logging."""
        clip_frac = (
            self._clipped_steps / self._total_steps
            if self._total_steps > 0
            else 0.0
        )
        return {
            "loss_ema": self.loss_ema,
            "grad_norm_ema": self.grad_norm_ema,
            "grad_clip_fraction": clip_frac,
            "steps_since_improvement": self.steps_since_improvement,
        }

    def get_state(self) -> dict:
        """Serialize internal state for checkpoint persistence."""
        return {
            "loss_ema": self.loss_ema,
            "grad_norm_ema": self.grad_norm_ema,
            "loss_ema_min": self.loss_ema_min,
            "steps_since_improvement": self.steps_since_improvement,
            "recent_losses": list(self._recent_losses),
            "ema_history": list(self._ema_history),
            "consecutive_vanish": self._consecutive_vanish,
            "total_steps": self._total_steps,
            "clipped_steps": self._clipped_steps,
            "val_history": self._val_history,
            "train_val_gaps": self._train_val_gaps,
            "alerts": self._alerts,
            "last_alert_step": self._last_alert_step,
            "first_step": self._first_step,
        }

    def load_state(self, state: dict) -> None:
        """Restore internal state from a checkpoint dict."""
        self.loss_ema = state["loss_ema"]
        self.grad_norm_ema = state["grad_norm_ema"]
        self.loss_ema_min = state["loss_ema_min"]
        self.steps_since_improvement = state["steps_since_improvement"]

        self._recent_losses = deque(
            state["recent_losses"], maxlen=self.config.divergence_window
        )
        self._ema_history = deque(
            state["ema_history"], maxlen=self.config.plateau_window
        )
        self._consecutive_vanish = state["consecutive_vanish"]
        self._total_steps = state["total_steps"]
        self._clipped_steps = state["clipped_steps"]
        self._val_history = state["val_history"]
        self._train_val_gaps = state["train_val_gaps"]
        self._alerts = state["alerts"]
        self._last_alert_step = state["last_alert_step"]
        self._first_step = state["first_step"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "warning",
        step: int = 0,
    ) -> None:
        """Emit an alert if the cooldown for this type has elapsed."""
        last = self._last_alert_step.get(alert_type, -self.config.alert_cooldown - 1)
        if step - last < self.config.alert_cooldown:
            return  # still in cooldown

        self._last_alert_step[alert_type] = step

        record = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "step": step,
            "time": time.time(),
        }
        self._alerts.append(record)

        print(f"[HealthMonitor][{severity.upper()}] {message}")

        if self.wandb_logger is not None:
            try:
                self.wandb_logger.alert(
                    title=alert_type, text=message, level=severity
                )
            except Exception:
                pass  # don't let wandb errors break training
