"""
Throughput Tracker for AnyMAL Training

Tracks training throughput metrics using a sliding window.
"""

import time
from collections import deque


class ThroughputTracker:
    """
    Tracks training throughput using a sliding window.

    Records timestamps, token counts, and sample counts per step,
    then computes rates over a configurable window.
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self._timestamps = deque(maxlen=window_size)
        self._token_counts = deque(maxlen=window_size)
        self._sample_counts = deque(maxlen=window_size)
        self._total_steps = 0
        self._total_tokens = 0
        self._total_samples = 0
        self._start_time = time.time()

    def step(self, batch_size: int, seq_len: int):
        """Record one optimizer step with timing."""
        now = time.time()
        tokens = batch_size * seq_len

        self._timestamps.append(now)
        self._token_counts.append(tokens)
        self._sample_counts.append(batch_size)

        self._total_steps += 1
        self._total_tokens += tokens
        self._total_samples += batch_size

    def get_metrics(self) -> dict:
        """Get throughput metrics over the sliding window."""
        if len(self._timestamps) < 2:
            return {
                "steps_per_sec": 0.0,
                "tokens_per_sec": 0.0,
                "samples_per_sec": 0.0,
            }

        # Window duration
        dt = self._timestamps[-1] - self._timestamps[0]
        if dt <= 0:
            return {
                "steps_per_sec": 0.0,
                "tokens_per_sec": 0.0,
                "samples_per_sec": 0.0,
            }

        # Count steps/tokens/samples in the window (exclude first timestamp as it's the baseline)
        n_steps = len(self._timestamps) - 1
        window_tokens = sum(list(self._token_counts)[1:])
        window_samples = sum(list(self._sample_counts)[1:])

        return {
            "steps_per_sec": n_steps / dt,
            "tokens_per_sec": window_tokens / dt,
            "samples_per_sec": window_samples / dt,
        }

    def get_total_metrics(self) -> dict:
        """Get overall throughput since start."""
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return {"total_steps": 0, "total_tokens": 0, "total_samples": 0, "elapsed_sec": 0}
        return {
            "total_steps": self._total_steps,
            "total_tokens": self._total_tokens,
            "total_samples": self._total_samples,
            "elapsed_sec": elapsed,
        }
