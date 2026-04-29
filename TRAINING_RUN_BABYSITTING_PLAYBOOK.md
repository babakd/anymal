# Training Run Babysitting Playbook

Last updated: 2026-04-28

This playbook is for agents supervising long training runs. The goal is to keep
the run moving when it is healthy, intervene quickly when the run is not
recoverable as-is, and leave enough evidence that the next operator can pick up
without guessing.

## Operator Posture

Treat single-batch train loss as a heartbeat, not a verdict. It tells you the
process is alive and whether loss/gradients are finite. It does not tell you, by
itself, whether the run is learning. Use validation loss, checkpoint integrity,
restart behavior, and gradient persistence for decisions.

The default action for a noisy but progressing run is to keep watching. Pausing
or restarting a healthy run is itself a risk: it can burn time, lose scheduler
state if resume is faulty, and create ambiguous experiment lineage.

## First Five Minutes

Before calling a run healthy, verify these gates:

- The expected architecture, dataset, checkpoint, GPU type, timeout, and run name
  are printed in logs.
- The model loads the intended checkpoint, not an auto-discovered legacy or
  mismatched checkpoint.
- The trainable parameter report matches the stage. For example, Stage 1 should
  not train the LLM; Stage 2 should train the intended adapter/LoRA modules.
- The dataset is the intended source, not a fallback path, unless the fallback is
  explicitly the experiment.
- At least one optimizer step completes with finite loss and finite grad norm.
- W&B or the chosen metrics sink is live, if expected.

If any of these are wrong, stop early. A wrong run that trains cleanly is still
a bad run.

## Green, Yellow, Red

### Green: Continue Watching

Continue when all of these are true:

- Optimizer steps are advancing.
- Loss is finite, even if it is jagged.
- Gradient spikes are isolated and settle within a few steps.
- Validation loss is flat-to-improving, or there has not been a validation point
  yet.
- Checkpoints save and commit on schedule.
- The app/process has not silently restarted.

Normal noise includes large single-batch jumps, especially with heterogeneous
image-caption data, mixed instruction datasets, sparse answer-token loss, small
batch sizes, gradient accumulation, and variable sequence lengths. A batch can
be much harder than the previous one. Do not stop just because one batch looks
ugly.

### Yellow: Inspect, But Do Not Stop Yet

Inspect more closely when one of these happens:

- A HealthMonitor plateau warning appears, but validation is still improving.
- A gradient spike appears once or twice and immediately settles.
- Throughput dips briefly around data loading, evaluation, or checkpointing.
- The log stream goes quiet at an eval/checkpoint boundary for a plausible
  amount of time.
- Train loss looks flat over a short window, but the next eval is near.

Yellow actions:

- Capture a wider log window.
- Check app status.
- Verify the latest checkpoint directory on the volume.
- Compare the next validation point before changing hyperparameters.
- Note the warning and the reason you are not intervening.

Do not restart just to make the curve prettier. Restart only when the current
process is wrong, stuck, or unrecoverable.

### Red: Pause, Fix, Resume Or Relaunch

Stop or pause the run when any of these are true:

- The app silently restarts or reinitializes and loses progress.
- The run is using the wrong architecture, dataset, checkpoint, compressor type,
  image preprocessing, or trainable modules.
- Loss or gradients become NaN/Inf.
- Gradient spikes persist across multiple steps or escalate rather than settle.
- Validation loss worsens across two consecutive evals after the warmup period,
  especially while train loss keeps improving.
- Checkpoint save or volume commit fails, or the checkpoint directory is missing
  expected artifacts.
- Eval appears hung beyond the expected duration for that run.
- The process throws OOM, timeout, data-corruption, missing-file, or traceback
  errors.
- The run is near a timeout without a recent recoverable checkpoint.

Red actions:

- Preserve evidence first: app id, run name, latest step, last good checkpoint,
  traceback, and any relevant W&B URL.
- Prefer resuming from the latest complete checkpoint over starting from zero.
- Patch the smallest root cause that makes the next attempt recoverable.
- Relaunch with an explicit resume/pretrain checkpoint path.
- Verify the first new checkpoint after relaunch.

## How To Read Noisy Loss Curves

Raw train loss is high variance because each logged point is one narrow sample
of the data distribution. With multimodal training, batch difficulty changes
with image content, caption length, answer-token count, prompt format, and
sequence length. Mixed instruction fine-tuning adds another source of variance:
some samples have short, easy answers while others require longer reasoning or
rare vocabulary.

Healthy noise has this shape:

- Raw train loss jumps around.
- A rolling mean or bucket mean drifts down or stays flat while learning rate
  decays.
- Validation loss improves over eval checkpoints.
- Grad norms stay finite and return to baseline after spikes.

Unhealthy noise has this shape:

- Raw train loss jumps and the rolling mean rises for a sustained window.
- Validation loss worsens repeatedly.
- Grad norms spike repeatedly or grow with the loss.
- The run becomes unstable around the same code path, such as every eval or
  checkpoint.

For decisions, rank signals in this order:

1. Correct run identity and data path.
2. Checkpoint recoverability.
3. Validation loss trend.
4. Persistent gradient behavior.
5. Rolling or bucketed train loss.
6. Individual train-loss points.

## Evaluation Boundaries

Eval and checkpoint boundaries are where many failures show up. At each boundary,
confirm:

- Eval starts within the expected step window.
- Eval makes progress through batches.
- Eval reports a finite value with nonzero valid batches.
- Checkpoint save logs appear.
- Volume commit logs appear when running on Modal.
- Training resumes after the boundary, unless the run is complete.
- The checkpoint directory contains the expected artifacts.

For AnyMAL-style checkpoints, a resumable checkpoint usually includes:

- `trainer_state.pt`
- `model_meta.json`
- `projector.pt`
- `token_compressor.pt` for V2
- `llm/` for Stage 2 LoRA checkpoints when applicable

## Suggested Monitoring Cadence

Use a tight cadence during risky periods:

- Launch through first optimizer step: continuous.
- First checkpoint: continuous.
- Eval/checkpoint boundaries: continuous until training resumes.
- After a relaunch: continuous until the first checkpoint after relaunch.

Use a lighter cadence during steady training:

- Check step progression every few minutes.
- Verify checkpoint directories at each save boundary.
- Compare validation points at each eval boundary.
- Keep an eye on wall time versus timeout.

## Modal Commands

Useful log stream:

```bash
modal app logs --timestamps APP_ID | grep --line-buffered -E 'Eval:|VQA|\[step [0-9]+\]|Model saved|Saved checkpoint|Committed checkpoint|Traceback|Error|ERROR|NaN|Training complete|Training completed|completed|HealthMonitor'
```

Checkpoint verification:

```bash
modal volume ls anymal-checkpoints /path/to/output/checkpoint-N
```

App status:

```bash
modal app list | rg 'APP_ID|anymal'
```

Short bounded log read when `timeout` is unavailable:

```bash
perl -e 'alarm(20); exec @ARGV' -- modal app logs --timestamps APP_ID | tail -200
```

## Reporting Template

When handing off or summarizing a supervised run, include:

- Run name and app id.
- W&B URL, if available.
- Stage, architecture, dataset, compressor, GPU type, timeout.
- Launch command or enough flags to reconstruct it.
- Latest step and whether the process is still live.
- Latest complete checkpoint path and verified artifact list.
- Validation trend.
- Any HealthMonitor warnings and why they were or were not actionable.
- Any intervention taken, including root cause and relaunch command.
- Next hard gate to watch.

## Practical Rule

Do not be hypnotized by jagged loss. A run with ugly batches, improving
validation, stable gradients, and good checkpoints is healthy. A run with smooth
loss but the wrong checkpoint, wrong data, missing artifacts, or silent restarts
is not.
