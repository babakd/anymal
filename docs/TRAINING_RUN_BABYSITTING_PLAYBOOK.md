# Training Run Babysitting Playbook

Last updated: 2026-05-09

This playbook is for agents supervising long training runs. The goal is to keep
the run moving when it is healthy, intervene quickly when the run is not
recoverable as-is, and leave enough evidence that the next operator can pick up
without guessing.

## Operator Posture

Treat single-batch train loss as a heartbeat, not a verdict. It tells you the
process is alive and whether loss/gradients are finite. It does not tell you, by
itself, whether the run is learning. Use validation loss, checkpoint integrity,
restart behavior, and gradient persistence for decisions.

Do not be charitable to anomalies. A spike is not automatically fatal, but it is
evidence to investigate. Your job is to falsify "healthy run" with W&B history
first, then logs, checkpoint artifacts, app status, and validation trend. Only
call an anomaly benign after checking those signals and writing down why it
cleared.

Delegated monitoring agents inherit this same posture. Their primary screen is
Weights & Biases, not the console log tail. A monitor report must include the
W&B run path, state, latest step, active alerts, recent-window loss movement,
loss EMA, grad norm EMA, grad clipping fraction, LR, validation loss when
available, and artifact/checkpoint status. If W&B and logs disagree, W&B wins
until the discrepancy is explained. Do not accept "the next few log lines look
fine" as a reason to downgrade an active W&B spike, clipping, loss-window, or
eval-regression alert.

The default action for a noisy but progressing run is to keep watching only
after the monitoring evidence agrees. Pausing or restarting a healthy run is a
risk, but letting a bad expensive run continue because the logs look familiar is
worse.

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
- W&B or the chosen metrics sink is live, if expected, and has current
  `train/loss`, `train/lr`, health, and gradient metrics. Do not rely only on
  console logs for paid runs.

If any of these are wrong, stop early. A wrong run that trains cleanly is still
a bad run.

## Green, Yellow, Red

### Green: Continue Watching

Continue when all of these are true:

- Optimizer steps are advancing.
- Loss is finite, even if it is jagged.
- W&B confirms the recent loss window is bounded, health EMAs are sane, and
  there are no unresolved loss/gradient alerts.
- Gradient spikes are isolated, settle within a few steps, and are not paired
  with rising loss EMA, high clipping, or throughput stalls.
- Validation loss is flat-to-improving, or there has not been a validation point
  yet.
- Checkpoints save and commit on schedule.
- The app/process has not silently restarted.

Normal noise can include large single-batch jumps, especially with
heterogeneous image-caption data, mixed instruction datasets, sparse
answer-token loss, small batch sizes, gradient accumulation, and variable
sequence lengths. That is a hypothesis, not an excuse. Check W&B before
classifying ugly batches as normal.

### Yellow: Inspect, But Do Not Stop Yet

Inspect more closely when one of these happens:

- A HealthMonitor plateau warning appears, but validation is still improving.
- A gradient spike appears once or twice and immediately settles.
- A loss point is `>= 2x` the W&B `health/loss_ema`, or a grad norm point is
  `>= 5x` `health/grad_norm_ema`, even if the next console line looks normal.
- The recent W&B loss-window mean rises by more than `25%` versus the previous
  comparable window.
- `health/grad_clip_fraction` rises above `0.20`, or increases steadily.
- Throughput dips briefly around data loading, evaluation, or checkpointing.
- The log stream goes quiet at an eval/checkpoint boundary for a plausible
  amount of time.
- Train loss looks flat over a short window, but the next eval is near.

Yellow actions:

- Capture a wider log window.
- Pull W&B history for the latest window and compare raw loss, loss EMA,
  grad norm, grad norm EMA, clipping fraction, LR, throughput, and eval loss.
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
- W&B shows repeated loss spikes, rising loss EMA, rising grad norm EMA, or
  high clipping that is not visible in the console sample.
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
  traceback, W&B URL, W&B metric snapshot, and checkpoint listing.
- Write the decision into the experiment ledger/playbook before launching the
  next expensive variant. Include why any spike was stopped, continued, or
  reclassified; silence in the ledger means the anomaly is still unresolved.
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
3. W&B-backed validation loss trend.
4. W&B-backed persistent gradient and clipping behavior.
5. Rolling or bucketed train loss from W&B history.
6. Console log samples.
7. Individual train-loss points.

## W&B Monitoring Requirements

For expensive Modal runs with W&B enabled, every monitoring report must include
both logs and W&B. If W&B is down or credentials are unavailable, say that
explicitly and increase log/checkpoint cadence until W&B access is restored.
This applies equally to delegated monitoring sub-agents: they should report the
W&B run path, latest metric snapshot, active alerts, spike/clip status, and the
log/checkpoint evidence used to judge whether the run continues. W&B is the
health source of truth when it disagrees with a calm-looking console tail.
When assigning a monitoring sub-agent, include this rubric in the prompt: be
non-charitable on spikes and clipping, use W&B before console logs for health
judgment, and write every stop/continue decision plus metric evidence back to
the playbook or experiment ledger. A delegated monitor is not authorized to
clear an active W&B alert, spike, or clipping event from console logs alone; the
event stays open until a later W&B window clears it with matching loss EMA, grad
EMA, clipping, LR, eval, and checkpoint evidence.

For gradient-accumulated runs, W&B health metrics must represent the completed
optimizer step, not the last microbatch in the accumulation window. If
`train/accumulation_micro_batches` is present, confirm it matches the configured
gradient accumulation count on every report. If a W&B train row exists and the
field is absent or wrong, stop or escalate the run immediately: sparse-label
loss-window and HealthMonitor metrics cannot be trusted as optimizer-step
accumulated means. If W&B has not synced any train rows yet, label the run
`pending first train row`, keep watching logs for progress, and re-check W&B at
the first logging boundary instead of calling it healthy or failed.

Minimum W&B fields to inspect:

- `train/loss`
- `train/raw_loss`, when objective normalization or loss scaling is enabled
- `train/objective_loss`, when distinct from raw loss
- `train/backward_loss`, when loss scaling is enabled
- `train/supervised_tokens` and `train/loss_normalization_multiplier`, when
  using supervised-token-aware normalization
- `train/accumulation_micro_batches`, when gradient accumulation is enabled
- `health/loss_ema`
- `train/grad_norm`
- `health/grad_norm_ema`
- `health/grad_clip_fraction`
- `train/lr`
- `eval/loss`, once evals exist
- throughput metrics, if present

Use `scripts/inspect_wandb_run.py` when local W&B access is unavailable:

```bash
modal run scripts/inspect_wandb_run.py \
  --run-path ENTITY/PROJECT/RUN_ID \
  --recent-window 100
```

Interpretation rules:

- Any loss point `>= 2x health/loss_ema` is yellow until the W&B window proves
  it was isolated.
- Any grad norm `>= 5x health/grad_norm_ema` is yellow; repeated occurrences
  or co-occurring loss growth are red.
- `health/grad_clip_fraction >= 0.20` is yellow; rising clip fraction across
  checks is red unless intentionally caused by a known hyperparameter change.
- `health/grad_clip_fraction` near `1.0` is red even when train loss is falling;
  clipping says the optimizer is not seeing the update the logs make tempting.
- A completed checkpoint does not rescue a run with pegged clipping. Mark the
  artifact as available but failed unless a later W&B window clears clipping
  below the declared gate.
- A recent-window loss mean up more than `25%` versus the previous comparable
  window is yellow even if logs show continued step progress.
- Any active W&B alert is unresolved until a later W&B window clears it. Do not
  dismiss it from console logs alone. Once an alert fires, keep it in the run's
  history even if the active alert list later returns to `[]`; a later healthy
  snapshot can change the judgment to `yellow-history/artifact-good`, but it
  cannot make the run clean green.
- Missing or incorrect `train/accumulation_micro_batches` on an existing W&B
  train row for a gradient-accumulated run is red because W&B may be logging
  last-microbatch metrics instead of optimizer-step accumulated means. Zero W&B
  train rows before the first logging boundary is pending, not green and not
  red; re-check at the first logged optimizer step.
- Three increasing eval losses or a widening train/eval gap after warmup is
  red unless the experiment explicitly predeclared that behavior as acceptable.
- Do not explain away spikes by dataset heterogeneity until W&B says the EMA,
  grad behavior, clipping, and checkpoint cadence are still healthy.
- Delegated monitors should be explicitly non-charitable about spikes: every
  spike/clipping exception needs W&B evidence in the ledger, not a reassuring
  prose interpretation based on console loss alone.
- Delegated monitors should preserve alert history separately from the latest
  state. A final clean W&B snapshot can clear an active stop condition, but it
  does not erase earlier active alerts; label those runs `yellow-history` or
  `artifact-good` unless the ledger explains why the history is irrelevant.
- Every stop/continue decision for an expensive run must be written back with
  the Modal app id, W&B run path/URL, and the W&B snapshot that justified it:
  latest step, active alerts, historical alerts, recent-window loss movement,
  raw/objective/train loss, `train/backward_loss`, `health/loss_ema`,
  `train/grad_norm`, `health/grad_norm_ema`,
  `health/grad_clip_fraction`, `train/accumulation_micro_batches`, `train/lr`,
  eval rows/loss if present, artifact/checkpoint status at declared boundaries,
  and an explicit green/yellow/red judgment.

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
- Compare W&B loss/grad/clipping windows at least once per checkpoint interval.
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

## Active V4 Monitoring Ledger

These notes are intentionally conservative. Do not remove yellow history just
because the latest few log lines look calm; clear it only with W&B window,
checkpoint, and eval evidence.

- `v4-stage1a-shared-20000-20260508-codex`
  (`babakdam/anymal-pretrain/6ffd2qwa`, app `ap-hTW04HKrFzV2fbDo6SVTBP`) was an
  H100 Stage 1A run and is now stopped by babysitting policy. The
  `checkpoint-4250` hard gate was checked with
  W&B, Modal app/log snapshots, and Modal volume listings on `2026-05-08`.
  Checkpoint contents were listed twice about 30 seconds apart and are complete:
  `trainer_state.pt` (`12.2 GiB`, modified `13:43 EDT`), `projector.pt`
  (`6.1 GiB`, modified `13:42 EDT`), and `model_meta.json` (`485 B`, modified
  `13:42 EDT`). The app remained live with one container, and logs after the
  gate showed training resumed through roughly step `4306`; no traceback, OOM,
  NaN, or checkpoint-volume failure was observed in the sampled logs.

  Status remains yellow-watch, not green. W&B was checked explicitly with
  `scripts/inspect_wandb_run.py --recent-window 220`: at step `4250`, train loss
  was `1.9976`, loss EMA `2.6590`, grad norm `0.7308`, grad EMA `1.0638`, clip
  fraction `0.1640`, LR `1.8137e-4`, and alerts were still
  `recent_loss_spikes` and `recent_grad_spikes`. A later W&B check at step
  `4360` still reported those alerts with train loss `1.8235`, loss EMA
  `2.8327`, grad norm `1.2129`, grad EMA `1.5173`, clip fraction `0.1732`, LR
  `1.8040e-4`, and recent 220-row loss window `2.8380 -> 2.7947` (no `+25%`
  mean rise). Eval/loss is still the last known improved pair, `2.9393` at step
  `2000` to `2.8393` at step `4000`; the next scheduled eval is around step
  `6000`. Recorded yellow events now include step `220` grad spike, step `960`
  loss spike, step `2540` grad spike, step `2860` loss spike, step `3948` grad
  spike, step `3996` loss spike, and a fresh post-4000 W&B loss spike at step
  `4140` (`6.3670` vs EMA `2.8839`, ratio `2.21`). Modal progress output also
  showed raw loss spikes after `4000`, including `7.3678` near displayed step
  `4169`. Clipping remains below the `0.20` stop/escalation line but has drifted
  upward from `0.1640` at step `4250` to `0.1732` at step `4360`, so treat the
  trend as yellow evidence rather than comfort.

  `checkpoint-4500` was checked on `2026-05-08` with W&B, Modal app/log
  snapshots, and Modal volume listings. The first volume listing found all three
  expected files, and a second listing after the save boundary reported
  `trainer_state.pt` (`12.2 GiB`, modified `13:53 EDT`), `projector.pt`
  (`6.1 GiB`, modified `13:53 EDT`), and `model_meta.json` (`485 B`, modified
  `13:53 EDT`), so the checkpoint is complete and not an obvious partial.
  Modal logs showed training resumed through about step `4517`; no traceback,
  OOM, NaN, inf, or checkpoint failure was observed in the sampled logs. The app
  remained live with one task.

  Status was escalated yellow/orange, not green. W&B was checked explicitly with
  `scripts/inspect_wandb_run.py --recent-window 220`: at step `4550`, train loss
  was `3.0209`, loss EMA `2.6327`, grad norm `1.2151`, grad EMA `1.7192`, clip
  fraction `0.1892`, LR `1.7869e-4`, eval/loss was still `2.8393`, and alerts
  remained `recent_loss_spikes` and `recent_grad_spikes`. The recent 220-row
  loss window still did not trip the `+25%` rule (`2.9019 -> 2.7948`), but
  clipping had continued upward from `0.161` at step `4000` to `0.1640` at
  `4250`, `0.1732` at `4360`, `0.1775` at `4450`, `0.1863` at `4520`, and
  `0.1892` at `4550`.

  Final stop update on `2026-05-08`: W&B was checked explicitly again before the
  stop. At step `4640`, W&B reported clip fraction `0.2000` with active
  `recent_loss_spikes`, `recent_grad_spikes`, and `high_grad_clip_fraction`;
  at step `4680`, clip fraction rose to `0.204059829`, train loss was `3.9157`,
  loss EMA `2.9647`, grad norm `0.7353`, grad EMA `2.4286`, LR `1.7748e-4`,
  eval/loss was still `2.8393`, and the 220-row loss window was not rising by
  `25%` (`2.9101 -> 2.7615`). The parent stopped app
  `ap-hTW04HKrFzV2fbDo6SVTBP` after W&B step `4730` crossed the playbook
  escalation line with clip fraction `0.2093` and active `recent_loss_spikes`,
  `recent_grad_spikes`, and `high_grad_clip_fraction`. A post-stop W&B check
  reached step `4750` with clip fraction `0.2112`, train loss `2.2418`, loss EMA
  `3.0296`, grad norm `0.9418`, grad EMA `1.4315`, LR `1.7682e-4`, active
  `recent_loss_spikes` and `high_grad_clip_fraction`, and recent 220-row loss
  window `2.9290 -> 2.7543`.

  Post-stop verification found the app stopped with zero tasks. The last
  verified complete checkpoint remains `checkpoint-4500`. A `checkpoint-4750`
  directory is present and contains `trainer_state.pt`, `projector.pt`, and
  `model_meta.json`, but it must be treated as partial/unverified: two Modal
  volume listings about one minute apart showed `trainer_state.pt` stuck at
  `1.0 GiB` (modified `14:03 EDT`) instead of the expected `12.2 GiB`, while
  `projector.pt` was `6.1 GiB` and `model_meta.json` was `485 B`. Do not resume
  from `checkpoint-4750`; use `checkpoint-4500` as the last safe checkpoint
  unless a later manual repair proves otherwise.
- `v4-stage1a-recovery4500-lr8e5-3000-20260508-codex`
  (`babakdam/anymal-pretrain/g607yqek`, app `ap-BR97ZXuooRK292Ja0Kgflv`) was a
  stopped H100 Stage 1A recovery. It used
  `/checkpoints/pretrain-output/v4-stage1a-shared-20000-20260508-codex/checkpoint-4500`
  as `--pretrain-checkpoint`, not `--resume-checkpoint`, so it loaded the last
  safe connector weights with a fresh optimizer and scheduler. LR was reduced to
  `8e-5`, max steps `3000`, warmup `100`, eval every `300`, expected checkpoint
  cadence every `250`. Logs verified W&B run `g607yqek`, real LLaVA-Pretrain
  captions/images, and successful Stage 1 connector load from `checkpoint-4500`.

  Parent stopped the recovery after the early W&B check showed the lower-LR
  restart did not clear the clipping failure: at W&B step `30`,
  `health/grad_clip_fraction` was about `0.80` while LR was only `2.4e-5` during
  warmup. Post-stop W&B was checked explicitly with
  `scripts/inspect_wandb_run.py --recent-window 220`: the run state was
  `finished`, latest W&B step `40`, active alert `high_grad_clip_fraction`, train
  loss `2.8424`, loss EMA `2.8923`, grad norm `2.0451`, grad EMA `2.3047`, clip
  fraction `0.825`, LR `3.2e-5`, no evals, no recent loss spikes, no recent grad
  spikes, and recent loss window `3.4226 -> 2.2750` across four train rows.
  Modal app status was `stopped` with zero tasks. Logs also showed a step-8 grad
  spike warning, then `[step 30]` with LR `2.40e-05` and `[step 40]` with LR
  `3.20e-05`, followed by `Stopping app - user stopped from CLI`; no traceback,
  OOM, NaN, or inf was observed in the sampled logs. No `checkpoint-50` or
  `checkpoint-250` directory was found in the recovery output, so no recovery
  checkpoint/eval gate landed. Do not restart this recipe without changing the
  clipping root cause.
- `v4-stage1a-recovery4000-lr5e5-canary300-20260508-codex`
  (`babakdam/anymal-pretrain/r4hx5xrs`, app `ap-E3Ke98Cu4xVqc7Aal55iSm`) was a
  diagnostic canary from earlier connector weights and is now stopped. It used
  `/checkpoints/pretrain-output/v4-stage1a-shared-20000-20260508-codex/checkpoint-4000`
  as `--pretrain-checkpoint`, not `--resume-checkpoint`, so optimizer/scheduler
  were fresh. LR was `5e-5`, max steps `300`, warmup `30`, eval every `50`. Logs
  verified real LLaVA-Pretrain data, successful Stage 1 connector load from
  `checkpoint-4000`, and V4 connector-only trainables. W&B was checked
  explicitly at step `40`: active `high_grad_clip_fraction`, clip fraction
  `0.35`, train loss `2.6687`, loss EMA `2.9513`, grad norm `1.0645`, grad EMA
  `1.0090`, LR `4.98e-5`, no evals, no recent loss/grad spikes, and recent loss
  window `3.3475 -> 2.2407` across four train rows. Parent stopped the app.
  Post-stop W&B was checked explicitly with `scripts/inspect_wandb_run.py
  --recent-window 220`: run state `finished`, latest W&B step `50`, active
  `high_grad_clip_fraction`, clip fraction `0.30`, train loss `3.3043`, loss EMA
  `2.8749`, grad norm `1.0543`, grad EMA `0.9180`, LR `4.939e-5`, no evals, no
  recent loss/grad spikes, and five-row loss window `3.3475 -> 2.5952`. Modal
  listed it `stopped` with zero tasks. Logs showed step `50` then `Stopping app -
  user stopped from CLI` as the first eval was just starting, so no completed
  eval row landed. The Modal volume run directory exists but is empty; both
  `checkpoint-50` and `checkpoint-75` are absent, so no checkpoint/eval gate
  landed. Also investigate the caption label previews: supervised text currently
  includes `<|begin_of_text|>` in Stage 1A captions, which may be a
  recipe/data-format contributor to unstable connector gradients.
- `v4-stage1a-fixedlabels-ckpt4000-lr5e5-canary100-20260508-codex`
  (`babakdam/anymal-pretrain/froy06eg`, app `ap-Du9jNqbTRtxsNAzf3X8ZVB`) was the
  patched-label V4 Stage 1A canary and is now stopped on W&B clipping
  escalation. It used
  `/checkpoints/pretrain-output/v4-stage1a-shared-20000-20260508-codex/checkpoint-4000`
  as `--pretrain-checkpoint`, not `--resume-checkpoint`, so optimizer/scheduler
  were fresh. LR was `5e-5`, max steps `100`, warmup `10`, eval every `50`,
  batch size `1`, H100. Logs verified W&B run `froy06eg`, real LLaVA-Pretrain
  captions/images, successful Stage 1 connector load from `checkpoint-4000`, and
  V4 connector-only trainables (`1,633,185,792` params).

  The fixed-label diagnostic did succeed: supervised previews no longer include
  `<|begin_of_text|>`. Log examples were `madonna tour is set to kick off in a
  performance in los on thursday`, `the bmw m3 sport car in front of mountains`,
  and `the wordpress template of a tool store`. W&B was checked explicitly early
  with `scripts/inspect_wandb_run.py --recent-window 100`: at step `10`, the
  run was still `running` but already had active `high_grad_clip_fraction`, clip
  fraction `0.60`, train loss `3.7993`, loss EMA `4.2794`, grad norm `0.8121`,
  grad EMA `1.2927`, LR `5.0e-5`, no evals, and no recent loss/grad spikes.
  Because clipping was far above the `0.20` escalation line, parent stopped the
  app. Post-stop W&B was checked explicitly again: run state `finished`, latest
  W&B step `40`, active `high_grad_clip_fraction`, clip fraction `0.60`, train
  loss `2.9152`, loss EMA `3.2533`, grad norm `0.7119`, grad EMA `1.3856`, LR
  `3.875e-5`, no eval rows, no recent loss spikes, no recent grad spikes, and
  four-row loss window `3.6531 -> 2.5407`. Modal listed the app `stopped` with
  zero tasks; logs showed `[step 10]`, `[step 20]`, `[step 30]`, `[step 40]`,
  then `Stopping app - user stopped from CLI`. The Modal volume run directory
  exists but is empty, so no `checkpoint-50` and no eval/checkpoint gate landed.
  Judgment: masking tokenizer special tokens from caption labels fixed that data
  formatting symptom, but it is insufficient by itself to clear the Stage 1A
  clipping pathology.
- `v4-stage1a-bottleneck1024-canary100-20260508-codex`
  (`babakdam/anymal-pretrain/dsv4mlbs`, app `ap-76TjvdhPvFm3LdHXdHJ6SC`) was the
  from-scratch bottleneck-connector V4 Stage 1A canary and is now stopped on W&B
  clipping escalation. It used 64 global + 64 local visual tokens, connector
  layers `3`, heads `8`, FF mult `2`, hidden dim `1024`, 2D position features,
  LR `1e-4`, max steps `100`, warmup `10`, eval every `50`, batch size `1`, and
  A100. Logs verified W&B run `dsv4mlbs`, real LLaVA-Pretrain captions/images,
  fixed-label supervised previews without `<|begin_of_text|>`, and a much smaller
  V4 connector-only trainable set (`44,374,016` params instead of `1.633B`).

  W&B was checked explicitly before stopping: at step `10`, the run was still
  `running` but already had active `high_grad_clip_fraction`, clip fraction
  `1.0`, train loss `5.0610`, loss EMA `9.5638`, grad norm `8.3869`, grad EMA
  `90.8336`, LR `1.0e-4`, no evals, and no recent loss/grad spikes. Parent
  stopped the app because clipping was far above the `0.20` escalation line.
  Post-stop W&B was checked explicitly again: run state `finished`, latest W&B
  step `20`, active `high_grad_clip_fraction`, clip fraction still `1.0`, train
  loss `4.2093`, loss EMA `7.8934`, grad norm `3.2777`, grad EMA `58.6991`, LR
  `9.7286e-5`, no eval rows, no recent loss spikes, no recent grad spikes, and
  two-row loss window `5.0610 -> 4.2093`. Modal listed the app `stopped` with
  zero tasks; the Modal volume run directory listing was empty (`[]`), so no
  `checkpoint-50` and no eval/checkpoint gate landed. Judgment: shrinking the
	  connector is architecturally useful for cost and memory, but it is insufficient
	  by itself to clear Stage 1A clipping; next run must change gradient
	  scale/initialization/loss recipe, not just width.
- `v4-stage1a-bottleneck1024-scale01-canary100-20260508-codex`
  (`babakdam/anymal-pretrain/wm6y5cvh`, app `ap-TOyngwQQ017RjE8vDGZbjo`) was the
  output-scale follow-up to the bottleneck connector canary and is now stopped on
  W&B clipping escalation. It used the same 64 global + 64 local, 3-layer,
  8-head, FF mult `2`, hidden dim `1024` connector, plus
  `connector_output_scale=0.1`, LR `1e-4`, max steps `100`, warmup `10`, eval
  every `50`, batch size `1`, and A100. Logs verified W&B run `wm6y5cvh`, real
  LLaVA-Pretrain captions/images, fixed-label supervised previews without
  `<|begin_of_text|>`, output scale `0.1`, and `44,374,016` connector-only
  trainables.

  W&B was checked explicitly before stopping: at step `10`, the run was still
  `running` but already had active `high_grad_clip_fraction`, clip fraction
  `1.0`, train loss `6.7240`, loss EMA `10.1241`, grad norm `48.4955`, grad EMA
  `132.7394`, LR `1e-4`, no eval rows, and no recent loss/grad spikes. Parent
  stopped the app because clipping was pegged, despite the local stream showing
  ordinary-looking loss movement. Post-stop W&B was checked explicitly again:
  run state `finished`, latest W&B step `20`, active `high_grad_clip_fraction`,
  clip fraction still `1.0`, train loss `3.9441`, loss EMA `8.0904`, grad norm
  `2.5491`, grad EMA `81.1468`, LR `9.7286e-5`, no eval rows, no recent loss
  spikes, no recent grad spikes, and two-row loss window `6.7240 -> 3.9441`.
  Modal listed the app `stopped` with zero tasks; the Modal volume run directory
  listing was empty, so no `checkpoint-50` and no eval/checkpoint gate landed.
  Judgment: output scaling at `0.1` is insufficient by itself. The next canary
  should change initialization/gating or the caption-alignment objective, and any
  monitor must keep W&B clipping above console loss trends.
- `v4-stage1a-bottleneck1024-gate001-canary100-20260508-codex`
  (`babakdam/anymal-pretrain/cdtr4gp6`, app `ap-h9cCm08DcrIDEqyIcJy1Sw`) tested
  a trainable output gate initialized at `0.01` on top of the bottleneck1024
  connector. Logs verified `connector_output_scale=1.0`,
  `connector_output_gate_init=0.01`, real LLaVA-Pretrain data, fixed labels, and
  `44,374,017` trainables. A delegated monitor was explicitly instructed to use
  W&B and be non-charitable about clipping. W&B step `10` already had active
  `high_grad_clip_fraction`, clip fraction `1.0`, train loss `5.8066`, loss EMA
  `5.7100`, grad norm `3.8188`, grad EMA `31.0132`, LR `1e-4`, and no eval
  rows. Parent stopped the app. Post-stop W&B reached step `30` with active
  `high_grad_clip_fraction` and `recent_loss_window_mean_up_gt_25pct`, clip
  fraction still `1.0`, train loss `9.0393`, loss EMA `8.4019`, grad norm
  `5.1590`, grad EMA `17.0959`, LR `8.9472e-5`, and no eval rows. Modal listed
  the app `stopped` with zero tasks and the output directory was empty.
- `v4-stage1a-bottleneck1024-gate0001-canary50-20260508-codex`
  (`babakdam/anymal-pretrain/us0ls2wu`, app `ap-753TNbms8Pv2wNLeL0qrut`) tested
  gate init `0.001`. Logs verified the same bottleneck1024 architecture, real
  data, fixed labels, and `44,374,017` trainables. A delegated monitor was again
  instructed to use W&B and reject pegged clipping even if loss improved. W&B
  step `10` had active `high_grad_clip_fraction`, clip fraction `1.0`, train
  loss `5.4899`, loss EMA `6.0910`, grad norm `2.3484`, grad EMA `62.3095`, LR
  `9.7286e-5`, and no eval rows. Post-stop W&B was `finished` at step `20` with
  active `high_grad_clip_fraction`, clip fraction still `1.0`, train loss
  `4.7958`, loss EMA `5.9601`, grad norm `21.4131`, grad EMA `39.3029`, LR
  `7.75e-5`, no eval rows, and no loss/grad spike alerts. Modal listed the app
  `stopped` with zero tasks and the output directory was empty.
- `v4-stage1a-bottleneck1024-gate00001-canary20-20260508-codex`
  (`babakdam/anymal-pretrain/2qdqg86p`, app `ap-FNaPaB3I8yEpFaF7fmLefB`) tested
  gate init `0.0001`. W&B step `10` already had active
  `high_grad_clip_fraction`, clip fraction `1.0`, train loss `6.2635`, loss EMA
  `6.2312`, grad norm `3.4968`, grad EMA `130.6474`, LR `6.2814e-5`, and no
  eval rows. The run completed around the stop request and saved
  `/checkpoints/pretrain-output/v4-stage1a-bottleneck1024-gate00001-canary20-20260508-codex/checkpoint-20`;
  Modal volume listing shows `trainer_state.pt`, `projector.pt`, and
  `model_meta.json`. Final W&B was `finished` at step `20` with active
  `high_grad_clip_fraction`, clip fraction still `1.0`, train loss `4.8986`,
  loss EMA `6.0846`, grad norm `1.4890`, grad EMA `78.9965`, LR `1e-5`, no eval
  rows, no loss/grad spike alerts, and two-row loss window `6.2635 -> 4.8986`.
  The checkpoint exists but fails the health gate. Judgment across gate
  experiments: trainable visual gating at `0.01`, `0.001`, and `0.0001` did not
  clear Stage 1A clipping; do not launch another amplitude-only canary as the
  next step.
- `v4-stage1a-outputwarmup30-gate00001-lr1e5-canary60-20260508-codex`
  (`babakdam/anymal-pretrain/iip0i3e3`, app `ap-6PUomRff3C9w5V8LMkNegC`) tested
  Stage 1A connector-gradient warmup: only `projector.output_proj.*` and
  `projector.output_gate_logit` were allowed to update for the first `30`
  optimizer steps on the 4x A100 distributed Stage 1 path, with
  `connector_output_gate_init=0.0001` and LR `1e-5`.
  Logs verified real LLaVA-Pretrain captions/images, fixed-label supervised
  previews, `3` active warmup tensors, and `71` masked tensors. W&B was checked
  during the warmup before the planned unfreeze: step `20` had active
  `high_grad_clip_fraction`, clip fraction `1.0`, train loss `6.0813`, loss EMA
  `6.5835`, grad norm `17.0686`, grad EMA `31.0523`, LR `8.5881e-6`, no eval
  rows, and two-row loss window `7.6394 -> 6.0813`. Parent stopped the app.
  Final W&B state was `finished`; Modal listed the app `stopped` with zero tasks
  and the output directory was empty. Judgment: output-only connector warmup did
  not clear caption-alignment clipping. The monitor correctly ignored falling
  loss because clipping was pegged.
- `v4-stage1a-groundingfirst-gate00001-lr1e5-canary80-20260508-codex`
  (`babakdam/anymal-pretrain/udsb1fr9`, app `ap-eE4Hll6DpCL0A6rqvc1fni`) tested
  the same bottleneck1024 V4 connector on the `v4_grounding` short-answer
  mixture instead of web captions, with `connector_output_gate_init=0.0001`,
  output scale `1.0`, LR `1e-5`, and no connector warmup. Logs confirmed the
  distributed 4-GPU Stage 1 path, real images, VQAv2 direct answers, COCO
  object/count/color direct answers, and short LLaVA direct-answer samples; a
  stale log label called these "V3 grounding-alignment" samples, and the code was
  patched afterward to print the requested dataset name. W&B step `10` already
  had active `high_grad_clip_fraction`, clip fraction `1.0`, train loss
  `10.8670`, loss EMA `8.0383`, grad norm `7.5836`, grad EMA `7.2998`, LR
  `9.9829e-6`, and no eval rows. Parent stopped before step-50 eval. Final W&B
  state was `finished`; Modal listed the app `stopped` with zero tasks and no
  checkpoint contents. Judgment: the current grounding-first objective also
  fails the immediate clipping gate; do not continue similar runs to eval.
- `v4-stage1a-groundingfirst-lossscale01-gate00001-lr1e5-canary80-20260508-codex`
  (`babakdam/anymal-pretrain/7lizvbnm`, app `ap-5eFp7HMB0KnQkoscsSxy6y`) tested
  `pretrain_loss_scale=0.1` on the same grounding-first recipe. Logs verified the
  4x A100 distributed Stage 1 path, `Stage 1 loss scale: 0.1`, and short-answer
  grounding labels. W&B was the source of truth: step `50` had no alerts, eval
  loss `8.0005`, train loss `12.1285`, grad norm `0.4288`, grad EMA `0.4147`,
  and clip fraction `0.04`, so the immediate clipping pathology was much better.
  At step `60`, W&B raised `recent_loss_window_mean_up_gt_25pct`; the monitor
  treated that as a stop condition instead of waving it away. Final W&B after the
  stop showed last step `70`, active `recent_loss_window_mean_up_gt_25pct`, train
  loss `7.4674`, loss EMA `8.5904`, grad norm `0.6900`, grad EMA `0.6184`, clip
  fraction `0.0714`, LR `1.4216e-6`, one eval row (`8.0005`), no grad spikes,
  and no pointwise loss spikes. Modal listed the app `stopped` with zero tasks.
  `checkpoint-50` exists with `trainer_state.pt`, `projector.pt`, and
  `model_meta.json`. Judgment: `loss_scale=0.1` is the first knob to clear early
  clipping, but this run remains yellow/red because W&B loss-window health
  failed before the requested 80 steps.
- `v4-stage1a-groundingfirst-lossscale01-accum16-gate00001-lr1e5-canary100-20260508-codex`
  (`babakdam/anymal-pretrain/a530s1tz`, `scarlet-yogurt-35`, app
  `ap-c1HAaQCQ5vroHYRlZ023ac`) tested whether larger Stage 1 gradient
  accumulation could smooth the loss-window failure while keeping
  `pretrain_loss_scale=0.1`. Logs verified the 4x A100 distributed Stage 1 path,
  effective batch size `64`, `Stage 1 gradient accumulation: 16`,
  `Stage 1 loss scale: multiplying backward loss by 0.1`, the real
  `v4_grounding` mixture, and the fixed dataset label (`Loaded 27446000
  v4_grounding instruction samples`). W&B was still the decision source: the run
  hit active `recent_loss_window_mean_up_gt_25pct` by optimizer step `20`, with
  train loss `11.5021`, loss EMA `8.0981`, grad norm `0.6147`, grad norm EMA
  `0.6188`, clip fraction `0.1`, LR `9.7286e-6`, no eval rows, no inspector
  grad spikes, and no inspector pointwise loss spikes. The console also reported
  `[HealthMonitor][WARNING] Loss spike at step 8: 16.6235 > 2.0x EMA (8.2828)`,
  but the stop decision was based on the active W&B alert, not console vibes.
  Modal listed the app `stopped` with zero tasks. The run directory exists but is
  empty, so no checkpoint landed. Judgment: red. Larger accumulation alone did
  not clear loss-window health; do not keep scaling effective batch as the next
  primary fix.
- `v4-stage1a-groundingfirst-lossscale01-lr3e6-canary120-20260508-codex`
  (`babakdam/anymal-pretrain/o72woiib`, `sandy-capybara-36`, app
  `ap-P2rKyVW0yk5S7IfmSo1I7K`) tested lower peak LR `3e-6` with the same
  `pretrain_loss_scale=0.1`, `v4_grounding`, gate `0.0001`, and accumulation `8`.
  Logs verified the 4x A100 distributed path, effective batch size `32`, real
  `v4_grounding` mixture, fixed dataset label, and V4 connector-only trainables
  (`44,374,017`). W&B at step `20` was clean but not enough: no alerts, clip
  fraction `0.0`, train loss `5.8749`, loss EMA `8.6810`, grad norm `0.4596`,
  grad EMA `0.5964`, LR `2.9636e-6`. W&B at step `50` still had no alerts, clip
  fraction `0.0`, eval loss `8.6358`, train loss `12.4370`, loss EMA `9.4071`,
  grad norm `0.3117`, grad EMA `0.4865`, and a flat recent window
  (`8.4460 -> 8.2972`); `checkpoint-50` saved and committed. By W&B step `80`,
  `recent_loss_window_mean_up_gt_25pct` was active and unresolved. Final W&B after
  stop: train loss `11.6854`, loss EMA `9.3434`, grad norm `0.6954`, grad EMA
  `0.4057`, clip fraction `0.0`, LR `1.1153e-6`, eval loss `8.6358`, no
  inspector loss/grad spikes, and recent window `7.3366 -> 11.5327`. Modal listed
  the app `stopped` with zero tasks. `checkpoint-50` exists with
  `trainer_state.pt`, `projector.pt`, and `model_meta.json`. Judgment: red/yellow.
  Lower LR plus loss scaling fixes clipping but not loss-window health; do not
  spend a long run on update-size shrinkage alone.
- `v4-stage1a-answerfocus-lossscale01-gate00001-lr1e5-canary120-20260508-codex`
  (`babakdam/anymal-pretrain/sjrrgvm6`, `fine-field-37`, app
  `ap-3HDCBEK0StEV7mn2T1GkXx`) tested whether the answer-type-focused Stage 1
  mixture could smooth the unstable grounding-first objective while preserving
  `pretrain_loss_scale=0.1`, LR `1e-5`, gate init `0.0001`, output scale `1.0`,
  and accumulation `8`. A delegated monitoring sub-agent, Descartes, was
  explicitly assigned to watch W&B before logs and to be non-charitable on
  spikes/clipping. The agent stopped the app because W&B clipping was
  persistently above the playbook gate: `health/grad_clip_fraction` was `0.30`
  at step `10`, `0.25` at step `20`, and `0.50` at latest synced step `30`.
  Final W&B snapshot: active `high_grad_clip_fraction`, train loss `1.2553`,
  loss EMA `8.8399`, grad norm `4.4440`, grad norm EMA `1.5574`, LR
  `9.397e-6`, no eval rows, no inspector loss spikes, no inspector grad spikes,
  and recent-window loss range `1.2553` to `12.7570` across three train rows.
  Logs verified 4-GPU distributed V4 connector-only training, dataset
  `v4_answer_type_focus`, effective batch size `32`, and `44,374,017`
  trainables. Modal app status after stop was `stopped` with zero tasks, and the
  run directory listing returned empty; no `checkpoint-50` or eval row landed.
  Judgment: red. Do not treat the falling latest train loss as recovery; W&B
  clipping is the source of truth here.
- `v4-stage1a-groundingfirst-tokennorm8-lossscale01-gate00001-lr1e5-canary120-20260508-codex`
  (`babakdam/anymal-pretrain/jwd997c5`, `dandy-bush-38`, app
  `ap-mj06o5UUMhVj7IrOaW0amS`) tested the new supervised-token-target objective:
  raw HF token-mean loss is multiplied by `supervised_tokens / 8`, clamped to
  `[0.05, 4.0]`, then `pretrain_loss_scale=0.1` is applied for backward. A
  delegated monitoring sub-agent, Hubble, was explicitly assigned to use W&B
  before logs and to be non-charitable on spikes/clipping. Step `50` passed the
  checkpoint/eval gate: W&B had no alerts, train/objective loss `2.9637`, raw
  loss `11.8549`, supervised tokens `2`, normalization multiplier `0.25`,
  backward loss `0.2964`, loss EMA `2.5360`, grad norm `0.1417`, grad EMA
  `0.1296`, clip fraction `0.0`, LR `7.5196e-6`, and eval loss `7.8830`.
  `checkpoint-50` exists with `trainer_state.pt`, `projector.pt`, and
  `model_meta.json`. Final W&B check at step `60` was red/yellow: active
  `recent_loss_window_mean_up_gt_25pct`, train/objective loss `2.6404`, raw
  loss `10.5617`, supervised tokens `2`, multiplier `0.25`, backward loss
  `0.2640`, loss EMA `2.5327`, grad norm `0.1086`, grad EMA `0.1496`, clip
  fraction `0.0`, LR `6.2814e-6`, eval loss still `7.8830`, and the run state
  ended `finished` with Modal app `stopped` and zero tasks. Judgment:
  red/yellow, do not continue this canary. Token normalization fixed clipping
  better than prior recipes and improved first eval, but it did not clear the
  W&B loss-window gate. Operational defect found: W&B display naming was random
  because `wandb_run_name` was not wired through Modal distributed pretrain;
  this was patched afterward so the next launch should use the requested
  `--run-name` as the W&B display name.
- `v4-stage1a-groundingfirst-tokennorm8-lossscale01-gate00001-lr5e6-canary120-20260508-codex`
  (`babakdam/anymal-pretrain/1isusl46`, app `ap-6NhnUkn2iXHsNwhdLqvGYh`) tested
  whether lowering peak LR to `5e-6` would keep the token-normalized recipe's
  clipping fix while smoothing the W&B loss window. W&B display naming was
  correct after the Modal `wandb_run_name` patch. Step `50` was clean but not
  sufficient: no alerts, clip fraction `0.0`, train/objective loss `3.1060`,
  raw loss `12.4238`, supervised tokens `2`, multiplier `0.25`, backward loss
  `0.3106`, loss EMA `2.6217`, grad norm `0.4639`, grad EMA `0.2449`, LR
  `3.7598e-6`, and flat five-row loss window (`2.1089 -> 2.1093`). Eval landed
  at `8.2909`, worse than the LR `1e-5` token-normalized canary. `checkpoint-50`
  exists with `trainer_state.pt`, `projector.pt`, and `model_meta.json`. At W&B
  step `60`, the monitor stopped/rejected the run for active
  `recent_loss_window_mean_up_gt_25pct`: train/objective loss `2.8325`, raw loss
  `11.3299`, supervised tokens `2`, multiplier `0.25`, backward loss `0.2832`,
  loss EMA `2.6547`, grad norm `0.3481`, grad EMA `0.3792`, clip fraction
  `0.0167`, LR `3.1407e-6`, eval loss `8.2909`, no inspector pointwise
  loss/grad spikes, and recent loss window `1.5663 -> 2.8931`. Judgment:
  red per the monitoring sub-agent's final report. Do not call low clipping a pass;
  this is another W&B loss-window
  failure, and LR-only shrink worsened first eval.
- `v4-stage1a-groundingfirst-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-canary120-20260508-codex`
  (`babakdam/anymal-pretrain/mqebqqc0`, app `ap-se9J3ORGBtFruPUtZKIfZF`) repeated
  the best LR `1e-5` token-normalized recipe after patching W&B/HealthMonitor
  reporting to use accumulated optimizer-step means instead of the last
  microbatch in each gradient-accumulation window. The monitoring sub-agent,
  Ptolemy, was explicitly instructed to be non-charitable on W&B spikes/alerts,
  to use W&B before console logs, and to reject missing or incorrect
  `train/accumulation_micro_batches`. W&B verified
  `train/accumulation_micro_batches=8.0`, no active alerts, no inspector
  loss/grad spikes, and `health/grad_clip_fraction=0.0` through the run. Step
  `50` eval was `7.8095`; step `100` eval improved to `5.1151`. Final W&B at
  step `120`: train/objective loss `1.2308`, raw loss `4.1529`, backward loss
  `0.1231`, supervised tokens `3.125`, multiplier `0.390625`, loss EMA
  `1.7561`, grad norm `0.2211`, grad EMA `0.2826`, LR `1e-6`, eval loss
  `5.1151`, and recent loss window `2.1678 -> 1.7965`. Modal listed the app
  `stopped` with zero tasks, and `checkpoint-50` plus `checkpoint-100` each
  contain `trainer_state.pt`, `projector.pt`, and `model_meta.json`. Judgment:
  green. Operational lesson: for gradient-accumulated sparse-label runs,
  last-microbatch W&B health is not acceptable evidence. Accumulated-step W&B
  metrics are mandatory before a run can be called healthy.
- `v4-stage1a-groundingfirst-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-extended1000b-20260508-codex`
  (`babakdam/anymal-pretrain/qjsw0606`, app
  `ap-jdJIdfkNWydB3SsjEs68b6`) is an operational abort, not a model failure.
  The monitoring sub-agent correctly used W&B before logs and was
  non-charitable about missing artifacts, but it assumed the step-100 eval
  boundary was also a checkpoint boundary. Code inspection showed the launch's
  implicit checkpoint cadence was `_checkpoint_save_interval(1000)=250`, while
  eval cadence was `100`, so `checkpoint-100` was not due. Final W&B after the
  stop was health-clean at step `150`: no active alerts,
  `train/accumulation_micro_batches=8.0`, train/objective loss `1.8965`, raw
  loss `6.3185`, backward loss `0.1897`, supervised tokens `2.375`,
  normalization multiplier `0.296875`, loss EMA `2.3580`, grad norm `0.1074`,
  grad EMA `0.2540`, clip fraction `0.04`, LR `9.9316e-6`, eval loss `7.8015`,
  no inspector loss/grad spikes, and recent window `2.5072 -> 2.3553`. Modal
  listed the app as `stopped` with zero tasks, and the run directory was empty
  because the first real checkpoint would have been step `250`. Do not count
  this as negative model evidence. Patch added `--pretrain-save-steps`; future
  monitors must use the printed save cadence, not infer checkpoint steps from
  eval cadence.
- `v4-stage1a-groundingfirst-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-extended1000c-save100-20260508-codex`
  (`babakdam/anymal-pretrain/npuz3sp8`, app
  `ap-gDtuS4QbsdA6Trg5BJ7GAc`) is the corrected active extension with explicit
  `--pretrain-save-steps 100`. Treat the save cadence and eval cadence as both
  every `100` optimizer steps. Through step `200`, W&B is healthy and artifact
  gates are passing. Step `100`: alerts `[]`,
  `train/accumulation_micro_batches=8.0`, clip fraction `0.0`, train/objective
  loss `1.6548`, raw loss `5.8442`, backward loss `0.1655`, supervised tokens
  `2.25`, multiplier `0.28125`, loss EMA `2.1994`, grad norm `0.2551`, grad
  EMA `0.3166`, eval loss `5.2072`, no W&B loss/grad spikes, and complete
  `checkpoint-100`. Step `200`: latest parent inspection at W&B step `210`
  still had alerts `[]`, accumulation `8.0`, clip fraction `0.0`, train/objective
  loss `1.1918`, raw loss `2.8130`, backward loss `0.1192`, supervised tokens
  `2.875`, multiplier `0.359375`, loss EMA `1.2096`, grad norm `0.1375`, grad
  EMA `0.1835`, LR `9.6723e-6`, no W&B loss/grad spikes, eval improved to
  `3.2008`, and the recent loss window improved `2.3684 -> 1.3782`.
  `checkpoint-200` contains `trainer_state.pt`, `projector.pt`, and
  `model_meta.json`. Step `250` monitor report still said continue: W&B state
  `running`, alerts `[]`, all `25` existing train rows had
  `train/accumulation_micro_batches=8.0`, clip fraction `0.0`, train/objective
  loss `1.1543`, raw loss `2.5884`, backward loss `0.1154`, supervised tokens
  `4.625`, multiplier `0.578125`, loss EMA `1.1451`, grad norm `0.0730`, grad
  EMA `0.1448`, LR `9.3971e-6`, no loss/grad spikes, eval still improving
  (`5.2072` at step `100` to `3.2008` at step `200`), and the recent loss
  window improved `2.1851 -> 1.3496`. A parent W&B re-check at step `260` again
  said continue: alerts `[]`, accumulation `8.0`, clip fraction `0.0`,
  train/objective loss `0.9854`, raw loss `3.0599`, backward loss `0.0985`,
  supervised tokens `2.5`, multiplier `0.3125`, loss EMA `1.1354`, grad norm
  `0.2356`, grad EMA `0.1553`, LR `9.3162e-6`, no loss/grad spikes, and recent
  window improved `2.1746 -> 1.2679`. Step `270` W&B was also clean: alerts
  `[]`, accumulation `8.0`, clip fraction `0.0`, train/objective loss `0.8212`,
  raw loss `2.5004`, backward loss `0.0821`, supervised tokens `2.5`,
  multiplier `0.3125`, loss EMA `1.0335`, grad norm `0.3569`, grad EMA
  `0.1999`, LR `9.2307e-6`, no loss/grad spikes, and recent window improved
  `2.1746 -> 1.2360`. Step `280` W&B still had no stop trigger: alerts `[]`,
  accumulation `8.0`, clip fraction `0.0`, train/objective loss `1.1318`, raw
  loss `2.1013`, backward loss `0.1132`, supervised tokens `4.875`, multiplier
  `0.609375`, loss EMA `1.0287`, grad norm `0.1368`, grad EMA `0.1865`, LR
  `9.1406e-6`, no loss/grad spikes, and recent window improved
  `2.0897 -> 1.2464`. Step `290` W&B also said continue: alerts `[]`,
  accumulation `8.0`, clip fraction `0.0`, train/objective loss `1.1997`, raw
  loss `2.4610`, backward loss `0.1200`, supervised tokens `4.75`, multiplier
  `0.59375`, loss EMA `0.9697`, grad norm `0.0716`, grad EMA `0.1587`, LR
  `9.0460e-6`, no loss/grad spikes, and recent window improved
  `2.0897 -> 1.2433`. Step `300` gate passed: W&B state `running`, alerts `[]`,
  accumulation `8.0`, clip fraction `0.0`, train/objective loss `1.1873`, raw
  loss `2.2154`, backward loss `0.1187`, supervised tokens `3.5`, multiplier
  `0.4375`, loss EMA `0.9823`, grad norm `0.1439`, grad EMA `0.1544`, LR
  `8.9472e-6`, no loss/grad spikes, eval improved again to `2.6205`
  (`5.2072 -> 3.2008 -> 2.6205`), and recent window improved
  `2.0225 -> 1.2503`. Independent monitor Russell verified all `30` W&B train
  rows from steps `10` through `300` had `train/accumulation_micro_batches=8.0`
  with missing `[]` and wrong `[]`. `checkpoint-300` contains
  `trainer_state.pt`, `projector.pt`, and `model_meta.json`. Modal app remains
  `ephemeral` with one active task. Console showed slower microbatch stretches
  after step `250`, so keep watching throughput and W&B sync health, but do not
  stop from console slowdown alone while W&B health remains clean. Step `320`
  soft W&B check was still clean: state `running`, alerts `[]`, accumulation
  `8.0`, clip fraction `0.0`, train/objective loss `0.5585`, raw loss `1.9328`,
  backward loss `0.0559`, supervised tokens `2.875`, multiplier `0.359375`,
  loss EMA `0.9289`, grad norm `0.0723`, grad EMA `0.1503`, LR `8.7370e-6`,
  no loss/grad spikes, and recent window improved `1.9974 -> 1.2017`.
  Step `330` soft W&B check remained clean: alerts `[]`, accumulation `8.0`,
  clip fraction `0.0`, train/objective loss `0.9805`, raw loss `1.9945`,
  backward loss `0.0981`, supervised tokens `4.625`, multiplier `0.578125`,
  loss EMA `0.8524`, grad norm `0.3951`, grad EMA `0.1747`, LR `8.6260e-6`,
  no loss/grad spikes, and recent window improved `1.9974 -> 1.1887`. The
  step-330 grad norm was above its EMA but below the playbook spike rule and
  not paired with clipping or W&B alerts. Step `340` fired the stop gate:
  active W&B alert `recent_loss_spikes`, train/objective loss `1.9874`, loss
  EMA `0.9362`, ratio `2.1227`, raw loss `3.0522`, backward loss `0.1987`,
  supervised tokens `6.0`, multiplier `0.75`, accumulation `8.0`, grad norm
  `0.0903`, grad EMA `0.1469`, clip fraction `0.0`, LR `8.5111e-6`, no grad
  spikes, and recent window still improving `1.9940 -> 1.1915`. The run was
  stopped anyway because an active W&B spike alert is unresolved until W&B
  clears it; console had reached about step `345` by termination, but no later
  W&B health row or checkpoint gate landed. Modal app ended `stopped` with zero
  tasks. Final verified checkpoint is `checkpoint-300`; `checkpoint-400` is
  absent. Judgment: stopped on strict W&B spike policy after a clean step-300
  candidate, not a clipping or eval-regression failure.
- `v4-stage1b-grounding-early250-from-stage1a2000-20260508-codex`
  (`babakdam/anymal-pretrain/ssoqebwc`, app `ap-A8Op30esUTAMBs2ZCokzU7`) is a
  completed A100 Stage 1B canary. Final W&B step `250`: eval/loss `0.9570`,
  train loss `0.415`, loss EMA `0.910`, grad norm `0.571`, grad EMA `0.678`,
  clip fraction `0.152`, LR `2.0e-5`, no grad spikes, active
  `recent_loss_spikes`. Eval improved monotonically across logged evals, and
  checkpoints `62`, `124`, `186`, and `248` are complete. Keep it labeled
  yellow-history because W&B recorded loss spikes at steps `80` and `170`.
- Stage 2A direct-calibration loss-scale setup is now guarded by W&B-first
  stop gates. The initial run
  `v4-stage2a-directcal-from-stage1a300-20260508-codex`
  (`babakdam/anymal-finetune/cv70gfmy`, app
  `ap-FwB5yLePPpMtMLRmPjVwFf`) was stopped at W&B step `8`: active
  `high_grad_clip_fraction`, clip fraction `1.0`, train loss `1.3366`, loss EMA
  `4.2684`, grad norm `17.6211`, grad EMA `17.0653`, LR `1.6e-4`, and
  accumulation `8.0`. The explicit-LoRA-LR follow-up
  `v4-stage2a-directcal-lora1e5-from-stage1a300-20260508-codex`
  (`babakdam/anymal-finetune/7iz3gqye`, app
  `ap-eXLMYJmlxBssgDMR5XHse0`) was also stopped at W&B step `5`: active
  `high_grad_clip_fraction`, clip fraction `1.0`, train loss `4.7371`, loss EMA
  `4.8712`, grad norm `18.9571`, grad EMA `20.0704`, LR `5e-6`, and
  accumulation `8.0`. The `loss_scale=0.05` follow-up
  `v4-stage2a-directcal-lossscale005-lora1e5-from-stage1a300-20260508-codex`
  (`babakdam/anymal-finetune/ivvpnfop`, app
  `ap-xDC5VyUHd6sA0JmwUxYrtG`) reduced the raw gradient scale but still failed
  the W&B gate: final W&B step `7` had active `high_grad_clip_fraction`, clip
  fraction `0.4286`, train/raw loss `4.5885`, backward loss `0.2294`, grad norm
  `0.8014`, grad EMA `0.9160`, LR `7e-6`, and accumulation `8.0`. None of these
  failed attempts produced checkpoint artifacts, and none should be rehabilitated
  from falling console loss.
- `v4-stage2a-directcal-lossscale003-lora1e5-from-stage1a300-20260508-codex`
  (`babakdam/anymal-finetune/fmrgmyy8`, app
  `ap-zvc4ZdZ0C1YHBSpyWvukVr`) completed as the Stage 2A loss-scale canary from
  Stage 1A `checkpoint-300`. Stop gates for this run were intentionally strict:
  any active W&B alert, sustained or recurring nonzero
  `health/grad_clip_fraction`, fresh inspector loss/grad spike, eval
  regression, missing/wrong accumulation rows, or missing checkpoint artifact at
  the declared save boundary is stop/escalate until W&B and artifacts clear it.
  Logs verified the intended Stage 1A projector path, dataset
  `v4_direct_calibration`, frozen connector, explicit projector and LoRA LR
  `1e-5`, Stage 2 backward loss scale `0.03`, and LoRA-only trainables:
  adapter `0`, LoRA `167,772,160`, other `0`. Step `50` passed the first
  checkpoint/eval gate: W&B had alerts `[]`, accumulation `8.0`, clip fraction
  `0.0`, train/raw loss `1.3340`, backward loss `0.0400`, loss EMA `1.9015`,
  grad norm `0.1807`, grad EMA `0.2761`, LR `6.2814e-6`, no loss/grad spikes,
  and eval loss `1.2776`; Modal volume `checkpoint-50` contains
  `trainer_state.pt`, `model_meta.json`, `llm/`, and `projector.pt`. Final W&B
  finished cleanly at step `100`: alerts `[]`, accumulation `8.0`, clip
  fraction `0.0`, train/raw loss `1.1870`, backward loss `0.0356`, loss EMA
  `1.1151`, grad norm `0.1255`, grad EMA `0.1138`, LR `1.0e-6`, no inspector
  loss/grad spikes, eval improved from `1.2776` at step `50` to `1.1988` at
  step `100`, and the 50-row recent loss window stayed bounded from `1.0693` to
  `1.0453`. Modal app completed normally. Modal volume
  `checkpoint-100` contains `trainer_state.pt`, `model_meta.json`, `llm/`, and
  `projector.pt`. Judgment: green completed Stage 2A canary; loss scale `0.03`
  is the first tested Stage 2A recipe here to clear the strict W&B clipping and
  spike gates through the requested 100 steps. It still needs external held-out
  VQA before promotion.
- `v4-stage2a-directcal-lossscale003-lora1e5-from-stage1b248-20260508-codex`
  (`babakdam/anymal-finetune/z7c3o6ug`, app
  `ap-lOnsPRzER2eCaTdpEpxEHu`) is the Stage 2A ablation from the Stage 1B
  grounding checkpoint `checkpoint-248`. Lorentz was assigned as the strict
  delegated monitor with W&B as the primary screen, and its early red call was
  preserved rather than talked away: at W&B step `7`, active
  `recent_loss_window_mean_up_gt_25pct` showed a small-window rise of about
  `40.5%`. Parent W&B re-checks at steps `9` and `12` showed the alert cleared,
  `health/grad_clip_fraction=0.0`, `train/accumulation_micro_batches=8.0`, and
  no pointwise loss/grad spikes, so the run continued under yellow-history.
  The step-50 gate passed with alerts `[]`, accumulation `8.0`, clip fraction
  `0.0`, train/raw loss `0.5935`, backward loss `0.0178`, loss EMA `0.7012`,
  grad norm `0.0122`, grad EMA `0.0142`, LR `6.2814e-6`, eval loss `0.7979`,
  and a complete `checkpoint-50`. A later W&B read around step `92` again
  reported active `recent_loss_window_mean_up_gt_25pct` (`0.5605 -> 0.8240`)
  while the app had already reached the final eval/checkpoint path; the final
  W&B snapshot at step `100` cleared alerts and showed eval `0.7956`, clip
  fraction `0.0`, train/raw loss `0.8841`, backward loss `0.0265`, loss EMA
  `0.6793`, grad norm `0.0144`, grad EMA `0.0156`, LR `1e-6`, no inspector
  loss/grad spikes, and recent 20-row window `0.7837 -> 0.6510`.
  `checkpoint-100` contains `trainer_state.pt`, `model_meta.json`, `llm/`, and
  `projector.pt`. Judgment: artifact-good/yellow-history, not green-pure. The
  completed checkpoint is valid for held-out VQA, but every downstream note must
  carry the transient W&B alerts from steps `7` and `92`.
- `v4-stage1b-clean500-from-a1ckpt300-20260509-codex`
  (`babakdam/anymal-pretrain/dsuz75tp`, app
  `ap-iU9sh2Zc43bqt7W4qcttjm`) is the active cleaner Stage 1B continuation from
  the accumulated-metrics A1 `checkpoint-300`. Carver is assigned as the strict
  delegated monitor and has been explicitly re-tasked to use W&B before logs,
  preserve historical alerts, and avoid charitable readings of spikes, clipping,
  loss-window alerts, and artifact lateness. Startup sync is yellow-history only:
  Carver's first report saw W&B temporarily return zero rows while console logs
  had started; later W&B reads populated normally, so this is not a model-health
  alert but remains recorded. Step `100`/`130` W&B gate is currently clean:
  alerts `[]`, accumulation `8.0`, clip fraction `0.0`, no inspector loss/grad
  spikes, eval improved from `2.5661` at step `50` to `2.3389` at step `100`,
  and the run is still `running`. Parent W&B step `130` showed train/objective
  loss `2.1259`, raw loss `2.6689`, backward loss `0.2126`, loss EMA `1.7974`,
  grad norm `0.1895`, grad EMA `0.3532`, LR `9.3162e-6`, and recent-window loss
  movement `1.6999 -> 1.8085` across 13 rows. Carver's next W&B check at step
  `140` remained alert-free with train/objective loss `2.2612`, raw loss
  `1.9379`, backward loss `0.2261`, loss EMA `1.6980`, grad norm `0.1818`,
  grad EMA `0.3226`, LR `9.1406e-6`, and recent-window movement
  `1.6190 -> 1.9696` (`21.6%`, below the `>25%` W&B alert threshold). That rise
  is on the watchlist; continue monitoring rather than calling it green.
  `checkpoint-100` contains `trainer_state.pt`, `projector.pt`, and
  `model_meta.json`; Carver noted W&B logged artifacts were still empty, so
  volume artifacts are currently the material artifact evidence. Step `200` gate
  passed: W&B alerts `[]`, accumulation `8.0`, clip fraction `0.0`, no
  inspector loss/grad spikes, eval improved again to `2.1605`
  (`2.5661 -> 2.3389 -> 2.2426 -> 2.1605`), train/objective loss `1.3378`, raw
  loss `2.2441`, backward loss `0.1338`, loss EMA `1.5344`, grad norm
  `0.2582`, grad EMA `0.2349`, LR `7.75e-6`, and recent-window movement
  `1.6604 -> 1.8989`, below the W&B alert threshold. `checkpoint-200` contains
  `trainer_state.pt`, `projector.pt`, and `model_meta.json`. Carver independently
  passed the step-300 gate: W&B step `300`, alerts `[]`, accumulation `8.0`, clip
  fraction `0.0`, no W&B loss/grad spikes, train/objective loss `1.1607`, raw
  loss `1.9990`, backward loss `0.1161`, loss EMA `1.5307`, grad norm `0.1923`,
  grad EMA `0.2195`, LR `4.7186e-6`, and recent-window movement
  `1.8337 -> 1.5904`. Eval trajectory at that watch was
  `2.3389 -> 2.2426 -> 2.1605 -> 2.0969 -> 2.0531`. `checkpoint-300` contains
  `trainer_state.pt`, `projector.pt`, and `model_meta.json`; Carver verified
  metadata matches V4 shape (`64` global / `64` local, hidden `1024`, output
  scale `1.0`, gate init `0.0001`, 2D positions enabled). Parent W&B step `330`
  remained clean with alerts `[]`, clip fraction `0.0`, accumulation `8.0`, no
  inspector loss/grad spikes, train/objective loss `1.5325`, raw loss `2.0363`,
  backward loss `0.1533`, loss EMA `1.5411`, grad norm `0.2107`, grad EMA
  `0.2146`, LR `3.8143e-6`, eval `2.0531`, and recent-window movement
  `1.7791 -> 1.6317`. Step `400` gate passed: W&B alerts `[]`, accumulation
  `8.0`, clip fraction `0.0`, no inspector loss/grad spikes, eval improved to
  `2.0128`, train/objective loss `2.5717`, raw loss `2.2020`, backward loss
  `0.2572`, loss EMA `1.4973`, grad norm `0.1209`, grad EMA `0.1950`, LR
  `2.0528e-6`, and recent-window movement `1.7796 -> 1.5779`. The step-400
  train row is elevated versus EMA but below W&B spike criteria and not paired
  with clipping; it stays on watch. `checkpoint-400` contains `trainer_state.pt`,
  `projector.pt`, and `model_meta.json`. The run did not complete to step `500`.
  Post-400 logs showed repeated auto-resume from `checkpoint-400`, including a
  costly `Skipping 3200 micro-batches to resume position...` path, while W&B
  stopped advancing after step `410` and reported state `crashed`. Final W&B
  step `410` had alerts `[]`, accumulation `8.0`, clip fraction `0.0`, no
  inspector loss/grad spikes, train/objective loss `1.3611`, raw loss `1.6318`,
  backward loss `0.1361`, loss EMA `1.5296`, grad norm `0.1832`, grad EMA
  `0.1969`, LR `1.8594e-6`, eval `2.0128`, and recent-window movement
  `1.7796 -> 1.5676`; the stop was operational/W&B-crashed, not a metric
  blow-up. Parent stopped the Modal app; post-stop app state was `stopped` with
  zero tasks. Verified checkpoints are `checkpoint-100`, `checkpoint-200`,
  `checkpoint-300`, and `checkpoint-400`; there is no `checkpoint-500`.
  Judgment: useful clean-through-400 Stage 1B checkpoint, but the run is
  yellow-history/operational-crash rather than completed green.
- `v4-stage2a-directcal-lossscale003-lora1e5-from-stage1b400clean-20260509-codex`
  (`babakdam/anymal-finetune/k00wp8wx`, app `ap-ukMv087X6GXABYwoH5Pet2`) was
  the first downstream Stage 2A gate from the clean-through-400 Stage 1B
  checkpoint. Startup contract was correct after Stage 2 adapter disabling:
  adapter/projector trainables `0`, LoRA trainables `167,772,160`, other `0`,
  connector frozen for the actual optimizer, batch size `2`, accumulation `8`,
  projector/LoRA LR `1e-5`, and Stage 2 backward `loss_scale=0.03`. W&B stopped
  this run early: step `3` already had active `high_grad_clip_fraction`, clip
  fraction `1.0`, grad norm `1.8728`, grad EMA `1.2073`, raw/train loss
  `3.7292`, backward loss `0.1119`, LR `3e-6`, and accumulation `8.0`. Parent
  stopped the Modal app; final W&B step `4` was `finished` but still had active
  `high_grad_clip_fraction`, clip fraction `1.0`, grad norm `1.4606`, grad EMA
  `1.2200`, raw/train loss `3.9609`, backward loss `0.1188`, loss EMA
  `3.7435`, LR `4e-6`, no eval rows, no inspector loss/grad spikes, and no
  checkpoints. Judgment: red optimization failure for `loss_scale=0.03` from
  Stage1B400; do not use for accuracy. If retrying this checkpoint, reduce
  Stage 2 update scale before any longer run.
- `v4-stage2a-directcal-lossscale001-lora1e5-from-stage1b400clean-20260509-codex`
  (`babakdam/anymal-finetune/qx0mcvwr`, app `ap-NIhQMJHaMAc1eEZbJvlBdJ`)
  completed the reduced Stage 2 scale retry from Stage1B400. Startup and final
  trainable summary matched the intended contract: frozen connector /
  adapter-projector `0`, LoRA `167,772,160`, other `0`, batch size `2`,
  accumulation `8`, projector/LoRA LR `1e-5`, and backward `loss_scale=0.01`.
  W&B step `2` cleared the early clipping gate where `loss_scale=0.03` failed:
  alerts `[]`, clip fraction `0.0`, grad norm `0.5715`, backward loss `0.0382`,
  accumulation `8.0`. Step `50` gate passed after eval/save settled: alerts
  `[]`, clip `0.0`, no inspector loss/grad spikes, eval `1.2514`, grad norm
  `0.0367`, grad EMA `0.1055`, raw/train loss `1.1271`, backward loss
  `0.0113`, loss EMA `1.5724`, and complete `checkpoint-50`. Final W&B step
  `100` finished cleanly with alerts `[]`, accumulation `8.0`, clip `0.0`, no
  inspector loss/grad spikes, eval improved to `1.1746`, raw/train loss
  `1.1429`, backward loss `0.0114`, loss EMA `1.0822`, grad norm `0.0310`,
  grad EMA `0.0404`, LR `1e-6`, and recent-window movement
  `1.0384 -> 1.0452` across 50 rows. `checkpoint-100` contains
  `trainer_state.pt`, `model_meta.json`, `llm/`, and `projector.pt`. Judgment:
  green optimization/artifact run for the Stage1B400 `loss_scale=0.01` recipe;
  direct-prompt VQA still failed afterward (overall `5.400`, EOS `0.618`,
  max-hit `0.382`), so the training run is green but the model candidate is not
  promotable.
- DeepStack-lite V4 launch attempts on `2026-05-09` are now in the ledger. The
  first command using `--background` (`ap-Wu3TPSPQYTOhr3HWPZHHSu`) failed before
  training because the local entrypoint rejects background mode. A detached
  120-step attempt (`ap-JLfSYAeQlCyjpqIGIvYA4B`) also failed operationally with
  zero Modal tasks and no W&B run. These are not model-health failures, but they
  stay recorded so missing W&B evidence is never mistaken for a green run.
- `v4-stage1a-deepstacklite-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-canary20-20260509-codex`
  (`babakdam/anymal-pretrain/w7ktjb94`, app `ap-PGBfgosRiESzTdPAwUKPua`)
  completed the first DeepStack-lite canary. Carver monitored it with W&B as the
  primary source and strict spike/clipping posture. Final W&B step `20`:
  state `finished`, alerts `[]`, accumulation `8.0`, clip `0.0`, no inspector
  loss/grad spikes, train/objective loss `2.3092`, raw loss `7.4254`,
  backward loss `0.2309`, loss EMA `2.7728`, grad norm `0.1844`, grad EMA
  `0.1474`, LR `1e-6`, and no eval rows because the canary ended before the
  eval cadence. `checkpoint-20` contains `trainer_state.pt`, `projector.pt`, and
  `model_meta.json`; metadata confirms
  `connector_type=deepstack_spatial_perceiver_resampler`,
  `vision_feature_strategy=deepstack_lite`, layers `[-3, -2, -1]`, `64/64`
  image-token split, hidden `1024`, gate init `0.0001`, and 2D positions.
  Judgment: pass/continue canary, yellow-history for startup W&B sync lag and
  empty W&B artifact table.
- `v4-stage1a-deepstacklite-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-canary120b-20260509-codex`
  (`babakdam/anymal-pretrain/kb8js5z0`, app `ap-aBJHyxraKrFQJiWcIOtK9n`)
  completed the longer DeepStack-lite Stage 1A canary. Startup had a W&B
  first-row sync gap, which is recorded as yellow-history rather than ignored.
  Step `50` gate passed: alerts `[]`, accumulation `8.0`, clip `0.0`, no
  inspector loss/grad spikes, train/objective loss `1.8737`, raw loss `6.2265`,
  backward loss `0.1874`, loss EMA `2.5355`, grad norm `0.1271`, grad EMA
  `0.1281`, LR `7.5196e-6`, eval `7.7001`, recent window `2.2928 -> 1.9261`,
  and complete `checkpoint-50`. Step `100` eval improved to `7.1668` and
  `checkpoint-100` was saved. Final W&B step `120`: state `finished`, alerts
  `[]`, accumulation `8.0`, clip `0.0`, no loss/grad spikes, train/objective
  loss `1.9442`, raw loss `6.7862`, backward loss `0.1944`, loss EMA `2.3480`,
  grad norm `0.1339`, grad EMA `0.1412`, LR `1e-6`, eval losses
  `7.7001 -> 7.1668`, and recent-window movement `2.1911 -> 2.3420` across
  `12` train rows, below the alert threshold. `checkpoint-100` contains
  `trainer_state.pt`, `projector.pt`, and `model_meta.json`; metadata confirms
  the DeepStack-lite connector and layers `[-3, -2, -1]`. Modal app ended
  `stopped` with zero tasks. Judgment: pass/continue, not final promotion;
  preserve yellow-history for startup W&B sync gaps and empty W&B artifacts.
- `v4-stage1a-deepstacklite-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-extended500-save100-20260509-codex`
  (`babakdam/anymal-pretrain/xb0silsi`, app
  `ap-B8Hvvx5yEIsO1sFO7LFJgU`) completed the longer DeepStack-lite Stage 1A run
  with explicit save/eval cadence every `100` optimizer steps. Carver was the
  strict W&B-first monitoring sub-agent for this run and independently reported
  continue/yellow-history at the step-400 gate. Startup had a temporary
  W&B first-row sync gap, which remains yellow-history. Early watch items also
  remain attached: step `50` eval `8.2100` was worse than the 120b canary's
  first eval `7.7001`; step `60` had a train/objective jump to `2.8969`; and
  the step-100 recent loss window rose about `16.9%`, below the alert threshold
  but not erased. Step `200`/`240` gate passed under parent and Carver checks:
  W&B state `running`, latest step `240`, train rows `24`, alerts `[]`,
  `train/accumulation_micro_batches=8.0`, clip fraction `0.004167`, no
  inspector loss/grad spikes, train/objective loss `1.0674`, raw loss
  `2.3201`, backward loss `0.1067`, loss EMA `1.3126`, grad norm `0.1738`,
  grad EMA `0.2475`, LR `6.5886e-6`, eval trend
  `8.2100 -> 6.9942 -> 6.1354 -> 3.6915`, and recent-window movement improving
  rather than rising. `checkpoint-200` contains `trainer_state.pt`,
  `projector.pt`, and `model_meta.json`; metadata matches DeepStack-lite V4
  with connector `deepstack_spatial_perceiver_resampler`, layers
  `[-3, -2, -1]`, `3` feature levels, `64/64` image tokens, hidden `1024`,
  gate init `0.0001`, and 2D positions. Judgment at this gate:
  continue/yellow-history. Step `280` soft W&B check stayed clean with alerts
  `[]`, accumulation `8.0`, clip fraction `0.003571`, no inspector loss/grad
  spikes, train/objective loss `1.2628`, raw loss `2.6646`, backward loss
  `0.1263`, loss EMA `1.1412`, grad norm `0.3273`, grad EMA `0.2437`, LR
  `5.3430e-6`, eval `3.3106`, and recent-window movement `2.0322 -> 1.2820`.
  Step `300`/`310` gate also passed: W&B state `running`, latest step `310`,
  alerts `[]`, accumulation `8.0`, clip fraction `0.003226`, no inspector
  loss/grad spikes, train/objective loss `1.5867`, raw loss `2.8101`,
  backward loss `0.1587`, loss EMA `1.1152`, grad norm `0.1575`, grad EMA
  `0.2317`, LR `4.4114e-6`, eval improved again to `3.0524`, and
  recent-window movement improved `1.8051 -> 1.3234`. `checkpoint-300`
  contains `trainer_state.pt`, `projector.pt`, and `model_meta.json`. The small
  nonzero clip fraction is not a stop condition, but it stays in the ledger and
  must be re-checked at the next gate. Step `350`/`360`/`390` soft checks
  stayed W&B-clean after the console-only suspicious microbatch loss around
  step `353`: alerts `[]`, no inspector loss/grad spikes, accumulation `8.0`,
  eval improved to `2.8963` at step `360`, and the loss window kept improving
  (`1.2450 -> 1.1550` at the step-390 check). Step `400`/`410` hard gate also
  passed under the same non-charitable W&B-first policy: W&B state `running`,
  latest checked step `410`, train rows `41`, alerts `[]`, accumulation `8.0`,
  clip fraction `0.002439`, no inspector loss/grad spikes, train/objective loss
  `0.9197`, raw loss `3.3103`, backward loss `0.0920`, loss EMA `1.0198`,
  grad norm `0.2138`, grad EMA `0.2031`, LR `1.8594e-6`, eval improved again
  to `2.8282`, and recent-window movement improved `1.3234 -> 1.0495`.
  `checkpoint-400` contains `trainer_state.pt`, `projector.pt`, and
  `model_meta.json`. Judgment remains continue/yellow-history, not green-pure:
  startup W&B gaps, earlier eval lag, the step-60 jump, the step-100 loss-window
  rise, console-only step-353 spike, and nonzero clipping remain preserved for
  the next watcher. Step `450` soft gate passed: W&B state `running`, latest
  checked step `450`, train rows `45`, alerts `[]`, accumulation `8.0`, clip
  fraction `0.002222`, no inspector loss/grad spikes, train/objective loss
  `1.1924`, raw loss `2.7074`, backward loss `0.1192`, loss EMA `0.9698`,
  grad norm `0.1380`, grad EMA `0.1882`, LR `1.2714e-6`, eval improved to
  `2.7581`, and recent-window movement improved `1.2361 -> 0.9460`. Final
  step `500` closeout passed: W&B state `finished`, latest step `500`, train
  rows `50`, alerts `[]`, accumulation `8.0`, clip fraction `0.002000`, no
  inspector loss/grad spikes, train/objective loss `1.9103`, raw loss `3.9376`,
  backward loss `0.1910`, loss EMA `1.0196`, grad norm `0.2567`, grad EMA
  `0.1705`, LR `1.0e-6`, eval improved again to `2.7120`, and recent-window
  movement was controlled (`1.1162 -> 1.0411`). `checkpoint-500` contains
  `trainer_state.pt`, `projector.pt`, and `model_meta.json`. Console showed a
  late microbatch loss of `5.8067` around step `499`; W&B did not flag a loss
  spike, but the console spike stays in yellow-history. Final judgment:
  completed artifact-good/yellow-history Stage 1A DeepStack checkpoint, suitable
  for downstream Stage 1B/Stage 2A testing but not a standalone promotion.
- `v4-stage1b-deepstacklite-from-stage1a500-tokennorm8-aggmetrics-lossscale01-gate00001-lr1e5-extended500-save100-20260509-codex`
  (`babakdam/anymal-pretrain/2wiqwnto`, app
  `ap-MEQM1e6xO2qip772hSqoHr`) was the DeepStack Stage 1B grounding run
  from the Stage 1A DeepStack `checkpoint-500`. It intentionally uses
  `--pretrain-checkpoint`, not `--resume-checkpoint`, so connector weights load
  while optimizer/scheduler state starts fresh. Rawls is the strict W&B-first
  monitoring sub-agent for this run. Startup verified DeepStack
  `deepstack_spatial_perceiver_resampler`, layers `[-3, -2, -1]`, `3` feature
  levels, `64/64` tokens, hidden `1024`, gate init `0.0001`, and Stage 1
  connector-only trainables `44,377,089`. Dataset diagnostics stayed clean for
  direct answers (`C<|eot_id|>`, `phone<|eot_id|>`, `no<|eot_id|>`) with `128`
  image placeholders. First W&B gate at step `20`: state `running`, train rows
  `2`, alerts `[]`, accumulation `8.0`, clip fraction `0.0`, no inspector
  loss/grad spikes, train/objective loss `0.7141`, raw loss `1.9351`,
  backward loss `0.0714`, loss EMA `0.8854`, grad norm `0.2802`, grad EMA
  `0.1625`, LR `4.0e-6`, and no eval due yet. Judgment: continue, with
  startup row timing watched but no active W&B health alert.
  Step `40` W&B strict-monitor check escalated to yellow: active alert
  `recent_loss_window_mean_up_gt_25pct` with recent-window movement
  `0.7099 -> 1.0576` across four train rows. Other health gates were still
  clean at that snapshot: state `running`, accumulation `8.0`, clip fraction
  `0.0`, no inspector loss/grad spikes, train/objective loss `1.0845`, raw
  loss `3.0657`, backward loss `0.1085`, loss EMA `0.9794`, grad norm
  `0.1741`, grad EMA `0.1814`, LR `8.0e-6`, and no eval/checkpoint due yet.
  Judgment: continue only as yellow-history/watch; the active W&B loss-window
  alert is preserved even if a later window clears.
  Step `90` W&B parent check cleared active alerts but stayed on yellow-watch:
  state `running`, train rows `9`, history rows `10`, alerts `[]`,
  accumulation `8.0`, clip fraction `0.0`, no inspector loss/grad spikes,
  train/objective loss `1.0392`, raw loss `3.4628`, backward loss `0.1039`,
  loss EMA `0.9083`, grad norm `0.3390`, grad EMA `0.2573`, LR
  `9.8257e-6`, first eval `2.6056`, and recent-window movement
  `0.8837 -> 1.0383` across nine rows. Because the window was rising, this
  stays yellow-watch even without an active W&B alert.
  Step `100` boundary passed W&B and artifact gates: alerts `[]`,
  accumulation `8.0`, clip fraction `0.0`, no inspector loss/grad spikes,
  train/objective loss `0.8114`, raw loss `2.8371`, backward loss `0.0811`,
  loss EMA `0.8870`, grad norm `0.2719`, grad EMA `0.2651`, LR
  `9.7286e-6`, eval improved to `2.4619`, and recent-window movement was
  `0.8730 -> 1.0345` across ten rows. `checkpoint-100` is visible with
  `trainer_state.pt`, `projector.pt`, and `model_meta.json`. Judgment:
  continue, but preserve the step-40 active loss-window alert and the later
  rising-window yellow-watch in run history.
  Step `130` W&B check after console microbatch spikes stayed formally clean:
  alerts `[]`, accumulation `8.0`, clip fraction `0.0`, no inspector
  loss/grad spikes, train/objective loss `1.3580`, raw loss `2.8378`,
  backward loss `0.1358`, loss EMA `0.9113`, grad norm `0.2510`, grad EMA
  `0.2052`, LR `9.3162e-6`, latest eval still `2.4619`, and recent-window
  movement `0.8506 -> 1.0112` across thirteen rows. Judgment: continue, but
  keep as yellow-watch because the window is still rising and console had
  isolated high microbatch losses.
  Step `150` W&B gate was clean and trend-improving: alerts `[]`,
  accumulation `8.0`, clip fraction `0.0`, no inspector loss/grad spikes,
  train/objective loss `0.7022`, raw loss `2.1585`, backward loss `0.0702`,
  loss EMA `0.8676`, grad norm `0.1485`, grad EMA `0.1933`, LR
  `8.9472e-6`, eval improved to `2.2104`, and recent-window movement
  flattened to `0.8970 -> 0.8756` across fifteen rows. Judgment: continue as
  green-at-gate/yellow-history; do not erase the earlier active W&B
  loss-window alert.
  Step `170` W&B check stayed clean despite console-only high microbatch losses
  around the 150-170 span, including a `4.7524` terminal loss near step `165`:
  alerts `[]`, accumulation `8.0`, clip fraction `0.0`, no inspector
  loss/grad spikes, train/objective loss `1.4308`, raw loss `2.9716`,
  backward loss `0.1431`, loss EMA `0.9009`, grad norm `0.2820`, grad EMA
  `0.2305`, LR `8.5111e-6`, latest eval still `2.2104`, and recent-window
  movement `0.9609 -> 0.8984` across seventeen rows. Judgment: continue;
  console-only spikes remain yellow-history but do not override clean W&B.
  Step `200` boundary had a console HealthMonitor warning: loss EMA did not
  decrease over the 200-step monitor window (`0.5253 -> 0.8337`). W&B did not
  turn that into an active stop condition: alerts `[]`, accumulation `8.0`,
  clip fraction `0.0`, no inspector loss/grad spikes, train/objective loss
  `0.4807`, raw loss `1.0720`, backward loss `0.0481`, loss EMA `0.8337`,
  grad norm `0.3981`, grad EMA `0.2169`, LR `7.75e-6`, eval improved to
  `2.1062`, and recent-window movement was down (`0.9538 -> 0.8641`) across
  twenty rows. `checkpoint-200` is visible with `trainer_state.pt`,
  `projector.pt`, and `model_meta.json`. Judgment: continue as
  artifact-good/yellow-history; the HealthMonitor warning remains recorded.
  Step `230` W&B check remained clean: alerts `[]`, accumulation `8.0`, clip
  fraction `0.0`, no inspector loss/grad spikes, train/objective loss
  `0.7526`, raw loss `1.8328`, backward loss `0.0753`, loss EMA `0.8181`,
  grad norm `0.2766`, grad EMA `0.1900`, LR `6.8906e-6`, latest eval
  `2.1062`, and recent-window movement `0.9732 -> 0.8606`. Judgment:
  continue; yellow-history remains from prior W&B/console warnings.
  Step `250` W&B eval gate was clean: alerts `[]`, accumulation `8.0`, clip
  fraction `0.0`, no inspector loss/grad spikes, train/objective loss
  `0.8676`, raw loss `1.8138`, backward loss `0.0868`, loss EMA `0.8477`,
  grad norm `0.3728`, grad EMA `0.2094`, LR `6.2814e-6`, eval improved to
  `2.0273`, and recent-window movement was nearly flat (`0.8919 -> 0.9030`).
  No checkpoint was due at step `250`. Judgment: continue as
  clean-at-gate/yellow-history.
  Step `120` W&B follow-up remained GO/yellow-history: state `running`, train
  rows `12`, history rows `14`, alerts `[]`, accumulation `8.0`, clip fraction
  `0.0`, no inspector loss/grad spikes, train/objective loss `0.5307`, raw loss
  `1.4477`, backward loss `0.0531`, loss EMA `0.8503`, grad norm `0.1693`,
  grad EMA `0.2333`, LR `9.4733e-6`, eval trend `2.6056 -> 2.4619`, and
  recent-window movement `0.8506 -> 0.9534` across twelve train rows. Modal app
  still had one live task, and `checkpoint-100` remained artifact-complete with
  `trainer_state.pt`, `projector.pt`, and `model_meta.json`. Judgment: GO, not
  green-pure; preserve step `40`/`50` active W&B loss-window alert and step
  `90` rising-window yellow-watch.
  Step `200`/`210` gate passed but added yellow-history: the first immediate
  `checkpoint-200` volume listing was missing while W&B had just reached step
  `200`, then a re-check after the boundary settled showed `checkpoint-200`
  complete with `trainer_state.pt`, `projector.pt`, and `model_meta.json`.
  W&B step `210` was GO: state `running`, train rows `21`, history rows `25`,
  alerts `[]`, accumulation `8.0`, clip fraction `0.0`, no inspector loss/grad
  spikes, train/objective loss `0.6561`, raw loss `1.3291`, backward loss
  `0.0656`, loss EMA `0.8077`, grad norm `0.1187`, grad EMA `0.2169`, LR
  `7.4727e-6`, eval trend `2.6056 -> 2.4619 -> 2.2104 -> 2.1062`, and
  recent-window movement `0.9538 -> 0.8452` across twenty-one train rows.
  Console also emitted `[HealthMonitor][WARNING] Loss EMA has not decreased by
  >= 0.01 over the last 200 steps (step 200)`, so the run remains
  GO/yellow-history rather than green-pure.
  Step `260` soft W&B check stayed GO/yellow-history: state `running`, train
  rows `26`, alerts `[]`, accumulation `8.0`, clip fraction `0.0`, no
  inspector loss/grad spikes, train/objective loss `0.8278`, raw loss `2.5899`,
  backward loss `0.0828`, loss EMA `0.8543`, grad norm `0.1434`, grad EMA
  `0.1809`, LR `5.9704e-6`, eval improved to `2.0273`, and recent-window
  movement improved `0.9371 -> 0.8430`. Step `300` hard gate also passed:
  state `running`, train rows `30`, history rows `36`, alerts `[]`,
  accumulation `8.0`, clip fraction `0.0`, no inspector loss/grad spikes,
  train/objective loss `0.9274`, raw loss `1.6377`, backward loss `0.0927`,
  loss EMA `0.7789`, grad norm `0.2612`, grad EMA `0.1736`, LR `4.7186e-6`,
  eval trend `2.4619 -> 2.2104 -> 2.1062 -> 2.0273 -> 1.9358`, and
  recent-window movement `0.8641 -> 0.8489` across the latest twenty rows.
  `checkpoint-300` contains `trainer_state.pt`,
  `projector.pt`, and `model_meta.json`. Judgment: GO/yellow-history, not
  green-pure, preserving earlier active W&B alert, step-90 rising-window watch,
  step-200 HealthMonitor warning, and transient checkpoint-200 visibility lag.
  Final step `350` strict W&B check was STOP/yellow-red history, not green:
  parent stopped app `ap-MEQM1e6xO2qip772hSqoHr` after W&B reported state
  `finished` at step `350` with active alert `recent_loss_spikes`. The active
  spike was step `340`, train/objective loss `1.6254` versus loss EMA `0.7558`
  for ratio `2.15`; this matches the console HealthMonitor loss-spike warning
  and wins over later calming signals. Final W&B still had accumulation `8.0`,
  clip fraction `0.0`, no grad spikes, train/objective loss `0.2877`, raw loss
  `1.0847`, backward loss `0.0288`, loss EMA `0.7313`, grad norm `0.1331`,
  grad EMA `0.1574`, LR `3.25e-6`, and eval improved to `1.9128`, but active
  W&B spike plus manual stop prevents any green judgment. Modal app is stopped
  with zero tasks. Verified checkpoints are `checkpoint-100`, `checkpoint-200`,
  and `checkpoint-300`; `checkpoint-400` and `checkpoint-500` are absent. Last
  safe artifact is `checkpoint-300`. Judgment: STOP/yellow-red history; do not
  promote as green, and carry the earlier step `40`/`50` loss-window alert,
  step `90` rising-window watch, step `200` HealthMonitor warning, transient
  `checkpoint-200` visibility lag, console-only step `307`-`342` spike
  candidates, and final active step-`340` W&B loss spike.

- `v4-stage2a-directcal-lossscale003-lora1e5-from-deepstack-stage1b300-fix1-20260509-codex`
  (`babakdam/anymal-finetune/3jp174rm`, app `ap-ktE7hOXSbNmpHZHEievcyN`)
  is STOP/red for Stage 2A. Startup loaded the DeepStack Stage 1B
  `checkpoint-300`, froze the connector, and confirmed LoRA-only trainables
  (`adapter 0`, `lora 167,772,160`, `other 0`) with `loss_scale=0.03`.
  W&B step `3` already had active alert `high_grad_clip_fraction`, clip
  fraction `1.0`, raw/train loss `6.2854`, backward loss `0.1886`, grad norm
  `1.1377`, grad EMA `1.8291`, LR `3e-6`, and accumulation `8.0`. Parent
  stopped the app at step `4`; final W&B still had active
  `high_grad_clip_fraction`, clip fraction `0.75`, raw/train loss `5.8650`,
  backward loss `0.1760`, grad norm `0.7287`, grad EMA `1.7741`, no loss/grad
  spikes, and no eval/checkpoint. Judgment: do not rerun the DeepStack Stage 2A
  `loss_scale=0.03` branch unchanged.
- `v4-stage2a-directcal-lossscale001-lora1e5-from-deepstack-stage1b300-20260509-codex`
  (`babakdam/anymal-finetune/zq2qg4em`, app `ap-9RS32uYnaDaBjDFWSdWEiZ`)
  is the active DeepStack Stage 2A `loss_scale=0.01` validation from the last
  safe DeepStack Stage 1B artifact, `checkpoint-300`. Startup verified the
  DeepStack metadata load, `v4_direct_calibration`, clean direct-answer labels,
  frozen connector, and LoRA-only trainables (`adapter 0`, `lora 167,772,160`,
  `other 0`). Early W&B gates at steps `26`, `33`, and `42` were clean:
  alerts `[]`, accumulation `8.0`, clip fraction `0.0`, no inspector
  loss/grad spikes, and improving recent-window means. Step `50`/`55` hard gate
  passed W&B and artifacts: alerts `[]`, accumulation `8.0`, clip fraction
  `0.0`, no loss/grad spikes, eval `1.3578`, raw/train loss `0.7100`,
  backward loss `0.0071`, loss EMA `1.9915`, grad norm `0.0354`, grad EMA
  `0.1278`, LR `5.5e-6`, recent-window movement `1.4241 -> 1.1655`, and
  complete `checkpoint-50` with `trainer_state.pt`, `model_meta.json`, `llm/`,
  and `projector.pt`. Final STOP override: W&B step `92` produced active alert
  `recent_loss_window_mean_up_gt_25pct` with recent-window movement
  `0.9461 -> 1.2977`; parent stopped the app immediately, and final W&B state
  is `finished` at step `95` with the same active alert still present. Final
  metrics: accumulation `8.0`, clip fraction `0.0`, no loss/grad spikes,
  raw/train loss `1.3408`, backward loss `0.0134`, loss EMA `1.2586`,
  grad norm `0.0769`, grad EMA `0.0566`, LR `1.068e-6`, eval still `1.3578`,
  and final recent-window movement `1.0201 -> 1.2794`. Modal app is stopped
  with zero tasks. Only `checkpoint-50` exists; `checkpoint-100` was never
  reached. The last clean artifact, `checkpoint-50`, was externally evaluated
  after patching the DeepStack eval path to capture SigLIP hidden states via
  hooks when Transformers exposes only `last_hidden_state`: direct-prompt
  seed-42 VQA scored overall `4.233`, number `2.546`, other `2.469`, yes/no
  `7.580`, EOS `1.0`, max-hit `0.0`, and average generated tokens `3.582`.
  Promotion guard failed V1 floor, incumbent, and yes/no recovery; hygiene
  passed. Judgment: STOP/yellow-red history and accuracy-red; do not promote.

## Practical Rule

Do not be hypnotized by jagged loss, and do not wave it away either. A run with
ugly batches can keep going only when W&B shows bounded EMAs, controlled clipping,
no recurring spikes, improving validation, and good checkpoints. A run with
smooth loss but the wrong checkpoint, wrong data, missing artifacts, or silent
restarts is not healthy.

Strict monitor posture, added for the V4 modernization work: the watcher is not
allowed to be charitable about spikes, clipping, loss-window alerts, restart
loops, or missing W&B rows. W&B is the source of truth before logs. Logs can
explain a failure mode, but they cannot clear a W&B alert or erase historical
spikes. Every monitor handoff must preserve any prior active alert or yellow/red
event in the run ledger even if the latest step later looks clean.

Next V4 recipe to watch, if launched: `v4_semantic_calibration`. It is a
frozen-connector, LoRA-only Stage 2A ablation, using balanced canonical yes/no
labels and weights
`0.40/0.20/0.15/0.20/0.05` for yes-no/number/COCO/other/short. Treat it as
recipe evidence only if W&B confirms the intended dataset, checkpoint branch,
LoRA-only trainables, `train/accumulation_micro_batches=8.0`, zero or bounded
clipping, no active alerts, and complete `checkpoint-50`/`checkpoint-100`
artifacts. The first isolation branch should beat the prior Stage1B248 direct
calibration result (`7.600` overall) while preserving EOS `>=0.98` and
max-token hits `<=0.02`; lower training/eval loss alone is not enough.

First semantic-calibration attempt:
`v4-stage2a-semanticcal-lossscale003-lora1e5-from-stage1b248-20260509-codex`
(`ap-JP6wrEiBQPEjVWxJ3eAQuE`, W&B
`https://wandb.ai/babakdam/anymal-finetune/runs/muyq9oz9`) was stopped early by
the parent. Startup contract was correct: Stage1B248 checkpoint, semantic
dataset, balanced yes/no source (`60,040` examples), clean direct labels,
frozen connector after Stage 2 adapter disabling, LoRA-only trainables
`167,772,160`, accumulation `8.0`, and clipping `0.0`. W&B inspections at steps
`6` and `10` both had active `recent_loss_window_mean_up_gt_25pct`; step `10`
showed recent-window movement `0.7242 -> 0.9392`, train/raw loss `0.9719`,
backward loss `0.0292`, grad norm `0.0162`, grad EMA `0.0145`, loss EMA
`0.8037`, LR `1e-5`, no point loss/grad spikes, and no eval rows. The app was
stopped immediately. Final W&B reached step `11` and cleared alerts after the
stop signal, but no eval/checkpoint exists; do not treat this as a green run.

Second semantic-calibration attempt:
`v4-stage2a-semanticcal-bs4-lossscale003-lora1e5-from-stage1b248-20260509-codex`
(`ap-0nOSPSoi635kK2vkRmuRpU`, W&B
`https://wandb.ai/babakdam/anymal-finetune/runs/oxuky3ss`) completed under
direct parent babysitting after the fresh monitoring-agent spawn hit the
agent/thread limit. This used the same Stage1B248 legacy V4 connector branch,
but raised per-device batch size to `4` for effective batch `32` while keeping
frozen connector, LoRA-only trainables (`167,772,160`), and Stage 2
`loss_scale=0.03`. Startup verified `v4_semantic_calibration`, balanced yes/no
source (`60,040` examples; `yes=30,549`, `no=29,491` after available-image
filtering), clean direct labels, accumulation `8.0`, and clipping `0.0`. W&B
was checked repeatedly, not just logs: steps `4`, `8`, `12`, `19`, `27`, `37`,
`50`, `62`, `68`, `70`, `73`, `79`, `86`, `92`, and `100` all had alerts `[]`,
clip fraction `0.0`, no inspector loss/grad spikes, and accumulation `8.0`.
Yellow-watch context still stays recorded: console-only microbatch losses
reached about `3.84` around step `86`, and the step-79 W&B recent window rose
slightly (`0.7903 -> 0.8280`), below the `+25%` stop rule. Final W&B state was
`finished` at step `100` with eval losses `0.8464 -> 0.8416`, train/raw loss
`0.6553`, backward loss `0.0197`, loss EMA `0.7909`, grad norm `0.0109`, grad
EMA `0.0136`, LR `1e-6`, and recent-window movement `0.8753 -> 0.6960`.
`checkpoint-100` contains `trainer_state.pt`, `model_meta.json`, `llm/`, and
`projector.pt`. Initial external direct-prompt seed-42 VQA reported overall
`7.333`, number `4.861`, other `6.888`, yes/no `9.038`, EOS `1.0`, max-hit
`0.0`, avg generated tokens `6.443`; however, a prediction dump showed the
evaluator was scoring duplicated decoded chat-role prefixes like
`assistant\n\nyes` as the answer `assistant`. After patching
`evaluation/vqa_eval.py` to strip a leading decoded `assistant` role prefix,
the corrected VQA file
`vqa_eval_v4_stage2a_semanticcal_bs4_stage1b248_ckpt100_directprompt_training_chat_postprocessfix.json`
scored overall `52.400`, number `37.037`, other `40.871`, yes/no `76.093`,
EOS `1.0`, max-hit `0.0`, avg generated tokens `6.443`. Corrected V1
direct-prompt/postprocess-fix floor is overall `21.100`, yes/no `28.960`;
corrected V3 direct-calibration checkpoint-100 is overall `9.400`, yes/no
`11.079`. Updated promotion guard passes V1 floor, incumbent, yes/no recovery,
and hygiene. Judgment: optimization/artifact green with yellow-history and
corrected VQA promotion PASS; keep the old `7.333` as scorer-bug history, not
as the current accuracy judgment. Seed-43 confirmation also held: overall
`51.367`, number `35.430`, other `40.370`, yes/no `72.043`, EOS `1.0`, max-hit
`0.0`.
