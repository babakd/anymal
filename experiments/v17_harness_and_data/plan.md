# V17 Harness and Training-Data Hardening Plan

Audience: execution agent with access to the AnyMAL working directory, Modal, W&B, prior checkpoints, and V8-V16 experiment artifacts.

---

# One-line objective

V17 is not a training campaign. It is the harness-and-data hardening pass that must complete before any new architecture or training campaign launches.

The audit of V8-V16 evaluation code, leakage audit, and training-data builders revealed multiple issues that make promotion-level statistical claims unreliable. Until these are fixed, no V17+ training move can be cleanly attributed.

The goal of V17 is to land the benchmark and data fixes, re-evaluate the existing V11 and V16 checkpoints under the corrected harness, and produce a clean baseline table that the next architecture/training campaign can use as ground truth.

No new training runs in V17 except those required to re-cache the V11 retention teacher under any newly-pinned dataset revisions.

---

# Non-negotiable operating rules

## 1. No architecture changes

Do not modify connector, vision encoder, or decoder configurations. Do not propose new substrates. The connector axis is closed (V12-V16); architecture work resumes only after V17 hardening completes.

## 2. No new training campaigns

Do not launch Phase 1/2/3-style training experiments. The only training-adjacent work allowed is rebuilding the V11 retention KL cache if a dataset revision is pinned.

## 3. Fix in place, do not invent new evaluators

Patch the existing files in `evaluation/checkpoint_eval/` and `evaluation/vqa_eval.py`. Do not start a parallel evaluation harness. Keep artifact schemas backward-compatible so prior V8-V16 artifacts can still be inspected.

## 4. Every fix has a verification test

Each item below specifies what artifact or unit test proves the fix works. Do not mark a fix complete without producing that artifact.

## 5. Re-evaluate before recommending V18

After all benchmark and data fixes land, re-evaluate V11 and the V16 best candidates (Phase 2A step-3000, Phase 2B step-2000/3000) under the corrected harness and produce a clean comparison table. This is the V17 deliverable.

---

# Part A: Benchmark fixes

## A1. Paired bootstrap utility

### Goal

Add a single shared paired-bootstrap utility that compares candidate-vs-baseline accuracy on identical examples and returns a 95% CI on the delta.

### Specification

```text
function: paired_bootstrap_ci(candidate_correct, baseline_correct, seed, n_resamples=10000, confidence=0.95)
inputs:
  candidate_correct: list[bool] of length N
  baseline_correct:  list[bool] of length N (paired by item index)
required: candidate_correct and baseline_correct must be aligned on question_id
output:
  observed_delta:  float (candidate_accuracy - baseline_accuracy)
  ci_low:          float
  ci_high:         float
  p_value_two_sided: float (proportion of resampled deltas with opposite sign)
```

### Where it goes

Create `evaluation/checkpoint_eval/paired_bootstrap.py`. Import it from gqa, chartqa, textvqa, pope, vqa evaluators.

### Implementation

Non-parametric bootstrap: resample N item indices with replacement, compute (candidate_acc - baseline_acc) on the resampled set, repeat n_resamples times, return the empirical 2.5/97.5 percentiles.

Default: `n_resamples=10000`, `seed=12345`.

### Replace the parametric GQA bootstrap

`evaluation/checkpoint_eval/gqa_checkpoint_eval.py:_bootstrap_mean_ci` currently uses `rng.binomial(total, p_hat, size=n_resamples)` which is parametric. Replace with a non-parametric bootstrap that resamples the actual `correct_values` list with replacement. Keep the function signature so callers don't break.

### Verification

```text
Write a unit test in tests/test_paired_bootstrap.py that:
  1. Constructs candidate_correct = baseline_correct (identical models).
     Asserts CI includes 0 and observed_delta == 0.
  2. Constructs candidate_correct strictly better than baseline_correct on every item.
     Asserts CI excludes 0 on the positive side.
  3. Constructs candidate vs baseline with known delta = 0.05 on n=1000 with paired structure.
     Asserts the 95% CI contains 0.05 in at least 95 of 100 seeded trials.
```

Also: rerun a V11 GQA n=1000 result through the new bootstrap and confirm the new bootstrap CI is wider than the old parametric one for identical inputs (because real binary outcomes have item-level structure the parametric version ignored).

## A2. Relaxed ChartQA accuracy

### Goal

Replace ChartQA exact-match-after-normalization with the standard ChartQA relaxed-accuracy metric.

### Specification

For a model prediction and a set of gold answers:

```text
def chartqa_relaxed_match(pred: str, gold: str) -> bool:
    # 1. Strip thinking/role/prefix tokens, lowercase, trim.
    # 2. Detect whether gold is numerical:
    #    - try float(gold.replace(',', '').rstrip('%'))
    #    - if successful, prediction is also parsed the same way
    # 3. If both parse as numbers:
    #    - relative_error = abs(pred_num - gold_num) / max(abs(gold_num), epsilon)
    #    - return relative_error <= 0.05
    # 4. If text answer:
    #    - apply current text normalization (lowercase, strip punctuation
    #      except decimal point in number-bearing text, strip articles)
    #    - return pred_text == gold_text
```

Special cases:
- Percent-marker handling: "23.5%" parses as the number 23.5.
- Comma thousands separators: "1,234" → 1234.
- Currency markers ("$", "€"): strip before parsing.
- Plain integers ("24") count as numerical and use 5% tolerance.

### Where it goes

Add `chartqa_relaxed_match` to `evaluation/checkpoint_eval/chartqa_checkpoint_eval.py`. Keep the existing exact-match path and report both metrics on every artifact:

```text
chartqa_relaxed_match: <new headline metric>
chartqa_exact_match:   <legacy metric, retained for cross-version comparability>
```

### Verification

```text
Write tests in tests/test_chartqa_relaxed.py that:
  1. ("24",   "23.5") → match (relative error 2.13%, under 5%)
  2. ("25",   "23.5") → no match (relative error 6.4%, over 5%)
  3. ("23.5%", "23.5") → match (percent stripped)
  4. ("$1,234", "1234") → match (currency and comma stripped)
  5. ("yes", "yes") → match (text path)
  6. ("red", "blue") → no match (text path)
  7. ("two", "2") → no match (current behavior; flag as known-limitation in docstring)
```

Then re-run V11 and V16 ChartQA val full on the new metric. Report both relaxed and exact in the artifact.

## A3. CIs on all evaluators

### Goal

Add Wilson and paired-bootstrap CIs to ChartQA, TextVQA, POPE, and VQA evaluators. Match the structure already present in GQA.

### Specification

Each `_compute_*_metrics` function returns:

```text
<metric>_accuracy
<metric>_correct
<metric>_total
<metric>_ci_confidence
<metric>_accuracy_ci95_binomial_low
<metric>_accuracy_ci95_binomial_high
<metric>_accuracy_ci95_bootstrap_low
<metric>_accuracy_ci95_bootstrap_high
<metric>_bootstrap_seed
<metric>_bootstrap_resamples
```

For ChartQA, compute CIs separately on `chartqa_relaxed_match` and `chartqa_exact_match`.

For TextVQA, compute CIs on both `textvqa_exact_match` and `textvqa_soft_accuracy`. Note that soft accuracy is a non-binary score (values in [0, 1] per item), so the binomial Wilson CI does not apply cleanly; use bootstrap only for soft.

### Where it goes

```text
evaluation/checkpoint_eval/chartqa_checkpoint_eval.py:_compute_chartqa_metrics
evaluation/checkpoint_eval/textvqa_checkpoint_eval.py:_compute_textvqa_metrics
evaluation/checkpoint_eval/pope_checkpoint_eval.py:_compute_pope_metrics
evaluation/vqa_eval.py: (in the metric-computation path used by checkpoint evaluators)
```

### Verification

Re-run V11 on each benchmark and verify the new CIs are present in the artifact JSON. Manual sanity check: at n=1000 with accuracy ~7%, the Wilson CI half-width should be roughly ±1.7 absolute points.

## A4. Dataset version pinning

### Goal

Pin HF dataset revisions so the search/confirm/eval slices are reproducible across time. A re-uploaded mirror must not silently change the eval slice.

### Specification

For each HF dataset reference, record and pin:

```text
dataset_id:       e.g., "Mineru/GQA"
revision:         git commit SHA from HuggingFace dataset repo
fingerprint:      sha256 over (dataset_id + revision + split + seed + offset + max_samples)
slice_artifact:   /checkpoints/v17_slices/<benchmark>_<split>_<fingerprint>.json
```

The `load_dataset` calls should be parameterized with `revision=<pinned SHA>`. If the HF API does not expose the revision through the cache file, materialize the slice into a static JSON once and load that artifact thereafter.

### Datasets to pin

```text
Mineru/GQA                  testdev_balanced
anhdang000/ChartQA-V2       train, val
lmms-lab/textvqa            train, validation
```

### Where it goes

Add a `_pinned_revision` lookup at the top of each evaluator and pretrain dataset builder that touches a HF dataset. Store pinned revisions in a single JSON file:

```text
configs/dataset_revisions.json
{
  "Mineru/GQA":               { "revision": "<sha>", "pinned_on": "<date>" },
  "anhdang000/ChartQA-V2":    { "revision": "<sha>", "pinned_on": "<date>" },
  "lmms-lab/textvqa":         { "revision": "<sha>", "pinned_on": "<date>" }
}
```

### Verification

```text
1. Record the current revision SHA of each dataset in configs/dataset_revisions.json.
2. Compute the slice fingerprint for V11's GQA search and confirm slices.
3. Materialize the slices to /checkpoints/v17_slices/.
4. Re-run V11 GQA n1000 search with the materialized slice and confirm accuracy
   reproduces the prior 44.9 within Wilson CI.
```

If the V11 historical number does not reproduce within CI, document the discrepancy explicitly. Either the mirror has drifted since V11 was evaluated, or the slice mechanics differ. Resolve before proceeding.

## A5. Per-bucket GQA reporting

### Goal

V11 search n=1000 has 65.5% yes/no questions. V11 confirm n=3000 has 60.8%. The slices are not directly comparable in raw accuracy. Report per-bucket scores so a candidate-vs-baseline comparison can be made on matched answer-type populations.

### Specification

`_compute_gqa_metrics` already produces per-bucket counts (`structural`, `semantic`, `detailed`). Add a `by_answer_type` block to every artifact:

```text
"gqa_by_answer_type": {
  "yes_no": {
    "accuracy": <float>,
    "correct": <int>,
    "total":   <int>,
    "ci95_low": <float>,
    "ci95_high": <float>
  },
  "number": { ... },
  "other":  { ... }
}
```

When reporting "V11 vs candidate" deltas in the final V17 report and any future campaign report, report per-bucket deltas alongside aggregate deltas.

### Verification

Re-run V11 on GQA search n=1000 and confirm n=3000. Confirm `gqa_by_answer_type` is populated and the bucket totals sum to the slice total.

## A6. VQAv2 normalization improvements

### Goal

Add digit-word equivalence and the official VQAv2 contraction/article handling so absolute VQAv2 numbers can be compared across published systems. Relative comparisons inside V8-V16 are not affected.

### Specification

In `evaluation/vqa_eval.py:_process_answer`:

```text
1. After lowercasing and stripping thinking/role/prefix tokens, apply the official
   VQAv2 contractions table (do not -> dont, can't -> cant, etc.) — the standard
   VQA eval applies this BEFORE punctuation stripping.
2. Apply digit ↔ word equivalence: "0" ↔ "zero", "1" ↔ "one", ..., "10" ↔ "ten".
   Normalize both gold and prediction to a canonical form (prefer digit-only).
3. Then apply existing punctuation-strip + article-removal.
```

A reference implementation lives in the official VQAv2 eval code; do not invent a new scheme.

### Verification

```text
Write tests in tests/test_vqa_normalization.py with:
  ("two cats", "2 cats") → match
  ("don't know", "dont know") → match
  ("yes.", "yes") → match
```

Then re-run V11 VQAv2 clean n=3000. Expected: absolute score increases slightly (typically 1-3 points) due to digit-word equivalence. Internal V8-V16 comparisons should not change qualitative ranking.

---

# Part B: Training-data fixes

## B1. POPE-style training image leakage audit and filter

### Goal

POPE eval uses COCO val2014. POPE-style training (`coco_pope_style_presence_train2017_*.json`, `coco_pope_style_absence_train2017_*.json`) uses COCO train2017, which contains ~35k val2014 images. Filter or rebuild to eliminate the overlap.

### Specification

```text
1. Compute the set of image IDs in:
   - all V11/V15/V16 POPE evaluation slices (random, popular, adversarial splits)
2. For each POPE-style training source:
   - For every image, parse the COCO image_id from the filename or the source record.
   - If the image_id is in the eval set, drop the row.
3. Rebuild the POPE-style training data with the leakage-clean subset:
   /checkpoints/llava_data/coco_pope_style_presence_train2017_leakclean_<N>.json
   /checkpoints/llava_data/coco_pope_style_absence_train2017_leakclean_<N>.json
4. Update train.py to reference the leak-clean files.
5. Audit the leak-clean files with audit_v9_leakage.py and confirm zero val2014 overlap.
```

### Estimated impact

About 30% of POPE-style training rows will be eliminated. The remainder is leakage-clean.

### Verification

```text
1. Run audit_v9_leakage.py with:
   --eval-artifacts <V11 POPE adversarial n1000 artifact>
   --train-sources  <new leak-clean presence + absence files>
2. Confirm: exact_val2014_overlap=0, numeric_id_overlap=0, raw_ref_overlap=0.
3. Print the count of dropped rows and remaining rows.
```

## B2. TextVQA training answer policy

### Goal

`scripts/modal/train.py:_textvqa_instruction_path` selects the first non-empty answer from the 10-annotator list. Switch to most-common-annotator for alignment with eval scoring.

### Specification

Replace the per-row answer selection in `_textvqa_instruction_path`:

```text
1. Take row.get("answers") as a list of 10 annotator strings.
2. Normalize each annotator answer (lowercase, strip whitespace, strip trailing punctuation).
3. Compute most_common via Counter; pick the highest-frequency normalized answer.
4. On tie, pick the lexicographically first to keep determinism.
5. Use the most_common answer as the training target.
```

Rebuild the cached training file under a new name so the change is auditable:

```text
old: /checkpoints/textvqa_data/v15_lmms-lab_textvqa_train_seed1502_n20000.json
new: /checkpoints/textvqa_data/v17_lmms-lab_textvqa_train_majority_seed1502_n20000.json
```

Update train.py V15 balanced and any future mixture configs to reference the v17 file.

### Verification

```text
1. Build the v17 file.
2. Inspect 50 random rows: print (first_annotator_answer, majority_answer) pairs.
3. Confirm at least 30% of rows show a different majority vs first-annotator answer
   (this is the expected disagreement rate on a noisy 10-annotator dataset).
```

## B3. Image-hash audit utility

### Goal

The current audit (`scripts/audit_v9_leakage.py`) checks image-ID overlap by parsing filenames. It does not detect cases where the same physical image appears under different IDs in different splits.

### Specification

Add a new audit utility:

```text
scripts/audit_image_hash_overlap.py

inputs:
  --train-sources <list of train data JSON paths>
  --eval-artifacts <list of eval result JSON paths>
  --image-dirs <map from source/eval to image directory>

logic:
  1. For each train source, list every image path referenced.
  2. For each eval artifact, list every image path referenced.
  3. Compute SHA256 of the first 1 MiB of each image file (saves time on large files
     while preserving uniqueness for JPEG/PNG headers + content).
  4. Cross-reference SHA256 sets.
  5. Report per-pair overlap counts and a sample of overlapping (train_path, eval_path) pairs.

outputs:
  JSON with per-pair overlap counts, total unique hashes seen, and overlap examples.
```

### Where it goes

```text
scripts/audit_image_hash_overlap.py    (Modal entrypoint, since image files live on the volume)
```

### Verification

Run on these train↔eval pairs:

```text
ChartQA train          vs ChartQA val
TextVQA train          vs TextVQA validation
GQA train_balanced     vs GQA testdev_balanced
COCO train2017         vs COCO val2014  (for POPE-style)
VQAv2 train2014        vs VQAv2 val2014
```

Expected results:

```text
ChartQA: 0 overlap (different chart images by construction)
TextVQA: 0 overlap (split is image-disjoint)
GQA:    0 overlap (testdev images are held out)
COCO 2017 vs 2014: substantial overlap (this is the issue B1 addresses)
VQAv2 train2014 vs val2014: 0 overlap (different splits of COCO 2014)
```

Any unexpected overlap is a hard finding. Document and resolve before promoting any V16+ result.

## B4. Reduce or replace "cannot determine" counterfactual data

### Goal

The current V15/V16 mixture uses 10% counterfactual data with target "cannot determine". This is documented to leak into clean VQA evaluation. Reduce the dose or replace with contrastive image-dependence loss.

### Specification

Two options, pick one. Default is Option A unless training infrastructure does not support contrastive pairs.

### Option A: contrastive image-dependence loss

Implement the V16 Phase 1C "counterfactual ablation" (skipped during V16 execution):

```text
For each clean VQA training example:
  positive: (correct_image, question, answer)
  negative: (wrong_image, same_question, same_answer)

Loss term:
  L_contrastive = max(0, margin - logp(answer | correct_image, question)
                              + logp(answer | wrong_image, question))

Apply with lambda=0.05, margin=0.5 initially. The negative is a wrong-image-same-answer-type
sample drawn from the existing _vqa_control_counterfactual_path pool.

Use this INSTEAD of teaching "cannot determine" as a target.
```

### Option B: reduce dose

If contrastive infrastructure is not ready, drop counterfactual weight in V15/V16-style mixtures from 10% to 2%:

```text
vqa_wrong_image_counterfactual_ce: weight 1.0 (was 5.0)
vqa_blank_image_counterfactual_ce: weight 1.0 (was 5.0)
```

This is a temporary measure. Option A is the correct long-term fix.

### Verification

```text
1. Train two short 500-step probes on V11 base with the current 10% "cannot determine"
   mixture and with the new (Option A or B) mixture.
2. After 500 steps, evaluate VQAv2 clean n=1000.
3. Inspect top-20 cleaned answers in each. Count occurrences of "cannot determine".
4. The new mixture should produce <1% of clean predictions as "cannot determine".
   The current mixture is expected to produce 3-10%.
```

This is a diagnostic probe, not a campaign launch. The probes are 500 steps each and exist only to verify the data fix.

## B5. Re-balance retention pressure in the training mixture

### Goal

The V15 balanced mixture has 70% non-retention data (capability + counterfactual) and 30% retention with V11 KL. This is why V16 Phase 1A retention crashed. Rebalance.

### Specification

Two options, pick one based on B4's outcome.

### Option I: raise retention share to 50%

Adjust V15 balanced weights:

```text
Retention pool (total 50, V11 KL = 1.0):
  vqa_replay_direct:     13
  gqa_replay_direct:     13
  coco_object_replay:     8
  short_llava_replay:     5
  pope_presence_replay:   5  (using B1-fixed leak-clean source)
  pope_absence_replay:    6  (using B1-fixed leak-clean source)

Capability pool (total 40, KL = 0.0):
  chartqa_capability:    17  (using B2-fixed if applicable)
  textvqa_capability:    13  (using B2-fixed source)
  gqa_spatial_capability: 10

Counterfactual pool (total 10, per B4 decision)
```

### Option II: small KL on capability data

Apply V11 KL with weight 0.05-0.1 on capability rows in addition to retention. This is a softer anchor that does not force the student to imitate V11 (V11 is bad at charts) but maintains a small pull toward V11's overall behavior:

```text
retention KL weight:   1.0
capability KL weight:  0.05
counterfactual KL weight: 0.0
```

### Verification

Train a 500-step probe on each option (and the current V15 mixture as control). Compare at step 500 on n=200:

```text
GQA n=200
VQAv2 clean n=200
ChartQA val n=200
TextVQA val n=200
drift MSE to V11 on the 0C probe set
```

Pick the option that:
- Preserves GQA within 1.0 of V11 at step 500
- Preserves VQAv2 clean within 1.5 of V11
- Allows ChartQA/TextVQA to move (drift > 0 on capability metrics)

Record the choice and the comparison table in the V17 report. The actual long training runs happen in V18, not V17.

## B6. GQA metadata-based focus filter

### Goal

`_gqa_focus_match` in `scripts/modal/train.py:4742` uses keyword matching to identify spatial/relation questions. GQA has `types.structural` and `types.semantic` metadata that classify questions precisely. Switch to metadata.

### Specification

```text
Replace _gqa_focus_match's spatial branch with:

def _gqa_focus_match(question, answer, focus, row=None):
    if focus in {"spatial", "spatial_relation", "relation", "left_right"}:
        if row is not None:
            types = row.get("types") or {}
            structural = (types.get("structural") or "").lower()
            semantic = (types.get("semantic") or "").lower()
            return semantic in {"rel", "relate"} or structural in {"verify", "logical"}
        # fall back to keyword filter if metadata is unavailable
        return <existing keyword check>
```

Update `_gqa_direct_answer_path` to pass the full row to `_gqa_focus_match` so the metadata path is reachable.

### Verification

```text
1. Rebuild GQA spatial training data using the metadata-based filter.
2. Compare:
   - keyword-filter dataset size
   - metadata-filter dataset size
3. Inspect 50 sampled rows from each. Confirm metadata-filter rows are
   meaningfully spatial/relation by manual eyeball.
4. Record both datasets so future campaigns can switch back if needed.
```

---

# Part C: Data acquisition

## C1. Required new sources

Acquire and audit the following sources. For each, produce:
- A `data_inventory_<name>.json` artifact with size, license, image format, train/val/test splits.
- A leakage audit using B3 (image-hash) against the current eval slices.
- A `data_path` + `image_dir` ready for the training mixture builder.

### Priority order

```text
1. OCR-VQA          (200k+ available; OCR capability)
2. DocVQA train     (document QA; capability)
3. AI2D train       (diagram QA; compositional + capability)
4. ChartQA augmented (HuggingFaceM4/ChartQA or vis-nlp/ChartQA; license check needed)
5. ShareGPT4V detailed captions (100k+; general visual instruction)
6. Visual Genome region/attribute/relationship (compositional grounding)
7. RefCOCO / RefCOCO+ / RefCOCOg (grounding)
```

### Per-source requirements

For each acquired source:

```text
1. Inventory file: source name, HF repo or URL, pinned revision SHA, license,
   row count, image count, train/val/test sizes.
2. Image-hash audit using B3 against all current eval slices.
3. License documentation: record license per source. If GPL-3.0 or other restrictive,
   flag for review before inclusion in any deployed-model training mix.
4. A converted training-instruction JSON in the same format as ChartQA/TextVQA training files.
5. An image cache directory on the Modal volume.
```

## C2. License-aware data tracking

### Goal

Add a license field to every training data source record. Make license restrictions visible at training-mixture-build time.

### Specification

Update train.py's data source dicts to include `license` and `commercial_use_allowed`:

```text
{
    "name": "chartqa_capability",
    "data_path": ...,
    "image_dir": ...,
    "license": "GPL-3.0",                  # from upstream vis-nlp/ChartQA
    "license_source": "https://huggingface.co/datasets/vis-nlp/ChartQA",
    "commercial_use_allowed": false,
    ...
}
```

When the training mixture is built, log a license summary so deployment decisions are informed.

### Verification

```text
1. For every source in v15_qwen_balanced_stage1b and v15_qwen_retention_replay_stage1b,
   add license metadata.
2. Log the aggregate license posture at mixture build time.
3. Confirm the license summary appears in W&B run config.
```

## C3. Verify ChartQA mirror provenance

### Goal

`anhdang000/ChartQA-V2` is a community mirror. Either pin its revision or switch to a more reputable source.

### Specification

```text
1. Compute SHA256 of the current ChartQA-V2 train and val split files in cache.
2. Compare row counts and question distributions against the published vis-nlp/ChartQA.
3. If they match, pin the revision and record the equivalence in configs/dataset_revisions.json.
4. If they do not match, switch to vis-nlp/ChartQA or HuggingFaceM4/ChartQA, rerun B1's
   leakage audit and B3's image-hash audit, and update all references.
```

### Verification

```text
1. Record the chosen ChartQA source, revision, and license in the inventory file.
2. Re-run V11 ChartQA val full eval against the chosen source.
3. Compare to the prior V11 ChartQA result (6%). Document any difference >2 points
   absolute as evidence of mirror divergence.
```

---

# Part D: Re-evaluation under the fixed harness

## D1. Reproduce V11 baselines under the fixed harness

After all Part A and B fixes are in, re-run V11 on:

```text
GQA search n1000          (with per-bucket reporting)
GQA confirm n3000         (with per-bucket reporting)
GQA paired-bootstrap delta vs self (sanity check, expected ~0)
ChartQA val full          (relaxed + exact, with CIs)
TextVQA val full          (exact + soft, with CIs)
POPE adversarial n1000    (with CIs)
POPE popular n1000        (with CIs)
VQAv2 clean n3000 seed42  (with improved normalization, with CIs)
VQAv2 blank n3000 seed42
VQAv2 shuffled n3000 seed42
VQAv2 wrong-image n3000 seed42
```

Record every value plus CI in a single V17 baseline table.

## D2. Re-evaluate the V16 candidates under the fixed harness

For each of these checkpoints, run the same eval suite as D1:

```text
V16 Phase 2A vision-only last-2 step-3000
V16 Phase 2B vision-only last-4 step-2000
V16 Phase 2B vision-only last-4 step-3000
```

For each candidate, compute paired-bootstrap deltas vs V11 on every matched slice. Report:

```text
candidate accuracy
V11 accuracy on same slice
observed delta
paired bootstrap 95% CI on delta
delta significance verdict (CI excludes 0?)
```

## D3. Decision table

Produce a final V17 decision table that classifies each V16 candidate as:

```text
clean_pareto_win       (candidate beats V11 with non-overlapping CI on at least one capability metric AND retains within CI on all retention metrics)
non_significant_change (all candidate-vs-V11 deltas have CI overlapping 0)
mixed_tradeoff         (some capability gains significant, some retention losses significant)
regression             (significant losses without significant gains)
```

If V16 produces any `clean_pareto_win`, that candidate is the V17 frontier and the seed for V18 work. If none does, V11 remains the frontier and V18 must propose something new.

---

# Part E: Final deliverables

The V17 final report must include:

```text
1. Per-fix completion status (A1-A6, B1-B6, C1-C3).
2. The verification artifact path for each fix.
3. The V17 baseline table (D1).
4. The V16 re-evaluation table (D2).
5. The decision table (D3).
6. A short list of any unexpected findings (e.g., V11 numbers that did not reproduce).
7. License summary across all training sources used in V11 and V16 training mixtures.
8. The acquired-data inventory (C1).
9. Recommendation for V18:
   - If clean_pareto_win exists: continue capability-led training around that candidate
     under the fixed harness, with newly acquired data.
   - If non_significant_change: V15/V16 results were noise; V18 should propose a
     materially different approach (e.g., larger-scale data, decoder LoRA, or vision encoder
     LoRA at higher rank than V16 tested).
   - If mixed_tradeoff: identify the binding constraint and propose targeted V18 work.
   - If regression: V11 remains frontier; V18 needs a new hypothesis.
```

## What V17 is not

V17 is not a training campaign. It produces:

```text
- Patched evaluation code in evaluation/
- Patched data builders in scripts/modal/train.py
- New audit utilities in scripts/
- Pinned dataset revisions in configs/dataset_revisions.json
- New acquired data sources on the Modal volume
- Re-evaluated baseline tables
- A clean V11 + V16 comparison under the fixed harness
```

It does not produce new training checkpoints (except possibly the V11 retention KL cache rebuilt under a pinned dataset revision, which is a cache regen, not a new model).

## What V17 unlocks

V18 (the next architecture/training campaign) can:

```text
- Use paired bootstrap to make statistically defensible claims.
- Use relaxed ChartQA to detect real chart-reading gains.
- Train on a leakage-clean POPE-style mixture.
- Train on a properly-balanced retention/capability mixture.
- Train on a substantially larger capability dataset (C1 acquisitions).
- Compare to a clean V11 baseline that was re-evaluated under the same harness.
```

---

# Hard rules

```text
1. No architecture changes.
2. No new training campaigns beyond the 500-step diagnostic probes in B4 and B5.
3. Every fix has a verification artifact.
4. The V17 baseline table is the new ground truth — supersede prior V11/V15/V16 absolute numbers.
5. License-restrictive sources are flagged but not silently included in deployed-model paths.
6. Image-hash overlap audits are run on every train↔eval pair, not just COCO-derived ones.
```

# One-sentence mandate

V17 fixes the eval harness, fixes the training data, re-evaluates V11 and V16 candidates under the fixed harness, and produces a clean baseline that V18 can build on — without launching any new training campaign.
