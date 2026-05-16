# V17 Harness and Training-Data Hardening

Date started: 2026-05-15

## Objective

V17 is the harness-and-data hardening pass described in
`V17_HARNESS_AND_DATA.md`. The goal is to fix benchmark semantics, harden data
sources against leakage and provenance drift, re-evaluate the existing V11 and
V16 checkpoints under the corrected harness, and leave a clean baseline for V18.

## Completion Audit

| ID | Requirement | Evidence | Status |
| --- | --- | --- | --- |
| A1 | Shared paired bootstrap and non-parametric GQA bootstrap | `evaluation/checkpoint_eval/paired_bootstrap.py`; GQA delegates to item-resampling bootstrap; `tests/test_paired_bootstrap.py`; `/checkpoints/v17_fixed_harness/v11/gqa_search_n1000.json`; `gqa_bootstrap_ci_compare_v11_search.json` | done |
| A2 | ChartQA relaxed accuracy with legacy exact retained | `chartqa_relaxed_match`; `tests/test_chartqa_relaxed.py`; V11/V16 ChartQA artifacts include `chartqa_relaxed_match` and `chartqa_exact_match` | done |
| A3 | CI fields on ChartQA, TextVQA, POPE, and VQAv2 | Corrected V11 artifacts under `/checkpoints/v17_fixed_harness/v11/` contain Wilson/bootstrap fields where applicable | done |
| A4 | HF dataset revision pinning and reproducible slices | `configs/dataset_revisions.json`; materialized GQA slices under `/checkpoints/v17_slices/`; V11 GQA search reproduced 44.9 exactly | done |
| A5 | GQA per-answer-type reporting | `gqa_by_answer_type` appears in GQA artifacts; V11 search yes/no 351 and other 649; confirm yes/no 1055 and other 1945 | done |
| A6 | VQAv2 normalization | `evaluation/vqa_eval.py`; `tests/test_vqa_normalization.py`; V11 clean n3000 rerun at 63.8667 | done |
| B1 | POPE-style COCO leak-clean data | `/checkpoints/llava_data/coco_pope_style_*_train2017_leakclean_10000.json`; `pope_leakclean_v9_audit.json` exact/numeric/raw overlaps all 0 | done |
| B2 | TextVQA majority-answer policy | `/checkpoints/textvqa_data/v17_lmms-lab_textvqa_train_majority_seed1502_n20000.json`; 20,000 rows; first-vs-majority disagreement 26.83% | done with finding |
| B3 | Image-hash audit utility and audits | `scripts/audit_image_hash_overlap.py`; all final leak-clean/training-ready audits show 0 overlaps and 0 missing refs | done |
| B4 | Reduce cannot-determine counterfactual pressure | Legacy 10%, Option I reduced dose, and Option II capability-KL probes evaluated; all have 0/1000 `cannot determine` on clean VQAv2 n1000 | done |
| B5 | Rebalance retention pressure and choose probe | Legacy, Option I, and Option II all fail retention thresholds; no V17 probe is promotable | done |
| B6 | GQA metadata spatial filter | Metadata-based GQA cache at `/checkpoints/gqa_data/v17_gqa_metadata_spatial_train_balanced_seed1503_n15000.json`; train.py uses it | done |
| C1 | Acquire new data-source inventories | `/checkpoints/v17_acquired/manifest.json`; `/checkpoints/v17_reports/data_artifacts_report.json`; acquired-source hash audit complete | done |
| C2 | License-aware source tracking | Source metadata flows through `InstructionMixtureDataset`, supervised-token filtering, `PretrainConfig`, and W&B config; W&B init smoke run `dgpl2p1j` verified non-null summary | done |
| C3 | ChartQA mirror provenance | `/checkpoints/v17_reports/chartqa_provenance.json`; ChartQA-V2 val is byte-identical to HuggingFaceM4 val; community train has +10,567 rows; `vis-nlp/ChartQA` inaccessible via HF API | done |
| D1 | Corrected V11 baseline table | V11 artifacts under `/checkpoints/v17_fixed_harness/v11/`; self-paired sanity outputs in `paired_self_v11/` | done |
| D2 | Corrected V16 re-evaluation | Phase 2A step3000, Phase 2B step2000, and Phase 2B step3000 complete with paired deltas | done |
| D3 | Final decision table | V16 candidates are mixed tradeoffs; all three probes fail retention gates; V11 remains frontier baseline | done |

## Dataset Revisions

| Dataset | Pinned revision |
| --- | --- |
| `Mineru/GQA` | `55fbe98d3474e07e0d1fe0078ba2d48c9ea7712e` |
| `anhdang000/ChartQA-V2` | `93b2f1f6bd69516c1be21faeefa05540768e0537` |
| `lmms-lab/textvqa` | `9c0699cd19768ac5ab97568f6b3cbac4c0062884` |

## D1 Corrected V11 Baseline

| Slice | Metric | Value | 95% CI / Notes |
| --- | --- | ---: | --- |
| GQA search n1000 | accuracy | 44.9000 | bootstrap 41.7000-48.0000; yes/no 230/351 = 65.5271; other 219/649 = 33.7442 |
| GQA confirm n3000 offset1000 | accuracy | 42.6000 | bootstrap 40.8333-44.4000; yes/no 641/1055 = 60.7583; other 637/1945 = 32.7506 |
| ChartQA val full | relaxed / exact | 14.6875 / 6.8229 | relaxed CI 13.1250-16.2500; exact CI 5.7279-7.9688 |
| TextVQA validation full | exact / soft | 30.9400 / 28.4000 | exact CI 29.6800-32.2400; soft CI 27.1933-29.6467 |
| POPE adversarial n1000 | accuracy | 80.3000 | Wilson 77.7209-82.6472; bootstrap 77.8000-82.8000 |
| POPE popular n1000 | accuracy | 81.8000 | Wilson 79.2883-84.0683; bootstrap 79.4000-84.2000 |
| VQAv2 clean n3000 seed42 | VQA accuracy | 63.8667 | bootstrap 62.2778-65.4889 |
| VQAv2 blank n3000 seed42 | VQA accuracy | 37.9222 | bootstrap 36.3000-39.5556; no `cannot determine` in top answers |
| VQAv2 shuffled n3000 seed42 | VQA accuracy | 36.1778 | bootstrap 34.5667-37.8114; no `cannot determine` in top answers |
| VQAv2 wrong-image n3000 seed42 | VQA accuracy | 35.5222 | bootstrap 33.9000-37.1889; no `cannot determine` in top answers |

Self-paired V11 comparisons pair every row, produce observed delta 0.0, and
have CI `[0.0, 0.0]` for every D1 slice.

## D2 V16 Paired Comparisons

| Candidate | Slice | Delta vs V11 | 95% paired CI | Significant |
| --- | --- | ---: | --- | --- |
| Phase 2A step3000 | GQA search n1000 | -1.2000 | -2.5000 to 0.1000 | no |
| Phase 2A step3000 | GQA confirm n3000 | -0.2333 | -0.9667 to 0.4675 | no |
| Phase 2A step3000 | ChartQA relaxed | +2.2917 | 1.4063 to 3.1771 | yes |
| Phase 2A step3000 | ChartQA exact | +1.0938 | 0.4167 to 1.7708 | yes |
| Phase 2A step3000 | TextVQA exact | +0.0400 | -0.5800 to 0.6800 | no |
| Phase 2A step3000 | TextVQA soft | +0.0000 | -0.5933 to 0.5933 | no |
| Phase 2A step3000 | POPE adversarial | -1.1000 | -2.4000 to 0.2000 | no |
| Phase 2A step3000 | POPE popular | -1.8000 | -2.9000 to -0.7000 | yes |
| Phase 2A step3000 | VQAv2 clean | -0.2222 | -0.8778 to 0.4222 | no |
| Phase 2A step3000 | VQAv2 blank | -0.2111 | -0.8000 to 0.3667 | no |
| Phase 2A step3000 | VQAv2 shuffled | -0.2556 | -0.6889 to 0.1889 | no |
| Phase 2A step3000 | VQAv2 wrong-image | +0.0556 | -0.3667 to 0.4778 | no |
| Phase 2B step2000 | GQA search n1000 | -1.2000 | -2.5000 to 0.1000 | no |
| Phase 2B step2000 | GQA confirm n3000 | -0.2333 | -0.9667 to 0.4667 | no |
| Phase 2B step2000 | ChartQA relaxed | +2.7604 | 1.8229 to 3.7500 | yes |
| Phase 2B step2000 | ChartQA exact | +1.1979 | 0.4688 to 1.9271 | yes |
| Phase 2B step2000 | TextVQA exact | +0.1600 | -0.5000 to 0.8200 | no |
| Phase 2B step2000 | TextVQA soft | +0.1067 | -0.5133 to 0.7267 | no |
| Phase 2B step2000 | POPE adversarial | -1.1000 | -2.4000 to 0.2000 | no |
| Phase 2B step2000 | POPE popular | -2.0000 | -3.1000 to -0.9000 | yes |
| Phase 2B step2000 | VQAv2 clean | +0.0111 | -0.6444 to 0.6778 | no |
| Phase 2B step2000 | VQAv2 blank | -0.1556 | -0.7222 to 0.4333 | no |
| Phase 2B step2000 | VQAv2 shuffled | -0.3222 | -0.7778 to 0.1333 | no |
| Phase 2B step2000 | VQAv2 wrong-image | +0.0444 | -0.4111 to 0.4889 | no |
| Phase 2B step3000 | GQA search n1000 | -1.2000 | -2.5000 to 0.1000 | no |
| Phase 2B step3000 | GQA confirm n3000 | -0.2333 | -0.9667 to 0.5000 | no |
| Phase 2B step3000 | ChartQA relaxed | +2.6042 | 1.5625 to 3.6458 | yes |
| Phase 2B step3000 | ChartQA exact | +0.9896 | 0.2604 to 1.7188 | yes |
| Phase 2B step3000 | TextVQA exact | +0.1800 | -0.5000 to 0.8800 | no |
| Phase 2B step3000 | TextVQA soft | +0.1867 | -0.4533 to 0.8333 | no |
| Phase 2B step3000 | POPE adversarial | -1.2000 | -2.6000 to 0.2000 | no |
| Phase 2B step3000 | POPE popular | -1.9000 | -3.0000 to -0.8000 | yes |
| Phase 2B step3000 | VQAv2 clean | +0.0333 | -0.6556 to 0.7111 | no |
| Phase 2B step3000 | VQAv2 blank | -0.1778 | -0.7667 to 0.4333 | no |
| Phase 2B step3000 | VQAv2 shuffled | -0.4111 | -0.8778 to 0.0447 | no |
| Phase 2B step3000 | VQAv2 wrong-image | +0.0556 | -0.4000 to 0.5222 | no |

## D3 V16 Decision

All corrected V16 candidates are mixed tradeoffs. They deliver statistically
significant ChartQA gains but also statistically significant POPE popular
regressions. GQA, TextVQA, and VQAv2 deltas overlap zero. V11 remains the
frontier baseline for V18 unless a targeted probe satisfies the retention gates.

## B4/B5 Probe Results

Retention gate: preserve GQA n200 within 1.0 point of V11 and VQAv2 clean n200
within 1.5 points of V11. V11 n200 baselines are GQA 46.0000, VQAv2 clean
65.8333, ChartQA relaxed/exact 13.0000/5.5000, and TextVQA exact/soft
30.0000/29.0000.

| Probe | GQA n200 | VQAv2 n200 | ChartQA n200 relaxed/exact | TextVQA n200 exact/soft | VQAv2 n1000 | Cannot-determine n1000 | Drift vs V11 | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Legacy 10% counterfactual | 42.0000 (-4.0000) | 61.6667 (-4.1666) | 17.0000 / 7.0000 | 36.5000 / 34.6667 | 62.5333, CI 59.7333-65.3333 | 0/1000 | MSE 4.2727e-6; agreement 0.8594 | fail |
| Option I retention 50/capability 40/counterfactual 2 | 41.5000 (-4.5000) | 62.3333 (-3.5000) | 15.5000 / 7.0000 | 37.0000 / 34.8333 | 62.8667, CI 60.0667-65.7000 | 0/1000 | MSE 3.5798e-6; agreement 0.8750 | fail |
| Option II capability KL 0.05 | 42.5000 (-3.5000) | 61.1667 (-4.6666) | 15.0000 / 6.5000 | 36.5000 / 35.0000 | 62.2333, CI 59.4667-65.0667 | 0/1000 | MSE 3.7765e-6; agreement 0.8750 | fail |

No probe satisfies the retention gate. Legacy, Option I, and Option II all
improve ChartQA/TextVQA n200 relative to V11, and all suppress the clean VQAv2
`cannot determine` answer to 0/1000, but each loses more than the allowed GQA
and VQAv2 n200 margin.

## Data Hardening Results

- POPE ID leakage: leak-clean files
  `/checkpoints/llava_data/coco_pope_style_presence_train2017_leakclean_10000.json`
  and `/checkpoints/llava_data/coco_pope_style_absence_train2017_leakclean_10000.json`
  have exact/numeric/raw overlap 0 against POPE eval IDs.
- POPE physical hashes: original POPE-style train vs adversarial eval had
  overlap 0, unexpectedly lower than the plan's expectation; the ID-based
  leak-clean sources are still retained.
- TextVQA majority: 20,000-row majority-answer cache built with 5,366 normalized
  first-vs-majority disagreements, rate 26.83%.
- GQA metadata spatial: metadata filter produced 14,999 rows versus 15,000 for
  keyword selection after scanning 36,902 rows; train.py now uses the metadata
  path.
- ChartQA old train leakage: old V15 ChartQA train cache overlapped val on 5
  unique image hashes; leak-clean replacement has 0 overlaps.
- ChartQA leak-clean: `/checkpoints/chartqa_data/v17_chartqa_train_leakclean_val_seed1501_n20000.json`
  wrote 20,000 rows and skipped 8 val-hash-overlap rows.
- Acquired training-ready sources vs corrected V11 evals: 70 pairwise audits,
  all overlap_count 0, all missing_refs 0. Total unique train hashes 10,724;
  total unique eval hashes 7,256.

## Hash Audit Inventory

| Audit | Train unique / checked / missing | Eval unique / checked / missing | Overlap |
| --- | --- | --- | ---: |
| POPE original vs adversarial | 10,000 + 10,000 / 20,000 / 0 | 455 / 4,000 / 0 | 0 |
| POPE leak-clean vs adversarial | 10,000 + 10,000 / 20,000 / 0 | 455 / 4,000 / 0 | 0 |
| TextVQA train vs validation | 15,773 / 20,000 / 0 | 3,166 / 20,000 / 0 | 0 |
| VQAv2 train2014 vs val2014 | 29,996 / 29,996 / 0 | 2,316 / 12,000 / 0 | 0 |
| Old ChartQA train vs val | 12,727 / 19,979 / 0 | 1,055 / 7,680 / 0 | 5 |
| ChartQA leak-clean vs val | 12,740 / 20,000 / 0 | 1,055 / 7,680 / 0 | 0 |
| GQA metadata train vs testdev | 12,683 / 29,998 / 0 | 349 / 4,000 / 0 | 0 |
| DocVQA inventory vs evals | 991 / 2,000 / 0 | ChartQA 1,055; TextVQA 3,166; GQA 349; VQA 2,316 | 0 |
| Acquired training-ready vs all evals | 10,724 total unique / 19,646 checked / 0 | 7,256 total unique / 100,680 checked / 0 | 0 |

## Acquired Sources

| Source | Rows | License | Commercial use | Training-ready note |
| --- | ---: | --- | --- | --- |
| OCR-VQA | 9,629 | unknown | unknown | training-ready inventory |
| DocVQA LMMS public | 2,000 | apache-2.0 | yes | validation/test inventory only |
| AI2D | 1,000 | apache-2.0 | yes | training-ready inventory |
| ChartQA augmented | 2,000 | gpl-3.0 | no | restrictive training-ready inventory |
| ShareGPT4V detailed | 1,018 | cc-by-nc-4.0 | no | restrictive training-ready inventory |
| Visual Genome regions | 0 | cc-by-4.0 | yes | blocked: HF script unsupported by current `datasets` |
| RefCOCO | 2,000 | unknown | unknown | training-ready inventory |
| RefCOCO+ | 1,999 | unknown | unknown | training-ready inventory |
| RefCOCOg | 2,000 | unknown | unknown | training-ready inventory |

## License Tracking

The V17 balanced mixture license summary is intentionally non-commercial in
aggregate because it includes LLaVA-Instruct mixed upstream and the ChartQA-V2
community mirror:

```json
{
  "aggregate_commercial_use_allowed": false,
  "license_counts": {
    "CC-BY-4.0 / COCO terms": 3,
    "CC-BY-4.0 / GQA terms": 2,
    "CC-BY-4.0 / TextVQA terms": 1,
    "CC-BY-4.0 / VQAv2 terms": 3,
    "GPL-3.0 / mirror provenance pending": 1,
    "LLaVA-Instruct mixed upstream": 1
  }
}
```

The Option II training run
`https://wandb.ai/babakdam/anymal-pretrain/runs/kidswrz0` was launched before the
final supervised-filter metadata preservation fix, so its W&B config retained
the old null value. The root cause was that `_filter_supervised_samples()`
returned a plain `Subset`, dropping the mixture metadata before `PretrainConfig`
was built. That metadata is now preserved through filtering, and
`PretrainConfig` carries it at W&B initialization. A W&B-only smoke run verified
the fixed initialization path with the full non-null summary:
`https://wandb.ai/babakdam/anymal-pretrain/runs/dgpl2p1j`.

## Unexpected Findings

- TextVQA majority disagreement was 26.83%, below the plan's heuristic
  expectation of at least 30%.
- Original POPE-style physical hash overlap with POPE adversarial eval was 0,
  not the expected substantial overlap. ID-based leak-cleaning remains useful
  because it encodes the intended split contract.
- Old ChartQA train data had real val image leakage: 5 unique image hashes. This
  was fixed with the leak-clean ChartQA train cache.
- `vis-nlp/ChartQA` was not public through the HF API; `anhdang000/ChartQA-V2`
  val is byte-identical to `HuggingFaceM4/ChartQA` val, while the community
  train split has 10,567 extra rows and GPL-3.0/provenance risk.
- Visual Genome region-caption acquisition is blocked under the current
  `datasets` runtime because the dataset script is unsupported.

## Local Verification

Passed on 2026-05-15 after the final Option II report fill:

```text
python3 -m compileall -q models training evaluation data scripts chartqa_checkpoint_eval.py
python3 -m pytest tests/test_paired_bootstrap.py tests/test_chartqa_relaxed.py tests/test_vqa_normalization.py -q
python3 scripts/repo_health_check.py
git diff --check
```

Result: `10 passed, 1 skipped`; repository health check passed; diff whitespace
check passed.

## Current V18 Recommendation

V16 should not be promoted over V11: its corrected-harness gains are real on
ChartQA but paired with POPE popular regressions. Legacy, Option I, and Option
II probes also fail the retention gate despite improved ChartQA/TextVQA and zero
`cannot determine` answers on VQAv2 n1000. V18 should target
retention-preserving capability learning rather than continuing the current
mixture shape.
