# V18 Mid-Training Data Acquisition Results

Last updated: 2026-05-16

## Objective

V18 is a data-acquisition and mixture-prep pass. It grows the Stage 1B-style
mid-training pool, audits the new images against the V17/V11 eval slices, wires
the `v18_qwen_midtraining_stage1b` mixture, and leaves all required artifacts on
the Modal training volume for the next training/eval round.

This was not a training campaign. The only training run launched was the required
1-step smoke test.

## Final Status

V18 is complete under the user's policy: use existing working authentication,
write data to the Modal training/cache volume, skip paid/GPL/NC/mixed or still
unverified sources, and allow missing HF license tags only when permissive
upstream terms are verified. The user approved Visual Genome direct download
from Stanford, but Visual Genome was not materialized in this cache set because
the completed pool already cleared the V18 readiness gates.

All final artifacts are on the `anymal-checkpoints` Modal volume under
`/checkpoints`.

| Artifact | Status |
| --- | --- |
| `/checkpoints/v18_qwen/auth_checklist.json` | written |
| `/checkpoints/v18_qwen/license_decisions.json` | written |
| `/checkpoints/v18_qwen/phase0_preflight.json` | written |
| `/checkpoints/v18_qwen/phase0_gate_status.json` | written |
| `/checkpoints/v18_qwen/source_manifest.json` | written |
| `/checkpoints/v18_qwen/final_mixture_weights.json` | written |
| `/checkpoints/v18_qwen/mixture_license_summary.json` | written |
| `/checkpoints/v18_qwen/audits/all_sources_vs_v17_v11_evals.json` | pass |
| `/checkpoints/v18_qwen/audits/gqa_leakclean_filter_summary.json` | written |
| `/checkpoints/v18_qwen/v18_v11_teacher_topk128_retention_300k.pt` | written |
| `/checkpoints/v18_qwen/smoke_1step/checkpoint-1` | written |

## Phase 0

Preflight passed before acquisition:

| Check | Result |
| --- | --- |
| Modal volume writable | pass |
| Modal volume free space gate | pass, `381.47 GiB` visible free |
| Required free space | `>= 250 GiB` |
| Modal `huggingface` secret present | pass |
| Modal `huggingface` secret valid | pass |

Policy decisions recorded in `/checkpoints/v18_qwen/license_decisions.json`:

| Source family | Decision | Reason |
| --- | --- | --- |
| ChartQA GPL-3.0 training data | excluded | GPL-3.0 is license-restrictive |
| ShareGPT4V CC-BY-NC sources | excluded | non-commercial restriction |
| LLaVA-Mix-665k NC-mixed optional subset | excluded | non-commercial/mixed restriction |
| LCS-558k / LLaVA-Pretrain | excluded | mixed/license-dependent posture |
| Missing HF license tags | allowed only when permissive upstream terms are verified | user-approved rule |
| Visual Genome direct download | approved | user approved Stanford CDN direct download; deferred from this cache set |

## Materialized Sources

The final manifest contains 10 sources and 546,588 rows. Every source was cached
on `anymal-checkpoints`.

| Source | Rows | Weight | KL | Data path | Image dir | License |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `coco_captions` | 81,479 | 30 | 1.0 | `/checkpoints/v18_data/coco_captions_train2017_seed1802_n100000.json` | `/checkpoints/coco_images` | `cc-by-4.0 / COCO terms` |
| `vqav2_train_broad` | 150,000 | 15 | 1.0 | `/checkpoints/v18_data/vqav2_train2014_direct_seed1802_n150000.json` | `/checkpoints/coco_train2014_vqa` | `cc-by-4.0 / VQAv2 and COCO terms` |
| `gqa_train_balanced_broad` | 199,638 | 28 | 1.0 | `/checkpoints/v18_data/gqa_train_balanced_broad_seed1803_n200000_leakclean_v17evals.json` | `/checkpoints/gqa_images_hf` | `cc-by-4.0 / GQA terms` |
| `aokvqa` | 17,000 | 4 | 0.0 | `/checkpoints/v18_data/aokvqa_train_seed1804_n17000.json` | `/checkpoints/aokvqa_images_hf` | `apache-2.0` |
| `okvqa` | 9,000 | 2 | 0.0 | `/checkpoints/v18_data/okvqa_train_seed1804_n9000.json` | `/checkpoints/okvqa_images_hf` | `cc-by-4.0 / OK-VQA and COCO terms` |
| `vsr` | 3,502 | 1 | 0.0 | `/checkpoints/v18_data/vsr_train_seed1805_n10000.json` | `/checkpoints/vsr_images_hf` | `cc-by-4.0` |
| `ocrvqa` | 50,000 | 9 | 0.0 | `/checkpoints/v18_data/ocrvqa_train_seed1807_n50000.json` | `/checkpoints/ocrvqa_images_hf` | `apache-2.0` |
| `textvqa_majority` | 20,000 | 5 | 0.0 | `/checkpoints/v18_data/textvqa_train_majority_seed1502_n20000.json` | `/checkpoints/textvqa_images_hf` | `cc-by-4.0 / TextVQA and OpenImages terms` |
| `ai2d` | 1,000 | 1 | 0.0 | `/checkpoints/v18_data/ai2d_train_seed1806_n5000.json` | `/checkpoints/ai2d_images_hf` | `apache-2.0` |
| `gqa_spatial_metadata` | 14,969 | 5 | 0.0 | `/checkpoints/v18_data/gqa_spatial_metadata_seed1503_n15000_leakclean_v17evals.json` | `/checkpoints/gqa_images_hf` | `cc-by-4.0 / GQA terms` |

`/checkpoints/v18_qwen/final_mixture_weights.json` records a weighted hash
mixture with `epoch_length=384000` and total weight `100.0`.

## License Summary

`/checkpoints/v18_qwen/mixture_license_summary.json` reports
`aggregate_commercial_use_allowed=true`.

License/source references recorded in the manifest:

| Source | License source |
| --- | --- |
| COCO | `https://cocodataset.org/#termsofuse` |
| VQAv2 | `https://visualqa.org/download.html` |
| GQA | `https://cs.stanford.edu/people/dorarad/gqa/about.html` |
| A-OKVQA | `https://github.com/allenai/aokvqa` |
| OK-VQA | `https://huggingface.co/datasets/HuggingFaceM4/OK-VQA/blob/main/OK-VQA.py` |
| VSR | `https://huggingface.co/datasets/cambridgeltl/vsr_zeroshot` |
| OCR-VQA | `https://huggingface.co/datasets/atc96/OCR-VQA-200K/blob/main/LICENCE.txt` |
| TextVQA | `https://textvqa.org/dataset/` |
| AI2D | `https://huggingface.co/datasets/LIME-DATA/ai2d` |

Excluded or deferred:

| Source family | Status |
| --- | --- |
| LCS-558k / LLaVA-Pretrain | excluded, mixed/license-dependent |
| ChartQA | excluded, GPL-3.0 |
| ShareGPT4V | excluded, CC-BY-NC |
| LLaVA-Mix-665k | excluded, NC-mixed |
| Visual Genome | approved by user, not materialized in this cache set |
| RefCOCO family | deferred, unverified HF license tags |
| VizWiz / NLVR2 | deferred, requested mirrors unavailable |
| DocVQA | deferred, train split unavailable in selected public HF mirror |

## Leakage Audits

All-source audit:

| Artifact | Result |
| --- | --- |
| `/checkpoints/v18_qwen/audits/all_sources_vs_v17_v11_evals.json` | `passed=true` |
| Train sources audited | 10 |
| Train/eval source pairs | 100 |
| Overlap pairs | 0 |

GQA leak-cleaning was performed before final integration:

| Source | Input rows | Output rows | Removed rows | Overlap hashes removed |
| --- | ---: | ---: | ---: | ---: |
| `gqa_train_balanced_broad` | 200,000 | 199,638 | 362 | 28 |
| `gqa_spatial_metadata` | 14,999 | 14,969 | 30 | 25 |

The OK-VQA and OCR-VQA audit records include numeric reference examples in
`missing_examples`; those are unresolved non-path refs from the audit resolver,
not missing training images. The final real-image filter kept all rows for every
source, and the all-source overlap audit passed.

## Wiring

Repository wiring is in place:

| File | Change |
| --- | --- |
| `scripts/modal/train.py` | added `v18_qwen_midtraining_stage1b` and `v18_qwen_retention_only_for_cache`; reads `/checkpoints/v18_qwen/source_manifest.json`; propagates license metadata; supports `--pretrain-output-dir` for smoke outputs |
| `data/instruction_dataset.py` | preserves source/license metadata and stable source-prefixed sample IDs in mixtures |
| `training/pretrain.py` | stores dataset license summary in pretrain config |
| `scripts/modal/v14_teacher_cache.py` | supports V18 retention-only cache building, skips `teacher_kl_weight=0` rows, and writes resumable partial checkpoints |
| `configs/dataset_revisions.json` | records V18 source/cache revision metadata |
| `scripts/audit_image_hash_overlap.py` | hash-audit utility restored for V18 source audits |
| `scripts/modal/v18_phase0.py` | Phase 0 checklist, license decision, and preflight entrypoint |
| `scripts/modal/v18_data_prep.py` | source acquisition, leak cleaning, manifest, license summary, and weights entrypoint |

## Teacher Cache

The expanded V18 retention teacher cache was built against the V11 Qwen
checkpoint:

| Field | Value |
| --- | --- |
| Output | `/checkpoints/v18_qwen/v18_v11_teacher_topk128_retention_300k.pt` |
| Entries | 215,680 |
| Answer tokens | 1,753,018 |
| Top-k | 128 |
| Processed batches | 12,000 |
| Skipped KL-disabled rows | 103,680 |
| Partial checkpoint leftover | none |

The cache intentionally stores unique sample IDs. The V18 epoch has 384,000
positions; KL-disabled sources are skipped, and repeated sample IDs collapse to
one cache entry.

## Smoke Test

The 1-step H100 Modal smoke completed successfully:

| Field | Value |
| --- | --- |
| Dataset | `v18_qwen_midtraining_stage1b` |
| Starting checkpoint | `/checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125` |
| Teacher cache | `/checkpoints/v18_qwen/v18_v11_teacher_topk128_retention_300k.pt` |
| Output checkpoint | `/checkpoints/v18_qwen/smoke_1step/checkpoint-1` |
| W&B run | `https://wandb.ai/babakdam/anymal-pretrain/runs/6zmf7d47` |
| Final reported train loss | `1.972456507384777` |

Smoke diagnostics confirmed all 10 V18 sources load with real images and the
weighted mixture exposes 384,000 samples.

## Local Checks

| Check | Result |
| --- | --- |
| `python3 -m compileall -q models training evaluation data scripts` | pass |
| `python3 -m py_compile scripts/modal/v14_teacher_cache.py` | pass |
| `python3 scripts/repo_health_check.py` | fails on pre-existing root files: `V15_QWEN3.md`, `V16_QWEN3.md`, `textvqa_checkpoint_eval.py` |

## V19 Launch Notes

Use `dataset=v18_qwen_midtraining_stage1b` for the next mid-training run. At
effective batch size 128, the configured V18 epoch length of 384,000 positions is
exactly 3,000 optimizer steps. A 3k-step run is one weighted epoch; a 6k-step run
is two weighted epochs.

The data work is complete. The next campaign can launch without further source
acquisition, audit, cache, or mixture wiring work.
