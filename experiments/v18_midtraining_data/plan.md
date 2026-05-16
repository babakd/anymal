# V18 Mid-Training Data Acquisition Plan

Audience: execution agent with access to the AnyMAL working directory, Modal, the AnyMAL checkpoint volume, and the V17 hardened harness/audit utilities.

---

# One-line objective

V18 is a **data-acquisition and mixture-prep pass**, not a training campaign.

The goal is to grow the training data pool from ~72k examples (the V15/V16/V17 mid-training mixture) to a properly-scoped **500k-800k unique examples** for Stage 1B-style mid-training, integrate the new sources into `scripts/modal/train.py` as a new mixture (`v18_qwen_midtraining_stage1b`), pass the V17 image-hash leakage audits on every new source, and leave the Modal volume + mixture wiring **ready for the next training run to launch** without further data work.

V18 does not launch any new training runs. It does not change architecture, evaluation, or the harness. It only acquires data, audits it, and wires it into the mixture.

---

# Why this exists

The V15/V16/V17 mid-training mixture has ~72k unique examples:

```
Retention pool   ~7,000   (V11 KL = 1.0)
Capability pool ~55,000   (V11 KL = 0.0)
Counterfactual ~10,000   (V11 KL = 0.0)
TOTAL          ~72,000
```

Stage 1B is mid-training, not SFT. Published open-VLM mid-training mixes range from LLaVA-1.5's 558k pretrain captions up to InternVL/Idefics2's millions to billions. **The campaign has been running mid-training-shaped runs with SFT-sized data**, which is the proximate cause of:

```
V16 Phase 1A retention collapse by step 250
V17 B4/B5 retention sweeps all failing the gate
The model overfitting the capability set in ~720 steps
```

V18 fixes the data scale before the next training campaign launches.

---

# Non-negotiable operating rules

## 1. Do not launch training runs

V18 produces data artifacts and a wired mixture. No `modal run modal_train.py` for the mid-training mixture itself. The only training-adjacent run allowed is rebuilding the V11 retention KL teacher cache against the expanded retention pool (Item E1 below).

## 2. Authentication checkpoint comes first

Before any acquisition starts, run Phase 0 and **halt for the user to confirm authentication**. Do not proceed past Phase 0 until the user has provided or confirmed:

```text
- HuggingFace token (for gated datasets, if any are gated)
- Confirmation that the agent may write to the anymal-checkpoints Modal volume
- License posture decisions (Tier 4 NC sources, GPL-3.0 sources)
- Visual Genome direct-download approval (Stanford CDN, ~30 GB)
```

This is a hard gate. List exactly what needs auth, present the license-decision options, and stop.

## 3. Every source passes the image-hash audit before integration

Use `scripts/audit_image_hash_overlap.py` (added in V17 B3) on every acquired source against every relevant eval slice before the source is added to the training mixture. A source fails closed if any image hash overlaps any eval image hash.

## 4. License-aware tracking on every source

Every source added to the mixture must include `license` and `commercial_use_allowed` fields per V17 C2 conventions. If a license is `unknown`, flag it explicitly and ask the user before integration.

## 5. Backward-compatible mixture

Add `v18_qwen_midtraining_stage1b` as a new dataset alongside the existing V15 mixtures in `scripts/modal/train.py`. Do not modify or delete existing V15/V16/V17 mixtures. The agent running V19 must be able to choose between V15-balanced and V18-midtraining.

---

# Target data scale and mixture composition

V18 builds a mid-training-shaped mixture, not an SFT mixture. The composition inverts the V15 priorities: broad alignment + broad VQA dominate; capability is the wedge.

```
Broad alignment / caption (50% of mixture, ~300,000 examples)
  LCS-558k (LLaVA pretrain captions)         200,000
  COCO Captions full                         100,000

Broad VQA / grounding (35% of mixture, ~280,000 examples)
  VQAv2 train                                 80,000
  GQA train_balanced (broad)                 100,000
  A-OKVQA + OK-VQA + VizWiz                   57,000
  VSR + NLVR2                                 60,000  (split: VSR ~10k, NLVR2 ~50k)

Capability (10% of mixture, ~80,000 examples)
  OCR-VQA                                     50,000
  TextVQA train (majority-answer)             20,000
  AI2D                                         5,000
  DocVQA train (if acquired)                  10,000

Compositional grounding (5% of mixture, ~40,000 examples)
  GQA spatial-relation (metadata-filter)      15,000
  RefCOCO / RefCOCO+ / RefCOCOg               25,000

TOTAL: ~700,000 unique examples
```

Optional Tier 4 (license-restrictive) additions only if the user approves:

```
ChartQA train (current 20k, GPL-3.0 mirror)
ShareGPT4V detailed captions (~100k, CC-BY-NC)
LLaVA-Mix-665k subset                          (NC-mixed upstream)
```

These are off by default. They go in only with explicit user sign-off.

---

# Phase 0: authentication and license-decision checkpoint

## 0A. Inventory what needs authentication

Before any acquisition work begins, produce:

```text
/checkpoints/v18_qwen/auth_checklist.json
```

It must list, per source:

```text
source name
HF dataset id or download URL
gated yes/no
estimated download size
authentication required: HF token / URL approval / none
license
commercial_use_allowed
```

Print the checklist to the run log and **stop**.

## 0B. License-decision questions

Present these to the user as explicit choices:

```text
Question 1: Should the V18 mixture include GPL-3.0 ChartQA training data?
  Option A: include (matches V15 behavior, blocks commercial deployment)
  Option B: exclude (V18 mixture has no chart data; ChartQA capability deferred)
  Option C: include only the leak-clean ChartQA train cache built in V17

Question 2: Should the V18 mixture include CC-BY-NC sources?
  Option A: include ShareGPT4V detailed captions (NC, broad alignment value)
  Option B: exclude (V18 stays commercial-friendly in aggregate)

Question 3: Direct download from Stanford for Visual Genome (~30 GB)?
  Option A: yes (HF script is blocked; this is the only path)
  Option B: defer Visual Genome to a future campaign
```

Wait for user response. Do not proceed past Phase 0 until all three questions are answered. Record the answers in:

```text
/checkpoints/v18_qwen/license_decisions.json
```

## 0C. Modal volume preflight

Confirm:

```text
- Modal volume `anymal-checkpoints` is writable from the agent
- Free space on the volume is at least 250 GB (the V18 mixture images may add ~150 GB)
- HF token (if needed) is set via `modal.Secret.from_name("huggingface")` and is valid
```

If free space is under 250 GB, **halt and ask the user** whether to expand the volume or trim old caches before proceeding.

---

# Phase 1: broad alignment / caption acquisition

## 1A. LCS-558k (LLaVA pretrain captions)

### Source

```text
HF dataset id:  liuhaotian/LLaVA-Pretrain (or equivalent mirror)
Original file:  blip_laion_cc_sbu_558k.json
License:        mixed; LLaVA license + CC-BY-4.0 for COCO/LAION components
                Commercial use: license_dependent — flag for V18 mixture
Image source:   LAION-CC-SBU subset (filtered)
```

### Action

```text
1. Check if /checkpoints/llava_data/blip_laion_cc_sbu_558k.json already exists.
2. If not, download to that path (it was used in V8 Stage 1A so it may already be cached).
3. Verify image cache at /checkpoints/llava_pretrain/images/ exists with at least
   90% of the 558k images present.
4. If images are missing, the HF mirror typically ships with images bundled;
   extract them.
```

### Sampling

```text
Use 200,000 of the 558,000 available rows, seeded.
Cache: /checkpoints/v18_data/lcs558k_seed1801_n200000.json
```

### Audit

Run image-hash audit:

```text
scripts/audit_image_hash_overlap.py \
  --train-sources /checkpoints/v18_data/lcs558k_seed1801_n200000.json \
  --eval-artifacts <all V17 corrected V11 baseline artifacts> \
  --output /checkpoints/v18_qwen/audits/lcs558k_vs_evals.json
```

Pass condition: 0 image-hash overlap with every eval slice.

## 1B. COCO Captions

### Source

```text
HF dataset id:  HuggingFaceM4/COCO  (or pin a specific commit)
                fallback: yerevann/coco-karpathy or nlphuji/coco_captions
License:        CC-BY-4.0 / COCO terms; commercial use allowed
Image source:   COCO train2017 (existing /checkpoints/coco_train2017/)
```

### Action

```text
1. Load COCO train2017 captions (5 captions per image, ~118k images = ~590k captions).
2. Pick 1 caption per image deterministically (longest, or seed-shuffled choice).
3. Build instruction-format JSON:
   {"id": ..., "image": ..., "conversations":
     [{"from": "human", "value": "<image>\nDescribe this image."},
      {"from": "gpt", "value": <caption>}]}
4. Sample 100,000 rows seeded.
```

### Sampling

```text
Cache: /checkpoints/v18_data/coco_captions_train2017_seed1802_n100000.json
```

### Audit

Verify zero image-hash overlap with VQAv2 val2014 (POPE eval uses val2014, COCO captions
uses train2017; these are disjoint by construction but verify).

---

# Phase 2: broad VQA / grounding acquisition

## 2A. VQAv2 train scaled to 80,000

### Source

```text
Already in repo as /checkpoints/vqa_data/vqa_train2014_direct_<N>.json
Currently using 2,000 of 440,000 available.
License: CC-BY-4.0 / VQAv2 terms; commercial use allowed
```

### Action

```text
Call _vqa_train_direct_answer_path(max_samples=80000) in scripts/modal/train.py
to materialize /checkpoints/vqa_data/vqa_train2014_direct_80000.json.
Existing builder logic already handles this — just call it with a larger max.
```

### Audit

`audit_v9_leakage.py` already covers val2014↔train2014. Re-run for the 80k file.

## 2B. GQA train_balanced broad subset to 100,000

### Source

```text
HF dataset id:  Mineru/GQA
Pinned revision: per configs/dataset_revisions.json
License:        CC-BY-4.0 / GQA terms
Available rows: ~944,000 in train_balanced
Currently used: 10k replay + 15k spatial-focus = 25k
```

### Action

```text
Call _gqa_direct_answer_path(split="train_balanced", max_samples=100000, focus="all")
to materialize /checkpoints/gqa_data/v9_qwen_gqa_train_balanced_100000.json.
```

### Audit

Image-hash audit against GQA testdev_balanced eval slice. Expected overlap: 0
(by GQA's image-disjoint split design).

## 2C. A-OKVQA + OK-VQA + VizWiz

### Sources

```text
A-OKVQA:
  HF dataset id:  HuggingFaceM4/A-OKVQA  (verify exact mirror; pin revision)
  License:        Apache-2.0; commercial use allowed
  Available:      ~17,000 train

OK-VQA:
  HF dataset id:  Multimodal-Fatima/OK-VQA_train  (verify mirror)
  License:        CC-BY-4.0; commercial use allowed
  Available:      ~9,000 train

VizWiz (VQA):
  HF dataset id:  lmms-lab/vizwiz_vqa  (verify mirror)
  License:        CC-BY-4.0; commercial use allowed; PII considerations
  Available:      ~31,000 train (image-based questions from blind users)
```

### Action

For each:

```text
1. Confirm the HF mirror exists and is public (no gating).
2. Pin the revision in configs/dataset_revisions.json.
3. Load the train split, build instruction-format JSON.
4. Cache images to /checkpoints/<source>_images_hf/.
```

### Sampling

```text
A-OKVQA: 17,000 (all train)
OK-VQA:   9,000 (all train)
VizWiz:  31,000 (all train)
TOTAL:   57,000 across these three
```

### Cache paths

```text
/checkpoints/v18_data/aokvqa_train_n17000.json
/checkpoints/v18_data/okvqa_train_n9000.json
/checkpoints/v18_data/vizwiz_train_n31000.json
```

### Audit

Image-hash audit against all corrected V11 eval slices. **Note**: VizWiz includes
images that may overlap with OpenImages-based eval data (rare but check). Document
any PII finding.

## 2D. VSR (Visual Spatial Reasoning) + NLVR2

### Sources

```text
VSR:
  HF dataset id:  cambridgeltl/vsr_zeroshot  (or similar; verify mirror)
  License:        CC-BY-4.0
  Available:      ~10,000 train + dev

NLVR2:
  HF dataset id:  lil-lab/nlvr2  (verify mirror; may need direct download)
  License:        CC-BY-4.0
  Available:      ~107,000 train; sample 50,000
```

### Action

```text
1. Verify HF mirrors (or fall back to direct download from official repos).
2. Build instruction-format JSON; both are caption/proposition-style.
3. NLVR2 has TWO images per question (it's "do these two images both show X?").
   Decide: handle as two-image prompts if the model supports it, OR drop NLVR2
   if multi-image is not yet implemented. If dropped, document and skip.
```

### Sampling

```text
VSR:    10,000 (all train)
NLVR2:  50,000 (sampled if mixture stays single-image; else drop)
```

### Audit

Standard image-hash audit. NLVR2 specifically: check if its images overlap any
COCO-based eval (they shouldn't — NLVR2 uses its own image corpus).

---

# Phase 3: capability acquisition

## 3A. OCR-VQA scaled to 50,000

### Source

```text
HF dataset id:  howard-hou/OCR-VQA  (already partially acquired in V17)
License:        CC-BY-4.0
V17 acquired:   9,629 rows (training-ready inventory)
Target:         50,000 rows
```

### Action

```text
1. Re-load the OCR-VQA train split.
2. Sample 50,000 deterministically (seed 1803).
3. Build instruction-format JSON.
4. Cache images to /checkpoints/ocrvqa_images_hf/.
```

### Cache path

```text
/checkpoints/v18_data/ocrvqa_train_seed1803_n50000.json
```

### Audit

Image-hash audit against TextVQA val (since OCR-VQA and TextVQA both reference
text-in-image; verify they don't share images). Against all other V11 eval slices.

## 3B. TextVQA train majority-answer to 20,000

### Source

```text
HF dataset id:  lmms-lab/textvqa
V17 built the majority-answer cache:
  /checkpoints/textvqa_data/v17_lmms-lab_textvqa_train_majority_seed1502_n20000.json
```

### Action

```text
Reuse the V17 cache directly. No new acquisition needed.
Verify the file exists and has 20,000 rows.
```

### Audit

V17 already audited; reconfirm the hash audit artifact path is valid.

## 3C. AI2D to 5,000

### Source

```text
HF dataset id:  lmms-lab/ai2d  (verify exact mirror; V17 used 1,000 of these)
License:        Apache-2.0
Available:      ~5,000 total
```

### Action

```text
1. Load the full AI2D train split (multiple-choice diagram QA).
2. Convert multiple-choice format to instruction format:
   "Question: <q>\nOptions: A. <a> B. <b> ...\nAnswer:" → "<correct letter>"
3. Cache images to /checkpoints/ai2d_images_hf/.
```

### Cache path

```text
/checkpoints/v18_data/ai2d_train_n5000.json
```

### Audit

Standard image-hash audit.

## 3D. DocVQA train (conditional on acquisition path)

### Source

```text
Primary:        HF dataset id  lmms-lab/DocVQA (verify what splits are available)
Alternative:    https://www.docvqa.org/ direct download
License:        Apache-2.0
Available:      ~40,000 train; sample 10,000
```

### Action

If the HF mirror exposes the train split:

```text
1. Load DocVQA train.
2. Build instruction-format JSON.
3. Cache images to /checkpoints/docvqa_images_hf/.
```

If not, halt and ask the user whether to:

```text
Option A: download from www.docvqa.org directly (requires registration; may need
          user credentials)
Option B: skip DocVQA in V18; defer to a future campaign
```

### Cache path

```text
/checkpoints/v18_data/docvqa_train_n10000.json
```

### Audit

Image-hash audit. DocVQA images are document scans, unlikely to overlap with COCO/VG.

---

# Phase 4: compositional grounding acquisition

## 4A. GQA spatial-relation (metadata-filtered)

### Source

```text
Already built by V17 B6:
/checkpoints/gqa_data/v17_gqa_metadata_spatial_train_balanced_seed1503_n15000.json
```

### Action

```text
Reuse directly. No new acquisition.
```

## 4B. RefCOCO / RefCOCO+ / RefCOCOg

### Source

```text
HF dataset ids (V17 partially acquired ~6,000 across all three):
  lmms-lab/RefCOCO         (verify)
  lmms-lab/RefCOCOplus     (verify)
  lmms-lab/RefCOCOg        (verify)
License:                   typically image-source-dependent — VERIFY before use
Available:                 ~390,000 expressions total
```

### Action

```text
1. Verify license per mirror. If unknown, halt and ask user.
2. Load train splits.
3. Build instruction-format JSON with referring-expression prompts:
   "Where is the <expression>?" → bounding box as text, OR
   "What is at <bbox>?" → object name
   Choose ONE format; document the choice.
4. Cache images (typically COCO 2014 train).
```

### Sampling

```text
RefCOCO:  8,000
RefCOCO+: 8,000
RefCOCOg: 9,000
TOTAL:   25,000
```

### Cache paths

```text
/checkpoints/v18_data/refcoco_train_seed1804_n8000.json
/checkpoints/v18_data/refcocoplus_train_seed1805_n8000.json
/checkpoints/v18_data/refcocog_train_seed1806_n9000.json
```

### Audit

RefCOCO uses COCO train2014 images. Audit against VQAv2 val2014 to confirm no overlap.

## 4C. Visual Genome (conditional on Phase 0 approval)

### Source

```text
Primary:        Stanford CDN direct download (~30 GB)
                https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/
                files:
                  region_descriptions.json.zip
                  attributes.json.zip
                  relationships.json.zip
                  image_data.json.zip
                  + image files from VG_100K and VG_100K_2 (already partially cached
                    for GQA — verify reuse)
License:        CC-BY-4.0
```

### Action

Only proceed if Phase 0 Question 3 was answered Option A.

```text
1. Download annotation JSONs to /checkpoints/visual_genome/.
2. Verify images are present (GQA already downloads from VG_100K/VG_100K_2 endpoints,
   so significant overlap may already be cached).
3. Build instruction-format JSON from region descriptions:
   per region: "Describe the region centered at (x, y) with size (w, h):" → <description>
   OR
   per region: "What is in this image?" with cropped region as the image → <description>
4. Build a separate JSON from attributes:
   "What attributes does the <object> have?" → <attribute list>
5. Build a separate JSON from relationships:
   "What is the relationship between <subject> and <object>?" → <relation>
```

### Sampling

```text
Total target across VG: 100,000 examples
  region descriptions:   50,000
  attributes:            25,000
  relationships:         25,000
```

### Cache paths

```text
/checkpoints/v18_data/vg_regions_seed1807_n50000.json
/checkpoints/v18_data/vg_attributes_seed1808_n25000.json
/checkpoints/v18_data/vg_relationships_seed1809_n25000.json
```

### Audit

VG image set overlaps heavily with GQA images (GQA is built on VG). Confirm that
VG training samples and GQA testdev_balanced eval do not share images — this is
the critical audit for VG.

---

# Phase 5: tier-4 optional sources

Run only if Phase 0 Questions 1 or 2 were answered Option A.

## 5A. ChartQA train (license-conditional)

If Question 1 was Option A or C:

```text
Reuse V17 leak-clean cache:
  /checkpoints/chartqa_data/v17_chartqa_train_leakclean_val_seed1501_n20000.json
```

If Question 1 was Option B: skip.

## 5B. ShareGPT4V detailed captions (license-conditional)

If Question 2 was Option A:

```text
HF dataset id:  Lin-Chen/ShareGPT4V
License:        CC-BY-NC-4.0
Sample 100,000 rows from the detailed-caption subset.
Cache: /checkpoints/v18_data/sharegpt4v_detailed_seed1810_n100000.json
```

If Question 2 was Option B: skip.

### Audit

ShareGPT4V uses a mix of images from LAION, SBU, COCO. Image-hash audit against
all V11 eval slices is mandatory.

---

# Phase 6: mixture integration in train.py

## 6A. Add `v18_qwen_midtraining_stage1b`

Add a new dataset block in `scripts/modal/train.py` modeled on the existing
`v15_qwen_balanced_stage1b` block, with:

```python
if dataset == "v18_qwen_midtraining_stage1b":
    # Builds the V18 mid-training mixture per the targets in V18_MIDTRAINING_DATA.md.
    # Weights are tuned for the mid-training role: 50% broad alignment,
    # 35% broad VQA, 10% capability, 5% compositional.
    ...
```

## 6B. Source weights

```text
Broad alignment (weight 50)
  lcs558k_caption:              30
  coco_captions:                20

Broad VQA / grounding (weight 35)
  vqav2_train_broad:             8
  gqa_train_balanced_broad:     10
  aokvqa:                        4
  okvqa:                         2
  vizwiz:                        4
  vsr:                           2
  nlvr2:                         5   (drop if multi-image not supported)

Capability (weight 10)
  ocrvqa:                        4
  textvqa_majority:              2
  ai2d:                          1
  docvqa:                        1   (drop if Phase 3D was skipped)
  chartqa_leakclean:             2   (only if Tier 4 Q1 = include)

Compositional grounding (weight 5)
  gqa_spatial_metadata:          2
  refcoco_combined:              2
  vg_regions_attrs_rels:         1   (only if Phase 4C approved)
```

If optional sources are skipped (Phase 0 decisions), redistribute weight proportionally
within the same family. Document the final weights in:

```text
/checkpoints/v18_qwen/final_mixture_weights.json
```

## 6C. Retention KL targeting

In V18 mid-training the **retention pool is now the broad alignment + broad VQA sections**,
not the small 7k V15 retention pool. Set `teacher_kl_weight=1.0` on:

```text
lcs558k_caption
coco_captions
vqav2_train_broad
gqa_train_balanced_broad
```

Set `teacher_kl_weight=0.0` on capability, compositional grounding, and optional sources.

Rationale: broad alignment/VQA is what V11 is good at; that's where V11 KL anchors
behavior. Capability data is where V11 is bad (ChartQA, OCR-VQA, TextVQA); applying
V11 KL there would suppress new learning.

## 6D. Cached V11 teacher distributions for the expanded retention pool

The V14/V15 teacher cache covered the small retention pool only. V18 needs a new cache
covering ~300k retention examples.

This is the **only training-adjacent run V18 is allowed to launch**:

```text
modal run scripts/modal/v14_teacher_cache.py \
  --output /checkpoints/v18_qwen/v18_v11_teacher_topk128_retention_300k.pt \
  --dataset v18_qwen_retention_only_for_cache \
  --teacher-checkpoint /checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125 \
  --top-k 128
```

Where `v18_qwen_retention_only_for_cache` is a new train.py dataset that loads
just the retention components of v18_qwen_midtraining_stage1b.

### Verification

```text
Cache file exists at /checkpoints/v18_qwen/v18_v11_teacher_topk128_retention_300k.pt
Cache entries ≥ 280,000 (allows for filtered samples)
top-k = 128
Loading the cache from train.py succeeds in a 1-step smoke
```

## 6E. Mixture epoch length and weighted-index mode

For ~700k unique examples and a planned 3k-6k step training budget:

```python
mixture_config = {
    "strategy": "weighted",
    "datasets": <source list above>,
    "epoch_length": 384000,
    "weighted_index_mode": "hash",
}
```

`epoch_length=384000` matches one 3k-step run at effective batch 128. The weighted-hash
mode ensures deterministic sampling regardless of source order.

---

# Phase 7: license-aware metadata flow

V17 C2 already wired `license` and `commercial_use_allowed` through the mixture config,
supervised-token filtering, and W&B init. V18 must:

```text
1. Tag every new source with license and commercial_use_allowed in scripts/modal/train.py.
2. Confirm the aggregate license summary appears in the W&B config when a smoke run
   loads the mixture.
3. Write the per-source license posture to:
   /checkpoints/v18_qwen/mixture_license_summary.json
4. If any restrictive source (NC, GPL-3.0) is included, the summary must flag the
   aggregate as commercial_use_allowed=false.
```

---

# Phase 8: end-to-end smoke

After all sources are acquired and the mixture is wired, run **one 1-step smoke** to
prove the pipeline:

```bash
modal run scripts/modal/train.py \
  --stage pretrain \
  --architecture anymal_v3 \
  --llm-backbone Qwen/Qwen3-8B \
  --gpu-type h100 \
  --pretrain-checkpoint /checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125 \
  --dataset v18_qwen_midtraining_stage1b \
  --run-name v18-midtraining-smoke-1step \
  --max-steps 1 \
  --learning-rate 2e-6 \
  --batch-size 4 \
  --pretrain-gradient-accumulation-steps 8 \
  --pretrain-image-tokens 128 \
  --v3-connector-type perceiver_resampler \
  --v3-connector-output-scale 1.125 \
  --pretrain-teacher-kl-weight 1.0 \
  --pretrain-teacher-kl-checkpoint /checkpoints/pretrain-output/v11-qwen3-c1-posscale000-scale1125/checkpoint-posscale000-scale1125 \
  --pretrain-teacher-kl-cache-path /checkpoints/v18_qwen/v18_v11_teacher_topk128_retention_300k.pt \
  --pretrain-teacher-kl-cache-top-k 128 \
  --pretrain-save-steps 1 \
  --use-wandb
```

### Smoke pass criteria

```text
no NaNs
no placeholder contract failures
checkpoint-1 saves and contains model_meta.json + projector.pt
W&B config shows the V18 license summary
training step completes in reasonable wall time (< 5 minutes)
generated answers on a tiny probe are non-empty and clean
```

If the smoke passes, V18 is **ready to hand off to the next training campaign**.

If the smoke fails, fix and re-run. Do not declare V18 complete with a failing smoke.

---

# Final deliverables

V18 produces:

```text
1. /checkpoints/v18_qwen/auth_checklist.json        (Phase 0A)
2. /checkpoints/v18_qwen/license_decisions.json     (Phase 0B)
3. /checkpoints/v18_data/*.json                     (per-source instruction caches)
4. /checkpoints/<source>_images_hf/                 (per-source image caches)
5. /checkpoints/v18_qwen/audits/*.json              (image-hash audit per source)
6. /checkpoints/v18_qwen/v18_v11_teacher_topk128_retention_300k.pt
7. /checkpoints/v18_qwen/final_mixture_weights.json (Phase 6B)
8. /checkpoints/v18_qwen/mixture_license_summary.json (Phase 7)
9. Updated scripts/modal/train.py with v18_qwen_midtraining_stage1b dataset block
10. configs/dataset_revisions.json with all new pinned revisions
11. /checkpoints/v18_qwen/smoke_1step/                (Phase 8)
12. experiments/v18_midtraining_data/results.md      (campaign report)
```

The report at item 12 must include:

```text
- Phase 0 authentication and license-decision outcomes
- Per-source acquisition status (acquired / skipped / blocked)
- Per-source row counts and image-hash audit verdicts
- Final mixture weights and license summary
- Smoke run W&B URL and pass verdict
- Recommendation for V19 training campaign:
    - What step budget is needed for ~1 epoch over the V18 mixture
    - What capability gains might be expected at this scale
    - Any sources that were blocked and should be re-tried in V20+
```

---

# What V18 does not do

```text
- No new training runs except the teacher-cache build (E1) and the 1-step smoke (Phase 8)
- No architecture changes
- No evaluation harness changes (V17 already hardened these)
- No new audit utilities (V17 added image-hash audit)
- No conclusions about model quality — V18 produces a ready mixture, not a trained model
```

---

# Hard rules summary

```text
1. Halt for user authentication and license decisions before any acquisition.
2. Image-hash audit every source before integration. Fail closed.
3. License-flag every source. Surface aggregate license posture clearly.
4. Backward-compatible: V18 mixture is added alongside V15 mixtures, not replacing.
5. Verify with a 1-step smoke before declaring V18 done.
6. No training campaigns. V18 is data + wiring, not model training.
```

# One-sentence mandate

V18 acquires, audits, and wires ~700k unique mid-training examples into a new `v18_qwen_midtraining_stage1b` mixture — paused at the authentication checkpoint until the user approves license posture and any direct downloads — and leaves the Modal volume and `train.py` ready for the next agent to launch a properly-scaled mid-training run without any further data work.
