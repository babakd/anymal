## Plan: Dual-Architecture Training Modernization (AnyMAL v1 + AnyMALv2 Core)

### Summary
Implement `AnyMALv2` as a separate architecture while keeping existing `AnyMAL` behavior stable. Add a model factory + config switch so all training paths can run either architecture. Milestone scope is **V2 Core**: SigLIP2-based v2 encoder, MLP bottleneck projector, learned token compressor, and variable image-token plumbing. Keep current v1 pipelines working unchanged.

### Implementation Steps
1. Add architecture selection contract and factory.
- Update `/Users/babakd/anymal/configs/base.yaml` to add `model.architecture` with default `anymal_v1`.
- Add `/Users/babakd/anymal/models/factory.py` with a single entrypoint that instantiates `AnyMAL` or `AnyMALv2` from config.
- Update `/Users/babakd/anymal/models/__init__.py` exports to include `AnyMALv2` and the factory.
- Acceptance: existing configs run without edits and still build `AnyMAL` v1.

2. Implement v2 encoder/projector/compressor modules.
- Add `/Users/babakd/anymal/models/encoders/siglip2_encoder.py` (default model: SigLIP2 So400m).
- Update `/Users/babakd/anymal/models/encoders/__init__.py` with encoder registry exports.
- Add `/Users/babakd/anymal/models/projectors/mlp_bottleneck_projector.py`.
- Add `/Users/babakd/anymal/models/projectors/token_compressor.py` with learned pooling default.
- Update `/Users/babakd/anymal/models/projectors/__init__.py`.
- Acceptance: module-level unit tests confirm forward shapes and gradient flow.

3. Add `AnyMALv2` model class with variable-token splice support.
- Add `/Users/babakd/anymal/models/anymal_v2.py`.
- Keep `AnyMAL` in `/Users/babakd/anymal/models/anymal.py` behaviorally stable.
- In v2: encode images -> compress tokens -> project to LLM space -> splice by per-sample placeholder length.
- Implement strict checks for placeholder/token mismatches to prevent silent label/mask corruption.
- Acceptance: v2 forward/generate pass works for text-only and multimodal batches, including variable placeholder lengths.

4. Wire variable token counts through instruction data path.
- Update `/Users/babakd/anymal/data/instruction_dataset.py` to support configurable token count policy and per-sample placeholder lengths.
- Plumb `num_image_tokens` (and optional min/max range) from training config into dataset constructors.
- Update `/Users/babakd/anymal/scripts/train_finetune.py` and `/Users/babakd/anymal/modal_train.py` to pass these values explicitly.
- Acceptance: dataloader batches contain valid masks/labels for mixed placeholder lengths.

5. Enable architecture selection in all training entrypoints.
- Update `/Users/babakd/anymal/scripts/train_pretrain.py` and `/Users/babakd/anymal/scripts/train_finetune.py` to use the model factory.
- Update `/Users/babakd/anymal/modal_train.py` model creation paths (`run_pretrain`, `run_finetune`) to use the same factory/config fields.
- Ensure evaluation-time model construction in training flows remains architecture-consistent.
- Acceptance: both local and Modal runs can switch via config only.

6. Enforce strict checkpoint compatibility policy.
- Add model metadata file (e.g., `model_meta.json`) from both v1 and v2 `save_pretrained`.
- Update `/Users/babakd/anymal/training/trainer.py` checkpoint resume load path to validate architecture before loading projector/LoRA.
- Update `/Users/babakd/anymal/training/finetune.py` pretrain projector load path with same validation.
- Legacy behavior: checkpoints without metadata are treated as v1 and only allowed for v1 models.
- Acceptance: architecture mismatch fails fast with clear error.

7. Add v2 configs and defaults.
- Add `/Users/babakd/anymal/configs/pretrain_v2_alignment.yaml` with v2 defaults and max visual token cap `256`.
- Add `/Users/babakd/anymal/configs/finetune_v2.yaml` with v2 defaults and max visual token cap `384`.
- Keep existing v1 configs intact.
- Acceptance: new configs are runnable without manual key patching.

8. Update tests and smoke checks.
- Add/extend tests in `/Users/babakd/anymal/tests/test_model.py` and `/Users/babakd/anymal/tests/test_training.py`.
- Add cases for: factory routing, v2 module shapes, variable-token splice correctness, dataset placeholder variability, checkpoint compatibility gating.
- Add two short smoke runs (debug mode): one v1 config and one v2 config.
- Acceptance: tests pass and both smoke runs complete.

### Public API / Interface Changes
- New config key: `model.architecture` (`anymal_v1` | `anymal_v2`).
- New v2 model config keys: `model.vision_encoder_type`, `model.token_compressor_type`, `model.bottleneck_dim`, `model.max_image_tokens`, optional `model.min_image_tokens`.
- New exported model class: `AnyMALv2`.
- New model factory API in `/Users/babakd/anymal/models/factory.py`.
- Checkpoints now include model metadata used for strict architecture validation.

### Test Cases and Scenarios
1. `AnyMAL` v1 regression test: old configs still instantiate and train one step.
2. `AnyMALv2` init test: SigLIP2 + MLP bottleneck + learned compressor build and run forward.
3. Variable placeholder splice test: mixed sample token lengths in one batch produce correct `inputs_embeds`, `attention_mask`, and `labels`.
4. Placeholder mismatch test: invalid sample raises explicit error.
5. Dataset test: per-sample placeholder lengths follow configured policy and remain collatable.
6. Checkpoint compatibility test: v1 checkpoint into v2 model fails; v1->v1 and v2->v2 succeed.
7. Modal path smoke: `run_pretrain` and `run_finetune` select correct architecture from config.

### Assumptions and Defaults
- Milestone scope is **V2 Core** (not full dynamic-resolution/eval modernization).
- `AnyMAL` v1 remains default and behavior-preserving.
- v2 default encoder is **SigLIP2 So400m**.
- v2 default compressor is **learned pooling**.
- Token-cap defaults are **256 (Stage 1)** and **384 (Stage 2)**.
- Checkpoint policy is **strict + explicit** architecture compatibility.
