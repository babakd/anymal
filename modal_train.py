"""
Modal Training Script for AnyMAL

Usage:
    1. Install Modal: pip install modal
    2. Setup Modal: modal setup
    3. Add HuggingFace secret: modal secret create huggingface HF_TOKEN=<your-token>
    4. Run training: modal run modal_train.py

Options:
    modal run modal_train.py --max-steps 100    # Quick test
    modal run modal_train.py --max-steps 1000   # Longer run
    modal run modal_train.py --stage pretrain   # Stage 1 pretraining
    modal run modal_train.py --stage pretrain --gpu-type h100  # Select H100
"""

import modal
import os
import random
import secrets
from datetime import datetime
from pathlib import Path

# Define the Modal app
app = modal.App("anymal-training")

# Create a volume to persist model weights between runs
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=True)

# Get the local project directory
PROJECT_DIR = Path(__file__).parent

# Define the container image with all dependencies + local code
# Layer order optimized: stable layers first (apt, pip), then local code mounted at runtime
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")  # Stable - rarely changes
    .pip_install(  # Stable - changes weekly at most
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.53.0,<5.0.0",
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "open_clip_torch>=2.23.0",
        "timm>=0.9.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "wandb>=0.16.0",
        "datasets>=2.15.0",
        "requests>=2.31.0",
        "sentencepiece>=0.1.99",
        "huggingface_hub>=0.19.0",
    )
    # Mount local code - changes frequently but won't invalidate pip cache
    .add_local_dir(PROJECT_DIR, remote_path="/root/anymal", copy=False)
)


# Logical GPU families selectable via --gpu-type.
GPU_TYPE_ALIASES = {
    "a100": "a100",
    "a100-80gb": "a100",
    "a100_80gb": "a100",
    "h100": "h100",
}

GPU_MODAL_RESOURCES = {
    "a100": {"single": "A100-80GB", "distributed": "A100-80GB:4"},
    "h100": {"single": "H100", "distributed": "H100:4"},
}

# Full V2/V3 runs are longer than the original smoke/canary budget. V3 Stage 1A
# is a 20k-step H100 run, so give it the same day-scale budget as Stage 2.
STAGE1_PRETRAIN_TIMEOUT_SECONDS = 24 * 60 * 60
STAGE2_TRAINER_TIMEOUT_SECONDS = 24 * 60 * 60

# V4 direct-to-LLM 4096-wide connector clipped persistently in Stage 1A. The
# active default is a bottlenecked spatial connector; explicit CLI flags can
# still reproduce the historical direct connector ablation.
V4_DEFAULT_CONNECTOR_LAYERS = 3
V4_DEFAULT_CONNECTOR_HEADS = 8
V4_DEFAULT_CONNECTOR_FF_MULT = 2
V4_DEFAULT_CONNECTOR_HIDDEN_DIM = 1024
V4_DEFAULT_CONNECTOR_OUTPUT_SCALE = 1.0
V4_DEFAULT_CONNECTOR_OUTPUT_GATE_INIT = 0.0001
V4_DEFAULT_CONNECTOR_TYPE = "spatial_perceiver_resampler"
V4_LEGACY_DIRECT_CONNECTOR_HIDDEN_DIM = 4096
V3_DEFAULT_CONNECTOR_TYPE = "perceiver_resampler"
CURRENT_LLAMA3_BACKBONE = "meta-llama/Meta-Llama-3-8B-Instruct"
QWEN3_8B_BACKBONE = "Qwen/Qwen3-8B"
MODAL_LLM_CACHE_DIRS = {
    CURRENT_LLAMA3_BACKBONE: "/checkpoints/llama3-8b-instruct",
    QWEN3_8B_BACKBONE: "/checkpoints/qwen3-8b",
}


def _normalize_gpu_type(gpu_type: str) -> str:
    """Normalize user-facing GPU type flag to known keys."""
    key = str(gpu_type).strip().lower()
    key = GPU_TYPE_ALIASES.get(key, key)
    if key not in GPU_MODAL_RESOURCES:
        supported = ", ".join(sorted(GPU_MODAL_RESOURCES.keys()))
        raise ValueError(f"Unsupported gpu_type '{gpu_type}'. Supported values: {supported}")
    return key


def _normalize_architecture_key(architecture: str) -> str:
    """Normalize Modal architecture flags to canonical model metadata names."""
    from model_metadata import normalize_architecture_name

    return normalize_architecture_name(architecture)


def _canonicalize_llm_backbone(value: str = None) -> str:
    raw = str(value or CURRENT_LLAMA3_BACKBONE).strip()
    lowered = raw.lower()
    base = os.path.basename(raw.rstrip("/")).lower()
    if "llama-3.1" in lowered or "llama3.1" in lowered:
        raise ValueError(
            "Llama 3.1 is intentionally excluded from this V8 execution. "
            "Use Qwen/Qwen3-8B or the current LLaMA-3-8B incumbent."
        )
    if lowered in {
        CURRENT_LLAMA3_BACKBONE.lower(),
        "meta-llama/meta-llama-3-8b-instruct",
        "llama3",
        "llama-3",
        "llama3-8b-instruct",
        "meta-llama-3-8b-instruct",
    } or base in {"llama3-8b-instruct", "meta-llama-3-8b-instruct"}:
        return CURRENT_LLAMA3_BACKBONE
    if lowered in {
        QWEN3_8B_BACKBONE.lower(),
        "qwen3",
        "qwen3-8b",
        "qwen/qwen3-8b",
    } or base in {"qwen3-8b", "qwen3_8b"}:
        return QWEN3_8B_BACKBONE
    return raw


def _metadata_llm_backbone(value: str = None) -> str:
    canonical = _canonicalize_llm_backbone(value)
    if os.path.isabs(canonical):
        return _canonicalize_llm_backbone(os.path.basename(canonical))
    return canonical


def _ensure_llm_backbone_cached(llm_backbone: str = None) -> str:
    canonical = _canonicalize_llm_backbone(llm_backbone)
    if os.path.isabs(canonical):
        if not os.path.exists(os.path.join(canonical, "config.json")):
            raise FileNotFoundError(f"Requested llm_backbone path has no config.json: {canonical}")
        return canonical
    if canonical not in MODAL_LLM_CACHE_DIRS:
        raise ValueError(
            f"Unsupported llm_backbone={llm_backbone!r}. "
            f"Supported values: {sorted(MODAL_LLM_CACHE_DIRS)}"
        )

    local_dir = MODAL_LLM_CACHE_DIRS[canonical]
    if not os.path.exists(os.path.join(local_dir, "config.json")):
        print(f"Downloading decoder backbone {canonical} to {local_dir}...")
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=canonical,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        volume.commit()
        print(f"Decoder backbone saved to {local_dir}")
    else:
        print(f"Using cached decoder backbone {canonical} from {local_dir}")
    return local_dir


def _resolve_v4_token_split(
    total_tokens=None,
    global_tokens=None,
    local_tokens=None,
):
    """Resolve V4 global/local visual-token split for Modal ablations."""
    if total_tokens is None and global_tokens is None and local_tokens is None:
        return 64, 64, 128
    if global_tokens is not None:
        global_tokens = int(global_tokens)
    if local_tokens is not None:
        local_tokens = int(local_tokens)
    if total_tokens is not None:
        total_tokens = int(total_tokens)

    if global_tokens is None and local_tokens is None:
        if total_tokens <= 0:
            raise ValueError(f"V4 total image tokens must be > 0, got {total_tokens}")
        if total_tokens % 2 != 0:
            raise ValueError(
                "V4 even split requires --pretrain-image-tokens to be even when "
                "global/local counts are not provided."
            )
        global_tokens = total_tokens // 2
        local_tokens = total_tokens // 2
    elif global_tokens is None:
        global_tokens = total_tokens - local_tokens
    elif local_tokens is None:
        local_tokens = total_tokens - global_tokens
    elif total_tokens is None:
        total_tokens = global_tokens + local_tokens

    if global_tokens < 0 or local_tokens < 0:
        raise ValueError(
            "V4 global/local image tokens must be >= 0, got "
            f"global={global_tokens}, local={local_tokens}."
        )
    if global_tokens + local_tokens <= 0:
        raise ValueError("V4 requires at least one image token.")
    if total_tokens != global_tokens + local_tokens:
        raise ValueError(
            "V4 token split mismatch: "
            f"total={total_tokens}, global={global_tokens}, local={local_tokens}."
        )
    return global_tokens, local_tokens, total_tokens


def _parse_v4_hidden_state_indices(value):
    """Parse Modal-friendly comma-separated V4 hidden-state indices."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value).strip()
    if not text:
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _resolve_modal_gpu(stage: str, gpu_type: str) -> str:
    """Resolve selected stage + gpu_type to a concrete Modal GPU resource string."""
    key = _normalize_gpu_type(gpu_type)
    mode = "distributed" if stage == "pretrain" else "single"
    return GPU_MODAL_RESOURCES[key][mode]


def _parse_checkpoint_step(name: str):
    """Parse step from checkpoint directory name like checkpoint-1234."""
    if not str(name).startswith("checkpoint-"):
        return None
    try:
        return int(str(name).split("-", 1)[1])
    except (IndexError, ValueError):
        return None


def _create_versioned_run_dir(base_dir: str, prefix: str = "run") -> str:
    """Create a monotonically numbered run directory under base_dir."""
    os.makedirs(base_dir, exist_ok=True)

    max_existing = 0
    for entry in os.listdir(base_dir):
        if not entry.startswith(f"{prefix}-"):
            continue
        suffix = entry.split("-", 1)[1]
        if suffix.isdigit():
            max_existing = max(max_existing, int(suffix))

    next_id = max_existing + 1
    while True:
        run_name = f"{prefix}-{next_id:04d}"
        run_dir = os.path.join(base_dir, run_name)
        try:
            os.mkdir(run_dir)
            return run_dir
        except FileExistsError:
            next_id += 1


def _generate_unique_run_name(prefix: str = "run") -> str:
    """
    Generate a collision-free run name: {prefix}-YYYYMMDD-HHMMSS-XXXX.

    Safe to call from multiple parallel processes — each invocation produces
    a globally unique name without coordination. Must be called in the local
    entrypoint (before any Modal .remote() call) so all containers share the
    same name and no container needs to read the shared volume to pick one.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rand = secrets.token_hex(2)  # 4 hex chars, 65k possibilities per second
    return f"{prefix}-{timestamp}-{rand}"


def _resolve_run_output_dir(
    base_dir: str,
    resume_checkpoint: str = None,
    prefix: str = "run",
    run_name: str = None,
) -> str:
    """
    Resolve output directory for a training run.

    - Resumed runs: write checkpoints into the run directory containing resume_checkpoint.
    - Explicit run_name: use {base_dir}/{run_name} directly (race-free — caller
      already generated a unique name, typically via _generate_unique_run_name()
      in the local entrypoint).
    - Fallback (run_name is None): create a numbered dir via _create_versioned_run_dir.
      NOTE: this fallback races across parallel containers on Modal Volumes because
      volumes are eventually consistent — os.mkdir appears atomic per container but
      multiple containers can each "succeed" on the same path and then clobber each
      other when the volume syncs. Prefer passing run_name.
    """
    if resume_checkpoint:
        ckpt_dir = os.path.abspath(str(resume_checkpoint))
        if os.path.basename(ckpt_dir).startswith("checkpoint-"):
            resumed_output_dir = os.path.dirname(ckpt_dir)
            os.makedirs(resumed_output_dir, exist_ok=True)
            return resumed_output_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        return ckpt_dir

    if run_name:
        run_dir = os.path.join(base_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    return _create_versioned_run_dir(base_dir, prefix=prefix)


def _collect_checkpoint_candidates(root_dir: str):
    """Collect checkpoint directories recursively under root_dir."""
    if not os.path.exists(root_dir):
        return []

    candidates = []
    for current_root, dirnames, _filenames in os.walk(root_dir):
        base = os.path.basename(current_root.rstrip("/"))
        step = _parse_checkpoint_step(base)
        if step is not None:
            projector_path = os.path.join(current_root, "projector.pt")
            if os.path.exists(projector_path):
                candidates.append((os.path.getmtime(current_root), step, current_root))
            # Do not recurse inside checkpoint dirs.
            dirnames[:] = []
    return candidates


def _latest_resumable_checkpoint(output_dir: str, max_steps: int = None):
    """Find the latest checkpoint in a run dir that has trainer state."""
    candidates = []
    for _mtime, step, path in _collect_checkpoint_candidates(output_dir):
        if max_steps is not None and step >= max_steps:
            continue
        if os.path.exists(os.path.join(path, "trainer_state.pt")):
            candidates.append((step, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _resolve_effective_resume_checkpoint(
    output_dir: str,
    resume_checkpoint: str = None,
    max_steps: int = None,
):
    """Use an explicit resume checkpoint, or auto-resume from the run dir."""
    if resume_checkpoint:
        return resume_checkpoint

    latest_checkpoint = _latest_resumable_checkpoint(output_dir, max_steps=max_steps)
    if latest_checkpoint:
        print(f"Auto-resuming from latest checkpoint in run dir: {latest_checkpoint}")
    return latest_checkpoint


def _invoke_modal_call(call, background: bool = False, **kwargs):
    """Run a Modal call now.

    Long-running jobs should be detached at the CLI level with
    ``modal run --detach ...``. Spawning from this local entrypoint can return
    before the app has a durable worker, which makes training appear launched
    even though no checkpoint is ever written.
    """
    if background:
        raise ValueError(
            "--background is disabled because Modal local-entrypoint spawn did "
            "not reliably keep training jobs alive. Use `modal run --detach "
            "modal_train.py ...` instead."
        )
    return call.remote(**kwargs)


def _checkpoint_save_interval(max_steps: int, max_interval: int = 250) -> int:
    """Pick a checkpoint cadence that also protects short canaries."""
    if max_steps <= 0:
        return max_interval
    return min(max_interval, max(50, max_steps // 4), max_steps)


def _checkpoint_matches_run_config(
    checkpoint_dir: str,
    expected_architecture: str,
    expected_llm_backbone: str = None,
    token_compressor_type: str = None,
    v3_connector_type: str = None,
    v4_connector_type: str = None,
    v4_global_image_tokens: int = None,
    v4_local_image_tokens: int = None,
    v4_connector_layers: int = None,
    v4_connector_heads: int = None,
    v4_connector_ff_mult: int = None,
    v4_connector_hidden_dim: int = None,
    v4_connector_output_scale: float = None,
    v4_connector_output_gate_init: float = None,
    v4_use_2d_position_features: bool = None,
    v4_deepstack_num_feature_levels: int = None,
    v4_deepstack_hidden_state_indices: str = None,
) -> bool:
    """Return whether a checkpoint is compatible with the requested run config."""
    from model_metadata import read_model_metadata, validate_checkpoint_architecture

    try:
        validate_checkpoint_architecture(
            checkpoint_dir=checkpoint_dir,
            expected_architecture=expected_architecture,
        )
    except RuntimeError:
        return False

    meta = read_model_metadata(checkpoint_dir) or {}
    if expected_llm_backbone:
        requested_backbone = _metadata_llm_backbone(expected_llm_backbone)
        checkpoint_backbone = meta.get("llm_backbone")
        if checkpoint_backbone != requested_backbone:
            print(
                f"Skipping checkpoint {checkpoint_dir}: llm_backbone="
                f"{checkpoint_backbone!r}, requested={requested_backbone!r}"
            )
            return False

    if expected_architecture == "anymal_v2" and token_compressor_type:
        checkpoint_compressor = meta.get("token_compressor_type")
        if checkpoint_compressor and checkpoint_compressor != token_compressor_type:
            print(
                f"Skipping checkpoint {checkpoint_dir}: token_compressor_type="
                f"{checkpoint_compressor!r}, requested={token_compressor_type!r}"
            )
            return False

    if expected_architecture == "anymal_v3" and v3_connector_type:
        checkpoint_connector = meta.get("connector_type")
        if checkpoint_connector and checkpoint_connector != v3_connector_type:
            print(
                f"Skipping checkpoint {checkpoint_dir}: connector_type="
                f"{checkpoint_connector!r}, requested={v3_connector_type!r}"
            )
            return False

    if expected_architecture == "anymal_v4":
        parsed_layers = _parse_v4_hidden_state_indices(v4_deepstack_hidden_state_indices)
        expected_values = {
            "connector_type": v4_connector_type,
            "num_global_image_tokens": v4_global_image_tokens,
            "num_local_image_tokens": v4_local_image_tokens,
            "connector_layers": v4_connector_layers,
            "connector_heads": v4_connector_heads,
            "connector_ff_mult": v4_connector_ff_mult,
            "connector_hidden_dim": v4_connector_hidden_dim,
            "connector_output_scale": v4_connector_output_scale,
            "connector_output_gate_init": v4_connector_output_gate_init,
            "use_2d_position_features": v4_use_2d_position_features,
            "deepstack_num_feature_levels": v4_deepstack_num_feature_levels,
            "deepstack_hidden_state_indices": parsed_layers,
            "vision_feature_layers": parsed_layers,
        }
        for key, expected in expected_values.items():
            if expected is None:
                continue
            checkpoint_value = meta.get(key)
            if checkpoint_value != expected:
                print(
                    f"Skipping checkpoint {checkpoint_dir}: {key}="
                    f"{checkpoint_value!r}, requested={expected!r}"
                )
                return False

    return True


def _auto_discover_pretrain_checkpoint(
    arch_key: str,
    llm_backbone: str = None,
    token_compressor_type: str = None,
    v3_connector_type: str = None,
    v4_connector_type: str = None,
    v4_global_image_tokens: int = None,
    v4_local_image_tokens: int = None,
    v4_connector_layers: int = None,
    v4_connector_heads: int = None,
    v4_connector_ff_mult: int = None,
    v4_connector_hidden_dim: int = None,
    v4_connector_output_scale: float = None,
    v4_connector_output_gate_init: float = None,
    v4_use_2d_position_features: bool = None,
    v4_deepstack_num_feature_levels: int = None,
    v4_deepstack_hidden_state_indices: str = None,
):
    pretrain_dir = "/checkpoints/pretrain-output"
    compatible_candidates = []
    for mtime, step, path in _collect_checkpoint_candidates(pretrain_dir):
        if _checkpoint_matches_run_config(
            checkpoint_dir=path,
            expected_architecture=arch_key,
            expected_llm_backbone=llm_backbone,
            token_compressor_type=token_compressor_type,
            v3_connector_type=v3_connector_type,
            v4_connector_type=v4_connector_type,
            v4_global_image_tokens=v4_global_image_tokens,
            v4_local_image_tokens=v4_local_image_tokens,
            v4_connector_layers=v4_connector_layers,
            v4_connector_heads=v4_connector_heads,
            v4_connector_ff_mult=v4_connector_ff_mult,
            v4_connector_hidden_dim=v4_connector_hidden_dim,
            v4_connector_output_scale=v4_connector_output_scale,
            v4_connector_output_gate_init=v4_connector_output_gate_init,
            v4_use_2d_position_features=v4_use_2d_position_features,
            v4_deepstack_num_feature_levels=v4_deepstack_num_feature_levels,
            v4_deepstack_hidden_state_indices=v4_deepstack_hidden_state_indices,
        ):
            compatible_candidates.append((mtime, step, path))
    if not compatible_candidates:
        return None
    compatible_candidates.sort(key=lambda x: (x[0], x[1]))
    _mtime, _step, checkpoint = compatible_candidates[-1]
    return checkpoint


@app.cls(
    image=image,
    gpu="A100-80GB",  # Use A100 80GB for large models
    timeout=STAGE2_TRAINER_TIMEOUT_SECONDS,
    volumes={"/checkpoints": volume},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
class Trainer:
    """
    AnyMAL Trainer class with lifecycle hooks for efficient Modal execution.

    Uses @modal.enter() to load model once per container, avoiding redundant
    model initialization on subsequent calls to the same warm container.
    """

    @modal.enter()
    def setup(self):
        """
        Called once when container starts. Downloads weights and initializes model.
        Subsequent calls to train() reuse the loaded model.
        """
        import sys
        sys.path.insert(0, "/root/anymal")

        print("=" * 60)
        print("Container startup - loading model (runs once per container)")
        print("=" * 60)

        # The requested decoder is resolved inside train(); V8 supports Qwen3
        # without an unconditional LLaMA download during container startup.
        self.llama_path = None

        # Download LLaVA dataset JSON if not cached
        self._ensure_llava_data_cached()

        # Download VQA evaluation data if not cached (non-fatal if it fails)
        try:
            self._ensure_vqa_data_cached()
        except Exception as e:
            print(f"Warning: VQA data download failed: {e}")
            print("Training will proceed without VQA evaluation.")

        # Pre-import heavy modules
        print("Pre-importing modules...")
        from models import create_model_from_config
        from data import build_dataloader, ImageTextCollator
        from training import FinetuneTrainer, PretrainTrainer

        # Store factory for later instantiation.
        self.create_model_from_config = create_model_from_config
        self._model_loaded = True
        print("Container setup complete!")

    def _ensure_llava_data_cached(self):
        """Download LLaVA dataset JSON files to volume if not already cached."""
        from huggingface_hub import hf_hub_download

        cache_dir = "/checkpoints/llava_data"
        os.makedirs(cache_dir, exist_ok=True)

        # LLaVA-Instruct-150K for Stage 2
        instruct_json = os.path.join(cache_dir, "llava_instruct_150k.json")
        if not os.path.exists(instruct_json):
            print("Downloading LLaVA-Instruct-150K JSON...")
            hf_hub_download(
                repo_id="liuhaotian/LLaVA-Instruct-150K",
                filename="llava_instruct_150k.json",
                repo_type="dataset",
                local_dir=cache_dir,
            )
            volume.commit()
            print(f"Saved to {instruct_json}")
        else:
            print(f"Using cached LLaVA-Instruct JSON from {instruct_json}")

        # LLaVA-1.5 Mix-665K for multi-task Stage 2
        mix665k_json = os.path.join(cache_dir, "llava_v1_5_mix665k.json")
        if not os.path.exists(mix665k_json):
            print("Downloading LLaVA-1.5 Mix-665K JSON...")
            try:
                hf_hub_download(
                    repo_id="liuhaotian/LLaVA-Instruct-150K",
                    filename="llava_v1_5_mix665k.json",
                    repo_type="dataset",
                    local_dir=cache_dir,
                )
                volume.commit()
                print(f"Saved to {mix665k_json}")
            except Exception as e:
                print(f"Warning: Could not download Mix-665K JSON: {e}")
                print("mix_665k dataset option will not be available.")
        else:
            print(f"Using cached LLaVA-1.5 Mix-665K JSON from {mix665k_json}")

        # LLaVA-Pretrain for Stage 1 (CC3M subset)
        pretrain_json = os.path.join(cache_dir, "blip_laion_cc_sbu_558k.json")
        if not os.path.exists(pretrain_json):
            print("Downloading LLaVA-Pretrain JSON...")
            try:
                hf_hub_download(
                    repo_id="liuhaotian/LLaVA-Pretrain",
                    filename="blip_laion_cc_sbu_558k.json",
                    repo_type="dataset",
                    local_dir=cache_dir,
                )
                volume.commit()
                print(f"Saved to {pretrain_json}")
            except Exception as e:
                print(f"Warning: Could not download pretrain JSON: {e}")
                print("Pretrain will use dummy data if real data requested.")
        else:
            print(f"Using cached LLaVA-Pretrain JSON from {pretrain_json}")

        # Download COCO images subset for real training
        self._ensure_coco_images_cached(instruct_json)

    def _ensure_coco_images_cached(self, json_path: str, num_images: int = 100000):
        """Download a subset of COCO images for training with real data."""
        import json
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed

        image_dir = "/checkpoints/coco_images"
        manifest_path = os.path.join(image_dir, "manifest.json")

        # Check if we already have enough images
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            existing_files = len([
                f for f in os.listdir(image_dir)
                if f.endswith((".jpg", ".jpeg", ".png"))
            ])
            required = min(num_images, manifest.get("total", num_images))
            if existing_files >= required:
                print(f"Using cached COCO images ({existing_files} images in {image_dir})")
                return

        os.makedirs(image_dir, exist_ok=True)
        print(f"Downloading {num_images} COCO images...")

        # Get image list from LLaVA JSON
        with open(json_path) as f:
            data = json.load(f)

        seen = set()
        images = []
        for sample in data:
            img = sample.get("image")
            if img and img not in seen:
                seen.add(img)
                images.append(img)
                if len(images) >= num_images:
                    break

        COCO_BASE_URL = "http://images.cocodataset.org/train2017"

        def download_one(img_name):
            path = os.path.join(image_dir, img_name)
            if os.path.exists(path):
                return (img_name, "skip")
            try:
                resp = requests.get(f"{COCO_BASE_URL}/{img_name}", timeout=30)
                resp.raise_for_status()
                with open(path, "wb") as f:
                    f.write(resp.content)
                return (img_name, "ok")
            except Exception as e:
                return (img_name, f"fail: {e}")

        # Download in parallel
        ok_count, skip_count, fail_count = 0, 0, 0
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(download_one, img) for img in images]
            for i, future in enumerate(as_completed(futures)):
                img_name, status = future.result()
                if status == "ok":
                    ok_count += 1
                elif status == "skip":
                    skip_count += 1
                else:
                    fail_count += 1
                if (i + 1) % 500 == 0:
                    print(f"  Progress: {i+1}/{len(images)} (ok={ok_count}, skip={skip_count}, fail={fail_count})")

        print(f"COCO images: downloaded={ok_count}, cached={skip_count}, failed={fail_count}")

        # Save manifest
        manifest = {
            "downloaded": ok_count,
            "skipped": skip_count,
            "failed": fail_count,
            "total": len(images),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        volume.commit()
        print(f"COCO images cached to {image_dir}")

    def _ensure_vqa_data_cached(self, num_val_images: int = 2000):
        """Download VQAv2 evaluation data and a subset of COCO val2014 images."""
        import json
        import zipfile
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed

        vqa_dir = "/checkpoints/vqa_data"
        val_image_dir = "/checkpoints/coco_val2014"
        os.makedirs(vqa_dir, exist_ok=True)
        os.makedirs(val_image_dir, exist_ok=True)

        # Download VQAv2 questions
        questions_path = os.path.join(vqa_dir, "v2_OpenEnded_mscoco_val2014_questions.json")
        if not os.path.exists(questions_path):
            print("Downloading VQAv2 val2014 questions...")
            url = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                zip_path = os.path.join(vqa_dir, "questions.zip")
                with open(zip_path, "wb") as f:
                    f.write(resp.content)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(vqa_dir)
                os.remove(zip_path)
                volume.commit()
                print(f"VQA questions saved to {vqa_dir}")
            except Exception as e:
                print(f"Warning: Could not download VQA questions: {e}")
        else:
            print(f"Using cached VQA questions from {vqa_dir}")

        # Download VQAv2 annotations
        annotations_path = os.path.join(vqa_dir, "v2_mscoco_val2014_annotations.json")
        if not os.path.exists(annotations_path):
            print("Downloading VQAv2 val2014 annotations...")
            url = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                zip_path = os.path.join(vqa_dir, "annotations.zip")
                with open(zip_path, "wb") as f:
                    f.write(resp.content)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(vqa_dir)
                os.remove(zip_path)
                volume.commit()
                print(f"VQA annotations saved to {vqa_dir}")
            except Exception as e:
                print(f"Warning: Could not download VQA annotations: {e}")
        else:
            print(f"Using cached VQA annotations from {vqa_dir}")

        # Download a subset of COCO val2014 images for VQA evaluation
        manifest_path = os.path.join(val_image_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            if manifest.get("downloaded", 0) + manifest.get("skipped", 0) >= num_val_images:
                print(f"Using cached COCO val2014 images ({manifest.get('downloaded', 0) + manifest.get('skipped', 0)} images)")
                return

        # Get image IDs from VQA questions to know which images to download
        if not os.path.exists(questions_path):
            print("Warning: VQA questions not available, skipping val image download")
            return

        with open(questions_path) as f:
            questions = json.load(f)

        # Get unique image IDs
        seen_ids = set()
        image_ids = []
        for q in questions["questions"]:
            img_id = q["image_id"]
            if img_id not in seen_ids:
                seen_ids.add(img_id)
                image_ids.append(img_id)
                if len(image_ids) >= num_val_images:
                    break

        COCO_VAL_URL = "http://images.cocodataset.org/val2014"

        def download_one(img_id):
            filename = f"COCO_val2014_{img_id:012d}.jpg"
            path = os.path.join(val_image_dir, filename)
            if os.path.exists(path):
                return (filename, "skip")
            try:
                resp = requests.get(f"{COCO_VAL_URL}/{filename}", timeout=30)
                resp.raise_for_status()
                with open(path, "wb") as f:
                    f.write(resp.content)
                return (filename, "ok")
            except Exception as e:
                return (filename, f"fail: {e}")

        print(f"Downloading {len(image_ids)} COCO val2014 images for VQA eval...")
        ok_count, skip_count, fail_count = 0, 0, 0
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(download_one, img_id) for img_id in image_ids]
            for i, future in enumerate(as_completed(futures)):
                _, status = future.result()
                if status == "ok":
                    ok_count += 1
                elif status == "skip":
                    skip_count += 1
                else:
                    fail_count += 1
                if (i + 1) % 100 == 0:
                    print(f"  Val images: {i+1}/{len(image_ids)} (ok={ok_count}, skip={skip_count}, fail={fail_count})")

        print(f"COCO val2014: downloaded={ok_count}, cached={skip_count}, failed={fail_count}")

        manifest = {"downloaded": ok_count, "skipped": skip_count, "failed": fail_count, "total": len(image_ids)}
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        volume.commit()

    @modal.method()
    def train(
        self,
        max_steps: int = 100,
        stage: str = "finetune",
        architecture: str = "anymal_v1",
        token_compressor_type: str = "learned",
        learning_rate: float = None,
        lora_learning_rate: float = None,
        batch_size: int = 4,
        use_wandb: bool = False,
        use_dummy_data: bool = False,
        wandb_api_key: str = None,
        track_per_layer_grad_norms: bool = True,
        run_eval_benchmarks: bool = False,
        pretrain_checkpoint: str = None,
        finetune_checkpoint: str = None,
        resume_checkpoint: str = None,
        dataset: str = "instruct_150k",
        run_name: str = None,
        pretrain_image_tokens: int = None,
        projector_warmup_steps: int = None,
        freeze_connector: bool = False,
        finetune_loss_scale: float = 1.0,
        finetune_gradient_accumulation_steps: int = 8,
        connector_warmup_steps: int = 0,
        pretrain_loss_scale: float = 1.0,
        pretrain_loss_normalization: str = "mean",
        pretrain_loss_normalization_target_tokens: float = 8.0,
        pretrain_gradient_accumulation_steps: int = 8,
        v3_connector_type: str = V3_DEFAULT_CONNECTOR_TYPE,
        v4_global_image_tokens: int = None,
        v4_local_image_tokens: int = None,
        v4_connector_layers: int = None,
        v4_connector_heads: int = None,
        v4_connector_ff_mult: int = None,
        v4_connector_hidden_dim: int = None,
        v4_connector_output_scale: float = None,
        v4_connector_output_gate_init: float = None,
        v4_use_2d_position_features: bool = True,
        v4_connector_type: str = None,
        v4_deepstack_num_feature_levels: int = None,
        v4_deepstack_hidden_state_indices: str = None,
        llm_backbone: str = CURRENT_LLAMA3_BACKBONE,
    ):
        """
        Run AnyMAL training on Modal.

        Args:
            max_steps: Number of training steps
            stage: "pretrain" for Stage 1, "finetune" for Stage 2
            token_compressor_type: V2 connector type ("learned", "perceiver", "perceiver2")
            learning_rate: Learning rate (default: 2e-5 for finetune, 2e-4 for pretrain)
            lora_learning_rate: Separate LR for LoRA params (default: 2e-4 for finetune)
            batch_size: Per-device batch size
            use_wandb: Enable Weights & Biases logging
            use_dummy_data: Use dummy data instead of real dataset (for testing)
            wandb_api_key: Weights & Biases API key (optional, pass directly)
            pretrain_checkpoint: Path to Stage 1 checkpoint for Stage 2 (auto-discovered if None)
            finetune_checkpoint: Full Stage 2 checkpoint to continue from in a new run
            resume_checkpoint: Path to checkpoint to resume training from
            dataset: Dataset to use for finetune ("instruct_150k" or "mix_665k")
            projector_warmup_steps: Zero connector gradients for this many Stage 2 steps
            freeze_connector: Freeze the multimodal connector during Stage 2 and train LoRA only
        """
        import sys
        sys.path.insert(0, "/root/anymal")

        # Setup W&B if requested
        if use_wandb:
            import wandb
            api_key = wandb_api_key or os.environ.get("WANDB_API_KEY")
            if api_key:
                wandb.login(key=api_key)
                print("Weights & Biases enabled!")
            else:
                print("WARNING: use_wandb=True but no WANDB_API_KEY found. Disabling W&B.")
                use_wandb = False

        from training.distributed import print_rank_0

        arch_key = _normalize_architecture_key(architecture)
        llm_path = _ensure_llm_backbone_cached(llm_backbone)
        resolved_llm_backbone = _metadata_llm_backbone(llm_backbone)
        self.llama_path = llm_path
        if arch_key in {"anymal_v3", "anymal_v4"} and dataset == "instruct_150k":
            dataset = "v3_grounded"
        if arch_key in {"anymal_v3", "anymal_v4"} and dataset in {
            "v3_direct_calibration",
            "v3_yesno_calibrated",
            "v4_direct_calibration",
            "v4_semantic_calibration",
            "v5_semantic_calibration",
            "v5_semantic_calibration_robust",
            "v7_semantic_calibration_counterfactual",
        }:
            freeze_connector = True
        if pretrain_image_tokens is None and (stage == "pretrain" or arch_key != "anymal_v4"):
            pretrain_image_tokens = 128 if arch_key in {"anymal_v3", "anymal_v4"} else 256
        v4_global_tokens = None
        v4_local_tokens = None
        if arch_key == "anymal_v4" and (
            stage == "pretrain"
            or pretrain_image_tokens is not None
            or v4_global_image_tokens is not None
            or v4_local_image_tokens is not None
        ):
            v4_global_tokens, v4_local_tokens, pretrain_image_tokens = _resolve_v4_token_split(
                total_tokens=pretrain_image_tokens,
                global_tokens=v4_global_image_tokens,
                local_tokens=v4_local_image_tokens,
            )

        print_rank_0("=" * 60)
        print_rank_0(f"AnyMAL Training on Modal")
        print_rank_0(f"Stage: {stage}")
        print_rank_0(f"LLM backbone: {resolved_llm_backbone} ({llm_path})")
        print_rank_0(f"Max steps: {max_steps}")
        print_rank_0(f"Batch size: {batch_size}")
        print_rank_0(f"Data: {'dummy' if use_dummy_data else 'LLaVA'}")
        print_rank_0("=" * 60)

        if stage == "finetune":
            lr = learning_rate or 2e-5
            lora_lr = lora_learning_rate or 2e-4
            run_finetune(
                llama_path=llm_path,
                llm_backbone=resolved_llm_backbone,
                architecture=arch_key,
                max_steps=max_steps,
                learning_rate=lr,
                lora_learning_rate=lora_lr,
                batch_size=batch_size,
                use_wandb=use_wandb,
                use_dummy_data=use_dummy_data,
                token_compressor_type=token_compressor_type,
                track_per_layer_grad_norms=track_per_layer_grad_norms,
                run_eval_benchmarks=run_eval_benchmarks,
                pretrain_checkpoint=pretrain_checkpoint,
                finetune_checkpoint=finetune_checkpoint,
                resume_checkpoint=resume_checkpoint,
                dataset=dataset,
                run_name=run_name,
                projector_warmup_steps=projector_warmup_steps,
                train_adapter=not freeze_connector,
                finetune_loss_scale=finetune_loss_scale,
                finetune_gradient_accumulation_steps=finetune_gradient_accumulation_steps,
                pretrain_image_tokens=pretrain_image_tokens,
                v3_connector_type=v3_connector_type,
                v4_global_image_tokens=v4_global_tokens,
                v4_local_image_tokens=v4_local_tokens,
                v4_connector_layers=v4_connector_layers,
                v4_connector_heads=v4_connector_heads,
                v4_connector_ff_mult=v4_connector_ff_mult,
                v4_connector_hidden_dim=v4_connector_hidden_dim,
                v4_connector_output_scale=v4_connector_output_scale,
                v4_connector_output_gate_init=v4_connector_output_gate_init,
                v4_use_2d_position_features=v4_use_2d_position_features,
                v4_connector_type=v4_connector_type,
                v4_deepstack_num_feature_levels=v4_deepstack_num_feature_levels,
                v4_deepstack_hidden_state_indices=v4_deepstack_hidden_state_indices,
            )
        else:
            lr = learning_rate or 2e-4
            effective_pretrain_image_tokens = (
                pretrain_image_tokens
                if pretrain_image_tokens is not None
                else (128 if arch_key in {"anymal_v3", "anymal_v4"} else 256)
            )
            pretrain_output_dir = _resolve_run_output_dir(
                base_dir="/checkpoints/pretrain-output",
                resume_checkpoint=resume_checkpoint,
                prefix="run",
                run_name=run_name,
            )
            run_pretrain(
                llama_path=llm_path,
                llm_backbone=resolved_llm_backbone,
                architecture=arch_key,
                max_steps=max_steps,
                learning_rate=lr,
                batch_size=batch_size,
                use_wandb=use_wandb,
                use_dummy_data=use_dummy_data,
                token_compressor_type=token_compressor_type,
                resume_checkpoint=resume_checkpoint,
                pretrain_checkpoint=pretrain_checkpoint,
                output_dir=pretrain_output_dir,
                pretrain_image_tokens=effective_pretrain_image_tokens,
                dataset=dataset,
                v3_connector_type=v3_connector_type,
                connector_warmup_steps=connector_warmup_steps,
                pretrain_loss_scale=pretrain_loss_scale,
                pretrain_loss_normalization=pretrain_loss_normalization,
                pretrain_loss_normalization_target_tokens=pretrain_loss_normalization_target_tokens,
                pretrain_gradient_accumulation_steps=pretrain_gradient_accumulation_steps,
                v4_global_image_tokens=v4_global_tokens,
                v4_local_image_tokens=v4_local_tokens,
                v4_connector_layers=v4_connector_layers,
                v4_connector_heads=v4_connector_heads,
                v4_connector_ff_mult=v4_connector_ff_mult,
                v4_connector_hidden_dim=v4_connector_hidden_dim,
                v4_connector_output_scale=v4_connector_output_scale,
                v4_connector_output_gate_init=v4_connector_output_gate_init,
                v4_use_2d_position_features=v4_use_2d_position_features,
                v4_connector_type=v4_connector_type,
                v4_deepstack_num_feature_levels=v4_deepstack_num_feature_levels,
                v4_deepstack_hidden_state_indices=v4_deepstack_hidden_state_indices,
            )

        # Save outputs to volume
        volume.commit()
        print("Training complete! Outputs saved to volume.")


def _diagnose_model(model):
    """Print diagnostic info about model configuration."""
    print("\n" + "=" * 60)
    print("MODEL DIAGNOSTICS")
    print("=" * 60)

    # Tokenizer / pad token
    tok = model.tokenizer
    print(f"  Tokenizer vocab size: {len(tok)}")
    print(f"  pad_token: {repr(tok.pad_token)} (id={tok.pad_token_id})")
    print(f"  eos_token: {repr(tok.eos_token)} (id={tok.eos_token_id})")
    print(f"  pad_token == eos_token? {tok.pad_token_id == tok.eos_token_id}")
    if tok.pad_token_id == tok.eos_token_id:
        print("  WARNING: pad_token equals eos_token - EOS labels will be masked!")

    # Image placeholder token
    placeholder_id = getattr(model, "image_placeholder_token_id", None)
    if placeholder_id is not None:
        placeholder_str = tok.decode([placeholder_id])
        print(f"  image_placeholder_token_id: {placeholder_id} ({repr(placeholder_str)})")
    else:
        print("  image_placeholder_token_id: None (will prepend image tokens)")

    # Parameter counts by component
    groups = {"projector": [0, 0], "token_compressor": [0, 0], "lora": [0, 0], "vision": [0, 0], "other": [0, 0]}
    for name, param in model.named_parameters():
        total = param.numel()
        trainable = total if param.requires_grad else 0
        if "projector" in name:
            groups["projector"][0] += total
            groups["projector"][1] += trainable
        elif "token_compressor" in name:
            groups["token_compressor"][0] += total
            groups["token_compressor"][1] += trainable
        elif "lora" in name.lower():
            groups["lora"][0] += total
            groups["lora"][1] += trainable
        elif "image_encoder" in name:
            groups["vision"][0] += total
            groups["vision"][1] += trainable
        else:
            groups["other"][0] += total
            groups["other"][1] += trainable

    print("\n  Parameter counts (total / trainable):")
    for g, (t, tr) in groups.items():
        print(f"    {g:12s}: {t:>12,} / {tr:>12,}")

    print("=" * 60 + "\n")


def _diagnose_dataset_sample(dataset, tokenizer, num_samples=3):
    """Sample a few items from the dataset and log diagnostics."""
    import torch
    print("\n" + "=" * 60)
    print("DATASET DIAGNOSTICS")
    print("=" * 60)

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        img = sample["image"]
        ids = sample["input_ids"]
        mask = sample["attention_mask"]
        labels = sample["labels"]

        total_tokens = mask.sum().item()
        supervised = (labels != -100).sum().item()
        pad_count = (ids == tokenizer.pad_token_id).sum().item()

        # Check if any image placeholder tokens exist
        placeholder_id = getattr(dataset, "image_placeholder_token_id", None)
        if placeholder_id is None:
            # try the tokenizer vocab
            vocab = tokenizer.get_vocab()
            for c in ["<|reserved_special_token_0|>", "<|image|>", "<image>"]:
                if c in vocab:
                    placeholder_id = vocab[c]
                    break
        has_placeholder = (ids == placeholder_id).sum().item() if placeholder_id else 0

        print(f"\n  Sample {i}:")
        print(f"    image shape: {list(img.shape)}, range: [{img.min():.2f}, {img.max():.2f}]")
        print(f"    input_ids shape: {list(ids.shape)}, non-pad: {total_tokens}, pad: {pad_count}")
        print(f"    labels: {supervised} supervised / {total_tokens} non-pad tokens ({100*supervised/max(total_tokens,1):.1f}%)")
        print(f"    image placeholder tokens: {has_placeholder}")

        # Decode a small window of supervised tokens to sanity-check
        supervised_mask = labels != -100
        if supervised_mask.any():
            first_sup_idx = supervised_mask.nonzero(as_tuple=True)[0][0].item()
            supervised_ids = labels[supervised_mask][:20]
            raw_window = ids[first_sup_idx:first_sup_idx+20]
            decoded_supervised = tokenizer.decode(supervised_ids, skip_special_tokens=False)
            decoded_window = tokenizer.decode(raw_window, skip_special_tokens=False)
            print(f"    first supervised-only tokens: {repr(decoded_supervised[:100])}")
            print(f"    raw window from first supervised token: {repr(decoded_window[:100])}")

        # Check that eos_token is NOT masked (should be supervised if at end of response)
        eos_id = tokenizer.eos_token_id
        vocab = tokenizer.get_vocab()
        for stop_token in ("<|eot_id|>", "<|im_end|>"):
            stop_id = vocab.get(stop_token, None)
            if stop_id is not None:
                stop_positions = (ids == stop_id).nonzero(as_tuple=True)[0]
                if len(stop_positions) > 0:
                    stop_supervised = sum(1 for pos in stop_positions if labels[pos] != -100)
                    print(
                        f"    {stop_token} tokens: {len(stop_positions)} total, "
                        f"{stop_supervised} supervised"
                    )

    print("=" * 60 + "\n")


def _diagnose_batch(batch, tokenizer, step_name="first batch"):
    """Log diagnostics for a collated batch."""
    print(f"\n--- Batch diagnostics ({step_name}) ---")
    for key, val in batch.items():
        if hasattr(val, "shape"):
            print(f"  {key}: shape={list(val.shape)}, dtype={val.dtype}, device={val.device}")
    if "labels" in batch:
        labels = batch["labels"]
        supervised = (labels != -100).sum().item()
        total = labels.numel()
        print(f"  labels: {supervised} supervised / {total} total ({100*supervised/max(total,1):.1f}%)")
    if "images" in batch:
        imgs = batch["images"]
        print(f"  images range: [{imgs.min():.2f}, {imgs.max():.2f}]")
    print("---\n")


def _load_full_finetune_checkpoint(model, checkpoint_path: str):
    """Load projector, compressor, and LoRA adapter from a full Stage 2 checkpoint."""
    import torch
    from peft import PeftModel
    from model_metadata import validate_checkpoint_metadata_values

    expected_arch = getattr(model, "architecture", "anymal_v1")
    expected_values = {}
    if expected_arch == "anymal_v2":
        expected_values = {
            "vision_encoder_type": getattr(model, "vision_encoder_type", None),
            "token_compressor_type": getattr(model, "token_compressor_type", None),
            "max_image_tokens": getattr(model, "max_image_tokens", None),
            "min_image_tokens": getattr(model, "min_image_tokens", None),
        }
    elif expected_arch in {"anymal_v3", "anymal_v4"}:
        expected_values = {
            "vision_encoder_type": getattr(model, "vision_encoder_type", None),
            "connector_type": getattr(model, "connector_type", None),
            "num_image_tokens": getattr(model, "num_image_tokens", None),
            "connector_layers": getattr(model, "connector_layers", None),
            "connector_heads": getattr(model, "connector_heads", None),
            "connector_ff_mult": getattr(model, "connector_ff_mult", None),
        }
        model_backbone = getattr(model, "llm_backbone", None)
        if model_backbone and model_backbone != CURRENT_LLAMA3_BACKBONE:
            expected_values["llm_backbone"] = model_backbone
        if expected_arch == "anymal_v4":
            expected_values.update(
                {
                    "num_global_image_tokens": getattr(model, "num_global_image_tokens", None),
                    "num_local_image_tokens": getattr(model, "num_local_image_tokens", None),
                    "use_2d_position_features": getattr(model, "use_2d_position_features", None),
                }
            )
            if getattr(model, "connector_type", None) == "deepstack_spatial_perceiver_resampler":
                expected_values.update(
                    {
                        "deepstack_num_feature_levels": getattr(
                            model,
                            "deepstack_num_feature_levels",
                            None,
                        ),
                        "deepstack_hidden_state_indices": list(
                            getattr(model, "deepstack_hidden_state_indices", [])
                        ),
                        "vision_feature_layers": list(
                            getattr(model, "deepstack_hidden_state_indices", [])
                        ),
                    }
                )
    validate_checkpoint_metadata_values(
        checkpoint_dir=checkpoint_path,
        expected_architecture=expected_arch,
        expected_values=expected_values,
    )

    projector_path = os.path.join(checkpoint_path, "projector.pt")
    if not os.path.exists(projector_path):
        raise FileNotFoundError(f"Missing projector weights: {projector_path}")
    print(f"Loading finetune projector from {projector_path}")
    model.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))

    compressor_path = os.path.join(checkpoint_path, "token_compressor.pt")
    if hasattr(model, "token_compressor"):
        if not os.path.exists(compressor_path):
            raise FileNotFoundError(f"Missing token compressor weights: {compressor_path}")
        print(f"Loading finetune token compressor from {compressor_path}")
        model.token_compressor.load_state_dict(torch.load(compressor_path, map_location="cpu"))

    llm_path = os.path.join(checkpoint_path, "llm")
    if not os.path.isdir(llm_path):
        raise FileNotFoundError(f"Missing LoRA adapter directory: {llm_path}")
    print(f"Loading finetune LoRA adapter from {llm_path}")
    base_model = model.llm.model
    if hasattr(base_model, "peft_config") and hasattr(base_model, "unload"):
        base_model = base_model.unload()
    model.llm.model = PeftModel.from_pretrained(base_model, llm_path, is_trainable=True)


def run_finetune(llama_path, architecture, max_steps, learning_rate, batch_size, use_wandb, use_dummy_data,
                  token_compressor_type="learned",
                  track_per_layer_grad_norms=True, run_eval_benchmarks=False,
                  pretrain_checkpoint=None, finetune_checkpoint=None, resume_checkpoint=None,
                  lora_learning_rate=None, dataset="instruct_150k",
                  run_name=None, projector_warmup_steps=None, train_adapter=True,
                  finetune_loss_scale=1.0,
                  finetune_gradient_accumulation_steps=8,
                  pretrain_image_tokens=None,
                  llm_backbone=None,
                  v3_connector_type=V3_DEFAULT_CONNECTOR_TYPE,
                  v4_global_image_tokens=None, v4_local_image_tokens=None,
                  v4_connector_layers=None, v4_connector_heads=None,
                  v4_connector_ff_mult=None, v4_connector_hidden_dim=None,
                  v4_connector_output_scale=None,
                  v4_connector_output_gate_init=None,
                  v4_use_2d_position_features=True,
                  v4_connector_type=None,
                  v4_deepstack_num_feature_levels=None,
                  v4_deepstack_hidden_state_indices=None):
    """Run Stage 2 fine-tuning with real COCO images."""
    import torch
    from models import create_model_from_config
    from data import build_dataloader, ImageTextCollator
    from data.dataset_splitter import deterministic_train_val_split
    from training import FinetuneTrainer
    from training.finetune import FinetuneConfig

    if use_dummy_data:
        print("WARNING: --use-dummy-data was passed but all training must use real images.")
        print("Ignoring --use-dummy-data flag and loading real COCO images.")

    # Initialize model
    print("Initializing model...")
    arch_key = _normalize_architecture_key(architecture)
    expected_llm_backbone = _metadata_llm_backbone(llm_backbone or llama_path)
    if finetune_checkpoint:
        pretrain_checkpoint = None
        print("Skipping Stage 1 checkpoint load because full Stage 2 checkpoint is loaded.")
    elif pretrain_checkpoint is None:
        pretrain_checkpoint = _auto_discover_pretrain_checkpoint(
            arch_key=arch_key,
            llm_backbone=(
                expected_llm_backbone
                if expected_llm_backbone != CURRENT_LLAMA3_BACKBONE
                else None
            ),
            token_compressor_type=token_compressor_type,
            v3_connector_type=v3_connector_type or V3_DEFAULT_CONNECTOR_TYPE,
            v4_global_image_tokens=v4_global_image_tokens,
            v4_local_image_tokens=v4_local_image_tokens,
            v4_connector_layers=v4_connector_layers,
            v4_connector_heads=v4_connector_heads,
            v4_connector_ff_mult=v4_connector_ff_mult,
            v4_connector_hidden_dim=v4_connector_hidden_dim,
            v4_connector_output_scale=v4_connector_output_scale,
            v4_connector_output_gate_init=v4_connector_output_gate_init,
            v4_connector_type=v4_connector_type,
            v4_use_2d_position_features=(
                bool(v4_use_2d_position_features)
                if v4_use_2d_position_features is not None
                else None
            ),
            v4_deepstack_num_feature_levels=v4_deepstack_num_feature_levels,
            v4_deepstack_hidden_state_indices=v4_deepstack_hidden_state_indices,
        )
        if pretrain_checkpoint:
            print(f"Auto-discovered pretrain checkpoint: {pretrain_checkpoint}")
    if arch_key == "anymal_v4" and pretrain_checkpoint:
        from model_metadata import read_model_metadata

        meta = read_model_metadata(pretrain_checkpoint) or {}
        if v4_global_image_tokens is None:
            v4_global_image_tokens = meta.get("num_global_image_tokens")
        if v4_local_image_tokens is None:
            v4_local_image_tokens = meta.get("num_local_image_tokens")
        if pretrain_image_tokens is None:
            pretrain_image_tokens = meta.get("num_image_tokens")
        if v4_connector_layers is None:
            v4_connector_layers = meta.get("connector_layers")
        if v4_connector_heads is None:
            v4_connector_heads = meta.get("connector_heads")
        if v4_connector_ff_mult is None:
            v4_connector_ff_mult = meta.get("connector_ff_mult")
        if v4_connector_hidden_dim is None:
            v4_connector_hidden_dim = meta.get("connector_hidden_dim")
            if v4_connector_hidden_dim is None and meta.get("project_directly_to_llm_dim", True):
                v4_connector_hidden_dim = V4_LEGACY_DIRECT_CONNECTOR_HIDDEN_DIM
        if v4_connector_output_scale is None:
            v4_connector_output_scale = meta.get("connector_output_scale")
        if v4_connector_output_gate_init is None:
            v4_connector_output_gate_init = meta.get("connector_output_gate_init")
        if v4_connector_type is None:
            v4_connector_type = meta.get("connector_type")
        if v4_deepstack_num_feature_levels is None:
            v4_deepstack_num_feature_levels = meta.get("deepstack_num_feature_levels")
        if v4_deepstack_hidden_state_indices is None:
            v4_deepstack_hidden_state_indices = (
                meta.get("deepstack_hidden_state_indices")
                or meta.get("vision_feature_layers")
            )
        if "use_2d_position_features" in meta:
            v4_use_2d_position_features = bool(meta["use_2d_position_features"])
    model_cfg = {
        "model": {
            "architecture": arch_key,
            "llm_model_name": llama_path,
            "llm_backbone": llm_backbone or llama_path,
            "cache_dir": "/checkpoints/hf_cache",
            "use_qlora": True,
            "lora_r": 64,
            "lora_alpha": 16,
            "use_lora": False if finetune_checkpoint else None,
            "gradient_checkpointing": True,
            "use_flash_attention": False,  # Skip flash-attn, use SDPA instead
        }
    }
    if arch_key == "anymal_v1":
        model_cfg["model"].update(
            {
                "vision_model_name": "ViT-L-14",
                "vision_pretrained": "openai",
                "projector_type": "perceiver",
                "num_image_tokens": 64,
            }
        )
        dataset_num_image_tokens = 64
        dataset_policy = "fixed"
        dataset_min_tokens = None
        dataset_max_tokens = None
        dataset_image_size = 224
        dataset_vision_type = "clip"
        dataset_vision_model = "ViT-L-14"
        dataset_max_length = 1024
    elif arch_key == "anymal_v2":
        model_cfg["model"].update(
            {
                "vision_encoder_type": "siglip2",
                "vision_model_name": "google/siglip2-so400m-patch14-384",
                "token_compressor_type": token_compressor_type,
                "bottleneck_dim": 2048,
                "max_image_tokens": 384,
                "min_image_tokens": 384,
            }
        )
        dataset_num_image_tokens = 384
        dataset_policy = "fixed"
        dataset_min_tokens = 384
        dataset_max_tokens = 384
        dataset_image_size = 384
        dataset_vision_type = "siglip2"
        dataset_vision_model = "google/siglip2-so400m-patch14-384"
        dataset_max_length = 2304
    elif arch_key == "anymal_v3":
        resolved_v3_connector_type = v3_connector_type or V3_DEFAULT_CONNECTOR_TYPE
        model_cfg["model"].update(
            {
                "vision_encoder_type": "siglip2",
                "vision_model_name": "google/siglip2-so400m-patch14-384",
                "connector_type": resolved_v3_connector_type,
                "num_image_tokens": 128,
                "connector_layers": 6,
                "connector_heads": 16,
                "connector_ff_mult": 4,
                "project_directly_to_llm_dim": True,
            }
        )
        dataset_num_image_tokens = 128
        dataset_policy = "fixed"
        dataset_min_tokens = 128
        dataset_max_tokens = 128
        dataset_image_size = 384
        dataset_vision_type = "siglip2"
        dataset_vision_model = "google/siglip2-so400m-patch14-384"
        dataset_max_length = 1536
    else:
        v4_global_tokens, v4_local_tokens, v4_total_tokens = _resolve_v4_token_split(
            total_tokens=pretrain_image_tokens,
            global_tokens=v4_global_image_tokens,
            local_tokens=v4_local_image_tokens,
        )
        resolved_v4_connector_type = v4_connector_type or V4_DEFAULT_CONNECTOR_TYPE
        resolved_deepstack_layers = _parse_v4_hidden_state_indices(
            v4_deepstack_hidden_state_indices
        )
        resolved_deepstack_levels = (
            int(v4_deepstack_num_feature_levels)
            if v4_deepstack_num_feature_levels is not None
            else (len(resolved_deepstack_layers) if resolved_deepstack_layers else None)
        )
        model_cfg["model"].update(
            {
                "vision_encoder_type": "siglip2",
                "vision_model_name": "google/siglip2-so400m-patch14-384",
                "connector_type": resolved_v4_connector_type,
                "num_global_image_tokens": v4_global_tokens,
                "num_local_image_tokens": v4_local_tokens,
                "num_image_tokens": v4_total_tokens,
                "connector_layers": int(v4_connector_layers or V4_DEFAULT_CONNECTOR_LAYERS),
                "connector_heads": int(v4_connector_heads or V4_DEFAULT_CONNECTOR_HEADS),
                "connector_ff_mult": int(v4_connector_ff_mult or V4_DEFAULT_CONNECTOR_FF_MULT),
                "connector_hidden_dim": (
                    int(v4_connector_hidden_dim)
                    if v4_connector_hidden_dim is not None
                    else V4_DEFAULT_CONNECTOR_HIDDEN_DIM
                ),
                "connector_output_scale": (
                    float(v4_connector_output_scale)
                    if v4_connector_output_scale is not None
                    else V4_DEFAULT_CONNECTOR_OUTPUT_SCALE
                ),
                "connector_output_gate_init": (
                    float(v4_connector_output_gate_init)
                    if v4_connector_output_gate_init is not None
                    else None
                ),
                "use_2d_position_features": bool(v4_use_2d_position_features),
                "project_directly_to_llm_dim": False,
            }
        )
        if resolved_v4_connector_type == "deepstack_spatial_perceiver_resampler":
            model_cfg["model"].update(
                {
                    "deepstack_num_feature_levels": resolved_deepstack_levels or 3,
                    "deepstack_hidden_state_indices": resolved_deepstack_layers,
                }
            )
        dataset_num_image_tokens = v4_total_tokens
        dataset_policy = "fixed"
        dataset_min_tokens = v4_total_tokens
        dataset_max_tokens = v4_total_tokens
        dataset_image_size = 384
        dataset_vision_type = "siglip2"
        dataset_vision_model = "google/siglip2-so400m-patch14-384"
        dataset_max_length = v4_total_tokens + 1408

    model = create_model_from_config(model_cfg)
    if finetune_checkpoint:
        print(f"Continuing from full Stage 2 checkpoint: {finetune_checkpoint}")
        _load_full_finetune_checkpoint(model, finetune_checkpoint)

    # Diagnose model
    _diagnose_model(model)

    # Always load real images - never use dummy data
    print(f"Loading dataset ({dataset}) with real COCO images...")
    ft_dataset = load_finetune_dataset(
        model.tokenizer,
        dataset=dataset,
        num_image_tokens=dataset_num_image_tokens,
        image_token_policy=dataset_policy,
        min_image_tokens=dataset_min_tokens,
        max_image_tokens=dataset_max_tokens,
        image_size=dataset_image_size,
        max_length=dataset_max_length,
        vision_encoder_type=dataset_vision_type,
        vision_model_name=dataset_vision_model,
    )

    # Split into train/val
    train_dataset, val_dataset = deterministic_train_val_split(ft_dataset, val_fraction=0.05)
    print(f"Dataset split: {len(train_dataset):,} train / {len(val_dataset):,} val")

    # Diagnose dataset
    _diagnose_dataset_sample(train_dataset, model.tokenizer)

    collator = ImageTextCollator(
        tokenizer=model.tokenizer,
        max_length=1024,
    )

    train_dataloader = build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        distributed=False,
        collate_fn=collator,
    )

    eval_dataloader = build_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        distributed=False,
        collate_fn=collator,
    )

    # Diagnose one collated batch
    print("Sampling one batch for diagnostics...")
    diag_iter = iter(train_dataloader)
    diag_batch = next(diag_iter)
    _diagnose_batch(diag_batch, model.tokenizer, "pre-training sample")

    finetune_output_dir = _resolve_run_output_dir(
        base_dir="/checkpoints/finetune-output",
        resume_checkpoint=resume_checkpoint,
        prefix="run",
        run_name=run_name,
    )
    resume_checkpoint = _resolve_effective_resume_checkpoint(
        finetune_output_dir,
        resume_checkpoint=resume_checkpoint,
        max_steps=max_steps,
    )
    print(f"Finetune checkpoints will be written to: {finetune_output_dir}")

    if pretrain_checkpoint:
        print(f"Will load Stage 1 projector from: {pretrain_checkpoint}")
    elif finetune_checkpoint:
        print("Full Stage 2 checkpoint loaded; no Stage 1 checkpoint load needed.")
    else:
        print("WARNING: No pretrain checkpoint found. Perceiver resampler has random weights.")
        print("Stage 1 pretraining is strongly recommended before Stage 2.")

    # Create trainer config
    eval_steps = max(50, max_steps // 10)  # ~10 eval points during training
    default_projector_warmup_steps = 0 if arch_key in {"anymal_v2", "anymal_v3", "anymal_v4"} else 200
    if projector_warmup_steps is None:
        projector_warmup_steps = default_projector_warmup_steps

    config = FinetuneConfig(
        max_steps=max_steps,
        gradient_accumulation_steps=int(finetune_gradient_accumulation_steps or 8),
        learning_rate=learning_rate,
        lora_learning_rate=lora_learning_rate,
        warmup_steps=min(100, max_steps // 10),
        weight_decay=0.01,
        use_amp=True,
        amp_dtype="bfloat16",
        logging_steps=1,  # Log every step for close monitoring
        save_steps=_checkpoint_save_interval(max_steps, max_interval=50),
        eval_steps=eval_steps,
        max_eval_batches=200,  # Clip eval to 200 batches (~55s) during training
        output_dir=finetune_output_dir,
        save_llm_checkpoint=True,
        save_llm_base_weights=False,
        commit_on_save=True,
        use_wandb=use_wandb,
        wandb_project="anymal-finetune",
        track_per_layer_grad_norms=track_per_layer_grad_norms,
        pretrain_checkpoint=pretrain_checkpoint,
        finetune_checkpoint=finetune_checkpoint,
        resume_from_checkpoint=resume_checkpoint,
        projector_warmup_steps=projector_warmup_steps,
        train_adapter=bool(train_adapter),
        loss_scale=float(finetune_loss_scale or 1.0),
    )

    # Train
    trainer = FinetuneTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    # Set up eval runner for VQA benchmarks
    eval_runner = None
    if run_eval_benchmarks:
        try:
            from evaluation.eval_runner import EvalRunner
            vqa_questions = "/checkpoints/vqa_data/v2_OpenEnded_mscoco_val2014_questions.json"
            vqa_annotations = "/checkpoints/vqa_data/v2_mscoco_val2014_annotations.json"
            vqa_image_dir = "/checkpoints/coco_val2014"
            if os.path.exists(vqa_questions) and os.path.exists(vqa_annotations) and os.path.exists(vqa_image_dir):
                eval_runner = EvalRunner(
                    model=model,
                    vqa_questions_file=vqa_questions,
                    vqa_annotations_file=vqa_annotations,
                    vqa_image_dir=vqa_image_dir,
                    max_eval_samples=1000,
                    min_eval_samples=500,
                    subset_seed=42,
                    raise_on_error=True,
                )
                print(f"VQA eval runner initialized (will run every {eval_steps} steps)")
            else:
                missing = [
                    path for path in (vqa_questions, vqa_annotations, vqa_image_dir)
                    if not os.path.exists(path)
                ]
                raise RuntimeError(f"VQA data not found for benchmark evaluation: {missing}")
        except Exception as e:
            raise RuntimeError(f"Could not initialize eval runner: {e}") from e

    metrics = trainer.train()

    # Run final VQA eval
    if eval_runner is not None:
        print("\nRunning final VQA evaluation...")
        vqa_metrics = eval_runner.run(["vqa"])
        if not vqa_metrics or "eval/vqa_error" in vqa_metrics:
            raise RuntimeError(f"Final VQA eval failed or returned empty metrics: {vqa_metrics}")
        if trainer.logger is not None:
            trainer.logger.log(vqa_metrics)
        print(f"Final VQA metrics: {vqa_metrics}")

    print(f"\nTraining complete! Final metrics: {metrics}")


def run_pretrain(
    llama_path,
    architecture,
    max_steps,
    learning_rate,
    batch_size,
    use_wandb,
    use_dummy_data=False,
    token_compressor_type="learned",
    distributed=False,
    resume_checkpoint=None,
    pretrain_checkpoint=None,
    output_dir="/checkpoints/pretrain-output",
    run_name=None,
    pretrain_image_tokens=None,
    llm_backbone=None,
    dataset="llava_pretrain",
    connector_warmup_steps=0,
    pretrain_loss_scale=1.0,
    pretrain_loss_normalization="mean",
    pretrain_loss_normalization_target_tokens=8.0,
    pretrain_gradient_accumulation_steps=8,
    pretrain_save_steps=None,
    v3_connector_type=V3_DEFAULT_CONNECTOR_TYPE,
    v4_global_image_tokens=None,
    v4_local_image_tokens=None,
    v4_connector_layers=None,
    v4_connector_heads=None,
    v4_connector_ff_mult=None,
    v4_connector_hidden_dim=None,
    v4_connector_output_scale=None,
    v4_connector_output_gate_init=None,
    v4_use_2d_position_features=True,
    v4_connector_type=None,
    v4_deepstack_num_feature_levels=None,
    v4_deepstack_hidden_state_indices=None,
):
    """Run Stage 1 pretraining with real COCO images."""
    import torch
    from models import create_model_from_config
    from data import build_dataloader, ImageTextCollator
    from data.dataset_splitter import deterministic_train_val_split
    from training import PretrainTrainer
    from training.pretrain import PretrainConfig

    if use_dummy_data:
        print("WARNING: --use-dummy-data was passed but Stage 1 must use real images.")
        print("Ignoring --use-dummy-data flag and loading real COCO images.")
    resume_checkpoint = _resolve_effective_resume_checkpoint(
        output_dir,
        resume_checkpoint=resume_checkpoint,
        max_steps=max_steps,
    )
    print(f"Pretrain checkpoints will be written to: {output_dir}")

    # Initialize model (no LoRA for pretraining)
    print("Initializing model...")
    # For DDP, disable device_map so each process places model on its own GPU
    device_map = None if distributed else "auto"
    arch_key = _normalize_architecture_key(architecture)
    expected_llm_backbone = _metadata_llm_backbone(llm_backbone or llama_path)
    if arch_key == "anymal_v4" and (pretrain_checkpoint or resume_checkpoint):
        from model_metadata import read_model_metadata

        meta = read_model_metadata(pretrain_checkpoint or resume_checkpoint) or {}
        if v4_global_image_tokens is None:
            v4_global_image_tokens = meta.get("num_global_image_tokens")
        if v4_local_image_tokens is None:
            v4_local_image_tokens = meta.get("num_local_image_tokens")
        if pretrain_image_tokens is None:
            pretrain_image_tokens = meta.get("num_image_tokens")
        if v4_connector_layers is None:
            v4_connector_layers = meta.get("connector_layers")
        if v4_connector_heads is None:
            v4_connector_heads = meta.get("connector_heads")
        if v4_connector_ff_mult is None:
            v4_connector_ff_mult = meta.get("connector_ff_mult")
        if v4_connector_hidden_dim is None:
            v4_connector_hidden_dim = meta.get("connector_hidden_dim")
            if v4_connector_hidden_dim is None and meta.get("project_directly_to_llm_dim", True):
                v4_connector_hidden_dim = V4_LEGACY_DIRECT_CONNECTOR_HIDDEN_DIM
        if v4_connector_output_scale is None:
            v4_connector_output_scale = meta.get("connector_output_scale")
        if v4_connector_output_gate_init is None:
            v4_connector_output_gate_init = meta.get("connector_output_gate_init")
        if v4_connector_type is None:
            v4_connector_type = meta.get("connector_type")
        if v4_deepstack_num_feature_levels is None:
            v4_deepstack_num_feature_levels = meta.get("deepstack_num_feature_levels")
        if v4_deepstack_hidden_state_indices is None:
            v4_deepstack_hidden_state_indices = (
                meta.get("deepstack_hidden_state_indices")
                or meta.get("vision_feature_layers")
            )
        if "use_2d_position_features" in meta:
            v4_use_2d_position_features = bool(meta["use_2d_position_features"])
    model_cfg = {
        "model": {
            "architecture": arch_key,
            "llm_model_name": llama_path,
            "llm_backbone": llm_backbone or llama_path,
            "cache_dir": "/checkpoints/hf_cache",
            "use_qlora": False,  # No LoRA for Stage 1
            "use_lora": False,
            "gradient_checkpointing": True,
            "use_flash_attention": False,  # Skip flash-attn, use SDPA instead
        }
    }
    if arch_key == "anymal_v1":
        model_cfg["model"].update(
            {
                "vision_model_name": "ViT-L-14",
                "vision_pretrained": "openai",
                "projector_type": "perceiver",
                "num_image_tokens": 64,
            }
        )
        pretrain_num_image_tokens = 64
        insert_placeholders = False
        pretrain_max_length = 256
        pretrain_image_size = 224
        pretrain_vision_type = "clip"
        pretrain_vision_model = "ViT-L-14"
    elif arch_key == "anymal_v2":
        pretrain_num_image_tokens = int(pretrain_image_tokens or 256)
        if pretrain_num_image_tokens <= 0:
            raise ValueError(f"pretrain_image_tokens must be > 0, got {pretrain_image_tokens}")
        model_cfg["model"].update(
            {
                "vision_encoder_type": "siglip2",
                "vision_model_name": "google/siglip2-so400m-patch14-384",
                "token_compressor_type": token_compressor_type,
                "bottleneck_dim": 2048,
                "max_image_tokens": pretrain_num_image_tokens,
                "min_image_tokens": pretrain_num_image_tokens,
            }
        )
        insert_placeholders = True
        pretrain_max_length = pretrain_num_image_tokens + 384
        pretrain_image_size = 384
        pretrain_vision_type = "siglip2"
        pretrain_vision_model = "google/siglip2-so400m-patch14-384"
    elif arch_key == "anymal_v3":
        pretrain_num_image_tokens = int(pretrain_image_tokens or 128)
        if pretrain_num_image_tokens <= 0:
            raise ValueError(f"pretrain_image_tokens must be > 0, got {pretrain_image_tokens}")
        resolved_v3_connector_type = v3_connector_type or V3_DEFAULT_CONNECTOR_TYPE
        model_cfg["model"].update(
            {
                "vision_encoder_type": "siglip2",
                "vision_model_name": "google/siglip2-so400m-patch14-384",
                "connector_type": resolved_v3_connector_type,
                "num_image_tokens": pretrain_num_image_tokens,
                "connector_layers": 6,
                "connector_heads": 16,
                "connector_ff_mult": 4,
                "project_directly_to_llm_dim": True,
            }
        )
        insert_placeholders = True
        pretrain_max_length = pretrain_num_image_tokens + 384
        pretrain_image_size = 384
        pretrain_vision_type = "siglip2"
        pretrain_vision_model = "google/siglip2-so400m-patch14-384"
    else:
        v4_global_tokens, v4_local_tokens, pretrain_num_image_tokens = _resolve_v4_token_split(
            total_tokens=pretrain_image_tokens,
            global_tokens=v4_global_image_tokens,
            local_tokens=v4_local_image_tokens,
        )
        resolved_v4_connector_type = v4_connector_type or V4_DEFAULT_CONNECTOR_TYPE
        resolved_deepstack_layers = _parse_v4_hidden_state_indices(
            v4_deepstack_hidden_state_indices
        )
        resolved_deepstack_levels = (
            int(v4_deepstack_num_feature_levels)
            if v4_deepstack_num_feature_levels is not None
            else (len(resolved_deepstack_layers) if resolved_deepstack_layers else None)
        )
        model_cfg["model"].update(
            {
                "vision_encoder_type": "siglip2",
                "vision_model_name": "google/siglip2-so400m-patch14-384",
                "connector_type": resolved_v4_connector_type,
                "num_global_image_tokens": v4_global_tokens,
                "num_local_image_tokens": v4_local_tokens,
                "num_image_tokens": pretrain_num_image_tokens,
                "connector_layers": int(v4_connector_layers or V4_DEFAULT_CONNECTOR_LAYERS),
                "connector_heads": int(v4_connector_heads or V4_DEFAULT_CONNECTOR_HEADS),
                "connector_ff_mult": int(v4_connector_ff_mult or V4_DEFAULT_CONNECTOR_FF_MULT),
                "connector_hidden_dim": (
                    int(v4_connector_hidden_dim)
                    if v4_connector_hidden_dim is not None
                    else V4_DEFAULT_CONNECTOR_HIDDEN_DIM
                ),
                "connector_output_scale": (
                    float(v4_connector_output_scale)
                    if v4_connector_output_scale is not None
                    else V4_DEFAULT_CONNECTOR_OUTPUT_SCALE
                ),
                "connector_output_gate_init": (
                    float(v4_connector_output_gate_init)
                    if v4_connector_output_gate_init is not None
                    else (
                        None
                        if pretrain_checkpoint or resume_checkpoint
                        else V4_DEFAULT_CONNECTOR_OUTPUT_GATE_INIT
                    )
                ),
                "use_2d_position_features": bool(v4_use_2d_position_features),
                "project_directly_to_llm_dim": False,
            }
        )
        if resolved_v4_connector_type == "deepstack_spatial_perceiver_resampler":
            model_cfg["model"].update(
                {
                    "deepstack_num_feature_levels": resolved_deepstack_levels or 3,
                    "deepstack_hidden_state_indices": resolved_deepstack_layers,
                }
            )
        insert_placeholders = True
        pretrain_max_length = pretrain_num_image_tokens + 384
        pretrain_image_size = 384
        pretrain_vision_type = "siglip2"
        pretrain_vision_model = "google/siglip2-so400m-patch14-384"

    model = create_model_from_config(
        model_cfg,
        llm_device_map=device_map,
    )

    # Configure for Stage 1: only train projector
    model.set_training_stage(1)

    if pretrain_checkpoint:
        from model_metadata import (
            validate_checkpoint_architecture,
            validate_checkpoint_metadata_values,
        )

        print(f"Loading Stage 1 connector weights from: {pretrain_checkpoint}")
        validate_checkpoint_architecture(
            checkpoint_dir=pretrain_checkpoint,
            expected_architecture=arch_key,
        )
        expected_values = {}
        if arch_key in {"anymal_v3", "anymal_v4"}:
            expected_values = {
                "connector_type": getattr(model, "connector_type", None),
                "num_image_tokens": getattr(model, "num_image_tokens", None),
                "connector_layers": getattr(model, "connector_layers", None),
                "connector_heads": getattr(model, "connector_heads", None),
                "connector_ff_mult": getattr(model, "connector_ff_mult", None),
            }
            if expected_llm_backbone != CURRENT_LLAMA3_BACKBONE:
                expected_values["llm_backbone"] = expected_llm_backbone
            if arch_key == "anymal_v4":
                expected_values.update(
                    {
                        "num_global_image_tokens": getattr(model, "num_global_image_tokens", None),
                        "num_local_image_tokens": getattr(model, "num_local_image_tokens", None),
                        "connector_hidden_dim": getattr(model, "connector_hidden_dim", None),
                        "connector_output_scale": getattr(model, "connector_output_scale", None),
                        "connector_output_gate_init": getattr(model, "connector_output_gate_init", None),
                        "use_2d_position_features": getattr(model, "use_2d_position_features", None),
                    }
                )
                if getattr(model, "connector_type", None) == "deepstack_spatial_perceiver_resampler":
                    expected_values.update(
                        {
                            "deepstack_num_feature_levels": getattr(
                                model,
                                "deepstack_num_feature_levels",
                                None,
                            ),
                            "deepstack_hidden_state_indices": list(
                                getattr(model, "deepstack_hidden_state_indices", [])
                            ),
                            "vision_feature_layers": list(
                                getattr(model, "deepstack_hidden_state_indices", [])
                            ),
                        }
                    )
        if expected_values:
            validate_checkpoint_metadata_values(
                checkpoint_dir=pretrain_checkpoint,
                expected_architecture=arch_key,
                expected_values=expected_values,
            )
        projector_path = os.path.join(pretrain_checkpoint, "projector.pt")
        if not os.path.exists(projector_path):
            raise FileNotFoundError(
                f"Expected projector.pt in pretrain checkpoint: {pretrain_checkpoint}"
            )
        model.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))
        print("Loaded Stage 1 connector weights")

    # Diagnose model (rank 0 only)
    from training.distributed import is_main_process as _is_main
    if _is_main():
        _diagnose_model(model)

    # Always load real images
    print(f"Loading dataset ({dataset}) with real images...")
    if dataset in {"llava_pretrain", "caption_alignment", "v3_caption_alignment", "v4_caption_alignment"}:
        dataset = load_llava_pretrain_dataset(
            model.tokenizer,
            insert_image_placeholders=insert_placeholders,
            num_image_tokens=pretrain_num_image_tokens,
            max_length=pretrain_max_length,
            image_size=pretrain_image_size,
            vision_encoder_type=pretrain_vision_type,
            vision_model_name=pretrain_vision_model,
        )
    elif dataset in {
        "v3_grounding",
        "v3_grounding_alignment",
        "v4_grounding",
        "v4_grounding_alignment",
        "v4_answer_type_focus",
        "v4_direct_calibration",
        "v4_semantic_calibration",
        "v5_semantic_calibration",
        "v5_semantic_calibration_robust",
    }:
        grounding_dataset_name = dataset
        dataset = load_finetune_dataset(
            model.tokenizer,
            dataset=grounding_dataset_name,
            num_image_tokens=pretrain_num_image_tokens,
            image_token_policy="fixed",
            min_image_tokens=pretrain_num_image_tokens,
            max_image_tokens=pretrain_num_image_tokens,
            image_size=pretrain_image_size,
            max_length=512,
            vision_encoder_type=pretrain_vision_type,
            vision_model_name=pretrain_vision_model,
        )
    else:
        raise ValueError(f"Unsupported pretrain dataset: {dataset}")

    # Split into train/val
    train_dataset, val_dataset = deterministic_train_val_split(dataset, val_fraction=0.05)
    print(f"Dataset split: {len(train_dataset):,} train / {len(val_dataset):,} val")

    # Diagnose dataset (rank 0 only)
    if _is_main():
        _diagnose_dataset_sample(train_dataset, model.tokenizer)

    collator = ImageTextCollator(
        tokenizer=model.tokenizer,
        max_length=256,
    )

    # When using mp.spawn for DDP, num_workers>0 causes nested fork issues
    dl_workers = 0 if distributed else 4

    train_dataloader = build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dl_workers,
        distributed=distributed,
        collate_fn=collator,
    )

    eval_dataloader = build_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if distributed else 2,
        distributed=False,
        collate_fn=collator,
    )

    # Diagnose one collated batch (only on rank 0)
    from training.distributed import is_main_process
    if is_main_process():
        print("Sampling one batch for diagnostics...")
        diag_iter = iter(train_dataloader)
        diag_batch = next(diag_iter)
        _diagnose_batch(diag_batch, model.tokenizer, "pre-training sample")

    # Create trainer config
    eval_steps = max(50, max_steps // 10)
    save_steps = int(pretrain_save_steps or _checkpoint_save_interval(max_steps))
    print(f"Stage 1 checkpoint save steps: {save_steps}")

    config = PretrainConfig(
        max_steps=max_steps,
        gradient_accumulation_steps=int(pretrain_gradient_accumulation_steps or 8),
        learning_rate=learning_rate,
        warmup_steps=min(100, max_steps // 10),
        use_amp=True,
        amp_dtype="bfloat16",
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=5,
        eval_steps=eval_steps,
        max_eval_batches=200,
        output_dir=output_dir,
        save_llm_checkpoint=False,
        save_llm_base_weights=False,
        commit_on_save=True,
        use_wandb=use_wandb,
        wandb_project="anymal-pretrain",
        wandb_run_name=run_name,
        resume_from_checkpoint=resume_checkpoint,
        loss_scale=float(pretrain_loss_scale or 1.0),
        loss_normalization=pretrain_loss_normalization or "mean",
        loss_normalization_target_tokens=float(
            pretrain_loss_normalization_target_tokens or 8.0
        ),
        connector_warmup_steps=int(connector_warmup_steps or 0),
    )

    # Train
    trainer = PretrainTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    metrics = trainer.train()
    print(f"Training complete! Final metrics: {metrics}")


def load_finetune_dataset(
    tokenizer,
    dataset="instruct_150k",
    num_image_tokens: int = 64,
    image_token_policy: str = "fixed",
    min_image_tokens: int = None,
    max_image_tokens: int = None,
    image_size: int = 224,
    max_length: int = 1024,
    vision_encoder_type: str = "clip",
    vision_model_name: str = None,
    image_augmentation_mode: str = "none",
):
    """
    Load finetune dataset using cached JSON from volume.

    Args:
        tokenizer: LLaMA tokenizer
        dataset: "instruct_150k" for LLaVA-Instruct-150K,
                 "mix_665k" for LLaVA-1.5 Mix-665K (filtered to cached COCO images)
    """
    import json as _json
    from data.instruction_dataset import InstructionDataset, create_instruction_dataset

    image_dir = "/checkpoints/coco_images"
    if not os.path.exists(image_dir):
        raise RuntimeError(
            f"COCO images directory not found at {image_dir}. "
            "Real images are required for training. "
            "The container setup should have downloaded them via _ensure_coco_images_cached()."
        )

    num_images = len([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    print(f"Found {num_images} real COCO images in {image_dir}")
    if num_images == 0:
        raise RuntimeError(f"No JPEG images found in {image_dir}. Cannot train without real images.")

    direct_system_prompt = (
        "Answer directly and briefly. Do not greet the user. Do not ask follow-up questions."
    )
    training_system_prompt = InstructionDataset.DEFAULT_SYSTEM_PROMPT

    def _build_instruction_dataset(json_path):
        return InstructionDataset(
            data_path=json_path,
            image_dir=image_dir,
            tokenizer=tokenizer,
            image_size=image_size,
            max_length=max_length,
            num_image_tokens=num_image_tokens,
            image_token_policy=image_token_policy,
            min_image_tokens=min_image_tokens,
            max_image_tokens=max_image_tokens,
            vision_encoder_type=vision_encoder_type,
            vision_model_name=vision_model_name,
            use_augmentation=(str(image_augmentation_mode or "none").lower() != "none"),
            image_augmentation_mode=image_augmentation_mode,
            filter_to_available_images=True,
        )

    def _filtered_mix665k_path():
        json_path = "/checkpoints/llava_data/llava_v1_5_mix665k.json"
        if not os.path.exists(json_path):
            raise RuntimeError(
                f"Mix-665K JSON not found at {json_path}. "
                "It should have been downloaded during container setup."
            )

        print(f"Loading LLaVA-1.5 Mix-665K from {json_path}")

        # The 665K JSON has image paths like "coco/train2017/000000123456.jpg",
        # "gqa/images/xxxxx.jpg", "ocr_vqa/images/xxxxx.jpg", etc.
        # Normalize to filenames to match our cached COCO image subset.
        available_images = set(
            f for f in os.listdir(image_dir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        )

        with open(json_path, "r") as f:
            raw_data = _json.load(f)

        print(f"Mix-665K total samples: {len(raw_data)}")

        filtered = []
        for sample in raw_data:
            img_path = sample.get("image", "")
            if not img_path:
                continue
            img_filename = os.path.basename(img_path)
            if img_filename in available_images:
                sample["image"] = img_filename
                filtered.append(sample)

        print(f"Mix-665K filtered to {len(filtered)}/{len(raw_data)} samples with cached COCO images")
        if len(filtered) == 0:
            raise RuntimeError(
                "No Mix-665K samples matched available COCO images. "
                "Check image filenames."
            )

        filtered_path = "/checkpoints/llava_data/mix665k_filtered.json"
        with open(filtered_path, "w") as f:
            _json.dump(filtered, f)
        return filtered_path

    def _first_human_and_gpt(sample):
        question = ""
        answer = ""
        for turn in sample.get("conversations", []):
            role = turn.get("from", turn.get("role", ""))
            value = turn.get("value", turn.get("content", ""))
            value = value.replace("<image>", "").strip()
            if role in {"human", "user"} and not question:
                question = value
            elif role in {"gpt", "assistant"} and not answer:
                answer = value
        return question, answer

    def _has_reserved_chat_markers(text):
        return any(
            marker in text
            for marker in ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>")
        )

    def _direct_task_type(question):
        q = " ".join(question.lower().split())
        if "how many" in q or q.startswith("count"):
            return "count"
        if "what color" in q or "colour" in q:
            return "color"
        if any(term in q for term in ("left", "right", "behind", "front", "next to", "where")):
            return "spatial"
        if any(term in q for term in ("what is", "what object", "what vehicle", "which")):
            return "short_object"
        if any(term in q for term in ("is there", "are there", "can you see")):
            return "yes_no"
        return "other"

    def _filtered_direct_answer_path():
        mix_path = _filtered_mix665k_path()
        direct_path = "/checkpoints/llava_data/mix665k_direct_answer_filtered.json"
        with open(mix_path, "r") as f:
            raw_data = _json.load(f)

        kept = []
        task_counts = {}
        skipped_reserved_markers = 0
        for sample in raw_data:
            question, answer = _first_human_and_gpt(sample)
            task = _direct_task_type(question)
            answer_words = answer.split()
            is_direct_task = task in {"count", "color", "spatial", "short_object", "yes_no"}
            is_short_answer = 0 < len(answer_words) <= 8
            if is_direct_task and is_short_answer:
                if _has_reserved_chat_markers(question) or _has_reserved_chat_markers(answer):
                    skipped_reserved_markers += 1
                    continue
                kept.append(
                    {
                        "id": f"{sample.get('id', 'sample')}_direct_{task}",
                        "image": sample["image"],
                        "conversations": [
                            {"from": "human", "value": f"<image>\n{question.strip()}"},
                            {"from": "gpt", "value": answer.strip()},
                        ],
                    }
                )
                task_counts[task] = task_counts.get(task, 0) + 1

        if not kept:
            raise RuntimeError("Direct-answer Mix-665K filter produced no samples.")

        with open(direct_path, "w") as f:
            _json.dump(kept, f)
        print(
            f"Mix-665K direct-answer filter kept {len(kept)}/{len(raw_data)} samples; "
            f"tasks={task_counts}; skipped_reserved_markers={skipped_reserved_markers}"
        )
        return direct_path

    def _pluralize_category(name):
        irregular = {
            "person": "people",
            "mouse": "mice",
            "sheep": "sheep",
            "skis": "skis",
            "scissors": "scissors",
        }
        if name in irregular:
            return irregular[name]
        if name.endswith("y") and name[-2:] not in {"ay", "ey", "oy", "uy"}:
            return f"{name[:-1]}ies"
        if name.endswith(("s", "x", "ch", "sh")):
            return f"{name}es"
        return f"{name}s"

    def _ensure_coco_instance_annotations():
        import requests
        import zipfile

        annotation_dir = "/checkpoints/coco_annotations"
        instances_path = os.path.join(annotation_dir, "annotations", "instances_train2017.json")
        if os.path.exists(instances_path):
            return instances_path

        os.makedirs(annotation_dir, exist_ok=True)
        zip_path = os.path.join(annotation_dir, "annotations_trainval2017.zip")
        if not os.path.exists(zip_path):
            url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            print(f"Downloading COCO instance annotations from {url}...")
            response = requests.get(url, timeout=600)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(response.content)

        print(f"Extracting COCO instance annotations from {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(annotation_dir)
        volume.commit()
        if not os.path.exists(instances_path):
            raise FileNotFoundError(f"Expected COCO instances at {instances_path}")
        return instances_path

    def _coco_object_direct_answer_path():
        """Build short-answer object/category supervision from COCO instances."""
        output_path = "/checkpoints/llava_data/coco_object_direct_train2017.json"
        if os.path.exists(output_path):
            return output_path

        instances_path = _ensure_coco_instance_annotations()
        print(f"Building COCO object direct-answer data from {instances_path}")
        with open(instances_path, "r") as f:
            instances = _json.load(f)

        available_images = {
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        images = {
            image["id"]: image
            for image in instances.get("images", [])
            if image.get("file_name") in available_images
        }
        categories = {
            category["id"]: category["name"]
            for category in instances.get("categories", [])
        }
        per_image = {}
        for ann in instances.get("annotations", []):
            if ann.get("iscrowd"):
                continue
            image_id = ann.get("image_id")
            if image_id not in images:
                continue
            category_name = categories.get(ann.get("category_id"))
            if not category_name:
                continue
            per_image.setdefault(image_id, {}).setdefault(
                category_name,
                {"count": 0, "area": 0.0},
            )
            per_image[image_id][category_name]["count"] += 1
            per_image[image_id][category_name]["area"] += float(ann.get("area", 0.0))

        vehicle_categories = {
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
        }
        samples = []
        for image_id, category_stats in sorted(per_image.items()):
            if not category_stats:
                continue
            image = images[image_id]
            filename = image["file_name"]
            ranked = sorted(
                category_stats.items(),
                key=lambda item: (item[1]["area"], item[1]["count"]),
                reverse=True,
            )
            dominant_name, dominant_stats = ranked[0]
            image_area = max(float(image.get("width", 1) * image.get("height", 1)), 1.0)
            dominant_area_fraction = dominant_stats["area"] / image_area

            # Singular generic object questions are noisy in crowded scenes. Keep
            # them only when one category is visually dominant.
            if dominant_area_fraction >= 0.18 or len(ranked) == 1:
                samples.append(
                    {
                        "id": f"coco_object_{image_id}_dominant",
                        "image": filename,
                        "conversations": [
                            {"from": "human", "value": "<image>\nWhat object is this?"},
                            {"from": "gpt", "value": dominant_name},
                        ],
                    }
                )

            visible = [name for name, _stats in ranked[:3]]
            if visible:
                samples.append(
                    {
                        "id": f"coco_object_{image_id}_visible",
                        "image": filename,
                        "conversations": [
                            {"from": "human", "value": "<image>\nWhat objects are visible?"},
                            {"from": "gpt", "value": ", ".join(visible)},
                        ],
                    }
                )

            vehicles = [
                (name, stats)
                for name, stats in ranked
                if name in vehicle_categories
            ]
            if vehicles:
                vehicle_name = vehicles[0][0]
                samples.append(
                    {
                        "id": f"coco_object_{image_id}_vehicle",
                        "image": filename,
                        "conversations": [
                            {"from": "human", "value": "<image>\nWhat vehicle is shown?"},
                            {"from": "gpt", "value": vehicle_name},
                        ],
                    }
                )

            for name, stats in ranked[:2]:
                count = int(stats["count"])
                if 1 <= count <= 5:
                    samples.append(
                        {
                            "id": f"coco_object_{image_id}_count_{name.replace(' ', '_')}",
                            "image": filename,
                            "conversations": [
                                {
                                    "from": "human",
                                    "value": f"<image>\nHow many {_pluralize_category(name)} are visible?",
                                },
                                {"from": "gpt", "value": str(count)},
                            ],
                        }
                    )

        if not samples:
            raise RuntimeError("COCO object direct-answer generation produced no samples.")

        with open(output_path, "w") as f:
            _json.dump(samples, f)
        print(f"Built {len(samples)} COCO object direct-answer samples at {output_path}")
        volume.commit()
        return output_path

    def _ensure_vqa_train_files():
        import requests
        import zipfile

        vqa_dir = "/checkpoints/vqa_data"
        os.makedirs(vqa_dir, exist_ok=True)

        files = {
            "questions": {
                "path": os.path.join(vqa_dir, "v2_OpenEnded_mscoco_train2014_questions.json"),
                "url": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
                "zip": os.path.join(vqa_dir, "train_questions.zip"),
            },
            "annotations": {
                "path": os.path.join(vqa_dir, "v2_mscoco_train2014_annotations.json"),
                "url": "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
                "zip": os.path.join(vqa_dir, "train_annotations.zip"),
            },
        }

        for name, info in files.items():
            if os.path.exists(info["path"]):
                continue
            print(f"Downloading VQAv2 train2014 {name}...")
            resp = requests.get(info["url"], timeout=600)
            resp.raise_for_status()
            with open(info["zip"], "wb") as f:
                f.write(resp.content)
            with zipfile.ZipFile(info["zip"], "r") as zf:
                zf.extractall(vqa_dir)
            os.remove(info["zip"])
            volume.commit()

        return files["questions"]["path"], files["annotations"]["path"]

    def _ensure_vqa_train_images(questions_path, num_images=30000, image_sample_seed=20260429):
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed

        image_dir_vqa = "/checkpoints/coco_train2014_vqa"
        os.makedirs(image_dir_vqa, exist_ok=True)
        manifest_path = os.path.join(
            image_dir_vqa,
            f"manifest_seed{image_sample_seed}_n{num_images}.json",
        )
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = _json.load(f)
            if manifest.get("requested", 0) >= num_images:
                print(f"Using cached VQAv2 train image manifest: {manifest_path}")
                return image_dir_vqa

        with open(questions_path) as f:
            questions = _json.load(f)["questions"]
        image_ids = sorted({int(q["image_id"]) for q in questions})
        rng = random.Random(int(image_sample_seed))
        rng.shuffle(image_ids)
        image_ids = image_ids[: int(num_images)]

        base_url = "http://images.cocodataset.org/train2014"

        def download_one(image_id):
            filename = f"COCO_train2014_{image_id:012d}.jpg"
            path = os.path.join(image_dir_vqa, filename)
            if os.path.exists(path):
                return filename, "skip"
            try:
                resp = requests.get(f"{base_url}/{filename}", timeout=30)
                resp.raise_for_status()
                with open(path, "wb") as f:
                    f.write(resp.content)
                return filename, "ok"
            except Exception as e:
                return filename, f"fail: {e}"

        print(
            f"Ensuring {len(image_ids)} VQAv2 train2014 images in {image_dir_vqa} "
            f"(image_sample_seed={image_sample_seed})"
        )
        ok_count, skip_count, fail_count = 0, 0, 0
        with ThreadPoolExecutor(max_workers=24) as executor:
            futures = [executor.submit(download_one, image_id) for image_id in image_ids]
            for i, future in enumerate(as_completed(futures)):
                _filename, status = future.result()
                if status == "ok":
                    ok_count += 1
                elif status == "skip":
                    skip_count += 1
                else:
                    fail_count += 1
                if (i + 1) % 1000 == 0:
                    print(
                        f"  VQAv2 train images: {i + 1}/{len(image_ids)} "
                        f"(ok={ok_count}, skip={skip_count}, fail={fail_count})"
                    )

        with open(manifest_path, "w") as f:
            _json.dump(
                {
                    "requested": int(num_images),
                    "image_sample_seed": int(image_sample_seed),
                    "downloaded": ok_count,
                    "skipped": skip_count,
                    "failed": fail_count,
                    "image_ids": image_ids,
                },
                f,
                indent=2,
            )
        volume.commit()
        return image_dir_vqa

    def _vqa_train_direct_answer_path(max_samples=150000):
        output_path = f"/checkpoints/vqa_data/vqa_train2014_direct_{int(max_samples)}.json"
        if os.path.exists(output_path):
            questions_path, _annotations_path = _ensure_vqa_train_files()
            vqa_image_dir = _ensure_vqa_train_images(questions_path)
            return output_path, vqa_image_dir

        questions_path, annotations_path = _ensure_vqa_train_files()
        vqa_image_dir = _ensure_vqa_train_images(questions_path)

        available_images = {
            f for f in os.listdir(vqa_image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        with open(questions_path) as f:
            questions = {
                int(q["question_id"]): q
                for q in _json.load(f)["questions"]
            }
        with open(annotations_path) as f:
            annotations = _json.load(f)["annotations"]

        samples = []
        skipped_reserved_markers = 0
        skipped_missing_image = 0
        for ann in annotations:
            question_id = int(ann["question_id"])
            question = questions.get(question_id)
            if question is None:
                continue
            filename = f"COCO_train2014_{int(question['image_id']):012d}.jpg"
            if filename not in available_images:
                skipped_missing_image += 1
                continue
            prompt = " ".join(str(question["question"]).split())
            answer = " ".join(str(ann.get("multiple_choice_answer", "")).split())
            if not prompt or not answer:
                continue
            if _has_reserved_chat_markers(prompt) or _has_reserved_chat_markers(answer):
                skipped_reserved_markers += 1
                continue
            samples.append(
                {
                    "id": f"vqa_train2014_{question_id}",
                    "image": filename,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{prompt}"},
                        {"from": "gpt", "value": answer},
                    ],
                }
            )

        rng = random.Random(4201)
        rng.shuffle(samples)
        samples = samples[: int(max_samples)]
        if not samples:
            raise RuntimeError("VQAv2 train direct-answer generation produced no samples.")

        with open(output_path, "w") as f:
            _json.dump(samples, f)
        print(
            f"Built {len(samples)} VQAv2 train direct-answer samples at {output_path}; "
            f"skipped_missing_image={skipped_missing_image}; "
            f"skipped_reserved_markers={skipped_reserved_markers}"
        )
        volume.commit()
        return output_path, vqa_image_dir

    def _vqa_train_direct_answer_path_by_type(answer_type, max_samples=150000):
        """Build VQAv2 train2014 direct-answer data for one answer_type bucket."""
        safe_answer_type = str(answer_type).replace("/", "_").replace(" ", "_")
        output_path = (
            f"/checkpoints/vqa_data/"
            f"vqa_train2014_direct_{safe_answer_type}_{int(max_samples)}.json"
        )
        if os.path.exists(output_path):
            questions_path, _annotations_path = _ensure_vqa_train_files()
            vqa_image_dir = _ensure_vqa_train_images(questions_path)
            return output_path, vqa_image_dir

        questions_path, annotations_path = _ensure_vqa_train_files()
        vqa_image_dir = _ensure_vqa_train_images(questions_path)

        available_images = {
            f for f in os.listdir(vqa_image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        with open(questions_path) as f:
            questions = {
                int(q["question_id"]): q
                for q in _json.load(f)["questions"]
            }
        with open(annotations_path) as f:
            annotations = _json.load(f)["annotations"]

        samples = []
        skipped_missing_image = 0
        skipped_reserved_markers = 0
        for ann in annotations:
            if str(ann.get("answer_type", "")).lower() != str(answer_type).lower():
                continue
            question_id = int(ann["question_id"])
            question = questions.get(question_id)
            if question is None:
                continue
            filename = f"COCO_train2014_{int(question['image_id']):012d}.jpg"
            if filename not in available_images:
                skipped_missing_image += 1
                continue
            prompt = " ".join(str(question["question"]).split())
            answer = " ".join(str(ann.get("multiple_choice_answer", "")).split())
            if not prompt or not answer:
                continue
            if _has_reserved_chat_markers(prompt) or _has_reserved_chat_markers(answer):
                skipped_reserved_markers += 1
                continue
            samples.append(
                {
                    "id": f"vqa_train2014_{safe_answer_type}_{question_id}",
                    "image": filename,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{prompt}"},
                        {"from": "gpt", "value": answer},
                    ],
                }
            )

        rng = random.Random(4217 + sum(ord(c) for c in str(answer_type)))
        rng.shuffle(samples)
        samples = samples[: int(max_samples)]
        if not samples:
            raise RuntimeError(
                f"VQAv2 train direct-answer generation produced no {answer_type} samples."
            )

        with open(output_path, "w") as f:
            _json.dump(samples, f)
        print(
            f"Built {len(samples)} VQAv2 train direct-answer {answer_type} samples "
            f"at {output_path}; skipped_missing_image={skipped_missing_image}; "
            f"skipped_reserved_markers={skipped_reserved_markers}"
        )
        volume.commit()
        return output_path, vqa_image_dir

    def _vqa_train_balanced_yes_no_path(max_per_label=40000):
        """Build a balanced yes/no VQAv2 source with canonical yes/no labels."""
        output_path = (
            f"/checkpoints/vqa_data/"
            f"vqa_train2014_direct_yes_no_balanced_{int(max_per_label)}.json"
        )
        if os.path.exists(output_path):
            questions_path, _annotations_path = _ensure_vqa_train_files()
            vqa_image_dir = _ensure_vqa_train_images(questions_path)
            return output_path, vqa_image_dir

        questions_path, annotations_path = _ensure_vqa_train_files()
        vqa_image_dir = _ensure_vqa_train_images(questions_path)
        available_images = {
            f for f in os.listdir(vqa_image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        }

        with open(questions_path) as f:
            questions = {
                int(q["question_id"]): q
                for q in _json.load(f)["questions"]
            }
        with open(annotations_path) as f:
            annotations = _json.load(f)["annotations"]

        buckets = {"yes": [], "no": []}
        skipped_missing_image = 0
        skipped_reserved_markers = 0
        for ann in annotations:
            if str(ann.get("answer_type", "")).lower() != "yes/no":
                continue
            question_id = int(ann["question_id"])
            question = questions.get(question_id)
            if question is None:
                continue
            filename = f"COCO_train2014_{int(question['image_id']):012d}.jpg"
            if filename not in available_images:
                skipped_missing_image += 1
                continue
            prompt = " ".join(str(question["question"]).split())
            answer = " ".join(
                str(ann.get("multiple_choice_answer", "")).lower().split()
            )
            if answer not in buckets or not prompt:
                continue
            if _has_reserved_chat_markers(prompt):
                skipped_reserved_markers += 1
                continue
            buckets[answer].append(
                {
                    "id": f"vqa_train2014_yes_no_balanced_{answer}_{question_id}",
                    "image": filename,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{prompt}"},
                        {"from": "gpt", "value": answer},
                    ],
                }
            )

        rng = random.Random(6211)
        samples = []
        for answer in ("yes", "no"):
            rng.shuffle(buckets[answer])
            samples.extend(buckets[answer][: int(max_per_label)])
        rng.shuffle(samples)
        if not samples:
            raise RuntimeError("Balanced VQAv2 yes/no generation produced no samples.")

        with open(output_path, "w") as f:
            _json.dump(samples, f)
        print(
            f"Built {len(samples)} balanced VQAv2 yes/no samples at {output_path}; "
            f"yes={min(len(buckets['yes']), int(max_per_label))}; "
            f"no={min(len(buckets['no']), int(max_per_label))}; "
            f"skipped_missing_image={skipped_missing_image}; "
            f"skipped_reserved_markers={skipped_reserved_markers}"
        )
        volume.commit()
        return output_path, vqa_image_dir

    def _vqa_counterfactual_grounding_path(max_samples=30000):
        """Build answer-type-balanced blank/wrong-image VQAv2 grounding probes."""
        output_path = (
            f"/checkpoints/vqa_data/"
            f"vqa_train2014_counterfactual_grounding_{int(max_samples)}.json"
        )
        questions_path, annotations_path = _ensure_vqa_train_files()
        vqa_image_dir = _ensure_vqa_train_images(questions_path)
        blank_filename = "anymal_blank_gray_384.jpg"
        blank_path = os.path.join(vqa_image_dir, blank_filename)
        if not os.path.exists(blank_path):
            from PIL import Image

            Image.new("RGB", (384, 384), color=(128, 128, 128)).save(blank_path)
            volume.commit()

        if os.path.exists(output_path):
            return output_path, vqa_image_dir

        available_images = {
            f for f in os.listdir(vqa_image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        with open(questions_path) as f:
            questions = {
                int(q["question_id"]): q
                for q in _json.load(f)["questions"]
            }
        with open(annotations_path) as f:
            annotations = _json.load(f)["annotations"]

        buckets = {"yes/no": [], "number": [], "other": []}
        skipped_missing_image = 0
        skipped_reserved_markers = 0
        for ann in annotations:
            answer_type = str(ann.get("answer_type", "")).lower()
            if answer_type not in buckets:
                continue
            question_id = int(ann["question_id"])
            question = questions.get(question_id)
            if question is None:
                continue
            filename = f"COCO_train2014_{int(question['image_id']):012d}.jpg"
            if filename not in available_images:
                skipped_missing_image += 1
                continue
            prompt = " ".join(str(question["question"]).split())
            if not prompt:
                continue
            if _has_reserved_chat_markers(prompt):
                skipped_reserved_markers += 1
                continue
            buckets[answer_type].append(
                {
                    "question_id": question_id,
                    "image_id": int(question["image_id"]),
                    "image": filename,
                    "prompt": prompt,
                }
            )

        rng = random.Random(7301)
        target_per_type = max(1, int(max_samples) // max(len(buckets), 1))
        samples = []
        for answer_type, rows in buckets.items():
            rng.shuffle(rows)
            if len(rows) < 2:
                continue
            wrong_target = int(round(target_per_type * 0.70))
            blank_target = target_per_type - wrong_target

            for i, row in enumerate(rows[:wrong_target]):
                candidates = [
                    candidate
                    for candidate in rows
                    if candidate["image_id"] != row["image_id"]
                ]
                if not candidates:
                    continue
                wrong = candidates[(i * 7919) % len(candidates)]
                samples.append(
                    {
                        "id": f"vqa_cf_wrong_{answer_type.replace('/', '_')}_{row['question_id']}",
                        "image": wrong["image"],
                        "source_question_id": row["question_id"],
                        "source_image": row["image"],
                        "control_image": wrong["image"],
                        "counterfactual_type": "wrong_image_same_answer_type",
                        "answer_type": answer_type,
                        "conversations": [
                            {
                                "from": "human",
                                "value": (
                                    f"<image>\n{row['prompt']}\n"
                                    "Answer based only on the image. If the answer cannot be "
                                    "determined from the image, answer \"not enough information\"."
                                ),
                            },
                            {"from": "gpt", "value": "not enough information"},
                        ],
                    }
                )

            for row in rows[wrong_target: wrong_target + blank_target]:
                samples.append(
                    {
                        "id": f"vqa_cf_blank_{answer_type.replace('/', '_')}_{row['question_id']}",
                        "image": blank_filename,
                        "source_question_id": row["question_id"],
                        "source_image": row["image"],
                        "counterfactual_type": "blank_image",
                        "answer_type": answer_type,
                        "conversations": [
                            {
                                "from": "human",
                                "value": (
                                    f"<image>\n{row['prompt']}\n"
                                    "Answer based only on the image. If the answer cannot be "
                                    "determined from the image, answer \"not enough information\"."
                                ),
                            },
                            {"from": "gpt", "value": "not enough information"},
                        ],
                    }
                )

        rng.shuffle(samples)
        samples = samples[: int(max_samples)]
        if not samples:
            raise RuntimeError("VQAv2 counterfactual grounding generation produced no samples.")

        with open(output_path, "w") as f:
            _json.dump(samples, f)
        print(
            f"Built {len(samples)} VQAv2 counterfactual grounding samples at {output_path}; "
            f"skipped_missing_image={skipped_missing_image}; "
            f"skipped_reserved_markers={skipped_reserved_markers}"
        )
        volume.commit()
        return output_path, vqa_image_dir

    def _coco_absence_pope_style_path(max_samples=10000):
        """Build POPE-style object absence probes from COCO train2017 instances."""
        output_path = (
            f"/checkpoints/llava_data/"
            f"coco_pope_style_absence_train2017_{int(max_samples)}.json"
        )
        if os.path.exists(output_path):
            return output_path

        instances_path = _ensure_coco_instance_annotations()
        with open(instances_path, "r") as f:
            instances = _json.load(f)

        available_images = {
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        images = {
            image["id"]: image
            for image in instances.get("images", [])
            if image.get("file_name") in available_images
        }
        categories = {
            category["id"]: category["name"]
            for category in instances.get("categories", [])
        }
        all_category_names = sorted(set(categories.values()))
        present_by_image = {image_id: set() for image_id in images}
        for ann in instances.get("annotations", []):
            if ann.get("iscrowd"):
                continue
            image_id = ann.get("image_id")
            if image_id not in present_by_image:
                continue
            category_name = categories.get(ann.get("category_id"))
            if category_name:
                present_by_image[image_id].add(category_name)

        rng = random.Random(7321)
        image_ids = list(images.keys())
        rng.shuffle(image_ids)
        samples = []
        for image_id in image_ids:
            absent = [
                name for name in all_category_names
                if name not in present_by_image.get(image_id, set())
            ]
            if not absent:
                continue
            category = absent[(image_id * 1543) % len(absent)]
            article = "an" if category[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
            samples.append(
                {
                    "id": f"coco_absence_{image_id}_{category.replace(' ', '_')}",
                    "image": images[image_id]["file_name"],
                    "counterfactual_type": "pope_style_object_absence",
                    "absent_category": category,
                    "conversations": [
                        {"from": "human", "value": f"<image>\nIs there {article} {category} in the image?"},
                        {"from": "gpt", "value": "no"},
                    ],
                }
            )
            if len(samples) >= int(max_samples):
                break

        if not samples:
            raise RuntimeError("COCO POPE-style absence generation produced no samples.")

        with open(output_path, "w") as f:
            _json.dump(samples, f)
        print(f"Built {len(samples)} COCO POPE-style absence samples at {output_path}")
        volume.commit()
        return output_path

    def _llava_caption_long_form_path(max_samples=50000):
        output_path = f"/checkpoints/llava_data/llava_caption_long_form_{int(max_samples)}.json"
        if os.path.exists(output_path):
            return output_path

        instruct_path = "/checkpoints/llava_data/llava_instruct_150k.json"
        if not os.path.exists(instruct_path):
            raise RuntimeError(f"LLaVA-Instruct JSON not found at {instruct_path}")

        available_images = {
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        with open(instruct_path, "r") as f:
            raw_data = _json.load(f)

        samples = []
        for sample in raw_data:
            image = sample.get("image")
            if image not in available_images:
                continue
            _question, answer = _first_human_and_gpt(sample)
            if not answer or _has_reserved_chat_markers(answer):
                continue
            if len(answer.split()) < 8:
                continue
            samples.append(
                {
                    "id": f"{sample.get('id', 'sample')}_caption_long",
                    "image": image,
                    "conversations": [
                        {"from": "human", "value": "<image>\nDescribe the image."},
                        {"from": "gpt", "value": answer.strip()},
                    ],
                }
            )

        rng = random.Random(4202)
        rng.shuffle(samples)
        samples = samples[: int(max_samples)]
        if not samples:
            raise RuntimeError("LLaVA caption long-form extraction produced no samples.")

        with open(output_path, "w") as f:
            _json.dump(samples, f)
        print(f"Built {len(samples)} LLaVA long-form caption samples at {output_path}")
        volume.commit()
        return output_path

    if dataset in {"v3_grounding", "v3_grounding_alignment", "v4_grounding", "v4_grounding_alignment"}:
        vqa_direct_path, vqa_image_dir = _vqa_train_direct_answer_path(max_samples=150000)
        coco_direct_path = _coco_object_direct_answer_path()
        short_direct_path = _filtered_direct_answer_path()

        ds = create_instruction_dataset(
            data_path=None,
            image_dir=image_dir,
            tokenizer=tokenizer,
            mixture_config={
                "strategy": "weighted",
                "datasets": [
                    {
                        "name": "vqa_train_direct",
                        "weight": 0.50,
                        "data_path": vqa_direct_path,
                        "image_dir": vqa_image_dir,
                        "system_prompt": training_system_prompt,
                    },
                    {
                        "name": "coco_object_count_color_direct",
                        "weight": 0.30,
                        "data_path": coco_direct_path,
                        "system_prompt": training_system_prompt,
                    },
                    {
                        "name": "short_llava_mix",
                        "weight": 0.20,
                        "data_path": short_direct_path,
                        "system_prompt": training_system_prompt,
                    },
                ],
            },
            image_size=image_size,
            max_length=max_length,
            num_image_tokens=num_image_tokens,
            image_token_policy=image_token_policy,
            min_image_tokens=min_image_tokens,
            max_image_tokens=max_image_tokens,
            vision_encoder_type=vision_encoder_type,
            vision_model_name=vision_model_name,
            filter_to_available_images=True,
        )
        print(f"Loaded {len(ds)} {dataset} instruction samples")
        return ds

    if dataset in {"v3_grounded", "v3_weighted_grounded", "grounded_weighted"}:
        instruct_path = "/checkpoints/llava_data/llava_instruct_150k.json"
        mix_path = _filtered_mix665k_path()
        vqa_direct_path, vqa_image_dir = _vqa_train_direct_answer_path(max_samples=150000)
        coco_direct_path = _coco_object_direct_answer_path()
        short_direct_path = _filtered_direct_answer_path()
        caption_path = _llava_caption_long_form_path()

        ds = create_instruction_dataset(
            data_path=None,
            image_dir=image_dir,
            tokenizer=tokenizer,
            mixture_config={
                "strategy": "weighted",
                "datasets": [
                    {
                        "name": "vqa_train_direct",
                        "weight": 0.30,
                        "data_path": vqa_direct_path,
                        "image_dir": vqa_image_dir,
                        "system_prompt": training_system_prompt,
                    },
                    {
                        "name": "coco_object_count_color_direct",
                        "weight": 0.20,
                        "data_path": coco_direct_path,
                        "system_prompt": training_system_prompt,
                    },
                    {
                        "name": "short_llava_mix",
                        "weight": 0.20,
                        "data_path": short_direct_path,
                        "system_prompt": training_system_prompt,
                    },
                    {
                        "name": "llava_instruct_general",
                        "weight": 0.20,
                        "data_path": mix_path,
                    },
                    {
                        "name": "caption_long_form",
                        "weight": 0.10,
                        "data_path": caption_path,
                    },
                ],
            },
            image_size=image_size,
            max_length=max_length,
            num_image_tokens=num_image_tokens,
            image_token_policy=image_token_policy,
            min_image_tokens=min_image_tokens,
            max_image_tokens=max_image_tokens,
            vision_encoder_type=vision_encoder_type,
            vision_model_name=vision_model_name,
            filter_to_available_images=True,
        )
        print(f"Loaded {len(ds)} V3 weighted grounded instruction samples")
        return ds

    if dataset in {
        "v3_yn_number_focus",
        "v3_answer_type_focus",
        "v3_direct_calibration",
        "v3_yesno_calibrated",
        "v4_answer_type_focus",
        "v4_direct_calibration",
        "v4_semantic_calibration",
        "v5_semantic_calibration",
        "v5_semantic_calibration_robust",
        "v7_semantic_calibration_counterfactual",
    }:
        if dataset in {
            "v5_semantic_calibration",
            "v5_semantic_calibration_robust",
            "v7_semantic_calibration_counterfactual",
        }:
            direct_weights = {
                "yes_no": 0.40,
                "number": 0.20,
                "coco": 0.15,
                "other": 0.20,
                "short": 0.05,
            }
            calibration_prompt = (
                "Answer with only the final answer. Do not include role labels, "
                "explanations, or the word assistant. End after the answer."
            )
            calibration_image_augmentation_mode = (
                "vqa_light"
                if dataset in {"v5_semantic_calibration_robust", "v7_semantic_calibration_counterfactual"}
                else image_augmentation_mode
            )
        elif dataset == "v4_semantic_calibration":
            direct_weights = {
                "yes_no": 0.40,
                "number": 0.20,
                "coco": 0.15,
                "other": 0.20,
                "short": 0.05,
            }
            calibration_prompt = (
                "Answer the image question directly and briefly. End after the answer."
            )
            calibration_image_augmentation_mode = image_augmentation_mode
        elif dataset in {"v3_direct_calibration", "v3_yesno_calibrated", "v4_direct_calibration"}:
            direct_weights = {
                "yes_no": 0.45,
                "number": 0.25,
                "coco": 0.20,
                "other": 0.05,
                "short": 0.05,
            }
            calibration_prompt = (
                "Answer the image question directly and briefly. End after the answer."
            )
            calibration_image_augmentation_mode = image_augmentation_mode
        else:
            # Historical answer-type recipe from the 2026-05-05 fast screen.
            direct_weights = {
                "yes_no": 0.35,
                "number": 0.30,
                "coco": 0.20,
                "other": 0.10,
                "short": 0.05,
            }
            calibration_prompt = training_system_prompt
            calibration_image_augmentation_mode = image_augmentation_mode

        if dataset in {
            "v4_semantic_calibration",
            "v5_semantic_calibration",
            "v5_semantic_calibration_robust",
            "v7_semantic_calibration_counterfactual",
        }:
            vqa_yes_no_path, vqa_image_dir = _vqa_train_balanced_yes_no_path(
                max_per_label=40000,
            )
        else:
            vqa_yes_no_path, vqa_image_dir = _vqa_train_direct_answer_path_by_type(
                "yes/no",
                max_samples=80000,
            )
        vqa_number_path, _ = _vqa_train_direct_answer_path_by_type(
            "number",
            max_samples=50000,
        )
        vqa_other_path, _ = _vqa_train_direct_answer_path_by_type(
            "other",
            max_samples=80000,
        )
        coco_direct_path = _coco_object_direct_answer_path()
        short_direct_path = _filtered_direct_answer_path()
        counterfactual_path, counterfactual_image_dir = (
            _vqa_counterfactual_grounding_path(max_samples=30000)
            if dataset == "v7_semantic_calibration_counterfactual"
            else (None, None)
        )
        absence_path = (
            _coco_absence_pope_style_path(max_samples=10000)
            if dataset == "v7_semantic_calibration_counterfactual"
            else None
        )

        mixture_sources = [
            {
                "name": "vqa_train_yes_no",
                "weight": direct_weights["yes_no"],
                "data_path": vqa_yes_no_path,
                "image_dir": vqa_image_dir,
                "system_prompt": calibration_prompt,
            },
            {
                "name": "vqa_train_number",
                "weight": direct_weights["number"],
                "data_path": vqa_number_path,
                "image_dir": vqa_image_dir,
                "system_prompt": calibration_prompt,
            },
            {
                "name": "coco_object_count_color_direct",
                "weight": direct_weights["coco"],
                "data_path": coco_direct_path,
                "system_prompt": calibration_prompt,
            },
            {
                "name": "vqa_train_other",
                "weight": direct_weights["other"],
                "data_path": vqa_other_path,
                "image_dir": vqa_image_dir,
                "system_prompt": calibration_prompt,
            },
            {
                "name": "short_llava_mix",
                "weight": direct_weights["short"],
                "data_path": short_direct_path,
                "system_prompt": calibration_prompt,
            },
        ]
        if dataset == "v7_semantic_calibration_counterfactual":
            for source in mixture_sources:
                source["weight"] = float(source["weight"]) * 0.80
            mixture_sources.extend(
                [
                    {
                        "name": "vqa_counterfactual_grounding",
                        "weight": 0.15,
                        "data_path": counterfactual_path,
                        "image_dir": counterfactual_image_dir,
                        "system_prompt": calibration_prompt,
                    },
                    {
                        "name": "coco_pope_style_absence",
                        "weight": 0.05,
                        "data_path": absence_path,
                        "system_prompt": calibration_prompt,
                    },
                ]
            )

        ds = create_instruction_dataset(
            data_path=None,
            image_dir=image_dir,
            tokenizer=tokenizer,
            mixture_config={
                "strategy": "weighted",
                "datasets": mixture_sources,
            },
            image_size=image_size,
            max_length=max_length,
            num_image_tokens=num_image_tokens,
            image_token_policy=image_token_policy,
            min_image_tokens=min_image_tokens,
            max_image_tokens=max_image_tokens,
            vision_encoder_type=vision_encoder_type,
            vision_model_name=vision_model_name,
            use_augmentation=(str(calibration_image_augmentation_mode or "none").lower() != "none"),
            image_augmentation_mode=calibration_image_augmentation_mode,
            filter_to_available_images=True,
        )
        print(f"Loaded {len(ds)} {dataset} instruction samples")
        return ds

    if dataset in {"balanced_mix", "balanced_mixture", "mixture"}:
        instruct_path = "/checkpoints/llava_data/llava_instruct_150k.json"
        if not os.path.exists(instruct_path):
            print("Warning: LLaVA JSON not pre-cached, downloading now...")
            from huggingface_hub import hf_hub_download
            cache_dir = "/checkpoints/llava_data"
            os.makedirs(cache_dir, exist_ok=True)
            hf_hub_download(
                repo_id="liuhaotian/LLaVA-Instruct-150K",
                filename="llava_instruct_150k.json",
                repo_type="dataset",
                local_dir=cache_dir,
            )

        mix_path = _filtered_mix665k_path()
        ds = create_instruction_dataset(
            data_path=None,
            image_dir=image_dir,
            tokenizer=tokenizer,
            mixture_config={
                "strategy": "balanced",
                "datasets": [
                    {"name": "instruct_150k", "data_path": instruct_path},
                    {"name": "mix_665k", "data_path": mix_path},
                ],
            },
            image_size=image_size,
            max_length=max_length,
            num_image_tokens=num_image_tokens,
            image_token_policy=image_token_policy,
            min_image_tokens=min_image_tokens,
            max_image_tokens=max_image_tokens,
            vision_encoder_type=vision_encoder_type,
            vision_model_name=vision_model_name,
            filter_to_available_images=True,
        )
        print(f"Loaded {len(ds)} balanced instruction-mixture samples")
        return ds

    if dataset in {
        "balanced_mix_direct",
        "direct_balanced_mix",
        "balanced_mix_direct_trainprompt",
        "direct_balanced_mix_trainprompt",
        "balanced_mix_direct_dualprompt",
        "direct_balanced_mix_dualprompt",
        "balanced_mix_object_trainprompt",
        "object_balanced_mix_trainprompt",
        "balanced_mix_direct_object_trainprompt",
        "direct_object_balanced_mix_trainprompt",
        "balanced_mix_vqa_direct_object_trainprompt",
        "vqa_direct_object_balanced_mix_trainprompt",
        "concat_mix_direct_object_light_trainprompt",
        "light_direct_object_concat_trainprompt",
    }:
        instruct_path = "/checkpoints/llava_data/llava_instruct_150k.json"
        mix_path = _filtered_mix665k_path()
        direct_path = _filtered_direct_answer_path()
        mixture_strategy = "balanced"
        if dataset in {"balanced_mix_direct_trainprompt", "direct_balanced_mix_trainprompt"}:
            direct_datasets = [
                {
                    "name": "mix_665k_direct_trainprompt",
                    "data_path": direct_path,
                    "system_prompt": training_system_prompt,
                }
            ]
        elif dataset in {"balanced_mix_object_trainprompt", "object_balanced_mix_trainprompt"}:
            direct_datasets = [
                {
                    "name": "coco_object_direct_trainprompt",
                    "data_path": _coco_object_direct_answer_path(),
                    "system_prompt": training_system_prompt,
                }
            ]
        elif dataset in {
            "balanced_mix_direct_object_trainprompt",
            "direct_object_balanced_mix_trainprompt",
        }:
            direct_datasets = [
                {
                    "name": "mix_665k_direct_trainprompt",
                    "data_path": direct_path,
                    "system_prompt": training_system_prompt,
                },
                {
                    "name": "coco_object_direct_trainprompt",
                    "data_path": _coco_object_direct_answer_path(),
                    "system_prompt": training_system_prompt,
                },
            ]
        elif dataset in {
            "balanced_mix_vqa_direct_object_trainprompt",
            "vqa_direct_object_balanced_mix_trainprompt",
        }:
            vqa_direct_path, vqa_image_dir = _vqa_train_direct_answer_path(max_samples=150000)
            direct_datasets = [
                {
                    "name": "mix_665k_direct_trainprompt",
                    "data_path": direct_path,
                    "system_prompt": training_system_prompt,
                },
                {
                    "name": "coco_object_direct_trainprompt",
                    "data_path": _coco_object_direct_answer_path(),
                    "system_prompt": training_system_prompt,
                },
                {
                    "name": "vqa_train2014_direct_trainprompt",
                    "data_path": vqa_direct_path,
                    "image_dir": vqa_image_dir,
                    "system_prompt": training_system_prompt,
                },
            ]
        elif dataset in {
            "concat_mix_direct_object_light_trainprompt",
            "light_direct_object_concat_trainprompt",
        }:
            mixture_strategy = "concat"
            direct_datasets = [
                {
                    "name": "mix_665k_direct_trainprompt_light",
                    "data_path": direct_path,
                    "system_prompt": training_system_prompt,
                    "max_samples": 75000,
                    "sample_seed": 2901,
                },
                {
                    "name": "coco_object_direct_trainprompt_light",
                    "data_path": _coco_object_direct_answer_path(),
                    "system_prompt": training_system_prompt,
                    "max_samples": 75000,
                    "sample_seed": 2902,
                },
            ]
        elif dataset in {"balanced_mix_direct_dualprompt", "direct_balanced_mix_dualprompt"}:
            direct_datasets = [
                {
                    "name": "mix_665k_direct_strict",
                    "data_path": direct_path,
                    "system_prompt": direct_system_prompt,
                },
                {
                    "name": "mix_665k_direct_trainprompt",
                    "data_path": direct_path,
                    "system_prompt": training_system_prompt,
                },
            ]
        else:
            direct_datasets = [
                {
                    "name": "mix_665k_direct",
                    "data_path": direct_path,
                    "system_prompt": direct_system_prompt,
                }
            ]
        ds = create_instruction_dataset(
            data_path=None,
            image_dir=image_dir,
            tokenizer=tokenizer,
            mixture_config={
                "strategy": mixture_strategy,
                "datasets": [
                    {"name": "instruct_150k", "data_path": instruct_path},
                    {"name": "mix_665k", "data_path": mix_path},
                    *direct_datasets,
                ],
            },
            image_size=image_size,
            max_length=max_length,
            num_image_tokens=num_image_tokens,
            image_token_policy=image_token_policy,
            min_image_tokens=min_image_tokens,
            max_image_tokens=max_image_tokens,
            vision_encoder_type=vision_encoder_type,
            vision_model_name=vision_model_name,
            filter_to_available_images=True,
        )
        print(f"Loaded {len(ds)} direct-balanced instruction-mixture samples")
        return ds

    if dataset in {
        "direct_answer_mix",
        "mix_665k_direct",
        "direct_answer_trainprompt",
        "mix_665k_direct_trainprompt",
        "coco_object_direct_trainprompt",
        "object_direct_trainprompt",
        "vqa_train_direct",
        "vqa_train_direct_trainprompt",
    }:
        system_prompt = (
            training_system_prompt
            if dataset in {
                "direct_answer_trainprompt",
                "mix_665k_direct_trainprompt",
                "coco_object_direct_trainprompt",
                "object_direct_trainprompt",
                "vqa_train_direct_trainprompt",
            }
            else direct_system_prompt
        )
        if dataset in {"coco_object_direct_trainprompt", "object_direct_trainprompt"}:
            data_path = _coco_object_direct_answer_path()
            direct_image_dir = image_dir
        elif dataset in {"vqa_train_direct", "vqa_train_direct_trainprompt"}:
            data_path, direct_image_dir = _vqa_train_direct_answer_path(max_samples=150000)
        else:
            data_path = _filtered_direct_answer_path()
            direct_image_dir = image_dir
        ds = create_instruction_dataset(
            data_path=data_path,
            image_dir=direct_image_dir,
            tokenizer=tokenizer,
            image_size=image_size,
            max_length=max_length,
            num_image_tokens=num_image_tokens,
            image_token_policy=image_token_policy,
            min_image_tokens=min_image_tokens,
            max_image_tokens=max_image_tokens,
            vision_encoder_type=vision_encoder_type,
            vision_model_name=vision_model_name,
            system_prompt=system_prompt,
            filter_to_available_images=True,
        )
        print(f"Loaded {len(ds)} direct-answer instruction samples")
        return ds

    if dataset == "mix_665k":
        ds = _build_instruction_dataset(_filtered_mix665k_path())
        print(f"Loaded {len(ds)} multi-task instruction samples")
        return ds

    else:
        # Default: LLaVA-Instruct-150K
        json_path = "/checkpoints/llava_data/llava_instruct_150k.json"

        if not os.path.exists(json_path):
            print("Warning: LLaVA JSON not pre-cached, downloading now...")
            from huggingface_hub import hf_hub_download
            cache_dir = "/checkpoints/llava_data"
            os.makedirs(cache_dir, exist_ok=True)
            hf_hub_download(
                repo_id="liuhaotian/LLaVA-Instruct-150K",
                filename="llava_instruct_150k.json",
                repo_type="dataset",
                local_dir=cache_dir,
            )

        print(f"Loading LLaVA-Instruct-150K from {json_path}")
        print("Filtering dataset to only samples with real images")

        ds = _build_instruction_dataset(json_path)

        if len(ds) == 0:
            raise RuntimeError(
                f"Dataset is empty after filtering. No LLaVA samples matched the "
                f"{num_images} available COCO images. Check image filenames."
            )

        print(f"Loaded {len(ds)} instruction samples with real images")
        return ds


class COCOCaptionDataset:
    """Caption dataset with real COCO images for Stage 1 pretraining.

    Defined at module level so it can be pickled by DataLoader workers.
    """

    def __init__(
        self,
        samples,
        image_dir,
        tokenizer,
        insert_image_placeholders: bool = False,
        num_image_tokens: int = 64,
        max_length: int = 256,
        image_size: int = 224,
        vision_encoder_type: str = "clip",
        vision_model_name: str = None,
    ):
        from data.data_utils import get_vision_transform, TextProcessor
        self.samples = samples
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = get_vision_transform(
            vision_encoder_type=vision_encoder_type,
            vision_model_name=vision_model_name,
            image_size=image_size,
            is_train=True,
        )
        self.max_length = max_length
        self.text_processor = TextProcessor(tokenizer, max_length=max_length)
        self.insert_image_placeholders = insert_image_placeholders
        self.num_image_tokens = num_image_tokens
        self.image_placeholder_token_id = self._resolve_placeholder_token_id()

    def _resolve_placeholder_token_id(self):
        if not self.insert_image_placeholders:
            return None
        vocab = self.tokenizer.get_vocab()
        if "<|reserved_special_token_0|>" in vocab:
            return vocab["<|reserved_special_token_0|>"]
        if "<|image|>" in vocab:
            return vocab["<|image|>"]
        if "<image>" in vocab:
            return vocab["<image>"]
        raise ValueError(
            "insert_image_placeholders=True but tokenizer has no placeholder token."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        import torch
        item = self.samples[idx]

        # Load real image
        image_path = os.path.join(self.image_dir, item["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        caption = item["caption"]
        encoding = self.text_processor.encode_for_training(
            caption,
            response_start_idx=[(0, len(caption))],
        )
        labels = encoding["labels"]

        if self.insert_image_placeholders:
            placeholder_block = torch.full(
                (self.num_image_tokens,),
                self.image_placeholder_token_id,
                dtype=encoding["input_ids"].dtype,
            )
            placeholder_mask = torch.ones(self.num_image_tokens, dtype=encoding["attention_mask"].dtype)
            placeholder_labels = torch.full((self.num_image_tokens,), -100, dtype=labels.dtype)

            input_ids = torch.cat([placeholder_block, encoding["input_ids"]], dim=0)
            attention_mask = torch.cat([placeholder_mask, encoding["attention_mask"]], dim=0)
            labels = torch.cat([placeholder_labels, labels], dim=0)

            max_len = self.max_length
            if input_ids.shape[0] > max_len:
                input_ids = input_ids[:max_len]
                attention_mask = attention_mask[:max_len]
                labels = labels[:max_len]
            elif input_ids.shape[0] < max_len:
                pad_len = max_len - input_ids.shape[0]
                input_ids = torch.cat(
                    [
                        input_ids,
                        torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids.dtype),
                    ],
                    dim=0,
                )
                attention_mask = torch.cat(
                    [attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)],
                    dim=0,
                )
                labels = torch.cat(
                    [labels, torch.full((pad_len,), -100, dtype=labels.dtype)],
                    dim=0,
                )
        else:
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _resolve_llava_pretrain_image_dir(annotation_path: str, pretrain_dir: str):
    """
    Pick the LLaVA-Pretrain image root that matches annotation image refs.

    Some copies of images.zip extract to `images/<shard>/<file>.jpg`, while
    others extract shard directories directly under the target directory.
    """
    import json

    if not os.path.exists(annotation_path) or not os.path.isdir(pretrain_dir):
        return None

    candidates = [
        os.path.join(pretrain_dir, "images"),
        pretrain_dir,
    ]

    refs = []
    try:
        with open(annotation_path, "r") as f:
            records = json.load(f)
        for item in records[:1000]:
            image_ref = item.get("image_path", item.get("image", item.get("file_name", "")))
            if image_ref:
                refs.append(str(image_ref))
            if len(refs) >= 100:
                break
    except Exception as exc:
        print(f"Could not inspect LLaVA-Pretrain annotations for image root: {exc}")

    if refs:
        for candidate in candidates:
            if os.path.isdir(candidate) and any(os.path.exists(os.path.join(candidate, ref)) for ref in refs):
                return candidate

    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        for _root, _dirs, files in os.walk(candidate):
            if any(f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")) for f in files):
                return candidate

    return None


def _resolve_llava_pretrain_zip_path(pretrain_dir: str):
    """Return a cached LLaVA-Pretrain images.zip path if one exists."""
    candidates = [
        os.path.join(pretrain_dir, "images.zip"),
        os.path.join(pretrain_dir, "images", "images.zip"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def load_llava_pretrain_dataset(
    tokenizer,
    insert_image_placeholders: bool = False,
    num_image_tokens: int = 64,
    max_length: int = 256,
    image_size: int = 224,
    vision_encoder_type: str = "clip",
    vision_model_name: str = None,
):
    """
    Load captioning dataset for Stage 1 alignment using real images.

    Prefer true LLaVA-Pretrain caption data if both annotations and images have
    been staged. Fall back to the existing COCO-backed instruction-caption
    extraction path so current Modal runs remain executable.
    """
    pretrain_json_path = "/checkpoints/llava_data/blip_laion_cc_sbu_558k.json"
    pretrain_dir = "/checkpoints/llava_pretrain"
    pretrain_zip_path = _resolve_llava_pretrain_zip_path(pretrain_dir)
    pretrain_image_dir = _resolve_llava_pretrain_image_dir(pretrain_json_path, pretrain_dir)
    pretrain_manifest_path = "/checkpoints/llava_pretrain/manifest.json"
    if (
        os.path.exists(pretrain_json_path)
        and (pretrain_zip_path or (pretrain_image_dir and os.path.isdir(pretrain_image_dir)))
        and os.path.exists(pretrain_manifest_path)
    ):
        from data.laion_dataset import create_laion_dataset

        print(f"Loading true LLaVA-Pretrain captions from {pretrain_json_path}")
        if pretrain_zip_path:
            print(f"Reading true LLaVA-Pretrain images directly from zip: {pretrain_zip_path}")
        else:
            print(f"Reading true LLaVA-Pretrain images from directory: {pretrain_image_dir}")
        dataset = create_laion_dataset(
            data_path=pretrain_json_path,
            image_dir=pretrain_image_dir,
            image_zip_path=pretrain_zip_path,
            tokenizer=tokenizer,
            dataset_type="llava_pretrain_caption",
            caption_prompt="",
            filter_to_available_images=True,
            min_caption_chars=3,
            deduplicate_captions=True,
            insert_image_placeholders=insert_image_placeholders,
            num_image_tokens=num_image_tokens,
            max_length=max_length,
            image_size=image_size,
            vision_encoder_type=vision_encoder_type,
            vision_model_name=vision_model_name,
        )
        if len(dataset) > 0:
            print(f"Loaded {len(dataset)} true LLaVA-Pretrain samples with real images")
            return dataset
        print("True LLaVA-Pretrain path produced 0 usable samples; falling back to COCO captions")

    json_path = "/checkpoints/llava_data/llava_instruct_150k.json"
    image_dir = "/checkpoints/coco_images"

    if not os.path.exists(json_path):
        raise RuntimeError(
            f"LLaVA JSON not found at {json_path}. "
            "Run training setup first to cache data."
        )
    if not os.path.exists(image_dir):
        raise RuntimeError(
            f"COCO images not found at {image_dir}. "
            "Run training setup first to cache images."
        )

    print(f"Loading pretrain captions from {json_path}")

    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Build set of available images
    available_images = set(f for f in os.listdir(image_dir) if f.endswith('.jpg'))
    print(f"Found {len(available_images)} COCO images in {image_dir}")

    # Filter to samples with available images and extract captions
    samples = []
    for item in raw_data:
        img_name = item.get("image", "")
        if img_name not in available_images:
            continue
        # Extract first GPT response as caption
        conversations = item.get("conversations", [])
        caption = ""
        for conv in conversations:
            if conv.get("from") == "gpt":
                caption = conv.get("value", "")
                break
        if not caption:
            continue
        samples.append({"image": img_name, "caption": caption})

    print(f"Filtered to {len(samples)} samples with real images")
    if len(samples) == 0:
        raise RuntimeError("No pretrain samples matched available COCO images.")

    dataset = COCOCaptionDataset(
        samples=samples,
        image_dir=image_dir,
        tokenizer=tokenizer,
        insert_image_placeholders=insert_image_placeholders,
        num_image_tokens=num_image_tokens,
        max_length=max_length,
        image_size=image_size,
        vision_encoder_type=vision_encoder_type,
        vision_model_name=vision_model_name,
    )
    print(f"Loaded {len(dataset)} pretrain samples with real COCO images")
    return dataset


def create_dummy_instruction_dataset(tokenizer, num_samples=1000):
    """Create a dummy instruction dataset for testing."""
    import torch
    from torch.utils.data import Dataset
    from data.data_utils import get_image_transform, TextProcessor

    class DummyInstructionDataset(Dataset):
        def __init__(self, tokenizer, num_samples):
            self.tokenizer = tokenizer
            self.num_samples = num_samples
            self.transform = get_image_transform(image_size=224, is_train=True)
            self.text_processor = TextProcessor(tokenizer, max_length=512)
            from data.data_utils import CLIP_MEAN, CLIP_STD
            from torchvision import transforms as T
            self._clip_normalize = T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            image = self._clip_normalize(torch.rand(3, 224, 224))
            conversations = [
                {"role": "user", "content": f"What do you see in this image? (sample {idx})"},
                {"role": "assistant", "content": "I see various objects in this image."},
            ]
            text, response_start = self.text_processor.format_conversation(conversations)
            encoding = self.text_processor.encode_for_training(text, response_start)
            return {
                "image": image,
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": encoding["labels"],
            }

    return DummyInstructionDataset(tokenizer, num_samples)


def create_dummy_caption_dataset(tokenizer, num_samples=1000):
    """Create a dummy caption dataset for testing."""
    import torch
    from torch.utils.data import Dataset
    from data.data_utils import get_image_transform, TextProcessor

    class DummyCaptionDataset(Dataset):
        def __init__(self, tokenizer, num_samples):
            self.tokenizer = tokenizer
            self.num_samples = num_samples
            self.text_processor = TextProcessor(tokenizer, max_length=256)
            from data.data_utils import CLIP_MEAN, CLIP_STD
            from torchvision import transforms as T
            self._clip_normalize = T.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            image = self._clip_normalize(torch.rand(3, 224, 224))
            caption = f"A photograph showing various objects, sample {idx}."
            encoding = self.text_processor.encode_text(caption)
            labels = encoding["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            return {
                "image": image,
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": labels,
            }

    return DummyCaptionDataset(tokenizer, num_samples)


def _pretrain_worker(local_rank, world_size, config):
    """Worker function for distributed Stage 1 pretraining."""
    import sys
    sys.path.insert(0, "/root/anymal")

    # Set environment variables for distributed training
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    import torch
    torch.cuda.set_device(local_rank)

    from training.distributed import setup_distributed, cleanup_distributed

    setup_distributed()

    try:
        run_pretrain(
            llama_path=config["llama_path"],
            llm_backbone=config.get("llm_backbone", config["llama_path"]),
            architecture=config.get("architecture", "anymal_v1"),
            max_steps=config["max_steps"],
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            use_wandb=config["use_wandb"] and local_rank == 0,  # Only rank 0 logs
            token_compressor_type=config.get("token_compressor_type", "learned"),
            distributed=True,
            resume_checkpoint=config.get("resume_checkpoint"),
            pretrain_checkpoint=config.get("pretrain_checkpoint"),
            output_dir=config["output_dir"],
            run_name=config.get("run_name"),
            pretrain_image_tokens=config.get("pretrain_image_tokens"),
            dataset=config.get("dataset", "llava_pretrain"),
            connector_warmup_steps=config.get("connector_warmup_steps", 0),
            pretrain_loss_scale=config.get("pretrain_loss_scale", 1.0),
            pretrain_loss_normalization=config.get("pretrain_loss_normalization", "mean"),
            pretrain_loss_normalization_target_tokens=config.get(
                "pretrain_loss_normalization_target_tokens",
                8.0,
            ),
            pretrain_gradient_accumulation_steps=config.get("pretrain_gradient_accumulation_steps", 8),
            pretrain_save_steps=config.get("pretrain_save_steps"),
            v3_connector_type=config.get("v3_connector_type", V3_DEFAULT_CONNECTOR_TYPE),
            v4_global_image_tokens=config.get("v4_global_image_tokens"),
            v4_local_image_tokens=config.get("v4_local_image_tokens"),
            v4_connector_layers=config.get("v4_connector_layers"),
            v4_connector_heads=config.get("v4_connector_heads"),
            v4_connector_ff_mult=config.get("v4_connector_ff_mult"),
            v4_connector_hidden_dim=config.get("v4_connector_hidden_dim"),
            v4_connector_output_scale=config.get("v4_connector_output_scale"),
            v4_connector_output_gate_init=config.get("v4_connector_output_gate_init"),
            v4_use_2d_position_features=config.get("v4_use_2d_position_features", True),
            v4_connector_type=config.get("v4_connector_type"),
            v4_deepstack_num_feature_levels=config.get("v4_deepstack_num_feature_levels"),
            v4_deepstack_hidden_state_indices=config.get("v4_deepstack_hidden_state_indices"),
        )
    finally:
        cleanup_distributed()


def _run_pretrain_distributed(
    max_steps,
    learning_rate,
    batch_size,
    use_wandb,
    architecture="anymal_v1",
    token_compressor_type="learned",
    wandb_api_key=None,
    resume_checkpoint=None,
    pretrain_checkpoint=None,
    run_name=None,
    pretrain_image_tokens=None,
    dataset="llava_pretrain",
    connector_warmup_steps=0,
    pretrain_loss_scale=1.0,
    pretrain_loss_normalization="mean",
    pretrain_loss_normalization_target_tokens=8.0,
    pretrain_gradient_accumulation_steps=8,
    pretrain_save_steps=None,
    v3_connector_type=V3_DEFAULT_CONNECTOR_TYPE,
    v4_global_image_tokens=None,
    v4_local_image_tokens=None,
    v4_connector_layers=None,
    v4_connector_heads=None,
    v4_connector_ff_mult=None,
    v4_connector_hidden_dim=None,
    v4_connector_output_scale=None,
    v4_connector_output_gate_init=None,
    v4_use_2d_position_features=True,
    v4_connector_type=None,
    v4_deepstack_num_feature_levels=None,
    v4_deepstack_hidden_state_indices=None,
    llm_backbone=CURRENT_LLAMA3_BACKBONE,
):
    """Shared implementation for Stage 1 distributed pretraining."""
    import sys
    sys.path.insert(0, "/root/anymal")
    import torch
    import torch.multiprocessing as mp

    # Setup W&B
    if use_wandb:
        import wandb
        api_key = wandb_api_key or os.environ.get("WANDB_API_KEY")
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
        else:
            print("WARNING: use_wandb=True but no WANDB_API_KEY found. Disabling W&B.")
            use_wandb = False

    llama_path = _ensure_llm_backbone_cached(llm_backbone)
    resolved_llm_backbone = _metadata_llm_backbone(llm_backbone)

    num_gpus = torch.cuda.device_count()
    print(f"Starting distributed pretraining on {num_gpus} GPUs")

    pretrain_output_dir = _resolve_run_output_dir(
        base_dir="/checkpoints/pretrain-output",
        resume_checkpoint=resume_checkpoint,
        prefix="run",
        run_name=run_name,
    )
    print(f"Pretrain checkpoints will be written to: {pretrain_output_dir}")

    config = {
        "llama_path": llama_path,
        "llm_backbone": resolved_llm_backbone,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "use_wandb": use_wandb,
        "architecture": architecture,
        "token_compressor_type": token_compressor_type,
        "resume_checkpoint": resume_checkpoint,
        "pretrain_checkpoint": pretrain_checkpoint,
        "output_dir": pretrain_output_dir,
        "run_name": run_name,
        "pretrain_image_tokens": pretrain_image_tokens,
        "dataset": dataset,
        "connector_warmup_steps": connector_warmup_steps,
        "pretrain_loss_scale": pretrain_loss_scale,
        "pretrain_loss_normalization": pretrain_loss_normalization,
        "pretrain_loss_normalization_target_tokens": pretrain_loss_normalization_target_tokens,
        "pretrain_gradient_accumulation_steps": pretrain_gradient_accumulation_steps,
        "pretrain_save_steps": pretrain_save_steps,
        "v3_connector_type": v3_connector_type,
        "v4_global_image_tokens": v4_global_image_tokens,
        "v4_local_image_tokens": v4_local_image_tokens,
        "v4_connector_layers": v4_connector_layers,
        "v4_connector_heads": v4_connector_heads,
        "v4_connector_ff_mult": v4_connector_ff_mult,
        "v4_connector_hidden_dim": v4_connector_hidden_dim,
        "v4_connector_output_scale": v4_connector_output_scale,
        "v4_connector_output_gate_init": v4_connector_output_gate_init,
        "v4_use_2d_position_features": v4_use_2d_position_features,
        "v4_connector_type": v4_connector_type,
        "v4_deepstack_num_feature_levels": v4_deepstack_num_feature_levels,
        "v4_deepstack_hidden_state_indices": v4_deepstack_hidden_state_indices,
    }

    mp.spawn(_pretrain_worker, nprocs=num_gpus, args=(num_gpus, config))

    volume.commit()
    print("Distributed pretraining complete! Outputs saved to volume.")


@app.function(
    image=image,
    gpu="A100-80GB:4",
    timeout=STAGE1_PRETRAIN_TIMEOUT_SECONDS,
    volumes={"/checkpoints": volume},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def pretrain_distributed(
    max_steps,
    learning_rate,
    batch_size,
    use_wandb,
    architecture="anymal_v1",
    token_compressor_type="learned",
    wandb_api_key=None,
    resume_checkpoint=None,
    pretrain_checkpoint=None,
    run_name=None,
    pretrain_image_tokens=None,
    dataset="llava_pretrain",
    connector_warmup_steps=0,
    pretrain_loss_scale=1.0,
    pretrain_loss_normalization="mean",
    pretrain_loss_normalization_target_tokens=8.0,
    pretrain_gradient_accumulation_steps=8,
    pretrain_save_steps=None,
    v3_connector_type=V3_DEFAULT_CONNECTOR_TYPE,
    v4_global_image_tokens=None,
    v4_local_image_tokens=None,
    v4_connector_layers=None,
    v4_connector_heads=None,
    v4_connector_ff_mult=None,
    v4_connector_hidden_dim=None,
    v4_connector_output_scale=None,
    v4_connector_output_gate_init=None,
    v4_use_2d_position_features=True,
    v4_connector_type=None,
    v4_deepstack_num_feature_levels=None,
    v4_deepstack_hidden_state_indices=None,
    llm_backbone=CURRENT_LLAMA3_BACKBONE,
):
    """Run Stage 1 pretraining on 4x A100-80GB using DDP."""
    return _run_pretrain_distributed(
        max_steps=max_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_wandb=use_wandb,
        architecture=architecture,
        token_compressor_type=token_compressor_type,
        wandb_api_key=wandb_api_key,
        resume_checkpoint=resume_checkpoint,
        pretrain_checkpoint=pretrain_checkpoint,
        run_name=run_name,
        pretrain_image_tokens=pretrain_image_tokens,
        dataset=dataset,
        connector_warmup_steps=connector_warmup_steps,
        pretrain_loss_scale=pretrain_loss_scale,
        pretrain_loss_normalization=pretrain_loss_normalization,
        pretrain_loss_normalization_target_tokens=pretrain_loss_normalization_target_tokens,
        pretrain_gradient_accumulation_steps=pretrain_gradient_accumulation_steps,
        pretrain_save_steps=pretrain_save_steps,
        v3_connector_type=v3_connector_type,
        v4_global_image_tokens=v4_global_image_tokens,
        v4_local_image_tokens=v4_local_image_tokens,
        v4_connector_layers=v4_connector_layers,
        v4_connector_heads=v4_connector_heads,
        v4_connector_ff_mult=v4_connector_ff_mult,
        v4_connector_hidden_dim=v4_connector_hidden_dim,
        v4_connector_output_scale=v4_connector_output_scale,
        v4_connector_output_gate_init=v4_connector_output_gate_init,
        v4_use_2d_position_features=v4_use_2d_position_features,
        v4_connector_type=v4_connector_type,
        v4_deepstack_num_feature_levels=v4_deepstack_num_feature_levels,
        v4_deepstack_hidden_state_indices=v4_deepstack_hidden_state_indices,
        llm_backbone=llm_backbone,
    )


@app.function(
    image=image,
    gpu="H100:4",
    timeout=STAGE1_PRETRAIN_TIMEOUT_SECONDS,
    volumes={"/checkpoints": volume},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def pretrain_distributed_h100(
    max_steps,
    learning_rate,
    batch_size,
    use_wandb,
    architecture="anymal_v1",
    token_compressor_type="learned",
    wandb_api_key=None,
    resume_checkpoint=None,
    pretrain_checkpoint=None,
    run_name=None,
    pretrain_image_tokens=None,
    dataset="llava_pretrain",
    connector_warmup_steps=0,
    pretrain_loss_scale=1.0,
    pretrain_loss_normalization="mean",
    pretrain_loss_normalization_target_tokens=8.0,
    pretrain_gradient_accumulation_steps=8,
    pretrain_save_steps=None,
    v3_connector_type=V3_DEFAULT_CONNECTOR_TYPE,
    v4_global_image_tokens=None,
    v4_local_image_tokens=None,
    v4_connector_layers=None,
    v4_connector_heads=None,
    v4_connector_ff_mult=None,
    v4_connector_hidden_dim=None,
    v4_connector_output_scale=None,
    v4_connector_output_gate_init=None,
    v4_use_2d_position_features=True,
    v4_connector_type=None,
    v4_deepstack_num_feature_levels=None,
    v4_deepstack_hidden_state_indices=None,
    llm_backbone=CURRENT_LLAMA3_BACKBONE,
):
    """Run Stage 1 pretraining on 4x H100 using DDP."""
    return _run_pretrain_distributed(
        max_steps=max_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_wandb=use_wandb,
        architecture=architecture,
        token_compressor_type=token_compressor_type,
        wandb_api_key=wandb_api_key,
        resume_checkpoint=resume_checkpoint,
        pretrain_checkpoint=pretrain_checkpoint,
        run_name=run_name,
        pretrain_image_tokens=pretrain_image_tokens,
        dataset=dataset,
        connector_warmup_steps=connector_warmup_steps,
        pretrain_loss_scale=pretrain_loss_scale,
        pretrain_loss_normalization=pretrain_loss_normalization,
        pretrain_loss_normalization_target_tokens=pretrain_loss_normalization_target_tokens,
        pretrain_gradient_accumulation_steps=pretrain_gradient_accumulation_steps,
        pretrain_save_steps=pretrain_save_steps,
        v3_connector_type=v3_connector_type,
        v4_global_image_tokens=v4_global_image_tokens,
        v4_local_image_tokens=v4_local_image_tokens,
        v4_connector_layers=v4_connector_layers,
        v4_connector_heads=v4_connector_heads,
        v4_connector_ff_mult=v4_connector_ff_mult,
        v4_connector_hidden_dim=v4_connector_hidden_dim,
        v4_connector_output_scale=v4_connector_output_scale,
        v4_connector_output_gate_init=v4_connector_output_gate_init,
        v4_use_2d_position_features=v4_use_2d_position_features,
        v4_connector_type=v4_connector_type,
        v4_deepstack_num_feature_levels=v4_deepstack_num_feature_levels,
        v4_deepstack_hidden_state_indices=v4_deepstack_hidden_state_indices,
        llm_backbone=llm_backbone,
    )


@app.function(
    image=image,
    timeout=14400,
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def stage_llava_pretrain_images(force: bool = False):
    """
    Stage true LLaVA-Pretrain annotations and images into the Modal volume.

    This is intentionally separate from normal training setup because images.zip
    is large and Stage 2 runs do not need it.
    """
    import os
    import json
    import zipfile
    from huggingface_hub import hf_hub_download

    data_dir = "/checkpoints/llava_data"
    pretrain_dir = "/checkpoints/llava_pretrain"
    pretrain_json_path = os.path.join(data_dir, "blip_laion_cc_sbu_558k.json")
    manifest_path = os.path.join(pretrain_dir, "manifest.json")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(pretrain_dir, exist_ok=True)

    def _count_images(root):
        total = 0
        for _dirpath, _dirnames, filenames in os.walk(root):
            total += sum(
                1 for f in filenames
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            )
        return total

    def _count_zip_images(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            return sum(
                1 for name in zf.namelist()
                if not name.endswith("/")
                and name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            )

    print("Staging LLaVA-Pretrain annotation JSON...")
    hf_hub_download(
        repo_id="liuhaotian/LLaVA-Pretrain",
        filename="blip_laion_cc_sbu_558k.json",
        repo_type="dataset",
        local_dir=data_dir,
    )

    zip_path = _resolve_llava_pretrain_zip_path(pretrain_dir)
    if zip_path and os.path.exists(manifest_path) and not force:
        num_images = _count_zip_images(zip_path)
        print(f"Using existing LLaVA-Pretrain image zip {zip_path}: {num_images:,} files")
        return {
            "image_zip_path": zip_path,
            "num_images": num_images,
            "storage": "zip",
            "skipped": True,
        }

    image_dir = _resolve_llava_pretrain_image_dir(pretrain_json_path, pretrain_dir)
    existing_images = _count_images(image_dir) if image_dir else 0
    if existing_images and os.path.exists(manifest_path) and not force:
        print(f"Using existing LLaVA-Pretrain images in {image_dir}: {existing_images:,} files")
        return {
            "image_dir": image_dir,
            "num_images": existing_images,
            "storage": "directory",
            "skipped": True,
        }
    if existing_images and not os.path.exists(manifest_path):
        print(
            f"Found {existing_images:,} partially staged images in {image_dir}; "
            "continuing extraction before marking data complete."
        )

    print("Downloading LLaVA-Pretrain images.zip...")
    zip_path = hf_hub_download(
        repo_id="liuhaotian/LLaVA-Pretrain",
        filename="images.zip",
        repo_type="dataset",
        local_dir=pretrain_dir,
    )

    num_images = _count_zip_images(zip_path)
    if num_images == 0:
        raise RuntimeError(f"No images found inside {zip_path}")

    with open(manifest_path, "w") as f:
        json.dump(
            {
                "image_zip_path": zip_path,
                "num_images": num_images,
                "storage": "zip",
            },
            f,
        )
        f.write("\n")

    volume.commit()
    print(f"Staged {num_images:,} LLaVA-Pretrain images as one zip: {zip_path}")
    return {
        "image_zip_path": zip_path,
        "num_images": num_images,
        "storage": "zip",
        "skipped": False,
    }


@app.function(
    image=image,
    timeout=14400,
    volumes={"/checkpoints": volume},
)
def cleanup_failed_llava_pretrain_staging():
    """Remove only the failed LLaVA-Pretrain image staging payload."""
    import os
    import shutil

    pretrain_dir = "/checkpoints/llava_pretrain"
    if not os.path.exists(pretrain_dir):
        print(f"No LLaVA-Pretrain staging directory found at {pretrain_dir}")
        return {"removed": False}

    shutil.rmtree(pretrain_dir)
    print(f"Removed failed LLaVA-Pretrain staging directory: {pretrain_dir}")

    volume.commit()
    return {"removed": pretrain_dir}


@app.local_entrypoint()
def main(
    max_steps: int = 100,
    stage: str = "finetune",
    architecture: str = "anymal_v1",
    token_compressor_type: str = "learned",
    gpu_type: str = "a100",
    learning_rate: float = None,
    lora_learning_rate: float = None,
    batch_size: int = 4,
    use_wandb: bool = False,
    use_dummy_data: bool = False,
    wandb_api_key: str = None,
    track_per_layer_grad_norms: bool = True,
    run_eval_benchmarks: bool = False,
    pretrain_checkpoint: str = None,
    finetune_checkpoint: str = None,
    resume_checkpoint: str = None,
    dataset: str = "instruct_150k",
    run_name: str = None,
    background: bool = False,
    pretrain_image_tokens: int = None,
    projector_warmup_steps: int = None,
    freeze_connector: bool = False,
    finetune_loss_scale: float = 1.0,
    finetune_gradient_accumulation_steps: int = 8,
    connector_warmup_steps: int = 0,
    pretrain_loss_scale: float = 1.0,
    pretrain_loss_normalization: str = "mean",
    pretrain_loss_normalization_target_tokens: float = 8.0,
    pretrain_gradient_accumulation_steps: int = 8,
    pretrain_save_steps: int = None,
    v3_connector_type: str = V3_DEFAULT_CONNECTOR_TYPE,
    v4_global_image_tokens: int = None,
    v4_local_image_tokens: int = None,
    v4_connector_layers: int = None,
    v4_connector_heads: int = None,
    v4_connector_ff_mult: int = None,
    v4_connector_hidden_dim: int = None,
    v4_connector_output_scale: float = None,
    v4_connector_output_gate_init: float = None,
    v4_use_2d_position_features: bool = True,
    v4_connector_type: str = None,
    v4_deepstack_num_feature_levels: int = None,
    v4_deepstack_hidden_state_indices: str = None,
    llm_backbone: str = CURRENT_LLAMA3_BACKBONE,
):
    """
    Entry point for Modal training.

    Examples:
        modal run modal_train.py                              # Quick test with LLaVA data
        modal run modal_train.py --use-dummy-data             # Test with dummy data
        modal run modal_train.py --max-steps 500              # Longer run
        modal run modal_train.py --stage pretrain             # Stage 1 (4 GPUs)
        modal run modal_train.py --stage pretrain --gpu-type h100
        modal run modal_train.py --stage finetune             # Stage 2 (auto-discovers pretrain ckpt)
        modal run modal_train.py --architecture anymal_v2 --token-compressor-type perceiver --stage pretrain
        modal run modal_train.py --architecture anymal_v3 --stage pretrain --max-steps 20000
        modal run modal_train.py --architecture anymal_v3 --llm-backbone Qwen/Qwen3-8B --stage pretrain --max-steps 20
        modal run modal_train.py --use-wandb --wandb-api-key YOUR_KEY  # With W&B
        modal run modal_train.py --stage pretrain --resume-checkpoint /checkpoints/pretrain-output/run-0001/checkpoint-250
        modal run modal_train.py --stage finetune --learning-rate 2e-5 --lora-learning-rate 2e-4 --dataset mix_665k
        modal run --detach modal_train.py --stage finetune --max-steps 3000
    """
    selected_gpu = _resolve_modal_gpu(stage, gpu_type)
    arch_key = _normalize_architecture_key(architecture)
    resolved_llm_backbone = _metadata_llm_backbone(llm_backbone)
    if pretrain_image_tokens is None and (stage == "pretrain" or arch_key != "anymal_v4"):
        pretrain_image_tokens = 128 if arch_key in {"anymal_v3", "anymal_v4"} else 256
    if arch_key in {"anymal_v3", "anymal_v4"} and dataset == "instruct_150k":
        dataset = "v3_grounded"
    if arch_key in {"anymal_v3", "anymal_v4"} and dataset in {
        "v3_direct_calibration",
        "v3_yesno_calibrated",
        "v4_direct_calibration",
        "v4_semantic_calibration",
        "v5_semantic_calibration",
        "v5_semantic_calibration_robust",
        "v7_semantic_calibration_counterfactual",
    }:
        freeze_connector = True
    resolved_v4_connector_type = v4_connector_type or V4_DEFAULT_CONNECTOR_TYPE
    resolved_deepstack_layers = _parse_v4_hidden_state_indices(
        v4_deepstack_hidden_state_indices
    )
    resolved_deepstack_levels = (
        int(v4_deepstack_num_feature_levels)
        if v4_deepstack_num_feature_levels is not None
        else (len(resolved_deepstack_layers) if resolved_deepstack_layers else None)
    )
    token_compressor_type = str(token_compressor_type).strip().lower()
    if token_compressor_type not in {"learned", "perceiver", "perceiver2", "avg"}:
        raise ValueError(
            "--token-compressor-type must be one of: learned, perceiver, perceiver2, avg"
        )

    # Generate unique run name locally BEFORE any .remote() call. All containers
    # receive the same name and write to the same directory — avoids a race
    # where parallel invocations each try to claim e.g. run-0010 on a Modal
    # Volume (which has eventual consistency, so mkdir atomicity doesn't hold
    # across containers).
    if run_name is None:
        run_name = _generate_unique_run_name(prefix="run")

    print(f"Starting AnyMAL training on Modal...")
    print(f"  Stage: {stage}")
    print(f"  Run name: {run_name}")
    print(f"  Architecture: {arch_key}")
    print(f"  LLM backbone: {resolved_llm_backbone}")
    print(f"  Token compressor: {token_compressor_type}")
    if arch_key == "anymal_v3":
        print(f"  V3 connector type: {v3_connector_type or V3_DEFAULT_CONNECTOR_TYPE}")
    print(f"  GPU type: {gpu_type}")
    print(f"  Modal GPU resource: {selected_gpu}")
    print(
        "  Modal timeout: "
        f"{(STAGE1_PRETRAIN_TIMEOUT_SECONDS if stage == 'pretrain' else STAGE2_TRAINER_TIMEOUT_SECONDS) // 3600}h"
    )
    print(f"  Max steps: {max_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Data: {'dummy' if use_dummy_data else 'LLaVA'}")
    print(f"  Dataset: {dataset}")
    print(f"  W&B: {'enabled' if use_wandb else 'disabled'}")
    print(f"  Background spawn: {background}")
    print(f"  Per-layer grad norms: {track_per_layer_grad_norms}")
    print(f"  Eval benchmarks: {run_eval_benchmarks}")
    if stage == "pretrain":
        print(f"  Stage 1 image tokens: {pretrain_image_tokens}")
    if arch_key == "anymal_v4" and (
        stage == "pretrain"
        or pretrain_image_tokens is not None
        or v4_global_image_tokens is not None
        or v4_local_image_tokens is not None
    ):
        v4_global_tokens, v4_local_tokens, v4_total_tokens = _resolve_v4_token_split(
            total_tokens=pretrain_image_tokens,
            global_tokens=v4_global_image_tokens,
            local_tokens=v4_local_image_tokens,
        )
        pretrain_image_tokens = v4_total_tokens
        print(f"  V4 connector type: {resolved_v4_connector_type}")
        print(f"  V4 global/local tokens: {v4_global_tokens}/{v4_local_tokens}")
        print(f"  V4 connector layers: {v4_connector_layers or V4_DEFAULT_CONNECTOR_LAYERS}")
        print(f"  V4 connector heads: {v4_connector_heads or V4_DEFAULT_CONNECTOR_HEADS}")
        print(f"  V4 connector FF multiplier: {v4_connector_ff_mult or V4_DEFAULT_CONNECTOR_FF_MULT}")
        print(
            "  V4 connector hidden dim: "
            f"{v4_connector_hidden_dim if v4_connector_hidden_dim is not None else V4_DEFAULT_CONNECTOR_HIDDEN_DIM}"
        )
        print(
            "  V4 connector output scale: "
            f"{v4_connector_output_scale if v4_connector_output_scale is not None else V4_DEFAULT_CONNECTOR_OUTPUT_SCALE}"
        )
        print(
            "  V4 connector output gate init: "
            f"{v4_connector_output_gate_init if v4_connector_output_gate_init is not None else V4_DEFAULT_CONNECTOR_OUTPUT_GATE_INIT}"
        )
        print(f"  V4 2D position features: {v4_use_2d_position_features}")
        if resolved_v4_connector_type == "deepstack_spatial_perceiver_resampler":
            print(f"  V4 DeepStack feature levels: {resolved_deepstack_levels or 3}")
            print(
                "  V4 DeepStack hidden layers: "
                f"{resolved_deepstack_layers or 'last 3'}"
            )
    elif arch_key == "anymal_v4":
        v4_global_tokens = None
        v4_local_tokens = None
        print("  V4 global/local tokens: infer from checkpoint metadata")
        print("  V4 connector shape: infer from checkpoint metadata")
        print("  V4 connector type: infer from checkpoint metadata")
        print("  V4 2D position features: infer from checkpoint metadata")
    else:
        v4_global_tokens = None
        v4_local_tokens = None
    if stage != "pretrain" and projector_warmup_steps is not None:
        print(f"  Projector warmup steps: {projector_warmup_steps}")
    if stage == "pretrain" and connector_warmup_steps:
        print(f"  Stage 1 connector warmup steps: {connector_warmup_steps}")
    if stage == "pretrain" and pretrain_loss_scale != 1.0:
        print(f"  Stage 1 loss scale: {pretrain_loss_scale}")
    if stage == "pretrain" and pretrain_loss_normalization != "mean":
        print(f"  Stage 1 loss normalization: {pretrain_loss_normalization}")
        print(
            "  Stage 1 loss normalization target tokens: "
            f"{pretrain_loss_normalization_target_tokens}"
        )
    if stage == "pretrain":
        print(f"  Stage 1 gradient accumulation: {pretrain_gradient_accumulation_steps}")
        print(
            "  Stage 1 checkpoint save steps: "
            f"{pretrain_save_steps or _checkpoint_save_interval(max_steps)}"
        )
    if stage == "finetune":
        print(f"  Freeze connector: {freeze_connector}")
    if finetune_loss_scale != 1.0:
        print(f"  Stage 2 loss scale: {finetune_loss_scale}")
    if stage == "finetune":
        print(f"  Stage 2 gradient accumulation: {finetune_gradient_accumulation_steps}")
    if learning_rate:
        print(f"  Learning rate: {learning_rate}")
    if lora_learning_rate:
        print(f"  LoRA learning rate: {lora_learning_rate}")
    if pretrain_checkpoint:
        print(f"  Pretrain checkpoint: {pretrain_checkpoint}")
    if finetune_checkpoint:
        print(f"  Finetune checkpoint: {finetune_checkpoint}")
    if resume_checkpoint:
        print(f"  Resume from: {resume_checkpoint}")

    if stage == "pretrain":
        # Stage 1 uses multi-GPU distributed pretraining
        lr = learning_rate or 2e-4
        gpu_key = _normalize_gpu_type(gpu_type)
        pretrain_runner = (
            pretrain_distributed_h100 if gpu_key == "h100" else pretrain_distributed
        )
        _invoke_modal_call(
            pretrain_runner,
            background=background,
            max_steps=max_steps,
            learning_rate=lr,
            batch_size=batch_size,
            use_wandb=use_wandb,
            architecture=architecture,
            token_compressor_type=token_compressor_type,
            wandb_api_key=wandb_api_key,
            resume_checkpoint=resume_checkpoint,
            pretrain_checkpoint=pretrain_checkpoint,
            run_name=run_name,
            pretrain_image_tokens=pretrain_image_tokens,
            dataset=dataset,
            connector_warmup_steps=connector_warmup_steps,
            pretrain_loss_scale=pretrain_loss_scale,
            pretrain_loss_normalization=pretrain_loss_normalization,
            pretrain_loss_normalization_target_tokens=pretrain_loss_normalization_target_tokens,
            pretrain_gradient_accumulation_steps=pretrain_gradient_accumulation_steps,
            pretrain_save_steps=pretrain_save_steps,
            v3_connector_type=v3_connector_type,
            v4_global_image_tokens=v4_global_tokens,
            v4_local_image_tokens=v4_local_tokens,
            v4_connector_layers=v4_connector_layers,
            v4_connector_heads=v4_connector_heads,
            v4_connector_ff_mult=v4_connector_ff_mult,
            v4_connector_hidden_dim=v4_connector_hidden_dim,
            v4_connector_output_scale=v4_connector_output_scale,
            v4_connector_output_gate_init=v4_connector_output_gate_init,
            v4_use_2d_position_features=v4_use_2d_position_features,
            v4_connector_type=v4_connector_type,
            v4_deepstack_num_feature_levels=v4_deepstack_num_feature_levels,
            v4_deepstack_hidden_state_indices=v4_deepstack_hidden_state_indices,
            llm_backbone=resolved_llm_backbone,
        )
    else:
        # Stage 2 uses single-GPU with QLoRA
        trainer_cls = Trainer.with_options(gpu=selected_gpu)
        trainer = trainer_cls()
        _invoke_modal_call(
            trainer.train,
            background=background,
            max_steps=max_steps,
            stage=stage,
            architecture=architecture,
            token_compressor_type=token_compressor_type,
            learning_rate=learning_rate,
            lora_learning_rate=lora_learning_rate,
            batch_size=batch_size,
            use_wandb=use_wandb,
            use_dummy_data=use_dummy_data,
            wandb_api_key=wandb_api_key,
            track_per_layer_grad_norms=track_per_layer_grad_norms,
            run_eval_benchmarks=run_eval_benchmarks,
            pretrain_checkpoint=pretrain_checkpoint,
            finetune_checkpoint=finetune_checkpoint,
            resume_checkpoint=resume_checkpoint,
            dataset=dataset,
            run_name=run_name,
            pretrain_image_tokens=pretrain_image_tokens,
            projector_warmup_steps=projector_warmup_steps,
            freeze_connector=freeze_connector,
            finetune_loss_scale=finetune_loss_scale,
            finetune_gradient_accumulation_steps=finetune_gradient_accumulation_steps,
            v3_connector_type=v3_connector_type,
            v4_global_image_tokens=v4_global_tokens,
            v4_local_image_tokens=v4_local_tokens,
            v4_connector_layers=v4_connector_layers,
            v4_connector_heads=v4_connector_heads,
            v4_connector_ff_mult=v4_connector_ff_mult,
            v4_connector_hidden_dim=v4_connector_hidden_dim,
            v4_connector_output_scale=v4_connector_output_scale,
            v4_connector_output_gate_init=v4_connector_output_gate_init,
            v4_use_2d_position_features=v4_use_2d_position_features,
            v4_connector_type=v4_connector_type,
            v4_deepstack_num_feature_levels=v4_deepstack_num_feature_levels,
            v4_deepstack_hidden_state_indices=v4_deepstack_hidden_state_indices,
            llm_backbone=resolved_llm_backbone,
        )
