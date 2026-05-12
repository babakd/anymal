"""Modal runner for V8 Stage 0 decoder-swap smoke tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal


app = modal.App("anymal-v8-llm-swap-smoke")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=True)
PROJECT_DIR = Path(__file__).resolve().parents[2]

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.53.0,<5.0.0",
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "sentencepiece>=0.1.99",
        "huggingface_hub>=0.19.0",
        "einops>=0.7.0",
        "open_clip_torch>=2.23.0",
        "timm>=0.9.0",
    )
    .add_local_dir(PROJECT_DIR, remote_path="/root/anymal", copy=False)
)


CURRENT_LLAMA3_BACKBONE = "meta-llama/Meta-Llama-3-8B-Instruct"
QWEN3_8B_BACKBONE = "Qwen/Qwen3-8B"
MODAL_LLM_CACHE_DIRS = {
    CURRENT_LLAMA3_BACKBONE: "/checkpoints/llama3-8b-instruct",
    QWEN3_8B_BACKBONE: "/checkpoints/qwen3-8b",
}


def _canonicalize_llm_backbone(value: str | None = None) -> str:
    raw = str(value or QWEN3_8B_BACKBONE).strip()
    lowered = raw.lower()
    base = os.path.basename(raw.rstrip("/")).lower()
    if "llama-3.1" in lowered or "llama3.1" in lowered:
        raise ValueError("Llama 3.1 is intentionally excluded from this V8 execution.")
    if lowered in {CURRENT_LLAMA3_BACKBONE.lower(), "llama3", "llama3-8b-instruct"} or base == "llama3-8b-instruct":
        return CURRENT_LLAMA3_BACKBONE
    if lowered in {QWEN3_8B_BACKBONE.lower(), "qwen3", "qwen3-8b"} or base == "qwen3-8b":
        return QWEN3_8B_BACKBONE
    return raw


def _ensure_llm_backbone_cached(llm_backbone: str) -> tuple[str, str]:
    canonical = _canonicalize_llm_backbone(llm_backbone)
    if os.path.isabs(canonical):
        if not os.path.exists(os.path.join(canonical, "config.json")):
            raise FileNotFoundError(f"Requested llm_backbone path has no config.json: {canonical}")
        return canonical, canonical
    if canonical not in MODAL_LLM_CACHE_DIRS:
        raise ValueError(
            f"Unsupported llm_backbone={llm_backbone!r}. "
            f"Supported values: {sorted(MODAL_LLM_CACHE_DIRS)}"
        )

    local_dir = MODAL_LLM_CACHE_DIRS[canonical]
    if not os.path.exists(os.path.join(local_dir, "config.json")):
        from huggingface_hub import snapshot_download

        print(f"Downloading decoder backbone {canonical} to {local_dir}...")
        snapshot_download(
            repo_id=canonical,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        volume.commit()
    else:
        print(f"Using cached decoder backbone {canonical} at {local_dir}")
    return canonical, local_dir


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=2 * 60 * 60,
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_stage0_smoke(
    llm_backbone: str = QWEN3_8B_BACKBONE,
    image_tokens: int = 128,
    max_new_tokens: int = 8,
    torch_dtype: str = "bfloat16",
    remote_output_root: str = "/checkpoints/outputs/v8_llm_swap_smoke",
) -> dict:
    import sys

    sys.path.insert(0, "/root/anymal")

    from scripts.v8_llm_swap_smoke import run_smoke

    canonical, local_model_path = _ensure_llm_backbone_cached(llm_backbone)
    result = run_smoke(
        llm_backbone=local_model_path,
        output_root=remote_output_root,
        image_tokens=int(image_tokens),
        device_map="auto",
        torch_dtype=torch_dtype,
        use_qlora=False,
        max_new_tokens=int(max_new_tokens),
    )
    result["llm_backbone"] = canonical
    result["remote_output_root"] = remote_output_root
    volume.commit()
    return result


@app.local_entrypoint()
def main(
    llm_backbone: str = QWEN3_8B_BACKBONE,
    image_tokens: int = 128,
    max_new_tokens: int = 8,
    torch_dtype: str = "bfloat16",
    output_root: str = "outputs/v8_llm_swap_smoke",
    remote_output_root: str = "/checkpoints/outputs/v8_llm_swap_smoke",
):
    result = run_stage0_smoke.remote(
        llm_backbone=llm_backbone,
        image_tokens=image_tokens,
        max_new_tokens=max_new_tokens,
        torch_dtype=torch_dtype,
        remote_output_root=remote_output_root,
    )

    local_dir = Path(output_root) / str(result["llm_backbone"]).replace("/", "__").replace(":", "_")
    local_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in {
        "tokenizer_report": "tokenizer_report.json",
        "prompt_contract_report": "prompt_contract_report.json",
        "inputs_embeds_report": "inputs_embeds_report.json",
        "generation_report": "generation_report.json",
    }.items():
        with (local_dir / filename).open("w", encoding="utf-8") as f:
            json.dump(result[key], f, indent=2, sort_keys=True)
            f.write("\n")

    print(f"Saved local V8 smoke reports to {local_dir}")
    print(f"Remote reports are under {result['output_dir']}")
