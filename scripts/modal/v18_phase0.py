"""Phase 0 preflight for the V18 mid-training data campaign.

This script intentionally does not acquire datasets. It writes the required
auth/license checklist onto the training Modal volume and reports whether the
volume and Hugging Face secret are usable before the campaign proceeds.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import modal


app = modal.App("anymal-v18-phase0")
image = modal.Image.debian_slim(python_version="3.10")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=False)


CHECKLIST = [
    {
        "source_name": "LCS-558k / LLaVA-Pretrain captions",
        "hf_dataset_id_or_download_url": "liuhaotian/LLaVA-Pretrain",
        "pinned_revision": "70f9d1e5e1a697fe35830875cfc7de1dd590d727",
        "gated": False,
        "estimated_download_size": "metadata JSON plus image bundle/cache; image cache may be tens of GB if not already present",
        "authentication_required": "none for public HF metadata; HF token may be used for robust download",
        "license": "other / mixed; V18 plan notes LLaVA license plus CC-BY-4.0 COCO/LAION components",
        "commercial_use_allowed": "license_dependent",
        "phase0_decision": "pending_user_confirmation_due_mixed_license",
    },
    {
        "source_name": "COCO Captions train2017",
        "hf_dataset_id_or_download_url": "HuggingFaceM4/COCO or COCO train2017 annotations/images",
        "pinned_revision": "4d0dfd4a3712a80e7b8498ed76e778f7cea9d21c",
        "gated": False,
        "estimated_download_size": "annotations small; train2017 images about 18 GB if not already cached",
        "authentication_required": "none",
        "license": "cc-by-4.0 / COCO terms",
        "commercial_use_allowed": True,
        "phase0_decision": "allowed",
    },
    {
        "source_name": "VQAv2 train2014 direct-answer cache",
        "hf_dataset_id_or_download_url": "existing /checkpoints/vqa_data plus COCO train2014",
        "pinned_revision": None,
        "gated": False,
        "estimated_download_size": "already cached or COCO train2014 about 13 GB",
        "authentication_required": "none",
        "license": "cc-by-4.0 / VQAv2 and COCO terms",
        "commercial_use_allowed": True,
        "phase0_decision": "allowed",
    },
    {
        "source_name": "GQA train_balanced broad subset",
        "hf_dataset_id_or_download_url": "Mineru/GQA",
        "pinned_revision": "55fbe98d3474e07e0d1fe0078ba2d48c9ea7712e",
        "gated": False,
        "estimated_download_size": "metadata plus Visual Genome image cache; substantial if images missing",
        "authentication_required": "none for HF dataset metadata",
        "license": "cc-by-4.0 / GQA terms per V18 plan; HF card has no license tag",
        "commercial_use_allowed": True,
        "phase0_decision": "allowed_with_license_note",
    },
    {
        "source_name": "A-OKVQA train",
        "hf_dataset_id_or_download_url": "HuggingFaceM4/A-OKVQA",
        "pinned_revision": "d1b0efa3a436e9101dfbde3752db7607da696c35",
        "gated": False,
        "estimated_download_size": "annotations small; uses COCO images if not already cached",
        "authentication_required": "none",
        "license": "apache-2.0 per V18 plan; HF card has no license tag",
        "commercial_use_allowed": True,
        "phase0_decision": "allowed_with_license_note",
    },
    {
        "source_name": "OK-VQA train",
        "hf_dataset_id_or_download_url": "Multimodal-Fatima/OK-VQA_train",
        "pinned_revision": "78e937b0afc333cfe8fee21c4d56aa405b4d0682",
        "gated": False,
        "estimated_download_size": "annotations small; uses COCO images if not already cached",
        "authentication_required": "none",
        "license": "cc-by-4.0 per V18 plan; HF card has no license tag",
        "commercial_use_allowed": True,
        "phase0_decision": "allowed_with_license_note",
    },
    {
        "source_name": "VizWiz VQA train",
        "hf_dataset_id_or_download_url": "lmms-lab/vizwiz_vqa",
        "pinned_revision": None,
        "gated": "unknown; unauthenticated HF API returned 401 during Phase 0",
        "estimated_download_size": "image cache likely several GB",
        "authentication_required": "HF token",
        "license": "cc-by-4.0 per V18 plan; verify after authenticated metadata access",
        "commercial_use_allowed": True,
        "phase0_decision": "pending_authenticated_metadata_check",
    },
    {
        "source_name": "VSR train/dev",
        "hf_dataset_id_or_download_url": "cambridgeltl/vsr_zeroshot",
        "pinned_revision": "148b3777ceef4a1bfe46377614980836db7d12f2",
        "gated": False,
        "estimated_download_size": "small annotations plus referenced images",
        "authentication_required": "none",
        "license": "cc-by-4.0",
        "commercial_use_allowed": True,
        "phase0_decision": "allowed",
    },
    {
        "source_name": "NLVR2 train",
        "hf_dataset_id_or_download_url": "lil-lab/nlvr2 or official direct download",
        "pinned_revision": None,
        "gated": "unknown; unauthenticated HF API returned 401 during Phase 0",
        "estimated_download_size": "image cache likely several GB",
        "authentication_required": "HF token or direct-download approval; also requires multi-image support decision",
        "license": "cc-by-4.0 per V18 plan; verify after authenticated metadata access",
        "commercial_use_allowed": True,
        "phase0_decision": "pending_authenticated_metadata_check_and_multi_image_decision",
    },
    {
        "source_name": "OCR-VQA train",
        "hf_dataset_id_or_download_url": "howard-hou/OCR-VQA",
        "pinned_revision": "88234cc092c5f6d199b5cf3b471e3b490a69c07b",
        "gated": False,
        "estimated_download_size": "image cache likely tens of GB for 50k rows",
        "authentication_required": "none",
        "license": "cc-by-4.0 per V18 plan; HF card has no license tag",
        "commercial_use_allowed": True,
        "phase0_decision": "allowed_with_license_note",
    },
    {
        "source_name": "TextVQA train majority-answer cache",
        "hf_dataset_id_or_download_url": "lmms-lab/textvqa plus existing /checkpoints/textvqa_data cache",
        "pinned_revision": "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        "gated": False,
        "estimated_download_size": "already partially cached; image cache several GB",
        "authentication_required": "none",
        "license": "cc-by-4.0 upstream TextVQA/OpenImages terms per V18/V16 notes; HF card has no license tag",
        "commercial_use_allowed": True,
        "phase0_decision": "allowed_with_license_note",
    },
    {
        "source_name": "AI2D train",
        "hf_dataset_id_or_download_url": "lmms-lab/ai2d",
        "pinned_revision": "c83a9b9692933aff8349157c88a413df9d02c4e5",
        "gated": False,
        "estimated_download_size": "small-to-moderate image cache",
        "authentication_required": "none",
        "license": "apache-2.0 per V18 plan; HF card has no license tag",
        "commercial_use_allowed": True,
        "phase0_decision": "allowed_with_license_note",
    },
    {
        "source_name": "DocVQA train",
        "hf_dataset_id_or_download_url": "lmms-lab/DocVQA or https://www.docvqa.org/",
        "pinned_revision": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
        "gated": False,
        "estimated_download_size": "document image cache likely several GB",
        "authentication_required": "none if HF train split is usable; registration if direct site is needed",
        "license": "apache-2.0",
        "commercial_use_allowed": True,
        "phase0_decision": "allowed_if_hf_train_split_available_else_skip_or_ask",
    },
    {
        "source_name": "GQA spatial metadata-filtered cache",
        "hf_dataset_id_or_download_url": "existing /checkpoints/gqa_data/v17_gqa_metadata_spatial_train_balanced_seed1503_n15000.json",
        "pinned_revision": None,
        "gated": False,
        "estimated_download_size": "already cached",
        "authentication_required": "none",
        "license": "cc-by-4.0 / GQA terms per V18 plan",
        "commercial_use_allowed": True,
        "phase0_decision": "allowed",
    },
    {
        "source_name": "RefCOCO train",
        "hf_dataset_id_or_download_url": "lmms-lab/RefCOCO",
        "pinned_revision": "a5dff0b3194715fda69b2d6a0d2aaafb41eaa407",
        "gated": False,
        "estimated_download_size": "annotations plus COCO train2014 images if missing",
        "authentication_required": "none",
        "license": "image-source-dependent / not declared on HF card",
        "commercial_use_allowed": "unknown",
        "phase0_decision": "skip_until_license_verified",
    },
    {
        "source_name": "RefCOCO+ train",
        "hf_dataset_id_or_download_url": "lmms-lab/RefCOCOplus",
        "pinned_revision": "a283082593c437efc2bb9876d43bc3075cd914fe",
        "gated": False,
        "estimated_download_size": "annotations plus COCO train2014 images if missing",
        "authentication_required": "none",
        "license": "image-source-dependent / not declared on HF card",
        "commercial_use_allowed": "unknown",
        "phase0_decision": "skip_until_license_verified",
    },
    {
        "source_name": "RefCOCOg train",
        "hf_dataset_id_or_download_url": "lmms-lab/RefCOCOg",
        "pinned_revision": "93af55545967808285637a67431e4971a6f9fd49",
        "gated": False,
        "estimated_download_size": "annotations plus COCO train2014 images if missing",
        "authentication_required": "none",
        "license": "image-source-dependent / not declared on HF card",
        "commercial_use_allowed": "unknown",
        "phase0_decision": "skip_until_license_verified",
    },
    {
        "source_name": "Visual Genome direct annotations/images",
        "hf_dataset_id_or_download_url": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/",
        "pinned_revision": None,
        "gated": False,
        "estimated_download_size": "about 30 GB",
        "authentication_required": "direct URL approval",
        "license": "cc-by-4.0 per V18 plan",
        "commercial_use_allowed": True,
        "phase0_decision": "pending_user_direct_download_approval",
    },
    {
        "source_name": "ChartQA train",
        "hf_dataset_id_or_download_url": "anhdang000/ChartQA-V2 or existing leak-clean V17 cache",
        "pinned_revision": "93b2f1f6bd69516c1be21faeefa05540768e0537",
        "gated": False,
        "estimated_download_size": "already partially cached",
        "authentication_required": "none",
        "license": "GPL-3.0 upstream per V18/V16 notes; HF repack card has no license tag",
        "commercial_use_allowed": False,
        "phase0_decision": "excluded_by_user_license_policy",
    },
    {
        "source_name": "ShareGPT4V detailed captions",
        "hf_dataset_id_or_download_url": "Lin-Chen/ShareGPT4V",
        "pinned_revision": "55d02b0bc53a2754095a14110dda6daedd95671d",
        "gated": False,
        "estimated_download_size": "large mixed image/text corpus; sample image cache may be tens of GB",
        "authentication_required": "none for public HF metadata",
        "license": "cc-by-nc-4.0",
        "commercial_use_allowed": False,
        "phase0_decision": "excluded_by_user_license_policy",
    },
    {
        "source_name": "LLaVA-Mix-665k optional subset",
        "hf_dataset_id_or_download_url": "LLaVA/LLaVA-Instruct-150K and LLaVA-NeXT/LLaVA-mix665k-style mirrors",
        "pinned_revision": None,
        "gated": False,
        "estimated_download_size": "large; depends on selected mirror",
        "authentication_required": "none or HF token depending on mirror",
        "license": "NC-mixed upstream per V18 plan",
        "commercial_use_allowed": False,
        "phase0_decision": "excluded_by_user_license_policy",
    },
]


LICENSE_DECISIONS = {
    "hf_secret_use": {
        "decision": "use_existing_modal_secret",
        "reason": "The Modal huggingface secret is present and valid; user said to ask only if authentication help is needed.",
    },
    "modal_volume_writes": {
        "decision": "approved",
        "reason": "User explicitly requested that everything be placed on the Modal volume used for training.",
    },
    "chartqa_gpl3_training_data": {
        "decision": "exclude",
        "reason": "User instructed to skip paid or license-restrictive sources.",
    },
    "cc_by_nc_sources": {
        "decision": "exclude",
        "reason": "User instructed to skip paid or license-restrictive sources.",
    },
    "llava_mix665k_nc_mixed": {
        "decision": "exclude",
        "reason": "User instructed to skip paid or license-restrictive sources.",
    },
    "unknown_or_image_source_dependent_license_sources": {
        "decision": "approved_when_permissive_upstream_verified",
        "affected_sources": ["RefCOCO", "RefCOCO+", "RefCOCOg"],
        "reason": "User later approved missing HF license tags when permissive upstream terms are verified.",
    },
    "visual_genome_direct_download": {
        "decision": "approved",
        "reason": "User explicitly approved Visual Genome Stanford direct download.",
    },
    "lcs558k_llava_pretrain": {
        "decision": "exclude",
        "reason": "HF license is other/mixed and commercial use is license-dependent; user instructed to skip restrictive sources.",
    },
}

PHASE0_PENDING_CONFIRMATIONS = [
    {
        "id": "hf_secret_use",
        "question": "May V18 use the existing Modal huggingface secret for dataset metadata and downloads?",
        "current_status": "resolved_use_existing_valid_secret",
        "default_if_unanswered": "use_existing_valid_secret",
    },
    {
        "id": "modal_volume_writes",
        "question": "May V18 continue writing data artifacts to the anymal-checkpoints Modal volume?",
        "current_status": "resolved_approved_by_user_request",
        "default_if_unanswered": "write_required_v18_artifacts_to_volume",
    },
    {
        "id": "visual_genome_direct_download",
        "question": "Approve Visual Genome Stanford CDN direct download (~30 GB), or defer it?",
        "current_status": "resolved_approved_by_user",
        "default_if_unanswered": "approved",
    },
    {
        "id": "lcs558k_mixed_license",
        "question": "Include LCS-558k / LLaVA-Pretrain despite mixed/license-dependent posture, or skip it?",
        "current_status": "resolved_skip_due_user_license_policy",
        "default_if_unanswered": "skip",
    },
    {
        "id": "missing_hf_license_tags",
        "question": "Proceed with sources that have missing HF license tags but permissive upstream notes, or skip unless fully verified first?",
        "current_status": "resolved_approved_when_permissive_upstream_verified",
        "default_if_unanswered": "proceed_after_upstream_verification",
    },
]

BLOCKED_WORK_UNTIL_PHASE0_CONFIRMED = [
    "GPL-3.0 sources",
    "CC-BY-NC / non-commercial sources",
    "paid sources",
    "license-dependent or unknown-license sources unless independently verified as permissive upstream",
]


def _json_dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _check_hf_token() -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return {"present": False, "valid": False, "whoami": None, "error": "HF_TOKEN not present"}
    request = urllib.request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "anymal-v18-phase0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {
            "present": True,
            "valid": True,
            "whoami": {
                "name": payload.get("name"),
                "fullname": payload.get("fullname"),
                "type": payload.get("type"),
            },
            "error": None,
        }
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        return {"present": True, "valid": False, "whoami": None, "error": str(exc)}


@app.function(
    image=image,
    volumes={"/checkpoints": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=5 * 60,
)
def write_phase0_artifacts() -> dict[str, Any]:
    out_dir = Path("/checkpoints/v18_qwen")
    out_dir.mkdir(parents=True, exist_ok=True)

    usage = shutil.disk_usage("/checkpoints")
    write_probe = out_dir / ".phase0_write_probe"
    write_probe.write_text(f"ok {time.time()}\n", encoding="utf-8")
    can_write = write_probe.exists() and write_probe.read_text(encoding="utf-8").startswith("ok ")
    write_probe.unlink(missing_ok=True)

    checklist_path = out_dir / "auth_checklist.json"
    decisions_path = out_dir / "license_decisions.json"
    preflight_path = out_dir / "phase0_preflight.json"
    gate_status_path = out_dir / "phase0_gate_status.json"

    checklist = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "volume": "anymal-checkpoints",
        "volume_mount": "/checkpoints",
        "policy": {
            "no_training_runs_launched": True,
            "paid_or_license_restrictive_sources": "skip",
            "unknown_licenses": "fail_closed_until_verified",
        },
        "sources": CHECKLIST,
    }
    decisions = {
        "created_at_utc": checklist["created_at_utc"],
        "source": "User instruction plus V18 Phase 0 defaults",
        "decisions": LICENSE_DECISIONS,
    }
    preflight = {
        "created_at_utc": checklist["created_at_utc"],
        "volume": {
            "name": "anymal-checkpoints",
            "mount": "/checkpoints",
            "can_write": can_write,
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "free_gib": round(usage.free / (1024**3), 2),
            "meets_250gb_free_gate": usage.free >= 250 * 1024**3,
        },
        "huggingface_secret": _check_hf_token(),
    }
    gate_status = {
        "created_at_utc": checklist["created_at_utc"],
        "phase": "0",
        "status": "cleared_with_conservative_user_policy",
        "halt_required_by": "V18_MIDTRAINING_DATA.md Phase 0",
        "no_training_or_acquisition_started": True,
        "preflight_artifacts": {
            "auth_checklist": str(checklist_path),
            "license_decisions": str(decisions_path),
            "phase0_preflight": str(preflight_path),
        },
        "resolved_confirmations": PHASE0_PENDING_CONFIRMATIONS,
        "blocked_work_under_conservative_policy": BLOCKED_WORK_UNTIL_PHASE0_CONFIRMED,
        "safe_pre_gate_work_completed": [
            "auth/license checklist written",
            "license-restrictive source defaults recorded",
            "Modal volume write preflight completed",
            "Hugging Face secret validity preflight completed",
        ],
    }

    _json_dump(checklist_path, checklist)
    _json_dump(decisions_path, decisions)
    _json_dump(preflight_path, preflight)
    _json_dump(gate_status_path, gate_status)
    volume.commit()

    result = {
        "auth_checklist": str(checklist_path),
        "license_decisions": str(decisions_path),
        "phase0_preflight": str(preflight_path),
        "phase0_gate_status": str(gate_status_path),
        "volume_can_write": can_write,
        "volume_free_gib": preflight["volume"]["free_gib"],
        "meets_250gb_free_gate": preflight["volume"]["meets_250gb_free_gate"],
        "hf_secret_present": preflight["huggingface_secret"]["present"],
        "hf_secret_valid": preflight["huggingface_secret"]["valid"],
        "phase0_decisions": PHASE0_PENDING_CONFIRMATIONS,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.local_entrypoint()
def main() -> None:
    write_phase0_artifacts.remote()
