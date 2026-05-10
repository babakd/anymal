"""Web viewer for AnyMAL side-by-side comparison predictions.

Usage:
    modal deploy modal_viewer.py
    # Visit the printed URL
"""

import base64
import json
import os

import modal

app = modal.App("anymal-viewer")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]",
    "pillow",
)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AnyMAL architecture side-by-side</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    max-width: 1760px; margin: 2em auto; padding: 0 1.2em;
    background: #f5f5f7; color: #1d1d1f; line-height: 1.5;
  }}
  h1 {{ font-size: 1.5em; margin: 0 0 0.3em 0; }}
  h1 small {{ font-weight: 400; color: #666; font-size: 0.7em; }}
  .header {{
    background: white; padding: 1.4em 1.8em; border-radius: 12px;
    margin-bottom: 2em; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  .header p {{ margin: 0.4em 0; color: #444; font-size: 0.95em; }}
  .stats {{ display: flex; gap: 2em; margin-top: 1em; font-size: 0.9em; flex-wrap: wrap; }}
  .stat {{ color: #666; }}
  .stat strong {{ color: #1d1d1f; }}
  .run-list {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 0.65em; margin-top: 1.1em;
  }}
  .run-pill {{
    border: 1px solid #e1e1e6; border-left-width: 4px; border-radius: 8px;
    padding: 0.7em 0.85em; background: #fbfbfd; min-width: 0;
  }}
  .run-pill .name {{ font-weight: 700; font-size: 0.9em; }}
  .run-pill .path {{
    color: #666; font-family: ui-monospace, Menlo, monospace; font-size: 0.72em;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 0.25em;
  }}
  .example {{
    background: white; margin: 1.5em 0; padding: 1.5em;
    border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    display: grid; grid-template-columns: 280px 1fr; gap: 1.5em;
  }}
  .image-col img {{ width: 100%; border-radius: 8px; display: block; }}
  .idx {{ font-size: 0.8em; color: #888; margin-top: 0.5em; font-family: ui-monospace, Menlo, monospace; }}
  .q {{ font-weight: 600; font-size: 1.05em; margin: 0.3em 0 0.8em 0; }}
  .meta {{ color: #777; font-size: 0.8em; margin: -0.3em 0 0.8em 0; }}
  .gt {{
    color: #1d1d1f; padding: 0.9em 1em; background: #eaf6ea;
    border-left: 4px solid #34a853; border-radius: 6px; margin-bottom: 0.9em;
    font-size: 0.95em;
  }}
  .responses {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 0.8em;
  }}
  .response {{
    padding: 0.8em 1em; border-radius: 6px;
    font-size: 0.92em; line-height: 1.55; background: #f7f7f9; border-left: 4px solid #999;
    min-width: 0;
  }}
  .response.v1, .run-pill.v1 {{
    background: #f0eaf7; border-left: 4px solid #7a4fbf;
  }}
  .response.v2, .run-pill.v2 {{
    background: #fdf3e5; border-left: 4px solid #f0a020;
  }}
  .response.v3, .run-pill.v3 {{
    background: #e7f1fb; border-left: 4px solid #1f7ce0;
  }}
  .response.v4, .run-pill.v4 {{
    background: #eef8f1; border-left: 4px solid #2f9e61;
  }}
  .label {{
    font-size: 0.72em; text-transform: uppercase; letter-spacing: 0.4px;
    font-weight: 700; margin-bottom: 0.4em;
  }}
  .label.gt {{ color: #2a7a2a; }}
  .label.v1 {{ color: #5a30a0; }}
  .label.v2 {{ color: #a06010; }}
  .label.v3 {{ color: #0f5aa8; }}
  .label.v4 {{ color: #207846; }}
  .response-meta {{ color: #777; font-size: 0.76em; margin-top: 0.55em; }}
  @media (max-width: 1100px) {{
    .example {{ grid-template-columns: 1fr; }}
    .responses {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="header">
  <h1>{title} <small>{subtitle}</small></h1>
  <p>{description}</p>
  <div class="stats">
    <span class="stat"><strong>{n}</strong> examples</span>
    <span class="stat">Source: <code>{source}</code></span>
    <span class="stat">Prompt: <code>{prompt}</code></span>
    <span class="stat">Artifact: <code>{artifact}</code></span>
  </div>
  <div class="run-list">{runs}</div>
</div>
{examples}
</body>
</html>
"""

EXAMPLE_TEMPLATE = """<div class="example">
  <div class="image-col">
    <img src="data:image/jpeg;base64,{img_b64}" alt="{filename}">
    <div class="idx">#{idx}/{total} · {filename}</div>
  </div>
  <div class="content-col">
    <div class="q">{question}</div>
    <div class="meta">{meta}</div>
    <div class="gt"><div class="label gt">Ground Truth</div>{gt}</div>
    <div class="responses">{responses}</div>
  </div>
</div>"""

RUN_TEMPLATE = """<div class="run-pill {class_name}">
  <div class="name">{label}</div>
  <div class="path">{checkpoint}</div>
</div>"""

RESPONSE_TEMPLATE = """<div class="response {class_name}">
  <div class="label {class_name}">{label}</div>{text}{response_meta}
</div>"""

DATA_PATHS = [
    "/checkpoints/arch_sxs_predictions.json",
    "/checkpoints/three_way_predictions.json",
]


def _escape(s: str) -> str:
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _load_image_b64(path: str, max_size: int = 384) -> str:
    from io import BytesIO
    from PIL import Image
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_size, max_size))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:
        print(f"Warning: could not load {path}: {exc}")
        return ""


def _first_existing_data_path() -> str:
    env_path = os.environ.get("ANYMAL_SXS_PREDICTIONS")
    candidates = [env_path] if env_path else []
    candidates.extend(DATA_PATHS)
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f"No side-by-side prediction artifact found in {candidates}")


def _class_name(run) -> str:
    arch = str(run.get("architecture", "")).lower().replace("anymal_", "")
    if arch in {"v1", "v2", "v3", "v4"}:
        return arch
    key = str(run.get("key", "")).lower()
    for arch in ("v1", "v2", "v3", "v4"):
        if arch in key:
            return arch
    return "other"


def _normalize_data(data):
    if data.get("runs") and all("responses" in row for row in data.get("results", [])):
        return data

    legacy_runs = [
        {
            "key": "response_baseline",
            "label": "Baseline · Stage-1 only",
            "architecture": "v1",
            "checkpoint": data.get("pretrain_projector", ""),
        },
        {
            "key": "response_a",
            "label": "A · + instruct_150k",
            "architecture": "v1",
            "checkpoint": data.get("checkpoint_a", ""),
        },
        {
            "key": "response_f",
            "label": "F · + mix_665k",
            "architecture": "v1",
            "checkpoint": data.get("checkpoint_f", data.get("checkpoint_b", "")),
        },
    ]
    results = []
    for row in data.get("results", []):
        new_row = dict(row)
        responses = {}
        for run in legacy_runs:
            if run["key"] in row:
                responses[run["key"]] = {"text": row.get(run["key"], "")}
        if "response_b" in row and "response_f" not in responses:
            responses["response_f"] = {"text": row.get("response_b", "")}
        new_row["responses"] = responses
        results.append(new_row)
    return {
        **data,
        "title": "AnyMAL three-way comparison",
        "description": "Legacy ablation comparison artifact.",
        "example_source": "llava_val",
        "image_dir": "/checkpoints/coco_images",
        "runs": legacy_runs,
        "results": results,
    }


@app.function(
    image=image,
    volumes={"/checkpoints": volume},
    min_containers=1,
)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    fastapi_app = FastAPI()

    @fastapi_app.get("/healthz")
    def healthz():
        return {"ok": True}

    @fastapi_app.get("/raw")
    def raw():
        with open(_first_existing_data_path()) as f:
            return JSONResponse(json.load(f))

    @fastapi_app.get("/", response_class=HTMLResponse)
    def index():
        artifact_path = _first_existing_data_path()
        with open(artifact_path) as f:
            data = _normalize_data(json.load(f))

        results = data["results"]
        runs = data["runs"]
        image_dir = data.get("image_dir", "/checkpoints/coco_images")
        n = len(results)
        run_parts = [
            RUN_TEMPLATE.format(
                class_name=_class_name(run),
                label=_escape(run.get("label", run.get("key", ""))),
                checkpoint=_escape(run.get("checkpoint", "")),
            )
            for run in runs
        ]
        parts = []
        for i, r in enumerate(results):
            img_path = r["image"] if os.path.isabs(r["image"]) else os.path.join(image_dir, r["image"])
            img_b64 = _load_image_b64(img_path)
            response_parts = []
            for run in runs:
                key = run["key"]
                response = r.get("responses", {}).get(key, {})
                text = response.get("text", response if isinstance(response, str) else "")
                generated_tokens = response.get("generated_tokens") if isinstance(response, dict) else None
                response_meta = (
                    f'<div class="response-meta">{int(generated_tokens)} generated tokens</div>'
                    if generated_tokens is not None
                    else ""
                )
                response_parts.append(
                    RESPONSE_TEMPLATE.format(
                        class_name=_class_name(run),
                        label=_escape(run.get("label", key)),
                        text=_escape(text),
                        response_meta=response_meta,
                    )
                )
            answer_type = r.get("answer_type", "")
            question_type = r.get("question_type", "")
            meta = " · ".join(_escape(x) for x in [answer_type, question_type] if x)
            parts.append(
                EXAMPLE_TEMPLATE.format(
                    idx=i + 1,
                    total=n,
                    filename=_escape(r["image"]),
                    question=_escape(r["question"]),
                    meta=meta,
                    gt=_escape(r["ground_truth"]),
                    responses="\n".join(response_parts),
                    img_b64=img_b64,
                )
            )

        return HTML_TEMPLATE.format(
            title=_escape(data.get("title", "AnyMAL architecture side-by-side")),
            subtitle=_escape(" · ".join(run.get("architecture", "").upper() for run in runs if run.get("architecture"))),
            description=_escape(data.get("description", "")),
            n=n,
            source=_escape(data.get("example_source", "")),
            prompt=_escape(data.get("system_prompt", "")),
            artifact=_escape(os.path.basename(artifact_path)),
            runs="\n".join(run_parts),
            examples="\n".join(parts),
        )

    return fastapi_app
