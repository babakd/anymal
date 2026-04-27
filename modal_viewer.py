"""
Web viewer for AnyMAL ablation comparison predictions.

Renders 20 val-set comparisons between Run A (instruct_150k) and Run F (mix_665k)
side-by-side with the COCO image.

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
<title>AnyMAL Ablation: Stage-1 baseline vs A (instruct_150k) vs F (mix_665k)</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    max-width: 1600px; margin: 2em auto; padding: 0 1.2em;
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
  .example {{
    background: white; margin: 1.5em 0; padding: 1.5em;
    border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    display: grid; grid-template-columns: 260px 1fr; gap: 1.5em;
  }}
  .image-col img {{ width: 100%; border-radius: 8px; display: block; }}
  .idx {{ font-size: 0.8em; color: #888; margin-top: 0.5em; font-family: ui-monospace, Menlo, monospace; }}
  .q {{ font-weight: 600; font-size: 1.05em; margin: 0.3em 0 0.8em 0; }}
  .gt {{
    color: #1d1d1f; padding: 0.9em 1em; background: #eaf6ea;
    border-left: 4px solid #34a853; border-radius: 6px; margin-bottom: 0.9em;
    font-size: 0.95em;
  }}
  .responses {{
    display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.8em;
  }}
  .response {{
    padding: 0.8em 1em; border-radius: 6px;
    font-size: 0.92em; line-height: 1.55;
  }}
  .response.baseline {{
    background: #f0eaf7; border-left: 4px solid #7a4fbf;
  }}
  .response.a {{
    background: #fdf3e5; border-left: 4px solid #f0a020;
  }}
  .response.f {{
    background: #e7f1fb; border-left: 4px solid #1f7ce0;
  }}
  .label {{
    font-size: 0.72em; text-transform: uppercase; letter-spacing: 0.6px;
    font-weight: 700; margin-bottom: 0.4em;
  }}
  .label.gt {{ color: #2a7a2a; }}
  .label.baseline {{ color: #5a30a0; }}
  .label.a {{ color: #a06010; }}
  .label.f {{ color: #0f5aa8; }}
  @media (max-width: 1100px) {{
    .example {{ grid-template-columns: 1fr; }}
    .responses {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="header">
  <h1>AnyMAL three-way comparison <small>Stage-1-only · Stage-2 instruct_150k · Stage-2 mix_665k</small></h1>
  <p><strong>Baseline</strong> (purple): Stage-1 pretrained perceiver + frozen LLaMA-3-8B-Instruct. No LoRA, no instruction finetuning.</p>
  <p><strong>A</strong> (orange): baseline + Stage-2 finetune on <code>instruct_150k</code> (500 steps, train_loss ≈ 1.29)</p>
  <p><strong>F</strong> (blue): baseline + Stage-2 finetune on <code>mix_665k</code> (500 steps, train_loss ≈ 1.19)</p>
  <p>Examples drawn from the deterministic val split (stride-sampled, same 20 for all three models).</p>
  <div class="stats">
    <span class="stat"><strong>{n}</strong> examples</span>
    <span class="stat">Pretrain: <code>{pretrain}</code></span>
    <span class="stat">A: <code>{ckpt_a}</code></span>
    <span class="stat">F: <code>{ckpt_b}</code></span>
  </div>
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
    <div class="gt"><div class="label gt">Ground Truth</div>{gt}</div>
    <div class="responses">
      <div class="response baseline"><div class="label baseline">Baseline · Stage-1 only</div>{response_baseline}</div>
      <div class="response a"><div class="label a">A · + instruct_150k</div>{response_a}</div>
      <div class="response f"><div class="label f">F · + mix_665k</div>{response_f}</div>
    </div>
  </div>
</div>"""


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
        with open("/checkpoints/three_way_predictions.json") as f:
            return JSONResponse(json.load(f))

    @fastapi_app.get("/", response_class=HTMLResponse)
    def index():
        with open("/checkpoints/three_way_predictions.json") as f:
            data = json.load(f)

        results = data["results"]
        n = len(results)
        parts = []
        for i, r in enumerate(results):
            img_path = os.path.join("/checkpoints/coco_images", r["image"])
            img_b64 = _load_image_b64(img_path)
            parts.append(
                EXAMPLE_TEMPLATE.format(
                    idx=i + 1,
                    total=n,
                    filename=_escape(r["image"]),
                    question=_escape(r["question"]),
                    gt=_escape(r["ground_truth"]),
                    response_baseline=_escape(r.get("response_baseline", "")),
                    response_a=_escape(r["response_a"]),
                    response_f=_escape(r.get("response_f", r.get("response_b", ""))),
                    img_b64=img_b64,
                )
            )

        return HTML_TEMPLATE.format(
            n=n,
            pretrain=_escape(data.get("pretrain_projector", "")),
            ckpt_a=_escape(data.get("checkpoint_a", "")),
            ckpt_b=_escape(data.get("checkpoint_f", data.get("checkpoint_b", ""))),
            examples="\n".join(parts),
        )

    return fastapi_app
