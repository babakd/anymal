"""
Modal web server for the AnyMAL Prediction Progression Viewer.

Serves the progression viewer HTML with predictions data loaded from
the Modal volume. Gives a shareable URL for anyone to view.

Usage:
    modal deploy modal_viewer.py          # Deploy (persistent URL)
    modal serve modal_viewer.py           # Dev mode (temporary URL)
"""

import modal
import os
import json
import glob

app = modal.App("anymal-viewer")
volume = modal.Volume.from_name("anymal-checkpoints", create_if_missing=True)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("fastapi")
    .add_local_file(
        os.path.join(PROJECT_DIR, "progression_viewer.html"),
        remote_path="/root/progression_viewer.html",
    )
)


@app.function(
    image=image,
    volumes={"/checkpoints": volume},
)
@modal.fastapi_endpoint(method="GET", label="anymal-viewer")
def viewer():
    from fastapi.responses import HTMLResponse

    # Find latest predictions file on volume
    pattern = "/checkpoints/predictions_*.json"
    files = sorted(glob.glob(pattern))

    # Fall back to predictions.json
    if not files:
        fallback = "/checkpoints/predictions.json"
        if os.path.exists(fallback):
            files = [fallback]

    if not files:
        return HTMLResponse("<h1>No predictions found</h1><p>Run inference first.</p>")

    latest = files[-1]
    print(f"Serving predictions from: {latest}")

    with open(latest) as f:
        predictions_json = f.read()

    with open("/root/progression_viewer.html") as f:
        html = f.read()

    # Inject predictions data into the HTML by replacing the auto-load fetch block
    inject_script = f"""
<script>
// Auto-injected predictions data from Modal volume
window.__PREDICTIONS_DATA__ = {predictions_json};
</script>
"""

    # Also add a small script that loads the injected data on DOMContentLoaded
    loader_script = """
<script>
document.addEventListener('DOMContentLoaded', function() {
  if (window.__PREDICTIONS_DATA__) {
    // Trigger the same init flow as file loading
    var evt = new CustomEvent('predictions-loaded', { detail: window.__PREDICTIONS_DATA__ });
    document.dispatchEvent(evt);
  }
});
</script>
"""

    # Insert before </body>
    html = html.replace("</body>", inject_script + loader_script + "</body>")

    return HTMLResponse(html)
