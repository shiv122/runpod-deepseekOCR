# DeepSeek-OCR on RunPod Serverless

Serverless handler for [`deepseek-ai/DeepSeek-OCR`](https://huggingface.co/deepseek-ai/DeepSeek-OCR), packaged for **RunPod's Deploy-from-GitHub flow** so you never have to build or push the multi-GB image from a local machine.

## What's here

| File | Purpose |
|---|---|
| `handler.py` | RunPod serverless entrypoint — loads the model on cold start, handles each job |
| `Dockerfile` | CUDA 11.8 + torch 2.6 + flash-attn 2.7.3. Bakes the model weights at build time |
| `builder/download_model.py` | Pulls weights from HF during `docker build` |
| `requirements.txt` | Python deps (torch / flash-attn pinned in the Dockerfile) |
| `test_input.json` | Sample job — also used by RunPod for local testing |

## Deploy from GitHub (recommended)

You push ~30KB of text to GitHub. RunPod's builders do the rest (download the 7GB of weights, build the image, push to their registry) on fast cloud bandwidth.

1. **Push this directory to GitHub** (see "Push to GitHub" below).
2. **Connect GitHub to RunPod** (one-time):
   RunPod console → **Settings → Connections → GitHub → Connect** and authorize the repo.
3. **Create the endpoint**:
   - **Serverless → New Endpoint → GitHub Repo**
   - Pick your repo + branch
   - Build context: `/` (root) — the `Dockerfile` is at the root
   - **GPU**: 16GB+ VRAM (L4, A4000, A5000, A100 — anything ≥16GB works; model is 3B BF16 ≈ 6GB plus activations)
   - **Container Disk**: ≥20GB (the baked-in weights need room)
   - **Active / Max Workers**: start with `0 / 3`; idle timeout `5s`
4. **Wait for the build to finish** in RunPod's UI (~10–15 min the first time). On every subsequent push to that branch, RunPod auto-rebuilds.

## Push to GitHub

```bash
cd /Users/shiveshtripathi/Desktop/runpod
git remote add origin git@github.com:<your-user>/deepseek-ocr-runpod.git
git push -u origin main
```

(`git init` and the initial commit have already been made for you — see the "First commit" section below if you'd like to start over.)

## Invoke

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "https://example.com/receipt.jpg",
      "task": "markdown",
      "size": "gundam"
    }
  }'
```

For long documents use `/run` (async) and poll `/status/<id>` instead of `/runsync`.

### Input fields

| Field | Type | Default | Notes |
|---|---|---|---|
| `image` | string | — (required) | HTTPS URL **or** base64 (with or without `data:` prefix) |
| `task` | `"ocr" \| "markdown"` | `"markdown"` | Shorthand for the prompt; ignored if `prompt` is provided |
| `prompt` | string | — | Full custom prompt. Must start with `<image>\n` |
| `size` | `"tiny"\|"small"\|"base"\|"large"\|"gundam"` | `"gundam"` | Resolution preset (see below) |
| `base_size`, `image_size`, `crop_mode` | int / int / bool | from preset | Override individual sizing params |
| `test_compress` | bool | `false` | Ask the model to report token-compression stats |
| `return_image` | bool | `false` | Return the grounding-overlay image as base64 |

### Size presets

| Preset | base_size | image_size | crop_mode |
|---|---|---|---|
| `tiny` | 512 | 512 | false |
| `small` | 640 | 640 | false |
| `base` | 1024 | 1024 | false |
| `large` | 1280 | 1280 | false |
| `gundam` (default) | 1024 | 640 | true |

### Response

```json
{
  "text": "## Extracted markdown...",
  "prompt": "<image>\n<|grounding|>Convert the document to markdown. ",
  "sizing": { "base_size": 1024, "image_size": 640, "crop_mode": true },
  "overlay_image_b64": null
}
```

## Build options

The Dockerfile takes one build-arg, `BAKE_MODEL` (default `1`):

- **`BAKE_MODEL=1`** (default, recommended): weights downloaded at build time, baked into the image → sub-second cold starts.
- **`BAKE_MODEL=0`**: skip bake-in. Attach a RunPod network volume mounted at `/runpod-volume` and set env var `HF_HOME=/runpod-volume/huggingface` on the endpoint. First cold start pulls weights; later containers reuse the volume.

If RunPod's GitHub-build UI exposes build-args, set them there. Otherwise the default (`BAKE_MODEL=1`) is what you want.

## Local test (optional)

Only useful if you happen to have an NVIDIA GPU on your laptop. Not required.

```bash
docker run --rm --gpus all \
  -v $PWD/test_input.json:/workspace/test_input.json \
  <image>
```

The RunPod SDK auto-detects `test_input.json` when run outside the serverless queue.
