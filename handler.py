"""RunPod Serverless handler for DeepSeek-OCR.

Input schema (event["input"]):
    image:        str  - HTTPS URL or base64-encoded image (required)
    prompt:       str  - full prompt, OR use `task` shorthand below (optional)
    task:         str  - "ocr" | "markdown" | "free"  (default: "markdown")
    size:         str  - "tiny" | "small" | "base" | "large" | "gundam"  (default: "gundam")
    base_size:    int  - override size preset
    image_size:   int  - override size preset
    crop_mode:    bool - override size preset
    test_compress:bool - return token compression stats (default: False)
    return_image: bool - return the grounding overlay image as base64 (default: False)

Returns:
    { "text": str, "stats": {...}, "overlay_image_b64": str|null }
"""
import base64
import io
import json
import os
import re
import tempfile
import traceback
from pathlib import Path
from typing import Any

import requests
import runpod
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-OCR")
# Defaults to the path the Dockerfile bakes weights into. To use a RunPod
# network volume instead, set HF_HOME=/runpod-volume/huggingface in the
# endpoint env vars and build the image with --build-arg BAKE_MODEL=0.
HF_CACHE = os.environ.get("HF_HOME", "/opt/hf-cache")


SIZE_PRESETS = {
    "tiny":   {"base_size": 512,  "image_size": 512,  "crop_mode": False},
    "small":  {"base_size": 640,  "image_size": 640,  "crop_mode": False},
    "base":   {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "large":  {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "gundam": {"base_size": 1024, "image_size": 640,  "crop_mode": True},
}

TASK_PROMPTS = {
    "ocr":      "<image>\nFree OCR. ",
    "free":     "<image>\nFree OCR. ",
    "markdown": "<image>\n<|grounding|>Convert the document to markdown. ",
}

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is not None:
        return
    print(f"[init] loading {MODEL_NAME} (cache={HF_CACHE})", flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, cache_dir=HF_CACHE
    )
    _model = AutoModel.from_pretrained(
        MODEL_NAME,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        use_safetensors=True,
        cache_dir=HF_CACHE,
    )
    _model = _model.eval().cuda().to(torch.bfloat16)
    print("[init] model ready", flush=True)


def _decode_image(image_input: str) -> Image.Image:
    """Accept HTTPS URL or base64 string (with or without data: prefix)."""
    if image_input.startswith("http://") or image_input.startswith("https://"):
        resp = requests.get(image_input, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    if image_input.startswith("data:"):
        image_input = image_input.split(",", 1)[1]
    raw = base64.b64decode(image_input)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _resolve_sizing(job_input: dict) -> dict:
    preset = SIZE_PRESETS[job_input.get("size", "gundam").lower()]
    return {
        "base_size":  int(job_input.get("base_size",  preset["base_size"])),
        "image_size": int(job_input.get("image_size", preset["image_size"])),
        "crop_mode":  bool(job_input.get("crop_mode", preset["crop_mode"])),
    }


def _resolve_prompt(job_input: dict) -> str:
    if "prompt" in job_input and job_input["prompt"]:
        return job_input["prompt"]
    task = job_input.get("task", "markdown").lower()
    if task not in TASK_PROMPTS:
        raise ValueError(f"unknown task '{task}'; expected one of {list(TASK_PROMPTS)}")
    return TASK_PROMPTS[task]


def _find_overlay_image(out_dir: Path) -> Path | None:
    """The model writes a grounding-overlay image when <|grounding|> is in the prompt.
    Pick the most recently modified image in out_dir, if any.
    """
    candidates = [p for p in out_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def handler(event: dict[str, Any]) -> dict[str, Any]:
    try:
        _load_model()
        job_input = event.get("input") or {}

        if "image" not in job_input:
            return {"error": "missing required field 'image' (URL or base64)"}

        image = _decode_image(job_input["image"])
        prompt = _resolve_prompt(job_input)
        sizing = _resolve_sizing(job_input)
        test_compress = bool(job_input.get("test_compress", False))
        return_image = bool(job_input.get("return_image", False))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            img_path = tmp_path / "input.png"
            out_path = tmp_path / "out"
            out_path.mkdir()
            image.save(img_path, format="PNG")

            with torch.inference_mode():
                res = _model.infer(
                    _tokenizer,
                    prompt=prompt,
                    image_file=str(img_path),
                    output_path=str(out_path),
                    base_size=sizing["base_size"],
                    image_size=sizing["image_size"],
                    crop_mode=sizing["crop_mode"],
                    save_results=True,
                    test_compress=test_compress,
                )

            text = _extract_text(res, out_path)

            overlay_b64 = None
            if return_image:
                overlay = _find_overlay_image(out_path)
                if overlay is not None:
                    overlay_b64 = base64.b64encode(overlay.read_bytes()).decode()

        return {
            "text": text,
            "prompt": prompt,
            "sizing": sizing,
            "overlay_image_b64": overlay_b64,
        }

    except Exception as e:
        print("[handler] error:", e, flush=True)
        traceback.print_exc()
        return {"error": str(e), "trace": traceback.format_exc()}


def _extract_text(res: Any, out_path: Path) -> str:
    """`model.infer` may return text directly OR write it to a file under output_path.
    Be tolerant of both behaviors across model revisions.
    """
    if isinstance(res, str) and res.strip():
        return _strip_grounding_tags(res)
    if isinstance(res, dict):
        for k in ("text", "output", "result"):
            if k in res and isinstance(res[k], str):
                return _strip_grounding_tags(res[k])

    for name in ("result.mmd", "result.md", "output.md", "output.txt", "result.txt"):
        f = out_path / name
        if f.exists():
            return _strip_grounding_tags(f.read_text())

    for f in sorted(out_path.iterdir()):
        if f.suffix.lower() in {".md", ".mmd", ".txt"}:
            return _strip_grounding_tags(f.read_text())

    return "" if res is None else str(res)


_GROUNDING_RE = re.compile(r"<\|ref\|>.*?<\|/ref\|>|<\|det\|>.*?<\|/det\|>", re.DOTALL)


def _strip_grounding_tags(text: str) -> str:
    """Optionally clean grounding tags. Kept conservative — only strips bbox/ref pairs."""
    return _GROUNDING_RE.sub("", text).strip()


if __name__ == "__main__":
    # RunPod SDK auto-picks up ./test_input.json when no job is in the queue,
    # so the same entrypoint works locally and in the serverless worker.
    runpod.serverless.start({"handler": handler})
