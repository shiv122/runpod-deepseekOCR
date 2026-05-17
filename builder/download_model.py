"""Pre-download DeepSeek-OCR weights into the image so cold starts don't pay
for a multi-GB download. Run during `docker build`.

If you would rather keep the image small and store weights on a RunPod network
volume mounted at /runpod-volume, skip this step and let the handler pull on
first invocation — subsequent containers attached to the same volume reuse it.
"""
import os

from huggingface_hub import snapshot_download

MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-OCR")
CACHE_DIR = os.environ.get("HF_HOME", "/opt/hf-cache")

if __name__ == "__main__":
    print(f"Downloading {MODEL_NAME} -> {CACHE_DIR}", flush=True)
    snapshot_download(
        repo_id=MODEL_NAME,
        cache_dir=CACHE_DIR,
        allow_patterns=[
            "*.json",
            "*.py",
            "*.safetensors",
            "*.model",
            "tokenizer*",
        ],
    )
    print("Done.", flush=True)
