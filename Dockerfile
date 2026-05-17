# Slim build: use the official PyTorch image (python 3.11 + torch 2.6 + cu118
# already inside), then add only what we need on top.
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/runpod-volume/huggingface \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface

# Minimal system deps for image decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt hf_transfer

# Prebuilt flash-attn wheel for py311 + torch 2.6 + cu118 — no compile needed
RUN pip install flash-attn==2.7.3 --no-build-isolation

WORKDIR /workspace
COPY handler.py /workspace/handler.py
COPY test_input.json /workspace/test_input.json

CMD ["python", "-u", "handler.py"]
