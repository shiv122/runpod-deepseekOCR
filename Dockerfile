# Matches versions DeepSeek-OCR was tested on:
#   torch 2.6.0 + CUDA 11.8 + flash-attn 2.7.3
# Using Python 3.11 (3.11 has prebuilt wheels for every dep here, mirroring 3.12.9).
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# BAKE_MODEL=1 (default) bakes weights into the image for fast cold start.
# Set to 0 if you'd rather attach a RunPod network volume mounted at
# /runpod-volume and let the handler download on first run.
ARG BAKE_MODEL=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/opt/hf-cache \
    TRANSFORMERS_CACHE=/opt/hf-cache \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# System deps + Python 3.11 from deadsnakes
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common ca-certificates curl git \
      libgl1 libglib2.0-0 \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-dev python3.11-distutils \
 && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel packaging ninja

# Torch must land before flash-attn so its build env is satisfied
RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt hf_transfer

# Prebuilt flash-attn wheel exists for py311 + torch 2.6 + cu118
RUN pip install flash-attn==2.7.3 --no-build-isolation

# Bake model weights (~7GB) for sub-second cold starts. Skip with --build-arg BAKE_MODEL=0.
COPY builder/download_model.py /tmp/download_model.py
RUN if [ "$BAKE_MODEL" = "1" ]; then \
      python /tmp/download_model.py ; \
    else \
      echo "Skipping model bake-in; handler will fetch on first call." ; \
    fi

WORKDIR /workspace
COPY handler.py /workspace/handler.py
COPY test_input.json /workspace/test_input.json

CMD ["python", "-u", "handler.py"]
