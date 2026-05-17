# Slim build for RunPod's GitHub flow:
#   - pytorch/pytorch -runtime base: torch 2.6 + torchvision + python 3.11, no dev toolchain
#   - flash-attn pinned to a direct wheel URL (ABI auto-detected), no source build
#   - Model is NOT baked; first request downloads it onto the attached volume / container disk
# Keeps every added layer well under 1 GB, which avoids the I/O error
# RunPod's registry hits when committing multi-GB layers.
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/opt/hf-cache \
    TRANSFORMERS_CACHE=/opt/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# One pip layer: requirements + ABI-matched flash-attn wheel. The wheel URL is
# resolved at build time because PyPI's resolver doesn't pick the right
# cxx11abi variant on its own.
COPY requirements.txt /tmp/requirements.txt
RUN ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')") \
 && pip install -r /tmp/requirements.txt hf_transfer \
      "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.6cxx11abi${ABI}-cp311-cp311-linux_x86_64.whl"

WORKDIR /workspace
COPY handler.py /workspace/handler.py
COPY test_input.json /workspace/test_input.json

CMD ["python", "-u", "handler.py"]
