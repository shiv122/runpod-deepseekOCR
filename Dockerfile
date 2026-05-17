# Slim build for RunPod's GitHub flow:
#   - pytorch/pytorch base has torch 2.6 + torchvision + python 3.11 preinstalled
#   - flash-attn pinned to a direct wheel URL, ABI auto-detected
#   - Model is NOT baked; first request downloads it onto the attached network volume
# Final image: ~2GB. Build push completes in 1-2 min and avoids the I/O error
# that RunPod's registry hits on multi-GB layers.
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/runpod-volume/huggingface \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install -r /tmp/requirements.txt hf_transfer

# Install flash-attn from a direct wheel URL matching this image's torch ABI.
# (Pip's resolver doesn't auto-pick the right cxx11abi variant from PyPI, so we
# detect the ABI and fetch the matching wheel explicitly.)
RUN ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')") \
 && echo "Detected cxx11abi=$ABI" \
 && pip install \
      "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.6cxx11abi${ABI}-cp311-cp311-linux_x86_64.whl"

WORKDIR /workspace
COPY handler.py /workspace/handler.py
COPY test_input.json /workspace/test_input.json

CMD ["python", "-u", "handler.py"]
