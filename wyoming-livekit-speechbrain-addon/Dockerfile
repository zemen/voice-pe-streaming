ARG BUILD_FROM
FROM $BUILD_FROM

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf-cache \
    SPEECHBRAIN_CACHE_DIR=/data/speechbrain

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        git \
        jq \
        ca-certificates \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --upgrade pip setuptools wheel
# torch + torchaudio must both come from the CPU index. If only torch is pinned
# here, speechbrain's later install pulls torchaudio from PyPI which is a CUDA
# build that dlopens libcudart.so at import time and crashes on CPU-only hosts.
RUN python3 -m pip install --break-system-packages \
        torch==2.4.* torchaudio==2.4.* --index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install --break-system-packages \
        numpy \
        onnxruntime \
        wyoming \
        livekit-wakeword \
        speechbrain \
        aiohttp
RUN python3 -m pip install --break-system-packages --no-deps \
        "git+https://github.com/zemen/voice-pe-streaming@master"

COPY sputnik.onnx /app/models/sputnik.onnx
COPY run.sh /run.sh
RUN chmod a+x /run.sh

CMD ["/run.sh"]
