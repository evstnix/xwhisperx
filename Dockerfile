FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    WHISPER_CACHE=/root/.cache/whisper \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

# base deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev ffmpeg libsndfile1 git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# ✅ Torch/Torchaudio >= 2.5 (ставим сразу 2.8.x; колёса подтянут nvidia-cu12 зависимости)
RUN pip install --no-cache-dir "torch==2.8.*" "torchaudio==2.8.*"

# faster-whisper + CTranslate2 (GPU)
RUN pip install --no-cache-dir "ctranslate2>=4.4,<5" "faster-whisper>=1.0,<2"

# WhisperX + утилиты + runpod SDK
RUN pip install --no-cache-dir "whisperx==3.4.2" runpod==1.7.13 requests srt numpy

WORKDIR /src
COPY handler.py /src/handler.py
RUN python3 -m py_compile /src/handler.py

CMD ["python3", "-u", "handler.py"]
