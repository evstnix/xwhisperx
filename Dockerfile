FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    WHISPER_CACHE=/root/.cache/whisper \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev ffmpeg libsndfile1 git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Torch/Torchaudio для cuDNN 9
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.4.0+cu124 torchaudio==2.4.0+cu124

# CTranslate2 + faster-whisper
RUN pip install --no-cache-dir 'ctranslate2>=4.4,<5' 'faster-whisper>=1.0,<2'

# WhisperX + utils
RUN pip install --no-cache-dir whisperx==3.4.2 runpod==1.7.13 requests srt numpy

# Предзагрузка весов в слой образа (ускоряет cold start)
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"
RUN python3 -c "import whisperx; whisperx.load_align_model(language_code='ru', device='cpu', model_name=None)"

WORKDIR /src
COPY handler.py /src/handler.py
RUN python3 -m py_compile /src/handler.py

CMD ["python3", "-u", "handler.py"]
