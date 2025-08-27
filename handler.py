# /src/handler.py
# Pro-mode: faster-whisper (ASR with full decoder control) + WhisperX alignment (word/char)
import os, json, base64, tempfile, time, math
import runpod
import requests

# -------- lazy heavy imports (ускоряет cold start) --------
_torch = None
_whisperx = None
_FWClass = None  # faster_whisper.WhisperModel

def _lazy_imports():
    global _torch, _whisperx, _FWClass
    if _torch is None:
        import torch as _t
        _torch = _t
    if _whisperx is None:
        import whisperx as _w
        _whisperx = _w
    if _FWClass is None:
        from faster_whisper import WhisperModel as _FW
        _FWClass = _FW
    return _torch, _whisperx, _FWClass

# -------- config / env --------
FORCE_DEVICE    = os.getenv("FORCE_DEVICE", "auto")   # auto|cuda|cpu
DEFAULT_MODEL   = os.getenv("MODEL_NAME", "large-v3")
DEFAULT_BATCH   = int(os.getenv("BATCH_SIZE", "8"))   # (в FW .transcribe не используем)
DEFAULT_COMPUTE = os.getenv("COMPUTE_TYPE", "float16")  # float16 на GPU, int8 на CPU
HF_TOKEN_ENV    = os.getenv("HF_TOKEN")
HF_HOME         = os.getenv("HF_HOME", "/root/.cache/huggingface")
WHISPER_CACHE   = os.getenv("WHISPER_CACHE", "/root/.cache/whisper")

# Разрешённые kwargs для faster-whisper .transcribe(...)
ALLOWED_FW_ARGS = {
    "beam_size","best_of","patience","length_penalty","temperature",
    "compression_ratio_threshold","log_prob_threshold","no_speech_threshold",
    "vad_filter","vad_parameters",
    "condition_on_previous_text","initial_prompt","prefix",
    "suppress_blank","suppress_tokens",
    "without_timestamps","max_initial_timestamp","word_timestamps",
    "prepend_punctuations","append_punctuations","clip_timestamps",
    "hallucination_silence_threshold","hotwords",
    "temperature_increment_on_fallback","chunk_length"
}

# -------- caches --------
_fw_model = None
_fw_cfg   = {}      # {'name','ctype','device'}
_align_model = None
_align_meta  = None
_align_cfg   = {}   # {'lang','model_name'}

# -------- helpers --------
def _device():
    t, _, _ = _lazy_imports()
    if FORCE_DEVICE == "cuda" and t.cuda.is_available():
        return "cuda"
    if FORCE_DEVICE == "cpu":
        return "cpu"
    return "cuda" if t.cuda.is_available() else "cpu"

def _normalize_compute_type(device, requested):
    if device == "cpu":
        return "int8"  # безопасный дефолт для CPU
    return requested or "float16"

def _download_to_tmp(p):
    if p.get("audio_url"):
        url = p["audio_url"]
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            suffix = os.path.splitext(url.split("?")[0])[1] or ".wav"
            fd, path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
        return path
    if p.get("audio_b64"):
        raw = base64.b64decode(p["audio_b64"])
        fd, path = tempfile.mkstemp(suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        return path
    if p.get("audio_path"):
        return p["audio_path"]
    raise ValueError("Provide 'audio_url' or 'audio_b64' or 'audio_path'.")

def _ensure_fw_model(model_name, compute_type):
    """Init faster-whisper WhisperModel once per worker with desired compute_type/device."""
    global _fw_model, _fw_cfg
    _, _, FW = _lazy_imports()
    dev = _device()
    ctype = _normalize_compute_type(dev, compute_type)
    changed = (
        _fw_model is None or
        _fw_cfg.get("name") != model_name or
        _fw_cfg.get("ctype") != ctype or
        _fw_cfg.get("device") != dev
    )
    if changed:
        _fw_model = FW(model_name, device=dev, compute_type=ctype,
                       download_root=WHISPER_CACHE)
        _fw_cfg = {"name": model_name, "ctype": ctype, "device": dev}
    return _fw_model

def _ensure_aligner(lang_code, model_name=None):
    """Init wav2vec2 aligner for given language (RU→ jonatasgrosman/...russian)."""
    global _align_model, _align_meta, _align_cfg
    _, wx, _ = _lazy_imports()
    dev = _device()
    if (
        _align_model is None or _align_meta is None or
        _align_cfg.get("lang") != lang_code or
        (model_name and _align_cfg.get("model_name") != model_name)
    ):
        _align_model, _align_meta = wx.load_align_model(
            language_code=lang_code, device=dev, model_name=model_name
        )
        _align_cfg = {"lang": lang_code, "model_name": model_name}
    return _align_model, _align_meta

def _ts_srt(t):
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _make_srt(segments):
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_ts_srt(seg.get('start', 0))} --> {_ts_srt(seg.get('end', 0))}")
        lines.append((seg.get("text") or "").strip()); lines.append("")
    return "\n".join(lines)

def _make_vtt(segments):
    def ts(t):
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{ts(seg.get('start',0))} --> {ts(seg.get('end',0))}")
        lines.append((seg.get("text") or "").strip()); lines.append("")
    return "\n".join(lines)

def _log_versions_once():
    if os.environ.get("VER_LOGGED"):
        return
    t, _, _ = _lazy_imports()
    try:
        import ctranslate2
        print(f"[versions] torch={t.__version__} cuda={t.version.cuda} cudnn={t.backends.cudnn.version()} ct2={ctranslate2.__version__}")
    except Exception as e:
        print(f"[versions] torch=? err={e}")
    os.environ["VER_LOGGED"] = "1"

def _preload_once():
    if os.environ.get("PRELOADED"):
        return
    _ensure_fw_model(os.getenv("MODEL_NAME", "large-v3"), os.getenv("COMPUTE_TYPE", "float16"))
    for lang in (os.getenv("PRELOAD_LANGS","ru").split()):
        _ensure_aligner(lang)
    os.environ["PRELOADED"] = "1"

# -------- main handler --------
def handler(job):
    _log_versions_once()
    if os.getenv("PRELOAD","1") == "1":
        _preload_once()

    t0 = time.time()
    p = job.get("input", {})

    model_name   = p.get("model", DEFAULT_MODEL)
    batch_size   = int(p.get("batch_size", DEFAULT_BATCH))  # (не используется FW, но оставили для совместимости API)
    compute_type = p.get("compute_type", DEFAULT_COMPUTE)

    language     = p.get("language")           # "ru" и т.п.
    align        = bool(p.get("align", True))
    char_align   = bool(p.get("char_align", False))
    diarize      = bool(p.get("diarize", False))
    hf_token     = p.get("hf_token") or HF_TOKEN_ENV

    align_model_name = p.get("align_model")    # явный HF aligner
    return_raw  = bool(p.get("return_raw", True))
    return_srt  = bool(p.get("return_srt", True))
    return_vtt  = bool(p.get("return_vtt", False))

    # faster-whisper kwargs (полный контроль декодера) — БЕЗ batch_size
    fw_over = p.get("whisper", {}) or {}
    fw_kwargs = {}
    for k, v in fw_over.items():
        if k in ALLOWED_FW_ARGS:
            fw_kwargs[k] = v

    # 1) Аудио (FW умеет принимать путь)
    audio_path = _download_to_tmp(p)

    # 2) ASR: faster-whisper напрямую
    fw = _ensure_fw_model(model_name, compute_type)
    segments_iter, info = fw.transcribe(audio_path, language=language, **fw_kwargs)

    # 2.1) normalize → список
    segments_raw = []
    for s in segments_iter:
        segments_raw.append({
            "id": s.id,
            "start": float(s.start or 0.0),
            "end": float(s.end or 0.0),
            "text": (s.text or "").strip()
        })

    detected_lang = info.language or language

    # Если потребуется align или diarize — загрузим аудио через whisperX (16k, float32)
    audio_wx = None
    wx = None
    if (align and segments_raw) or diarize:
        _, wx, _ = _lazy_imports()
        audio_wx = wx.load_audio(audio_path)

    # 3) Alignment (wav2vec2)
    segments_aligned = segments_raw
    diarize_segments = None
    if align and segments_raw:
        lang = (language or detected_lang or "ru")
        align_model, meta = _ensure_aligner(lang, model_name=align_model_name)
        aligned = wx.align(segments_raw, align_model, meta, audio_wx, _device(),
                           return_char_alignments=char_align)
        segments_aligned = aligned.get("segments", aligned)

    # 4) (Optional) diarization
    if diarize:
        if not hf_token:
            raise ValueError("Diarization requested but no HF token provided (env HF_TOKEN or input.hf_token)")
        diar = wx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=_device())
        diarize_segments = diar(audio_wx,
                                min_speakers=p.get("min_speakers"),
                                max_speakers=p.get("max_speakers"))
        segments_aligned = wx.assign_word_speakers(diarize_segments, {"segments": segments_aligned})["segments"]

    # 5) build output
    out = {
        "device": _device(),
        "model": model_name,
        "compute_type": _fw_cfg.get("ctype"),
        "language": detected_lang,
        "timing": {"total_sec": round(time.time() - t0, 3)}
    }
    if return_raw:
        out["segments_raw"] = segments_raw
    out["segments"] = segments_aligned
    if diarize_segments is not None:
        out["diarization"] = diarize_segments
    if return_srt:
        out["srt"] = _make_srt(segments_aligned)
    if return_vtt:
        out["vtt"] = _make_vtt(segments_aligned)

    try:
        os.remove(audio_path)
    except Exception:
        pass

    return out

# -------- runpod entry --------
runpod.serverless.start({"handler": handler})
