# /src/handler.py
# faster-whisper (ASR) + WhisperX align + постпроцесс для таймингов

import os, json, base64, tempfile, time, re
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
FORCE_DEVICE    = os.getenv("FORCE_DEVICE", "auto")        # auto|cuda|cpu
DEFAULT_MODEL   = os.getenv("MODEL_NAME", "large-v3")
DEFAULT_BATCH   = int(os.getenv("BATCH_SIZE", "8"))        # (в FW .transcribe не используем)
DEFAULT_COMPUTE = os.getenv("COMPUTE_TYPE", "float16")     # float16 на GPU, int8 на CPU
HF_TOKEN_ENV    = os.getenv("HF_TOKEN")
HF_HOME         = os.getenv("HF_HOME", "/root/.cache/huggingface")
WHISPER_CACHE   = os.getenv("WHISPER_CACHE", "/root/.cache/whisper")

# allowed kwargs для faster-whisper .transcribe(...)
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
_PUNCT_TRAIL = r"\.\,\!\?\:\;\)\]\»”…"
_PUNCT_LEAD  = r"\(\[\«“"

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
        _fw_model = FW(model_name, device=dev, compute_type=ctype, download_root=WHISPER_CACHE)
        _fw_cfg = {"name": model_name, "ctype": ctype, "device": dev}
    return _fw_model

def _ensure_aligner(lang_code, model_name=None):
    global _align_model, _align_meta, _align_cfg
    _, wx, _ = _lazy_imports()
    dev = _device()
    if (
        _align_model is None or _align_meta is None or
        _align_cfg.get("lang") != lang_code or
        (model_name and _align_cfg.get("model_name") != model_name)
    ):
        _align_model, _align_meta = wx.load_align_model(language_code=lang_code, device=dev, model_name=model_name)
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

# -------- постпроцесс таймингов --------
def _norm_text(t: str) -> str:
    return re.sub(r'[\s\W]+', ' ', (t or '')).strip().lower()

def _snap_to_confident_words(seg, min_score=0.75):
    ws = seg.get("words") or []
    good = [w for w in ws if w.get("score", 1.0) >= min_score]
    if good:
        seg["start"] = float(good[0]["start"])
        seg["end"]   = float(good[-1]["end"])
    return seg

def _limit_long_words(segments, max_word_dur=1.2):
    for seg in segments:
        ws = seg.get("words") or []
        for i, w in enumerate(ws):
            s = float(w["start"]); e = float(w["end"])
            if e - s > max_word_dur:
                e_new = s + max_word_dur
                w["end"] = e_new
                # не даём следующему слову пересекать насильно подрезанный хвост
                if i + 1 < len(ws) and float(ws[i+1]["start"]) < e_new:
                    ws[i+1]["start"] = e_new
        if ws:
            seg["start"] = float(ws[0]["start"])
            seg["end"]   = float(ws[-1]["end"])
    return segments

def _round_and_fix(segments, nd=3, eps=0.001, min_word_dur=0.06):
    def r(x): return round(float(x or 0.0), nd)
    for s in segments:
        ws = s.get("words") or []
        # round words
        for w in ws:
            w["start"] = r(w.get("start"))
            w["end"]   = r(w.get("end"))
        # enforce monotonic after rounding
        if ws:
            last_end = ws[0]["start"] - eps
            for w in ws:
                if w["start"] < last_end + eps:
                    w["start"] = last_end + eps
                if w["end"] <= w["start"] + min_word_dur:
                    w["end"] = w["start"] + min_word_dur
                last_end = w["end"]
            s["start"] = ws[0]["start"]
            s["end"]   = ws[-1]["end"]
        # round segment bounds too
        s["start"] = r(s.get("start"))
        s["end"]   = r(s.get("end"))
    return segments

def _deoverlap_words(segments, eps=0.02, min_word_dur=0.06, pad=0.0):
    for seg in segments:
        ws = seg.get("words") or []
        if not ws:
            continue
        last_end = float(ws[0]["start"]) - eps
        for w in ws:
            s = float(w["start"]); e = float(w["end"])
            if s < last_end + eps:
                s = last_end + eps                  # сдвинуть старт вперёд
            if e <= s + min_word_dur:
                e = s + min_word_dur               # обеспечить минимальную длительность
            w["start"] = s; w["end"] = e
            last_end = e + pad
        seg["start"] = float(ws[0]["start"])
        seg["end"]   = float(ws[-1]["end"])
    return segments



def _join_words(words):
    t = " ".join(x["word"] for x in words)
    t = re.sub(rf"\s+([{_PUNCT_TRAIL}])", r"\1", t)  # пробелы перед .,!?:;… и закрывающими
    t = re.sub(rf"([{_PUNCT_LEAD}])\s+", r"\1", t)   # пробелы после открывающих «“([ 
    t = re.sub(r"\s+", " ", t).strip()
    return t



def _split_segments_by_gaps(segments, gap=0.60):
    out = []
    for seg in segments:
        ws = seg.get("words") or []
        if not ws:
            out.append(seg); continue
        cur = [ws[0]]
        for p, w in zip(ws, ws[1:]):
            if float(w["start"]) - float(p["end"]) > gap:
                out.append({
                    "start": float(cur[0]["start"]),
                    "end":   float(cur[-1]["end"]),
                    "text": _join_words(cur),
                    "words": cur
                })
                cur = [w]
            else:
                cur.append(w)
        out.append({
            "start": float(cur[0]["start"]),
            "end":   float(cur[-1]["end"]),
            "text":  _join_words(cur),
            "words": cur
        })
    return out

# -------- main handler --------
def handler(job):
    _log_versions_once()

    t0 = time.time()
    p = job.get("input", {}) or {}

    model_name   = p.get("model", DEFAULT_MODEL)
    batch_size   = int(p.get("batch_size", DEFAULT_BATCH))  # (зарезервировано)
    compute_type = p.get("compute_type", DEFAULT_COMPUTE)

    language     = p.get("language")
    align        = bool(p.get("align", True))
    char_align   = bool(p.get("char_align", False))
    diarize      = bool(p.get("diarize", False))
    hf_token     = p.get("hf_token") or HF_TOKEN_ENV

    align_model_name = p.get("align_model")
    return_raw  = bool(p.get("return_raw", True))
    return_srt  = bool(p.get("return_srt", True))
    return_vtt  = bool(p.get("return_vtt", False))

    # --- постпроцесс (всегда передаёшь сам) ---
    pp = p.get("postprocess", {}) or {}
    PP_SNAP  = float(pp.get("snap_min_score", 0.75))
    PP_MAXW  = float(pp.get("max_word_dur", 1.2))
    _raw_gap = pp.get("gap_split")                  # важный момент: None -> не резать
    PP_GAP   = None if _raw_gap is None else float(_raw_gap)
    PP_DEOVERLAP   = float(pp.get("deoverlap_eps", 0.02))
    PP_MINW_FLOOR  = float(pp.get("min_word_dur_floor", 0.06))
    PP_PAD         = float(pp.get("overlap_pad", 0.0))

    # --- faster-whisper kwargs — БЕЗ batch_size ---
    fw_over = p.get("whisper", {}) or {}
    fw_kwargs = {k: v for k, v in fw_over.items() if k in ALLOWED_FW_ARGS}

    # 1) Аудио (FW принимает путь)
    audio_path = _download_to_tmp(p)

    # 2) ASR
    fw = _ensure_fw_model(model_name, compute_type)
    segments_iter, info = fw.transcribe(audio_path, language=language, **fw_kwargs)

    segments_raw = []
    for s in segments_iter:
        segments_raw.append({
            "id": s.id,
            "start": float(s.start or 0.0),
            "end": float(s.end or 0.0),
            "text": (s.text or "").strip()
        })

    detected_lang = info.language or language

    # 3) Alignment (и подготовка аудио)
    audio_wx = None; wx = None
    segments_aligned = segments_raw
    diarize_segments = None

    if (align and segments_raw) or diarize:
        _, wx, _ = _lazy_imports()
        audio_wx = wx.load_audio(audio_path)

    if align and segments_raw:
        lang = (language or detected_lang or "ru")
        align_model, meta = _ensure_aligner(lang, model_name=align_model_name)
        aligned = wx.align(
            segments_raw, align_model, meta, audio_wx, _device(),
            return_char_alignments=char_align
        )
        segments_aligned = aligned.get("segments", aligned)

        # 3.1) поджать края по уверенным словам
        segments_aligned = [_snap_to_confident_words(s, PP_SNAP) for s in segments_aligned]

        # 3.2) ограничить длительность слова и убрать наезды
        segments_aligned = _limit_long_words(segments_aligned, max_word_dur=PP_MAXW)
        segments_aligned = _deoverlap_words(
            segments_aligned,
            eps=PP_DEOVERLAP,
            min_word_dur=PP_MINW_FLOOR,
            pad=PP_PAD
        )

        # 3.3) режем по запросу
        if PP_GAP is not None:
            if PP_GAP > 0:
                # режим "фразы между паузами"
                segments_aligned = _split_segments_by_gaps(segments_aligned, gap=PP_GAP)
            else:
                # PP_GAP == 0.0 → режим "по словам": взрываем в сегменты-слова
                exploded = []
                next_id = 0
                for seg in segments_aligned:
                    ws = seg.get("words") or []
                    if not ws:
                        # если слов нет — оставим как есть
                        seg = dict(seg)
                        seg["id"] = next_id; next_id += 1
                        exploded.append(seg)
                        continue
                    for w in ws:
                        exploded.append({
                            "id": next_id,
                            "start": float(w.get("start", 0.0)),
                            "end":   float(w.get("end",   0.0)),
                            "text":  (w.get("word") or "").strip(),
                            "words": [{
                                "word":  (w.get("word") or "").strip(),
                                "start": float(w.get("start", 0.0)),
                                "end":   float(w.get("end",   0.0)),
                                "score": float(w.get("score", 1.0)),
                            }],
                        })
                        next_id += 1
                segments_aligned = exploded

    # 4) Диаризация (опционально)
    if diarize:
        if not hf_token:
            raise ValueError("Diarization requested but no HF token provided (env HF_TOKEN or input.hf_token)")
        diar = wx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=_device())
        diarize_segments = diar(
            audio_wx,
            min_speakers=p.get("min_speakers"),
            max_speakers=p.get("max_speakers")
        )
        segments_aligned = wx.assign_word_speakers(diarize_segments, {"segments": segments_aligned})["segments"]

    # 5) финальный фикс/округление (один раз, в самом конце)
    segments_aligned = _round_and_fix(
        segments_aligned,
        nd=3,
        eps=max(0.001, PP_DEOVERLAP / 2),
        min_word_dur=PP_MINW_FLOOR
    )

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

runpod.serverless.start({"handler": handler})
