import os, re, json, uuid, time, wave, shutil, pathlib, tempfile, subprocess, threading
from typing import Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from google import genai
    from google.genai import types
except Exception:  # surfaced clearly by /health if dependency is missing
    genai = None
    types = None

BASE = pathlib.Path(__file__).parent
WORK = pathlib.Path(tempfile.gettempdir()) / "ai_dubbing_studio_gemini"
WORK.mkdir(exist_ok=True)
KEY = os.getenv("GEMINI_API_KEY", "").strip()
TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.6-flash").strip()
TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts").strip()
JOBS: dict[str, dict[str, Any]] = {}
LOCK = threading.Lock()
app = FastAPI(title="AI Dubbing Studio PRO - Gemini")

@app.get("/health")
def health():
    return {
        "ok": True,
        "engine": "Google Gemini",
        "api_key_configured": bool(KEY),
        "ffmpeg_available": bool(shutil.which("ffmpeg")),
        "sdk_available": genai is not None,
        "text_model": TEXT_MODEL,
        "tts_model": TTS_MODEL,
    }

def update_job(job_id: str, **values):
    with LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(values)

def clean_json(text: str):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("لم يرجع Gemini بيانات حوار منظمة.")
    return json.loads(text[start:end+1])

def save_upload(upload: UploadFile, destination: pathlib.Path):
    size = 0
    with destination.open("wb") as out:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > 1024 * 1024 * 1024:
                raise HTTPException(413, "الحد الأقصى لحجم الملف 1 جيجابايت.")
            out.write(chunk)
    if size == 0:
        raise HTTPException(400, "الملف فارغ.")
    return size

def make_audio_input(source: pathlib.Path, suffix: str, workdir: pathlib.Path):
    ffmpeg = shutil.which("ffmpeg")
    if suffix in {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".mp4", ".mov", ".mkv", ".webm", ".at3"} and not ffmpeg and suffix not in {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}:
        raise RuntimeError("يلزم FFmpeg على الخادم لمعالجة الفيديو أو AT3.")
    if suffix in {".mp4", ".mov", ".mkv", ".webm"}:
        target = workdir / "source_audio.wav"
        cmd = [ffmpeg, "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(target)]
    elif suffix == ".at3":
        target = workdir / "source_audio.wav"
        cmd = [ffmpeg, "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(target)]
    elif suffix in {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}:
        # Gemini accepts these common audio formats directly; avoid needless conversion.
        return source
    else:
        raise RuntimeError("صيغة الملف غير مدعومة.")
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if run.returncode or not target.exists() or target.stat().st_size == 0:
        raise RuntimeError("تعذر استخراج الصوت أو تحويله. " + (run.stderr or "")[-900:])
    return target

def transcribe_translate(client, audio_path: pathlib.Path, target_language: str, source_language: str):
    uploaded = client.files.upload(file=str(audio_path))
    lang = "Egyptian Arabic (Masri), natural everyday Egyptian dialect" if target_language == "ar-EG" else target_language
    source = source_language or "detect the source language automatically"
    prompt = f'''Listen to the attached audio carefully. Transcribe the spoken dialogue, identify speakers, and translate every spoken line into {lang}. Source language: {source}.
Return ONLY valid JSON in this exact schema:
{{"source_language":"...","segments":[{{"speaker":"Speaker 1","original":"...","translated":"..."}}]}}
Rules: preserve meaning, names, emotion, humor and conversational intent; use natural spoken language, not literal machine translation; keep separate speakers distinct; do not invent dialogue; omit music and sound effects; do not summarize; include all intelligible spoken dialogue in chronological order. Use at most two speaker labels (Speaker 1 and Speaker 2), assigning consistently. If only one speaker, use Speaker 1. For Egyptian Arabic, use natural Egyptian colloquial vocabulary and phrasing, not Modern Standard Arabic. Keep each segment reasonably short (one sentence or utterance).'''
    response = client.models.generate_content(model=TEXT_MODEL, contents=[uploaded, prompt], config=types.GenerateContentConfig(response_mime_type="application/json"))
    data = clean_json(response.text)
    segments = data.get("segments") or []
    cleaned = []
    for item in segments:
        translated = str(item.get("translated", "")).strip()
        original = str(item.get("original", "")).strip()
        if translated:
            speaker = str(item.get("speaker", "Speaker 1")).strip()
            if speaker not in {"Speaker 1", "Speaker 2"}:
                speaker = "Speaker 1"
            cleaned.append({"speaker": speaker, "original": original, "translated": translated})
    if not cleaned:
        raise RuntimeError("لم يتمكن Gemini من استخراج حوار واضح من الملف.")
    return data.get("source_language", "غير محددة"), cleaned

def synthesize(client, segments, output_path: pathlib.Path, dialect: str, voice1: str, voice2: str):
    # Gemini TTS supports two-speaker dialogue. Each utterance is tagged with its speaker.
    parts = []
    for seg in segments:
        parts.append(types.Part(text=seg["translated"], speech_metadata=types.SpeechMetadata(speaker=seg["speaker"])))
    cfg = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(speaker="Speaker 1", voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice1))),
                    types.SpeakerVoiceConfig(speaker="Speaker 2", voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice2))),
                ]
            )
        ),
    )
    prompt = f"Speak the following translated dialogue naturally in {dialect}. Keep the two speakers clearly distinct. Use expressive, conversational delivery and natural pauses. Read only the dialogue, do not add words."
    response = client.models.generate_content(model=TTS_MODEL, contents=[types.Content(role="user", parts=[types.Part(text=prompt), *parts])], config=cfg)
    audio_bytes = None
    mime = ""
    for candidate in (response.candidates or []):
        for part in (candidate.content.parts or []):
            if getattr(part, "inline_data", None) and part.inline_data.data:
                audio_bytes = part.inline_data.data
                mime = getattr(part.inline_data, "mime_type", "") or ""
                break
        if audio_bytes:
            break
    if not audio_bytes:
        raise RuntimeError("لم يرجع نموذج Gemini ملفاً صوتياً. تحقق من توفر نموذج TTS في حسابك.")
    if audio_bytes[:4] == b"RIFF":
        output_path.write_bytes(audio_bytes)
    else:
        # Gemini TTS commonly returns raw 24 kHz, 16-bit mono PCM.
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio_bytes)

def process_job(job_id: str, source: pathlib.Path, suffix: str, target_language: str, source_language: str, voice1: str, voice2: str):
    jobdir = WORK / job_id
    jobdir.mkdir(parents=True, exist_ok=True)
    try:
        if not KEY or genai is None:
            raise RuntimeError("لم يتم إعداد GEMINI_API_KEY على الخادم أو مكتبة Gemini غير مثبتة.")
        client = genai.Client(api_key=KEY)
        update_job(job_id, status="processing", progress="تحضير الملف الصوتي...")
        audio_input = make_audio_input(source, suffix, jobdir)
        update_job(job_id, progress="Gemini يستمع إلى الملف ويستخرج الحوار ويترجمه...")
        source_lang, segments = transcribe_translate(client, audio_input, target_language, source_language)
        (jobdir / "transcript.json").write_text(json.dumps({"source_language": source_lang, "segments": segments}, ensure_ascii=False, indent=2), encoding="utf-8")
        update_job(job_id, progress=f"تم استخراج {len(segments)} مقطعاً حوارياً. جارٍ توليد الأصوات...")
        dialect = "Egyptian Arabic dialect (Masri)" if target_language == "ar-EG" else target_language
        out_audio = jobdir / "dubbed_audio.wav"
        synthesize(client, segments, out_audio, dialect, voice1, voice2)
        video_out = None
        if suffix in {".mp4", ".mov", ".mkv", ".webm"}:
            ffmpeg = shutil.which("ffmpeg")
            video_out = jobdir / "dubbed_video.mp4"
            cmd = [ffmpeg, "-y", "-i", str(source), "-i", str(out_audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(video_out)]
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if run.returncode:
                raise RuntimeError("اكتمل الصوت لكن تعذر دمجه بالفيديو: " + (run.stderr or "")[-700:])
        update_job(job_id, status="completed", progress="اكتملت الدبلجة بنجاح.", audio_url=f"/api/download/{job_id}/audio", video_url=f"/api/download/{job_id}/video" if video_out else None, segment_count=len(segments), source_language=source_lang)
    except Exception as exc:
        update_job(job_id, status="failed", progress="توقفت العملية بسبب خطأ.", error=str(exc))

@app.post("/api/dub")
async def create_dub(background_tasks: BackgroundTasks, file: UploadFile = File(...), target_language: str = Form("ar-EG"), source_language: str = Form(""), voice1: str = Form("Kore"), voice2: str = Form("Puck")):
    suffix = pathlib.Path(file.filename or "media.bin").suffix.lower()
    allowed = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".mp4", ".mov", ".mkv", ".webm", ".at3"}
    if suffix not in allowed:
        raise HTTPException(400, "صيغة الملف غير مدعومة.")
    job_id = uuid.uuid4().hex[:12]
    jobdir = WORK / job_id
    jobdir.mkdir(parents=True, exist_ok=True)
    source = jobdir / ("original" + suffix)
    try:
        save_upload(file, source)
    except Exception:
        shutil.rmtree(jobdir, ignore_errors=True)
        raise
    with LOCK:
        JOBS[job_id] = {"status": "queued", "progress": "تم استلام الملف، وسيبدأ Gemini معالجته.", "filename": file.filename}
    background_tasks.add_task(process_job, job_id, source, suffix, target_language, source_language, voice1, voice2)
    return {"job_id": job_id, "status": "queued", "filename": file.filename}

@app.get("/api/status/{job_id}")
def job_status(job_id: str):
    with LOCK:
        item = JOBS.get(job_id)
    if not item:
        raise HTTPException(404, "لم يتم العثور على مشروع الدبلجة.")
    return {"job_id": job_id, **item}

@app.get("/api/download/{job_id}/{kind}")
def download(job_id: str, kind: str):
    if kind not in {"audio", "video", "transcript"}:
        raise HTTPException(400, "نوع التنزيل غير صحيح.")
    folder = WORK / job_id
    paths = {"audio": folder / "dubbed_audio.wav", "video": folder / "dubbed_video.mp4", "transcript": folder / "transcript.json"}
    path = paths[kind]
    if not path.exists():
        raise HTTPException(409, "الملف غير جاهز للتنزيل بعد.")
    names = {"audio": "dubbed_audio.wav", "video": "dubbed_video.mp4", "transcript": "dubbed_transcript.json"}
    media = {"audio": "audio/wav", "video": "video/mp4", "transcript": "application/json"}
    return FileResponse(path, filename=names[kind], media_type=media[kind])

app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
