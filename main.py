import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from typing import Optional

import anthropic
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Whisper model cache
# ---------------------------------------------------------------------------
_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


# ---------------------------------------------------------------------------
# In-memory job store
# ---------------------------------------------------------------------------
jobs: dict = {}

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="UGC Ad Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    ad_id: Optional[str] = None
    video_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Claude prompt
# ---------------------------------------------------------------------------
CLAUDE_PROMPT = """\
Du bist ein erfahrener UGC-Ad-Stratege und Direct-Response-Marketing-Experte.

Analysiere das folgende Transkript einer Video-Ad (Quelle: {label}) und erstelle eine vollständige Analyse auf Deutsch.

TRANSKRIPT:
{transcript}

Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt (kein Text davor oder danach, nur reines JSON):

{{
  "analyse": {{
    "hook_typ": "Typ des Hooks (z.B. Problem-Hook, Curiosity-Hook, Social-Proof-Hook, Story-Hook, Shock-Hook, Transformation-Hook)",
    "hook_beschreibung": "2-3 Sätze: Warum dieser Hook-Typ, wie er eingesetzt wird und wie wirkungsvoll er ist",
    "staerken": ["Stärke 1", "Stärke 2", "Stärke 3", "Stärke 4"],
    "schwaechen": ["Schwäche 1", "Schwäche 2", "Schwäche 3"],
    "bewertung": "B",
    "bewertung_begruendung": "Ausführliche Begründung der Note A-F mit konkreten Verbesserungsvorschlägen",
    "funnel_stage": "ToFu",
    "funnel_erklaerung": "Warum genau diese Funnel-Stage (ToFu/MoFu/BoFu) und was das für die Zielseite bedeutet",
    "zielgruppe": "Detaillierte Zielgruppen-Beschreibung: Alter, Interessen, Probleme, Kaufbereitschaft, Awareness-Level",
    "psychologische_trigger": ["FOMO", "Social Proof", "Authority", "Scarcity"],
    "gesamtbewertung": "Zusammenfassende Bewertung der Ad in 2-3 prägnanten Sätzen"
  }},
  "replika_script": {{
    "hook": "Hook-Text mit Platzhaltern [DEINE MARKE], [DEIN PRODUKT]",
    "problem_agitation": "Problem-Agitation Abschnitt – schmerzhaft und spezifisch",
    "loesung": "Lösungs-Präsentation mit klarem Nutzenversprechen",
    "social_proof": "Social-Proof Abschnitt mit konkreten Zahlen/Beispielen",
    "cta": "Call-to-Action – klar, dringend, handlungsauffordernd",
    "vollstaendiges_script": "Vollständiger Script-Text zum direkt Ablesen, mindestens 200 Wörter. Verwende [DEINE MARKE], [DEIN PRODUKT], [PREIS], [ANZAHL KUNDEN] als Platzhalter. Conversion-optimiert, natürliche Sprache, professionell."
  }},
  "hook_varianten": [
    {{
      "typ": "Problem-Hook",
      "text": "Vollständiger Hook-Text (2-3 packende Sätze die sofort stoppen)",
      "warum": "Psychologische Erklärung warum dieser Hook konvertiert"
    }},
    {{
      "typ": "Curiosity-Hook",
      "text": "Vollständiger Hook-Text (2-3 packende Sätze die Neugier wecken)",
      "warum": "Psychologische Erklärung warum dieser Hook konvertiert"
    }},
    {{
      "typ": "Transformation-Hook",
      "text": "Vollständiger Hook-Text (2-3 packende Sätze über Transformation)",
      "warum": "Psychologische Erklärung warum dieser Hook konvertiert"
    }}
  ],
  "tool_prompts": {{
    "heygen": "Detaillierter HeyGen Avatar-Prompt: Beschreibe Avatar-Typ (Alter, Geschlecht, Ethnie, Kleidungsstil), Setting/Hintergrund, Körpersprache, Energie-Level, Präsentationstempo und emotionalen Ausdruck. Mindestens 80 Wörter.",
    "elevenlabs": "Detaillierter ElevenLabs Voice-Prompt: Stimm-Charakteristik (männlich/weiblich, Alter, Akzent), Sprechtempo (langsam/mittel/schnell), Emotionslevel (warm/professionell/aufgeregt), Betonungsmuster, Pausen-Timing und empfohlene Voice-IDs. Mindestens 60 Wörter.",
    "kling_ai": "Detaillierter Kling AI B-Roll Prompt: Beschreibe alle Szenen einzeln mit Kamerawinkel (Close-up/Wide/POV), Beleuchtung, Farbpalette, Bewegungsgeschwindigkeit, visuellen Elementen und Gesamtstil. Mindestens 100 Wörter.",
    "capcut": "Detaillierte CapCut Editing-Anweisungen: Schnittrhythmus (Cuts pro Sekunde), Caption-Stil (Font, Farbe, Größe, Animations-Typ), Musik-Genre und BPM, Farbkorrektur-Preset, Übergangstypen und konkrete CapCut-Template-Empfehlungen. Mindestens 80 Wörter."
  }}
}}"""


# ---------------------------------------------------------------------------
# Shared core: transcribe + Claude  (raises on error)
# ---------------------------------------------------------------------------
def _transcribe_and_analyze(job_id: str, video_path: str, label: str) -> None:
    def update(step: str, progress: int):
        jobs[job_id].update({"step": step, "progress": progress})

    update("Audio wird transkribiert...", 40)
    model = get_whisper_model()
    segments, _ = model.transcribe(video_path, beam_size=5)
    transcript = "".join(seg.text for seg in segments).strip()
    if not transcript:
        transcript = "[Kein gesprochener Text erkannt – möglicherweise nur Musik/Hintergrundgeräusche]"

    update("KI-Analyse läuft...", 70)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise Exception("ANTHROPIC_API_KEY Umgebungsvariable ist nicht gesetzt.")

    result = call_claude(api_key, transcript, label)

    jobs[job_id].update({
        "status": "completed",
        "step": "Analyse abgeschlossen!",
        "progress": 100,
        "result": result,
        "transcript": transcript,
    })


# ---------------------------------------------------------------------------
# Background task: download via yt-dlp, then analyze
# ---------------------------------------------------------------------------
def process_via_ytdlp(job_id: str, url: str, label: str) -> None:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs[job_id].update({"step": "Video wird heruntergeladen...", "progress": 10})

            dl = subprocess.run(
                [
                    "yt-dlp",
                    "--no-playlist",
                    "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                    "--merge-output-format", "mp4",
                    "-o", os.path.join(tmpdir, "video.%(ext)s"),
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )

            video_file = next(
                (os.path.join(tmpdir, f) for f in os.listdir(tmpdir)
                 if os.path.isfile(os.path.join(tmpdir, f))),
                None,
            )

            if not video_file or dl.returncode != 0:
                stderr = (dl.stderr or "")[:800]
                raise Exception(
                    "Video-Download fehlgeschlagen.\n\n"
                    "Mögliche Ursachen:\n"
                    "• Facebook blockiert automatische Downloads (häufig bei Ad Library)\n"
                    "• Die URL ist nicht mehr gültig oder privat\n\n"
                    "Tipp: Nutze stattdessen die Option 'Video-URL' mit einem direkten \n"
                    "Link zum Video, oder lade die Datei direkt hoch ('Datei hochladen').\n\n"
                    f"Technische Details: {stderr}"
                )

            _transcribe_and_analyze(job_id, video_file, label)

    except Exception as exc:
        jobs[job_id].update({"status": "error", "step": "Fehler aufgetreten", "error": str(exc)})


# ---------------------------------------------------------------------------
# Background task: process already-saved file (cleans up tmpdir afterwards)
# ---------------------------------------------------------------------------
def process_uploaded_file(job_id: str, video_path: str, tmpdir: str) -> None:
    try:
        jobs[job_id].update({"step": "Datei wird verarbeitet...", "progress": 15})
        _transcribe_and_analyze(job_id, video_path, "hochgeladen")
    except Exception as exc:
        jobs[job_id].update({"status": "error", "step": "Fehler aufgetreten", "error": str(exc)})
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Claude call
# ---------------------------------------------------------------------------
def call_claude(api_key: str, transcript: str, label: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": CLAUDE_PROMPT.format(transcript=transcript, label=label)}],
    )
    text = message.content[0].text

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    s, e = text.find("{"), text.rfind("}") + 1
    if s != -1 and e > s:
        return json.loads(text[s:e])

    raise Exception("Claude-Antwort konnte nicht als JSON geparst werden.")


# ---------------------------------------------------------------------------
# Job helper
# ---------------------------------------------------------------------------
def _new_job() -> tuple[str, dict]:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "processing",
        "step": "Analyse wird gestartet...",
        "progress": 5,
        "result": None,
        "transcript": None,
        "error": None,
    }
    return job_id, jobs[job_id]


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.post("/analyze")
async def analyze_ad(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    """Accepts either ad_id (Facebook Ad Library) or video_url (any yt-dlp URL)."""
    if request.ad_id:
        ad_id = request.ad_id.strip()
        if not re.match(r"^\d{10,20}$", ad_id):
            raise HTTPException(status_code=400,
                detail="Ungültige Ad-ID. Bitte nur die numerische ID eingeben (10–20 Stellen).")
        url = f"https://www.facebook.com/ads/library/?id={ad_id}"
        label = f"Ad-ID {ad_id}"

    elif request.video_url:
        url = request.video_url.strip()
        if not url.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail="Ungültige URL.")
        label = url

    else:
        raise HTTPException(status_code=400, detail="Bitte ad_id oder video_url angeben.")

    job_id, _ = _new_job()
    background_tasks.add_task(process_via_ytdlp, job_id, url, label)
    return {"job_id": job_id}


@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Accept a direct video file upload."""
    if not file.content_type or not file.content_type.startswith("video/"):
        # Also allow generic octet-stream (some browsers send this for video files)
        if file.content_type != "application/octet-stream":
            raise HTTPException(status_code=400,
                detail="Bitte eine Video-Datei hochladen (MP4, MOV, AVI, WebM, ...).")

    MAX_SIZE = 200 * 1024 * 1024  # 200 MB
    tmpdir = tempfile.mkdtemp()
    try:
        filename = file.filename or "upload.mp4"
        safe_name = re.sub(r"[^\w.\-]", "_", filename)
        video_path = os.path.join(tmpdir, safe_name)

        size = 0
        with open(video_path, "wb") as out:
            while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                size += len(chunk)
                if size > MAX_SIZE:
                    out.close()
                    shutil.rmtree(tmpdir, ignore_errors=True)
                    raise HTTPException(status_code=413,
                        detail="Datei zu groß. Maximale Größe: 200 MB.")
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Upload fehlgeschlagen: {exc}")

    job_id, _ = _new_job()
    background_tasks.add_task(process_uploaded_file, job_id, video_path, tmpdir)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job nicht gefunden.")
    return jobs[job_id]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/version")
async def version():
    return {"version": "v4", "features": ["ad-id", "video-url", "file-upload"]}


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")
