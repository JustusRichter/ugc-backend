import json
import os
import re
import subprocess
import tempfile
import uuid
from typing import Optional

import anthropic
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Whisper model cache (loaded lazily on first request)
# ---------------------------------------------------------------------------
_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper_model


# ---------------------------------------------------------------------------
# In-memory job store  (fine for single-instance Railway deployment)
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
# Models
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    ad_id: str


# ---------------------------------------------------------------------------
# Claude prompt template
# ---------------------------------------------------------------------------
CLAUDE_PROMPT = """\
Du bist ein erfahrener UGC-Ad-Stratege und Direct-Response-Marketing-Experte.

Analysiere das folgende Transkript einer Facebook Ad (Ad-ID: {ad_id}) und erstelle eine vollständige Analyse auf Deutsch.

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
# Core processing  (runs as background task)
# ---------------------------------------------------------------------------
def process_ad(job_id: str, ad_id: str) -> None:
    def update(step: str, progress: int, **kwargs):
        jobs[job_id].update({"step": step, "progress": progress, **kwargs})

    try:
        with tempfile.TemporaryDirectory() as tmpdir:

            # ── Step 1: Download video via yt-dlp ──────────────────────────
            update("Video wird heruntergeladen...", 10)
            ad_url = f"https://www.facebook.com/ads/library/?id={ad_id}"

            dl = subprocess.run(
                [
                    "yt-dlp",
                    "--no-playlist",
                    "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                    "--merge-output-format", "mp4",
                    "-o", os.path.join(tmpdir, "video.%(ext)s"),
                    ad_url,
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )

            video_file = next(
                (
                    os.path.join(tmpdir, f)
                    for f in os.listdir(tmpdir)
                    if os.path.isfile(os.path.join(tmpdir, f))
                ),
                None,
            )

            if not video_file or dl.returncode != 0:
                stderr = (dl.stderr or "")[:600]
                raise Exception(
                    "Video-Download fehlgeschlagen. Mögliche Ursachen:\n"
                    "• Die Ad-ID ist falsch oder nicht mehr aktiv\n"
                    "• Das Video ist nicht öffentlich zugänglich\n"
                    "• Facebook hat den Zugriff blockiert\n\n"
                    f"Technische Details: {stderr}"
                )

            # ── Step 2: Transcribe with faster-whisper ──────────────────────
            update("Audio wird transkribiert...", 40)
            model = get_whisper_model()
            segments, _ = model.transcribe(video_file, beam_size=5)
            transcript = "".join(seg.text for seg in segments).strip()

            if not transcript:
                transcript = "[Kein gesprochener Text erkannt – möglicherweise nur Musik/Hintergrundgeräusche]"

            # ── Step 3: Analyze with Claude ─────────────────────────────────
            update("KI-Analyse läuft...", 70)
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise Exception("ANTHROPIC_API_KEY Umgebungsvariable ist nicht gesetzt.")

            result = call_claude(api_key, transcript, ad_id)

            jobs[job_id].update(
                {
                    "status": "completed",
                    "step": "Analyse abgeschlossen!",
                    "progress": 100,
                    "result": result,
                    "transcript": transcript,
                }
            )

    except Exception as exc:
        jobs[job_id].update(
            {
                "status": "error",
                "step": "Fehler aufgetreten",
                "error": str(exc),
            }
        )


def call_claude(api_key: str, transcript: str, ad_id: str) -> dict:
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": CLAUDE_PROMPT.format(transcript=transcript, ad_id=ad_id),
            }
        ],
    )
    text = message.content[0].text

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Last resort: extract outermost {}
    s, e = text.find("{"), text.rfind("}") + 1
    if s != -1 and e > s:
        return json.loads(text[s:e])

    raise Exception("Claude-Antwort konnte nicht als JSON geparst werden.")


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.post("/analyze")
async def analyze_ad(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    ad_id = request.ad_id.strip()
    if not re.match(r"^\d{10,20}$", ad_id):
        raise HTTPException(
            status_code=400,
            detail="Ungültige Ad-ID. Bitte nur die numerische ID eingeben (10–20 Stellen).",
        )

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "processing",
        "step": "Analyse wird gestartet...",
        "progress": 5,
        "result": None,
        "transcript": None,
        "error": None,
    }
    background_tasks.add_task(process_ad, job_id, ad_id)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job nicht gefunden.")
    return jobs[job_id]


@app.get("/health")
async def health():
    return {"status": "ok"}


# Serve frontend
@app.get("/")
async def root():
    return FileResponse("frontend/index.html")
