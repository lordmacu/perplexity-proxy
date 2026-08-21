"""
POST /v1/audio/speech — compatible con OpenAI TTS API.

Backend: Perplexity /rest/sse/audio/text_to_speech
- 8 voces (nombres Perplexity) = 8 voces OpenAI mapeadas 1:1
- Siempre devuelve MP3
- No soporta: speed, response_format != mp3
"""

import base64
import json
import random
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from auth import require_api_key
from config import settings, PPLX_HEADERS, VOICE_MAP
from curl_cffi.requests import AsyncSession

import capabilities
import soniox

router = APIRouter()

TTS_URL = f"{settings.pplx_base}/rest/sse/audio/text_to_speech"

# ── El contrato de audio, común a los cinco proxies ──────────────────────────
# 4096 es el límite de la API de OpenAI. Acá ya se declaraba en el esquema; se
# publica ahora en /v1/audio/voices para que un cliente pueda preguntarlo en vez
# de descubrirlo con un 422.
MAX_INPUT_CHARS = 4096
SUPPORTED_FORMATS = ("mp3",)

# Las 8 voces nativas de Perplexity, deducidas de los valores de VOICE_MAP.
NATIVE_VOICES = tuple(sorted(set(VOICE_MAP.values())))
DEFAULT_VOICE = "Tylis-mp3"


def resolve_voice(voice: str) -> tuple[str, bool]:
    """Devuelve (voz_nativa, hubo_sustitución).

    Una voz desconocida NO es un error acá: se elige una al azar entre las
    nativas. Es una decisión deliberada del operador -- prefiere audio en una
    voz cualquiera antes que un rechazo -- y tiene un costo que conviene tener
    presente: la misma petición repetida puede sonar distinta. Por eso la
    sustitución se informa en cabeceras (`X-Voice-Fallback`, `X-Voice-Used`) en
    lugar de ocurrir en silencio, que era el comportamiento anterior.
    """
    name = (voice or "").strip().lower()
    if not name:
        return DEFAULT_VOICE, False
    if voice in NATIVE_VOICES:
        return voice, False
    if name in VOICE_MAP:
        return VOICE_MAP[name], False
    return random.choice(NATIVE_VOICES), True


class SpeechRequest(BaseModel):
    model: str = Field(
        "tts-1",
        description=(
            "~ **Ignorado** en el backend. Cualquier valor es aceptado como alias.\n\n"
            "Siempre usa el motor TTS de Perplexity vía SSE."
        ),
        examples=["tts-1", "tts-1-hd", "perplexity-tts"],
    )
    input: str = Field(
        ...,
        description="✓ **Soportado**. Texto a sintetizar.",
        max_length=4096,
    )
    voice: str = Field(
        "alloy",
        description=(
            "✓ **Soportado** — mapeado a voces Perplexity.\n\n"
            "| Voz OpenAI | Voz Perplexity | Nota |\n"
            "|---|---|---|\n"
            "| `alloy` | Tylis | ✓ Nativa |\n"
            "| `echo` | Gravo | ✓ Nativa |\n"
            "| `ash` | Torma | ✓ Nativa |\n"
            "| `ballad` | Mylva | ✓ Nativa |\n"
            "| `coral` | Syla | ✓ Nativa |\n"
            "| `sage` | Solva | ✓ Nativa |\n"
            "| `cedar` | Kyrin | ✓ Exclusiva Perplexity |\n"
            "| `marin` | Velox | ✓ Exclusiva Perplexity |\n"
            "| `fable` | Kyrin | ~ Fallback |\n"
            "| `onyx` | Gravo | ~ Fallback |\n"
            "| `nova` | Syla | ~ Fallback |\n"
            "| `shimmer` | Velox | ~ Fallback |"
        ),
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        "mp3",
        description="✗ **Siempre MP3** — el backend Perplexity solo entrega MP3. El campo es aceptado pero ignorado.",
    )
    speed: float = Field(
        1.0,
        description="✗ **No soportado**. El campo es aceptado pero ignorado.",
        ge=0.25,
        le=4.0,
    )


@router.get("/audio/voices", summary="Voces y límites de TTS")
def list_voices():
    """Lo que este proxy acepta para texto a voz. Estándar en los cinco.

    No existía forma de preguntarlo, que es precisamente por qué se terminó
    mandando una voz que dos de los cinco proxies no tenían.
    """
    return {
        "default": DEFAULT_VOICE,
        "voices": list(NATIVE_VOICES),
        "openai_aliases": VOICE_MAP,
        "selection": "random",
        "max_input_chars": MAX_INPUT_CHARS,
        "formats": list(SUPPORTED_FORMATS),
        "default_format": "mp3",
    }


@router.post(
    "/audio/speech",
    summary="Text-to-Speech",
    description="""
Sintetiza texto a voz MP3 usando el motor TTS interno de Perplexity.

**Backend:** `POST /rest/sse/audio/text_to_speech` (SSE → MP3 concatenado)

**Parámetros OpenAI:**
- ✓ `input` — texto a sintetizar
- ✓ `voice` — 12 voces OpenAI mapeadas a 8 voces Perplexity
- ~ `model` — ignorado (siempre motor Perplexity)
- ✗ `speed` — no soportado
- ✗ `response_format` — siempre MP3

**Nota:** `cedar` y `marin` son voces exclusivas del ecosistema Perplexity/OpenAI no disponibles en la API pública de OpenAI.
""",
    response_class=Response,
    responses={
        200: {
            "content": {"audio/mpeg": {}},
            "description": "Audio MP3 sintetizado",
        }
    },
)
async def create_speech(
    body: SpeechRequest,
    request: Request,
    _=Depends(require_api_key),
):
    capabilities.require("audio_speech")

    if len(body.input) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=400, detail={"error": {
            "type": "invalid_request_error",
            "message": f"input is {len(body.input)} characters, the limit is {MAX_INPUT_CHARS}",
            "param": "input",
            "max_input_chars": MAX_INPUT_CHARS}})

    preset, substituted = resolve_voice(body.voice)

    async with AsyncSession() as session:
        response = await session.post(
            TTS_URL,
            headers={
                **PPLX_HEADERS,
                "Cookie": f"__Secure-next-auth.session-token={settings.perplexity_session}",
                "Accept": "text/event-stream",
            },
            json={
                "uuid":       str(uuid.uuid4()),
                "text":       body.input,
                "preset":     preset,
                "source":     "assistant",
                "visitor_id": str(uuid.uuid4()),
            },
            impersonate="chrome120",
            timeout=60,
            stream=True,
        )

        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Perplexity TTS error: {response.status_code}")

        audio_chunks: list[bytes] = []
        async for line in response.aiter_lines():
            # curl_cffi entrega bytes, no str: decodificar antes de tocar el texto.
            if isinstance(line, (bytes, bytearray)):
                line = line.decode("utf-8", "replace")
            line = line.strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data in ("[DONE]", ""):
                break
            try:
                event = json.loads(data)
                raw = event.get("audio") or event.get("data") or event.get("chunk")
                if raw:
                    audio_chunks.append(base64.b64decode(raw))
            except (json.JSONDecodeError, Exception):
                # fallback: try treating data directly as base64
                try:
                    audio_chunks.append(base64.b64decode(data))
                except Exception:
                    pass

    mp3 = b"".join(audio_chunks)
    if not mp3:
        raise HTTPException(status_code=502, detail="No audio data received from Perplexity TTS")

    headers = {"X-Voice-Used": preset}
    if substituted:
        headers["X-Voice-Fallback"] = body.voice
    return Response(content=mp3, media_type="audio/mpeg", headers=headers)


# ── Speech-to-Text ────────────────────────────────────────────────────────────

@router.post(
    "/audio/transcriptions",
    summary="Speech-to-Text",
    description="""
Transcribe audio a texto, compatible con la API de OpenAI.

**Backend:** `wss://stt-rt.soniox.com/transcribe-websocket` — el mismo motor que
usa la app de Perplexity (APK v2.95.0: `e3o.java`, `a4o.java`, `x2o.java`).

**Ojo, esto es distinto del resto del proxy:** la transcripción NO la hace
Perplexity, la hace **Soniox**, un tercero. Perplexity solo emite la credencial
(`POST /rest/realtime/v1/transcription/soniox-api-key`). La cuota y la
disponibilidad son de Soniox.

**Parámetros OpenAI:**
- ✓ `file` — el audio a transcribir
- ✓ `language` — se manda como `language_hints` (59 idiomas; ver `soniox.py`)
- ✓ `prompt` — se manda como `context` de Soniox
- ✓ `response_format` — `json` (default) o `text`
- ~ `model` — ignorado; siempre `stt-rt-v4`
- ✗ `temperature` — no soportado
""",
)
async def create_transcription(
    file: UploadFile = File(..., description="✓ Audio a transcribir"),
    model: str = Form("whisper-1", description="~ Ignorado; siempre `stt-rt-v4`"),
    language: Optional[str] = Form(None, description="✓ Código ISO-639-1, ej. `es`"),
    prompt: Optional[str] = Form(None, description="✓ Se manda como `context` de Soniox"),
    response_format: str = Form("json", description="✓ `json` o `text`"),
    _=Depends(require_api_key),
):
    capabilities.require("audio_transcription")

    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="archivo de audio vacío")

    try:
        api_key = await soniox.fetch_api_key()
        result = await soniox.transcribe(
            audio, api_key=api_key, language=language, context=prompt,
        )
    except soniox.SonioxError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    if response_format == "text":
        return Response(content=result.text, media_type="text/plain; charset=utf-8")
    return {"text": result.text, "language": result.language}
