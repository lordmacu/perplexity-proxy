"""
POST /v1/audio/speech — compatible con OpenAI TTS API.

Backend: Perplexity /rest/sse/audio/text_to_speech
- 8 voces (nombres Perplexity) = 8 voces OpenAI mapeadas 1:1
- Siempre devuelve MP3
- No soporta: speed, response_format != mp3
"""

import base64
import json
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from auth import require_api_key
from config import settings, PPLX_HEADERS, VOICE_MAP
from curl_cffi.requests import AsyncSession

router = APIRouter()

TTS_URL = f"{settings.pplx_base}/rest/sse/audio/text_to_speech"


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
    preset = VOICE_MAP.get(body.voice.lower(), "Tylis-mp3")

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

    return Response(content=mp3, media_type="audio/mpeg")
