"""
Endpoints nativos de Perplexity — proxy directo sin transformación OpenAI.

Todos los endpoints descubiertos via RE del APK v2.95.0.
"""


from __future__ import annotations

import time
import uuid as _uuid

from curl_cffi.requests import AsyncSession
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from auth import require_api_key
from config import settings, PPLX_HEADERS

router = APIRouter()

BASE = settings.pplx_base


def _headers():
    return {
        **PPLX_HEADERS,
        "Cookie": f"__Secure-next-auth.session-token={settings.perplexity_session}",
    }


async def _post(path: str, body: dict) -> dict:
    async with AsyncSession() as s:
        r = await s.post(
            f"{BASE}{path}", headers=_headers(), json=body,
            impersonate="chrome120", timeout=20,
        )
        try:
            return {"status": r.status_code, "body": r.json()}
        except Exception:
            return {"status": r.status_code, "body": r.text}


async def _get(path: str, params: dict | None = None) -> dict:
    async with AsyncSession() as s:
        r = await s.get(
            f"{BASE}{path}", headers=_headers(), params=params or {},
            impersonate="chrome120", timeout=20,
        )
        try:
            return {"status": r.status_code, "body": r.json()}
        except Exception:
            return {"status": r.status_code, "body": r.text}


# ── Tokens ────────────────────────────────────────────────────────────────────

class TokenRequest(BaseModel):
    timezone: str = Field("America/Bogota", description="Zona horaria del dispositivo")
    version: str = Field("2.95.0", description="Versión del APK")


@router.post(
    "/tokens/gemini",
    summary="Obtener token Gemini Live",
    description="""
Obtiene un token `auth_tokens/...` para conectarse a `BidiGenerateContentConstrained`.

**Fuente APK:** `q46.java` (CreateGeminiApiKeyRequest) / `tv9.java` (GeminiApiKeyResponse)

**Campos de respuesta:**
- `api_key`: token `auth_tokens/<hex>` — uso único por sesión WS, ~24h de vida
- `expires_at`: timestamp ISO 8601
- `first_entry_uuid`: UUID del slot pre-creado en el historial
- `first_entry_uuid_token`: JWT `{entry_uuid, user_id, exp}` — necesario para `create-entry` con UUID específico

**Nota:** sin rate limit detectado (15+ requests en segundos sin 429).
""",
)
async def get_gemini_token(body: TokenRequest, _=Depends(require_api_key)):
    return await _post(
        "/rest/realtime/v1/transcription/gemini-api-key",
        {"source": "android", "timezone": body.timezone, "version": body.version},
    )


@router.post(
    "/tokens/soniox",
    summary="Obtener token Soniox STT",
    description="""
Obtiene una API key de Soniox para conectar directamente al WebSocket de transcripción.

**Fuente APK:** `u56.java` (CreateSonioxApiKeyRequest) / `t2o.java` (SonioxApiKeyResponse)

**Campos de respuesta:**
- `apiKey`: token para `wss://stt-rt.soniox.com/transcribe-websocket`
- `expiresAt`: timestamp de expiración

**Nota:** el endpoint Soniox NO devuelve `first_entry_uuid*` — esos campos solo están en el endpoint Gemini.
""",
)
async def get_soniox_token(body: TokenRequest, _=Depends(require_api_key)):
    return await _post(
        "/rest/realtime/v1/transcription/soniox-api-key",
        {"source": "android", "timezone": body.timezone, "version": body.version},
    )


# ── Sesiones de voz ───────────────────────────────────────────────────────────

class V2SessionRequest(BaseModel):
    sdp: str = Field(..., description="SDP offer (WebRTC) generado por el cliente")
    voice: str = Field(
        "alloy",
        description="Voz OpenAI: cedar | alloy | marin | fable | shimmer | ash | ballad | coral",
    )
    timezone: str = Field("America/Bogota")
    turn_detection_threshold: float = Field(0.8, description="VAD threshold [0-1]")


@router.post(
    "/session/v2",
    summary="Crear sesión OpenAI Realtime (WebRTC)",
    description="""
Crea una sesión de voz usando el path OpenAI Realtime (WebRTC).

**Fuente APK:** `f91.java` líneas 1658-1666 / `wvi.java` (response)

**Requiere:** SDP offer WebRTC válido (usar `aiortc` u otro cliente WebRTC).

**Respuesta incluye:**
- `sdp`: SDP answer de OpenAI — establecer como remote description
- `model_name`: `gpt-realtime-mini-2025-12-15`
- `first_entry_uuid` + `first_entry_uuid_token`: para vincular a historial

**Flujo post-sesión:**
```
WebRTC P2P → data channel "openai-events" (JSON bidireccional)
```

**Rate limit:** ~3 sesiones/hora por cuenta.
""",
)
async def create_v2_session(body: V2SessionRequest, _=Depends(require_api_key)):
    return await _post(
        "/rest/realtime/v2/session",
        {
            "timezone": body.timezone,
            "source": "android",
            "voice": body.voice,
            "turn_detection_threshold": body.turn_detection_threshold,
            "sdp": body.sdp,
        },
    )


class V4SessionRequest(BaseModel):
    sdp: str = Field(..., description="SDP offer WebRTC")
    voice: str = Field("alloy")
    prompt: dict = Field(
        {"id": "perplexity", "version": 1},
        description='Referencia al prompt interno. Estructura: `{"id": "perplexity", "version": 1}`',
    )
    timezone: str = Field("America/Bogota")


@router.post(
    "/session/v4",
    summary="Crear sesión OaiLive v4",
    description="""
Tercer voice provider descubierto: `OaiLive` en `FfiVoiceProvider`.

**Fuente APK:** strings de `libmultimodal_uniffi.so` + `PromptRef.java`

**Estado:** parcialmente explorado. Requiere campo `prompt` o `agent`.

**Diferencia con v2:** usa un endpoint distinto en el backend Perplexity,
posiblemente con routing a un modelo diferente o config de agente.
""",
)
async def create_v4_session(body: V4SessionRequest, _=Depends(require_api_key)):
    return await _post(
        "/rest/realtime/v4/session",
        {
            "timezone": body.timezone,
            "source": "android",
            "voice": body.voice,
            "sdp": body.sdp,
            "prompt": body.prompt,
        },
    )


# ── Realtime helpers ──────────────────────────────────────────────────────────

class ExecuteToolsRequest(BaseModel):
    session_id: str = Field(..., description="UUID de la sesión de voz activa")
    tool_name: str = Field("search_web", description="Nombre de la herramienta a ejecutar")
    tool_input: dict = Field(
        {"query": "..."},
        description="Input de la herramienta (para `search_web`: `{query: '...'}`).",
    )


@router.post(
    "/realtime/execute-tools",
    summary="Ejecutar herramienta en sesión de voz",
    description="""
Ejecuta una herramienta (búsqueda web, etc.) en el contexto de una sesión de voz activa.

**Fuente APK:** `g3j.java` (Retrofit interface)

**Campo requerido:** `source: "android"` — añadido automáticamente por este proxy.

**Herramientas conocidas:** `search_web`

**Nota:** requiere un `session_id` de una sesión activa (v2 o v4).
Con `session_id` inválido devuelve 200 con body vacío o resultado nulo.
""",
)
async def execute_tools(body: ExecuteToolsRequest, _=Depends(require_api_key)):
    return await _post(
        "/rest/realtime/execute-tools",
        {
            "session_id": body.session_id,
            "source": "android",
            "tool_name": body.tool_name,
            "tool_input": body.tool_input,
        },
    )


class CreateEntryRequest(BaseModel):
    session_id: str = Field(
        default_factory=lambda: str(_uuid.uuid4()),
        description="UUID de la sesión de voz",
    )
    query: str = Field(..., description="Texto de la pregunta del usuario")
    answer: str = Field(..., description="Texto de la respuesta del asistente")
    tool_ids: list[str] = Field(
        [],
        description="IDs de herramientas usadas durante el turno. Vacío si ninguna.",
    )
    model_name: str = Field(
        "gpt-realtime-mini-2025-12-15",
        description="Nombre del modelo que generó la respuesta",
    )
    entry_uuid: str | None = Field(
        None,
        description=(
            "UUID del slot pre-creado (obtenido de `first_entry_uuid` en `/tokens/gemini`). "
            "Si se omite, el servidor asigna un UUID aleatorio."
        ),
    )
    entry_uuid_token: str | None = Field(
        None,
        description=(
            "JWT de autenticación (`first_entry_uuid_token`). "
            "**Requerido** para que el UUID retornado coincida con `entry_uuid`. "
            "Sin este campo el servidor devuelve un UUID aleatorio (entrada 'flotante')."
        ),
    )
    input_audio_item_id: str | None = Field(None, description="Item ID del audio de entrada (WebRTC)")
    output_item_id: str | None = Field(None, description="Item ID del audio de salida (WebRTC)")


@router.post(
    "/realtime/create-entry",
    summary="Guardar turno de voz en historial",
    description="""
Guarda un turno Q&A de sesión de voz en el historial de conversación de Perplexity.

**Fuente APK:** `xvi.java` (RealtimeSessionEntry serializer)

**Con `entry_uuid_token`:** la entrada se vincula al slot pre-creado — el UUID retornado coincide con el enviado.

**Sin `entry_uuid_token`:** entrada "flotante" con UUID aleatorio — aparece en historial pero no vinculada a sesión.

**Verificado:** sin rate limit aparente. Sin `source` → 422 (añadido automáticamente).

**JWT payload:** `{entry_uuid, user_id, exp}` — ~1h de validez desde emisión.
""",
)
async def create_entry(body: CreateEntryRequest, _=Depends(require_api_key)):
    payload: dict = {
        "source": "android",
        "session_id": body.session_id,
        "query": body.query,
        "answer": body.answer,
        "tool_ids": body.tool_ids,
        "timestamp": int(time.time()),
        "model_name": body.model_name,
    }
    if body.entry_uuid:
        payload["entry_uuid"] = body.entry_uuid
    if body.entry_uuid_token:
        payload["entry_uuid_token"] = body.entry_uuid_token
    if body.input_audio_item_id:
        payload["inputAudioItemId"] = body.input_audio_item_id
    if body.output_item_id:
        payload["outputItemId"] = body.output_item_id

    return await _post("/rest/realtime/create-entry", payload)


# ── SDK Config ────────────────────────────────────────────────────────────────

@router.get(
    "/sdk/config",
    summary="SDK Config dinámico",
    description="""
Configuración dinámica del SDK multimodal. TTL de 900 segundos.

**Fuente APK:** strings de `libmultimodal_uniffi.so` + endpoint verificado

**Incluye:**
- URL del WebSocket Gemini
- Modelo y config de Soniox STT
- Modelo de Gemini para speech synthesis
- Modelo on-device de AI Coustics (noise reduction)
- `default_connection_mode`: `direct` (P2P) o `relay` (vía Perplexity)
""",
)
async def get_sdk_config(
    platform: str = "android",
    sdk_version: str = "3.13.1",
    _=Depends(require_api_key),
):
    return await _get(
        "/rest/multimodal/sdk/config",
        {"platform": platform, "sdk_version": sdk_version},
    )


# ── Auth Exchange ─────────────────────────────────────────────────────────────

class AuthExchangeRequest(BaseModel):
    token: str = Field(..., description="Token a intercambiar (formato desconocido)")
    grant_type: str | None = Field(None, description="Tipo de grant (experimental)")


@router.post(
    "/auth/exchange",
    summary="Auth Token Exchange (experimental)",
    description="""
Intercambia un token de autenticación. Endpoint descubierto en APK pero no completamente explorado.

**Estado:** experimental — propósito exacto desconocido. Posiblemente para intercambiar
un token de sesión por un token de corta duración específico para cierto recurso.

**Fuente:** strings de la librería nativa Rust.
""",
)
async def auth_exchange(body: AuthExchangeRequest, _=Depends(require_api_key)):
    payload = {"token": body.token}
    if body.grant_type:
        payload["grant_type"] = body.grant_type
    return await _post("/rest/auth/exchange", payload)


# ── Models ───────────────────────────────────────────────────────────────────

@router.get(
    "/models/config",
    summary="Config completa de modelos (v2)",
    description="""
Devuelve la configuración completa de todos los modelos disponibles en Perplexity.

**Fuente APK:** `rp2.java` → `@GET /rest/models/config/v2` / `ank.java` (RemoteModelsConfigResponse)

**Campos de respuesta:**
- `config_schema`: versión del esquema (`v2`)
- `models`: dict con todos los modelos y sus metadatos (provider, mode, label, etc.)
- `default_models`: modelo por defecto para cada modo (`search`, `research`, `asi`, etc.)
- `search_config`: configuración de modelos visibles en el selector de búsqueda
- `computer_config`: modelos disponibles para el modo "Computer" (ASI) con tiers y descripciones
- `agentic_research_compare_models`: modelos usados en "Model council"

**Providers encontrados:** PERPLEXITY, OPENAI, ANTHROPIC, GOOGLE, XAI, MOONSHOT_AI,
SONAR, NVIDIA, FIREWORKS, ZAI

**Nota:** incluye modelos internos/testing y futuros (gpt5, gpt56_sol, pplx_asi_fable_5, etc.).
""",
)
async def get_models_config(_=Depends(require_api_key)):
    return await _get("/rest/models/config/v2")


@router.get(
    "/models/modes",
    summary="Modos de búsqueda disponibles",
    description="""
Devuelve los modos de búsqueda disponibles para el usuario autenticado.

**Fuente APK:** `nvm.java` → `@GET /rest/models/modes` / `i6l.java` (RemoteSearchModesResponse)
/ `f6l.java` (RemoteSearchMode)

**Campos de cada modo:**
- `id`: identificador interno (`search`, `research`, `agentic_research`, `study`, `browser_agent`, `asi`)
- `label`: nombre visible
- `description`: descripción larga
- `subtitle`: subtítulo (nullable)
- `badge`: etiqueta de badge (nullable)
- `subscriptionTier`: tier requerido (`null` = free, `max` = plan Max)

**Modos conocidos:** Search, Deep research, Model council (Max), Learn step by step,
Control browser, Computer.
""",
)
async def get_models_modes(_=Depends(require_api_key)):
    return await _get("/rest/models/modes")


# ── AI Coustics Token ─────────────────────────────────────────────────────────

@router.post(
    "/tokens/ai-coustics",
    summary="Obtener token AI Coustics (noise reduction)",
    description="""
Obtiene un token para el SDK de reducción de ruido on-device de AI Coustics.

**Fuente APK:** campo `noise_reduction` en RealtimeSessionConfig, endpoint en `libmultimodal_uniffi.so`

**Modelo usado:** `quail-vf-2.1-l-48khz` @ 48 kHz

**Tipo:** on-device (el modelo corre localmente en el dispositivo, no en servidor).
Este endpoint solo entrega el token de autenticación para el SDK.
""",
)
async def get_ai_coustics_token(_=Depends(require_api_key)):
    return await _post(
        "/rest/realtime/v1/ai-coustics/token",
        {"source": "android"},
    )


# ── TTS SSE (raw) ─────────────────────────────────────────────────────────────

class TTSRawRequest(BaseModel):
    text: str = Field(..., description="Texto a sintetizar")
    preset: str = Field(
        "Kyrin-mp3",
        description=(
            "Preset de voz Perplexity. Formato: `NombreVoz-mp3`.\n\n"
            "| Preset | Voz OpenAI |\n"
            "|---|---|\n"
            "| Kyrin-mp3 | cedar |\n"
            "| Velox-mp3 | marin |\n"
            "| Tylis-mp3 | alloy |\n"
            "| Torma-mp3 | ash |\n"
            "| Mylva-mp3 | ballad |\n"
            "| Syla-mp3 | coral |\n"
            "| Gravo-mp3 | echo |\n"
            "| Solva-mp3 | sage |"
        ),
    )
    visitor_id: str = Field(
        default_factory=lambda: str(_uuid.uuid4()),
        description="ID de visitante (UUIDv4). Se genera automáticamente si se omite.",
    )


@router.post(
    "/tts",
    summary="TTS SSE nativo (proxy directo)",
    description="""
Proxy directo al endpoint SSE de TTS de Perplexity. Devuelve el stream SSE tal cual.

**Para audio MP3 completo** usa `/v1/audio/speech` (OpenAI-compatible).

**Este endpoint** devuelve el stream SSE crudo, útil para debugging o integración directa.

**Fuente APK:** `f91.java` líneas 314-324 / enum `j0g.java`
""",
)
async def tts_raw(body: TTSRawRequest, _=Depends(require_api_key)):
    async with AsyncSession() as s:
        r = await s.post(
            f"{BASE}/rest/sse/audio/text_to_speech",
            headers={**_headers(), "Accept": "text/event-stream"},
            json={
                "uuid": str(_uuid.uuid4()),
                "text": body.text,
                "preset": body.preset,
                "source": "assistant",
                "visitor_id": body.visitor_id,
            },
            impersonate="chrome120",
            timeout=30,
        )
        try:
            return {"status": r.status_code, "content_type": r.headers.get("content-type"), "preview": r.text[:500]}
        except Exception:
            return {"status": r.status_code, "error": "no response"}
