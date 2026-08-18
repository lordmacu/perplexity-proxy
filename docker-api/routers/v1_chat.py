"""
POST /v1/chat/completions — compatible con OpenAI Chat API.

Routing automático por modelo:
  - perplexity-gemini-live*  → Gemini Live WebSocket (audio→transcripción)
  - cualquier otro modelo    → Perplexity Search SSE (/rest/sse/perplexity_ask)

Conversaciones multi-turno (Perplexity backend):
  Cada respuesta entrega backend_uuid + read_write_token. Los guardamos en
  _THREAD_CACHE indexados por SHA-256 del historial completo (incluyendo la
  respuesta del asistente). El siguiente turno busca por SHA-256 de
  messages[:-1] y, si hay hit, envía last_backend_uuid + is_related_query=true
  para que Perplexity mantenga el hilo server-side.
"""


from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import AsyncGenerator, Literal

import websockets
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from auth import require_api_key
from config import settings, GEMINI_VOICES, PPLX_HEADERS
from curl_cffi.requests import AsyncSession

_SEARCH_URL = f"{settings.pplx_base}/rest/sse/perplexity_ask"

GEMINI_LIVE_MODELS = {
    "perplexity-gemini-live",
    "perplexity-gemini-live-charon",
    "perplexity-gemini-live-fenrir",
    "perplexity-gemini-live-zephyr",
}

router = APIRouter()

# ── Thread cache ───────────────────────────────────────────────────────────────
# key  → SHA-256(messages incluyendo última respuesta del asistente)
# value → {backend_uuid, read_write_token, device_id, expires}
_THREAD_CACHE: dict[str, dict] = {}
_THREAD_TTL = 3600  # 1 h


def _cache_key(messages: list[ChatMessage]) -> str:
    raw = json.dumps(
        [{"r": m.role, "c": m.content.strip()} for m in messages],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _get_thread(messages: list[ChatMessage]) -> tuple[str | None, str | None, str]:
    """Busca estado previo del hilo. Devuelve (backend_uuid, rwt, device_id)."""
    prior = messages[:-1]
    if not prior:
        return None, None, str(uuid.uuid4())
    key = _cache_key(prior)
    state = _THREAD_CACHE.get(key)
    if state and state["expires"] > time.time():
        return state["backend_uuid"], state.get("read_write_token"), state["device_id"]
    return None, None, str(uuid.uuid4())


def _save_thread(
    messages: list[ChatMessage],
    response_content: str,
    backend_uuid: str,
    read_write_token: str | None,
    device_id: str,
) -> None:
    if not backend_uuid:
        return
    all_msgs = list(messages) + [
        ChatMessage(role="assistant", content=response_content.strip())
    ]
    key = _cache_key(all_msgs)
    now = time.time()
    _THREAD_CACHE[key] = {
        "backend_uuid": backend_uuid,
        "read_write_token": read_write_token,
        "device_id": device_id,
        "expires": now + _THREAD_TTL,
    }
    # Limpieza de entradas expiradas
    expired = [k for k, v in list(_THREAD_CACHE.items()) if v["expires"] <= now]
    for k in expired:
        _THREAD_CACHE.pop(k, None)


# ── Modelos de request/response ───────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        ...,
        description=(
            "✓ **Soportado**. Roles `system`, `user`, `assistant`.\n\n"
            "- `system` → primer mensaje de sistema se usa como `system_instruction` en Gemini.\n"
            "- `user` / `assistant` → convertidos a turnos Gemini (`user` / `model`).\n"
            "- El último mensaje **debe** ser de rol `user`."
        ),
    )
    model: str = Field(
        "turbo",
        description=(
            "✓ **Routing automático por modelo:**\n\n"
            "| Modelo | Backend | Notas |\n"
            "|---|---|---|\n"
            "| `turbo` | Perplexity Search | **Gratis** — web-grounded |\n"
            "| `gpt4o`, `claude46sonnet`, `gemini25pro`, `grok4`, ... | Perplexity Search | Gratis → cae a `turbo` |\n"
            "| `perplexity-gemini-live` | Gemini Live WebSocket | Audio→transcripción |\n"
            "| `perplexity-gemini-live-charon` | Gemini Live (voz Charon) | |\n"
            "| `perplexity-gemini-live-fenrir` | Gemini Live (voz Fenrir) | |\n"
            "| `perplexity-gemini-live-zephyr` | Gemini Live (voz Zephyr) | |\n\n"
            "Los modelos `perplexity-gemini-live*` usan Gemini 3.1 Flash Live Preview. "
            "Todos los demás usan Perplexity Search SSE (recomendado)."
        ),
        examples=["turbo", "gpt4o", "claude46sonnet", "perplexity-gemini-live"],
    )
    stream: bool = Field(
        False,
        description=(
            "✓ **Soportado**. `true` → SSE con chunks de transcripción en tiempo real.\n"
            "`false` → espera la respuesta completa."
        ),
    )
    temperature: float | None = Field(None, description="✗ No soportado.")
    max_tokens: int | None = Field(None, description="✗ No soportado.")
    top_p: float | None = Field(None, description="✗ No soportado.")
    frequency_penalty: float | None = Field(None, description="✗ No soportado.")
    presence_penalty: float | None = Field(None, description="✗ No soportado.")
    stop: list[str] | None = Field(None, description="✗ No soportado.")
    n: int | None = Field(None, description="✗ No soportado. Siempre devuelve 1 completion.")
    tools: list | None = Field(None, description="✗ No soportado.")
    functions: list | None = Field(None, description="✗ No soportado (legacy).")
    logprobs: bool | None = Field(None, description="✗ No soportado.")
    x_voice: str | None = Field(
        None,
        alias="x-voice",
        description=(
            "✓ **Extensión propia** (no OpenAI). Voz Gemini explícita: "
            "`Aoede` | `Charon` | `Fenrir` | `Zephyr`. "
            "Tiene prioridad sobre la voz extraída del campo `model`."
        ),
    )

    model_config = {"extra": "allow", "populate_by_name": True}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_voice(model: str, x_voice: str | None) -> str:
    if x_voice and x_voice in GEMINI_VOICES:
        return x_voice
    for voice in GEMINI_VOICES:
        if model.lower().endswith(f"-{voice.lower()}"):
            return voice
    return settings.default_voice


def _build_gemini_turns(messages: list[ChatMessage]) -> tuple[str, list[dict]]:
    """Extrae system instruction y convierte mensajes a turnos Gemini."""
    system_parts = []
    turns = []
    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        elif msg.role == "user":
            turns.append({"role": "user", "parts": [{"text": msg.content}]})
        elif msg.role == "assistant":
            turns.append({"role": "model", "parts": [{"text": msg.content}]})
    system_text = "\n\n".join(system_parts) if system_parts else (
        "Eres un asistente conversacional. Responde de forma concisa y directa."
    )
    return system_text, turns


def _chunk_event(request_id: str, content: str, finish: str | None = None) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "perplexity-gemini-live",
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": content} if content else {},
            "finish_reason": finish,
        }],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _extract_query(messages: list[ChatMessage]) -> str:
    """Construye el query_str para Perplexity.

    Perplexity no tiene campo nativo de system prompt (confirmado en hs0.java del APK).
    Si hay mensajes de sistema, los prepende al query del usuario para que el modelo
    los procese como instrucciones de contexto.
    """
    system_parts = [m.content.strip() for m in messages if m.role == "system"]
    user_msgs = [m for m in messages if m.role == "user"]
    query = user_msgs[-1].content if user_msgs else ""
    if system_parts:
        system_text = "\n".join(system_parts)
        return f"{system_text}\n\n---\n\n{query}"
    return query


def _build_search_body(
    query: str,
    model: str,
    request_id: str,
    device_id: str,
    last_backend_uuid: str | None,
    read_write_token: str | None,
) -> dict:
    params: dict = {
        "query_mode": "COPILOT",
        "search_mode": "search",
        "model_api_name": model,
        "search_mode_supports_reasoning": False,
        "frontend_uuid": request_id,
        "version": "2.9",
        "android_device_id": device_id,
        "is_related_query": last_backend_uuid is not None,
    }
    if last_backend_uuid:
        params["last_backend_uuid"] = last_backend_uuid
    if read_write_token:
        params["read_write_token"] = read_write_token
    return {"query_str": query, "params": params}


# ── Lógica Gemini Live ────────────────────────────────────────────────────────

async def _gemini_stream(
    request_id: str,
    system_text: str,
    turns: list[dict],
    voice: str,
    token_manager,
) -> AsyncGenerator[str, None]:
    api_key = await token_manager.get_token()
    url = f"{settings.gemini_ws}?access_token={api_key}"

    try:
        async with websockets.connect(url, open_timeout=15) as ws:
            await ws.send(json.dumps({
                "setup": {
                    "model": settings.gemini_model,
                    "generation_config": {
                        "response_modalities": ["AUDIO"],
                        "speech_config": {
                            "voice_config": {
                                "prebuilt_voice_config": {"voice_name": voice}
                            }
                        },
                    },
                    "system_instruction": {"parts": [{"text": system_text}]},
                    "output_audio_transcription": {},
                },
            }))

            async for raw in ws:
                msg = json.loads(raw)
                if "setupComplete" in msg:
                    break

            token_manager.mark_used()

            await ws.send(json.dumps({
                "client_content": {
                    "turns": turns,
                    "turn_complete": True,
                }
            }))

            yield _chunk_event(request_id, "", None)

            async for raw in ws:
                msg = json.loads(raw)
                sc = msg.get("serverContent", {})

                if "outputTranscription" in sc:
                    text = sc["outputTranscription"].get("text", "")
                    if text:
                        yield _chunk_event(request_id, text)

                if sc.get("turnComplete"):
                    yield _chunk_event(request_id, "", "stop")
                    yield "data: [DONE]\n\n"
                    return

    except websockets.exceptions.ConnectionClosedError as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


async def _gemini_sync(
    system_text: str,
    turns: list[dict],
    voice: str,
    token_manager,
) -> str:
    api_key = await token_manager.get_token()
    url = f"{settings.gemini_ws}?access_token={api_key}"

    full_text = ""
    async with websockets.connect(url, open_timeout=15) as ws:
        await ws.send(json.dumps({
            "setup": {
                "model": settings.gemini_model,
                "generation_config": {
                    "response_modalities": ["AUDIO"],
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {"voice_name": voice}
                        }
                    },
                },
                "system_instruction": {"parts": [{"text": system_text}]},
                "output_audio_transcription": {},
            },
        }))

        async for raw in ws:
            if "setupComplete" in json.loads(raw):
                break

        token_manager.mark_used()

        await ws.send(json.dumps({
            "client_content": {
                "turns": turns,
                "turn_complete": True,
            }
        }))

        async for raw in ws:
            msg = json.loads(raw)
            sc = msg.get("serverContent", {})
            if "outputTranscription" in sc:
                full_text += sc["outputTranscription"].get("text", "")
            if sc.get("turnComplete"):
                break

    return full_text.strip()


# ── Endpoint ──────────────────────────────────────────────────────────────────

def _pplx_headers() -> dict:
    return {
        **PPLX_HEADERS,
        "Cookie": f"__Secure-next-auth.session-token={settings.perplexity_session}",
    }


def _extract_answer(blocks) -> str:
    if not isinstance(blocks, list):
        return ""
    for block in blocks:
        mb = block.get("markdown_block") or {}
        ans = mb.get("answer", "")
        if ans:
            return ans
        chunks = mb.get("chunks", [])
        if chunks:
            return "".join(str(c) for c in chunks)
    return ""


async def _search_stream_chat(
    messages: list[ChatMessage],
    model: str,
    request_id: str,
    device_id: str,
    last_backend_uuid: str | None,
    read_write_token: str | None,
) -> AsyncGenerator[str, None]:
    query = _extract_query(messages)
    body = _build_search_body(query, model, request_id, device_id, last_backend_uuid, read_write_token)

    accumulated: list[str] = []

    async with AsyncSession(impersonate="chrome120") as s:
        async with s.stream("POST", _SEARCH_URL, headers=_pplx_headers(),
                            json=body, timeout=60) as resp:
            if resp.status_code != 200:
                yield f"data: {json.dumps({'error': f'HTTP {resp.status_code}'})}\n\ndata: [DONE]\n\n"
                return
            _emitido = ""
            async for raw in resp.aiter_lines():
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8", "replace")
                raw = raw.strip()
                if not raw or not raw.startswith("data:"):
                    continue
                try:
                    evt = json.loads(raw[5:].strip())
                except Exception:
                    continue
                completo = "".join(
                    chunk
                    for block in evt.get("blocks", [])
                    for chunk in ((block.get("markdown_block") or {}).get("chunks", []))
                    if chunk
                )
                if completo.startswith(_emitido):
                    nuevo = completo[len(_emitido):]
                else:
                    nuevo = completo
                if nuevo:
                    _emitido = completo if completo.startswith(_emitido) else _emitido + nuevo
                    accumulated.append(nuevo)
                    delta = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "model": evt.get("display_model", model),
                        "choices": [{"index": 0, "delta": {"content": nuevo}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(delta)}\n\n"
                if evt.get("final_sse_message"):
                    # Persistir estado del hilo para el siguiente turno
                    final_uuid = evt.get("backend_uuid", "")
                    final_rwt = evt.get("read_write_token")
                    _save_thread(messages, "".join(accumulated), final_uuid, final_rwt, device_id)

                    stop = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "model": evt.get("display_model", model),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "_pplx": {
                            "display_model": evt.get("display_model"),
                            "status": evt.get("status"),
                            "backend_uuid": final_uuid,
                        },
                    }
                    yield f"data: {json.dumps(stop)}\n\ndata: [DONE]\n\n"
                    return


async def _search_sync_chat(
    messages: list[ChatMessage],
    model: str,
    request_id: str,
    device_id: str,
    last_backend_uuid: str | None,
    read_write_token: str | None,
) -> dict:
    query = _extract_query(messages)
    body = _build_search_body(query, model, request_id, device_id, last_backend_uuid, read_write_token)

    async with AsyncSession(impersonate="chrome120") as s:
        r = await s.post(_SEARCH_URL, headers=_pplx_headers(), json=body, timeout=60)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=f"Perplexity HTTP {r.status_code}")
        display_model = model
        for line in r.text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            try:
                evt = json.loads(line[5:].strip())
            except Exception:
                continue
            if evt.get("display_model"):
                display_model = evt["display_model"]
            if evt.get("final_sse_message"):
                content = _extract_answer(evt.get("blocks", []))
                backend_uuid = evt.get("backend_uuid", "")
                rwt = evt.get("read_write_token")
                # Persistir estado del hilo para el siguiente turno
                _save_thread(messages, content, backend_uuid, rwt, device_id)
                return {
                    "content": content,
                    "display_model": display_model,
                    "status": evt.get("status"),
                    "backend_uuid": backend_uuid,
                }
    raise HTTPException(status_code=502, detail="No final_sse_message received")


@router.post(
    "/chat/completions",
    summary="Chat Completions (Search + Gemini Live)",
    description="""
OpenAI-compatible endpoint con **routing automático** por modelo y soporte de
**conversaciones multi-turno** para el backend Perplexity Search.

### Cómo funciona el threading

El servidor mantiene un caché en memoria (TTL 1 h) que asocia el historial de
mensajes con el `backend_uuid` + `read_write_token` de la última respuesta de
Perplexity. En cada follow-up envía `last_backend_uuid` + `is_related_query=true`
para que Perplexity mantenga el hilo server-side — igual que la app Android.

No se requiere ningún campo extra: el cliente solo necesita pasar los mensajes
en formato OpenAI estándar (incluyendo los turnos anteriores del asistente).

### Routing

| Modelo | Backend | Notas |
|---|---|---|
| `turbo` | **Perplexity Search** (recomendado) | Gratis, web-grounded |
| `gpt4o`, `gpt5`, `claude46sonnet`, `gemini25pro`, `grok4`, ... | **Perplexity Search** | Gratis → cae a `turbo` en free tier |
| `perplexity-gemini-live` | **Gemini Live** WebSocket | Audio → transcripción |
| `perplexity-gemini-live-charon/fenrir/zephyr` | **Gemini Live** (voz específica) | |

**Source APK:** `hs0.java` (params) | `fs0.java` + `es0.java` (search SSE) | `fvi.java` (Gemini Live)
""",
)
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    _=Depends(require_api_key),
):
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    use_gemini = body.model in GEMINI_LIVE_MODELS or body.model.startswith("perplexity-gemini-live")

    if use_gemini:
        token_manager = request.app.state.token_manager
        system_text, turns = _build_gemini_turns(body.messages)
        voice = _resolve_voice(body.model, body.x_voice)
        if not turns or turns[-1]["role"] != "user":
            raise HTTPException(status_code=400, detail="El último mensaje debe ser del rol 'user'.")
        if body.stream:
            return StreamingResponse(
                _gemini_stream(request_id, system_text, turns, voice, token_manager),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no"},
            )
        content = await _gemini_sync(system_text, turns, voice, token_manager)
        return {
            "id": request_id, "object": "chat.completion", "created": int(time.time()),
            "model": body.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1,
                      "_note": "Token counts not available — Gemini Live constrained endpoint."},
        }

    # Search backend — recuperar estado de hilo si existe
    if not any(m.role == "user" for m in body.messages):
        raise HTTPException(status_code=400, detail="Se requiere al menos un mensaje con rol 'user'.")

    last_backend_uuid, read_write_token, device_id = _get_thread(body.messages)

    if body.stream:
        return StreamingResponse(
            _search_stream_chat(
                body.messages, body.model, request_id,
                device_id, last_backend_uuid, read_write_token,
            ),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    result = await _search_sync_chat(
        body.messages, body.model, request_id,
        device_id, last_backend_uuid, read_write_token,
    )
    return {
        "id": request_id, "object": "chat.completion", "created": int(time.time()),
        "model": result["display_model"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result["content"]}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1,
                  "_note": "Token counts unavailable from Perplexity search SSE."},
        "_pplx": {
            "display_model": result["display_model"],
            "status": result["status"],
            "backend_uuid": result.get("backend_uuid", ""),
            "thread": "follow-up" if last_backend_uuid else "new",
        },
    }
