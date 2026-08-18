"""
API OpenAI-compatible sobre Grok gRPC.

Endpoints:
  GET  /health
  GET  /v1/models
  GET  /v1/models/{model_id}
  POST /v1/chat/completions     streaming y no-streaming, tools básico
  POST /auth/login              email + contraseña
  POST /auth/otp/send           envía OTP (cuentas Twitter/Google)
  POST /auth/otp/verify         verifica OTP → devuelve session_token

Parámetros OpenAI soportados:
  model            → mapeado al pool de alta tasa o al modelo específico
  messages         → convertido a prompt único con roles
  stream           → SSE token a token
  tools            → inyectados como contexto de sistema; respuesta parseada
  tool_choice      → ignorado (se detectan tool calls en la respuesta)
  is_reasoning     → pasa is_reasoning=True a Grok (modelos con razonamiento)
  disable_search   → campo nativo Grok (default: True)

Parámetros ignorados (Grok no los soporta en gRPC):
  temperature, top_p, max_tokens, stop, frequency_penalty, presence_penalty,
  n, logprobs, logit_bias, seed, response_format
"""
from __future__ import annotations
import os, time, uuid, json, base64
from typing import AsyncIterator, Optional, Union, Any

from fastapi import FastAPI, HTTPException, Depends, Header, Path, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

import grok_backend as backend
import auth_backend as auth

API_KEY     = os.environ.get("API_KEY", "")
APP_VERSION = "1.0.0"

app = FastAPI(
    title="Grok OpenAI Proxy",
    version=APP_VERSION,
    description=__doc__,
    docs_url="/docs",
)


# ── Auth middleware ───────────────────────────────────────────────────────────
def verify_key(authorization: str = Header(default="")):
    if not API_KEY:
        return
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── Schemas ───────────────────────────────────────────────────────────────────
class Message(BaseModel):
    role:     str
    content:  Union[str, list, None] = None
    name:     Optional[str]          = None
    tool_calls: Optional[list]       = None
    tool_call_id: Optional[str]      = None


class ChatRequest(BaseModel):
    model:              str              = "grok-4.5"
    messages:           list[Message]
    stream:             bool             = False
    tools:              Optional[list]   = None
    tool_choice:        Optional[Any]    = None
    # Parámetros OpenAI aceptados pero sin efecto en Grok gRPC:
    temperature:        Optional[float]  = None
    top_p:              Optional[float]  = None
    max_tokens:         Optional[int]    = None
    stop:               Optional[Any]    = None
    frequency_penalty:  Optional[float]  = None
    presence_penalty:   Optional[float]  = None
    n:                  Optional[int]    = None
    seed:               Optional[int]    = None
    # Extensiones propias
    is_reasoning:       bool             = False
    disable_search:     bool             = True


class LoginRequest(BaseModel):
    email:    str
    password: str

class OtpSendRequest(BaseModel):
    email: str

class OtpVerifyRequest(BaseModel):
    email: str
    code:  str


# ── Helpers de respuesta OpenAI ───────────────────────────────────────────────
def _cid() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _chunk(cid: str, model: str, content: str = "", finish: Optional[str] = None,
           tool_calls: Optional[list] = None) -> str:
    delta: dict = {}
    if content:
        delta["content"] = content
    if tool_calls:
        delta["tool_calls"] = tool_calls
    if not delta and not finish:
        delta["role"] = "assistant"
    return "data: " + json.dumps({
        "id":      cid,
        "object":  "chat.completion.chunk",
        "created": int(time.time()),
        "model":   model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }) + "\n\n"


def _full(cid: str, model: str, content: str,
          tool_calls: Optional[list] = None) -> dict:
    message: dict = {"role": "assistant", "content": content or None}
    finish = "stop"
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"]    = None
        finish = "tool_calls"
    return {
        "id":      cid,
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage":   {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":               "ok",
        "version":              APP_VERSION,
        "session_configured":   bool(os.environ.get("GROK_SESSION_TOKEN")),
        "high_rate_pool_size":  len(backend.HIGH_RATE_POOL),
    }


# ── /v1/models ────────────────────────────────────────────────────────────────
@app.get("/v1/models", dependencies=[Depends(verify_key)])
def list_models():
    ts     = int(time.time())
    models = []

    # Aliases OpenAI-compatibles → round-robin sobre el pool completo
    for alias in backend.MODEL_ALIASES:
        models.append({
            "id":       alias,
            "object":   "model",
            "created":  ts,
            "owned_by": "grok",
            "notes":    "round-robin over 13×999/h pool" if backend.MODEL_ALIASES[alias] is None else "",
        })

    # Modelos internos directos (se puede pedir uno específico)
    for mid, info in backend.MODELS_CATALOG.items():
        models.append({
            "id":            mid,
            "object":        "model",
            "created":       ts,
            "owned_by":      "grok",
            "rate_per_hour": info["rate"],
            "window_hours":  info["window_h"],
            "notes":         info["notes"],
        })

    return {"object": "list", "data": models}


# ── /v1/models/rates — rate limits en tiempo real (ANTES del wildcard) ───────
@app.get("/v1/models/rates", dependencies=[Depends(verify_key)])
def models_rates():
    """Rate limits actuales consultados en tiempo real a Grok."""
    live = backend.get_rate_limits_live()
    result = []
    for mid, info in backend.MODELS_CATALOG.items():
        live_data = live.get(mid, {})
        result.append({
            "model":          mid,
            "notes":          info["notes"],
            "rate_per_hour":  info["rate"],
            "window_hours":   info["window_h"],
            "remaining":      live_data.get("remaining"),
            "total":          live_data.get("total"),
            "window_seconds": live_data.get("window_seconds"),
        })
    return {"models": result, "pool_current": backend._rotator.current()}


@app.get("/v1/models/{model_id:path}", dependencies=[Depends(verify_key)])
def get_model(model_id: str):
    ts = int(time.time())
    info = backend.MODELS_CATALOG.get(model_id)
    if not info and model_id not in backend.MODEL_ALIASES:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    base = {"id": model_id, "object": "model", "created": ts, "owned_by": "grok"}
    if info:
        base.update({
            "rate_per_hour": info["rate"],
            "window_hours":  info["window_h"],
            "notes":         info["notes"],
        })
    return base


def _upload_images_from_messages(messages: list[dict]) -> list[str]:
    """
    Extrae imágenes base64 de los mensajes y las sube a Grok.
    Retorna lista de file_ids listos para pasar a stream_chat/complete_chat.
    Ignora silenciosamente las imágenes que fallen al subirse.
    """
    image_data = backend.extract_images_from_messages(messages)
    if not image_data:
        return []
    file_ids = []
    for i, (img_bytes, mime) in enumerate(image_data):
        ext = mime.split("/")[-1].replace("jpeg", "jpg")
        filename = f"image_{i}.{ext}"
        try:
            result = backend.upload_file(filename, content=img_bytes, mime_type=mime)
            fid = result.get("file_id")
            if fid:
                file_ids.append(fid)
        except Exception:
            pass
    return file_ids


# ── /v1/chat/completions ──────────────────────────────────────────────────────
@app.post("/v1/chat/completions", dependencies=[Depends(verify_key)])
async def chat_completions(req: ChatRequest):
    msgs_raw = [m.model_dump() for m in req.messages]
    model_id        = backend.resolve_model(req.model)
    prompt, system  = backend.messages_to_prompt(msgs_raw, tools=req.tools)

    # Subir imágenes adjuntas (content tipo lista con image_url)
    image_file_ids = _upload_images_from_messages(msgs_raw)

    cid = _cid()

    if req.stream:
        async def sse() -> AsyncIterator[str]:
            yield _chunk(cid, req.model)   # primer chunk con role
            full_text = []
            try:
                for token in backend.stream_chat(
                    prompt, model_id,
                    is_reasoning=req.is_reasoning,
                    disable_search=req.disable_search,
                    system=system,
                    image_file_ids=image_file_ids or None,
                ):
                    full_text.append(token)
                    # Si hay tools, no emitir hasta el final (necesitamos parsear)
                    if not req.tools:
                        yield _chunk(cid, req.model, token)
            except Exception as e:
                yield _chunk(cid, req.model, f"\n[Error: {e}]", finish="stop")
                yield "data: [DONE]\n\n"
                return

            if req.tools:
                text, calls = backend.parse_tool_calls("".join(full_text))
                if calls:
                    yield _chunk(cid, req.model, tool_calls=calls, finish="tool_calls")
                else:
                    yield _chunk(cid, req.model, text)
            yield _chunk(cid, req.model, finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        try:
            content = backend.complete_chat(
                prompt, model_id,
                is_reasoning=req.is_reasoning,
                disable_search=req.disable_search,
                system=system,
                image_file_ids=image_file_ids or None,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

        tool_calls = None
        if req.tools:
            content, tool_calls = backend.parse_tool_calls(content)
            if not tool_calls:
                tool_calls = None

        return JSONResponse(_full(cid, req.model, content, tool_calls))


# ── /v1/images/generations ───────────────────────────────────────────────────
class ImageRequest(BaseModel):
    prompt:  str
    n:       int            = Field(default=1, ge=1, le=4)
    size:    str            = "1024x1024"   # ignorado (Grok decide el tamaño)
    model:   Optional[str]  = None          # None = rota entre los 3 imagine models
    quality: str            = "standard"    # ignorado
    style:   str            = "vivid"       # ignorado


@app.post("/v1/images/generations", dependencies=[Depends(verify_key)])
def image_generations(req: ImageRequest):
    """
    Genera imágenes con Grok Imagine (Aurora).
    Rota automáticamente entre los 3 imagine-agent models (3×999/hora).
    model puede ser: imagine-agent-mode | imagine-agent-mode-dev | imagine-agent-mode-grok-4-5
    Si no se especifica, rota entre los tres.
    """
    prompt = req.prompt
    if req.n > 1:
        prompt = f"Generate {req.n} variations of: {req.prompt}"

    try:
        images = backend.generate_image(prompt, model_id=req.model)
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    if not images:
        raise HTTPException(
            status_code=503,
            detail="No images returned — the account may not have image generation access",
        )

    return {
        "created": int(time.time()),
        "data": [
            {
                "url":      img["url"],
                "image_id": img.get("image_id"),
                "asset_id": img.get("asset_id"),
                "model":    img.get("model"),
            }
            for img in images
        ],
    }


# ── /grok/skills ─────────────────────────────────────────────────────────────
@app.get("/grok/skills", dependencies=[Depends(verify_key)])
def grok_skills(locale: str = "en-US"):
    """Lista los skills disponibles (Excel, Word, PDF, etc.)."""
    try:
        return {"skills": backend.list_skills(locale)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ── /grok/settings ────────────────────────────────────────────────────────────
@app.get("/grok/settings", dependencies=[Depends(verify_key)])
def grok_settings():
    """Configuración del usuario en la cuenta de Grok."""
    try:
        return backend.get_user_settings()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ── /grok/modes ───────────────────────────────────────────────────────────────
@app.get("/grok/modes", dependencies=[Depends(verify_key)])
def grok_modes(locale: str = "en-US"):
    """Modos de conversación (fast, auto, expert, heavy, build)."""
    try:
        return backend.list_modes(locale)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ── /grok/voices ──────────────────────────────────────────────────────────────
@app.get("/grok/voices", dependencies=[Depends(verify_key)])
def grok_voices():
    """Lista de voces disponibles para síntesis de voz."""
    try:
        return backend.list_voices()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ── /grok/rate-limits/{model_id} ─────────────────────────────────────────────
@app.get("/grok/rate-limits/{model_id:path}", dependencies=[Depends(verify_key)])
def grok_rate_limit(model_id: str, kind: int = 0):
    """Rate limits en tiempo real para un modelo específico."""
    try:
        return backend.get_rate_limit_single(model_id, kind)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ── /grok/conversations ───────────────────────────────────────────────────────
@app.get("/grok/conversations", dependencies=[Depends(verify_key)])
def grok_list_conversations(limit: int = 20, cursor: str = ""):
    """Lista conversaciones del usuario (título, fecha, ID)."""
    try:
        return backend.list_conversations(limit=limit, cursor=cursor)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/grok/conversations/{conv_id}", dependencies=[Depends(verify_key)])
def grok_get_conversation(conv_id: str):
    """Metadatos de una conversación (título, fechas)."""
    try:
        return backend.get_conversation(conv_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.delete("/grok/conversations/{conv_id}", dependencies=[Depends(verify_key)])
def grok_delete_conversation(conv_id: str):
    """Elimina una conversación permanentemente."""
    try:
        return backend.delete_conversation(conv_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


class ConversationUpdateRequest(BaseModel):
    title: str


@app.patch("/grok/conversations/{conv_id}", dependencies=[Depends(verify_key)])
def grok_update_conversation(conv_id: str, req: ConversationUpdateRequest):
    """Renombra una conversación."""
    try:
        return backend.update_conversation(conv_id, req.title)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/grok/conversations/{conv_id}/messages", dependencies=[Depends(verify_key)])
def grok_list_responses(conv_id: str):
    """
    Historial completo de mensajes de una conversación.
    Incluye response_id, rol, texto, parent_id y modelo.
    """
    try:
        msgs = backend.list_responses(conv_id)
        return {"conversation_id": conv_id, "messages": msgs, "count": len(msgs)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


class FileUploadRequest(BaseModel):
    filename:  str
    content_b64: Optional[str] = None  # base64-encoded file content
    mime_type:   Optional[str] = None

class AddResponseRequest(BaseModel):
    message: str
    model:   Optional[str] = None
    stream:  bool          = False
    disable_search: bool   = True


@app.post("/grok/conversations/{conv_id}/messages", dependencies=[Depends(verify_key)])
async def grok_add_response(conv_id: str, req: AddResponseRequest):
    """
    Añade un mensaje a una conversación existente y obtiene la respuesta.
    Soporta streaming (SSE) igual que /v1/chat/completions.
    """
    model_id = backend.resolve_model(req.model) if req.model else None
    cid = _cid()

    if req.stream:
        async def sse() -> AsyncIterator[str]:
            yield _chunk(cid, req.model or "grok", finish=None)
            try:
                for token in backend.stream_add_response(
                    conv_id, req.message,
                    model_id=model_id,
                    disable_search=req.disable_search,
                ):
                    yield _chunk(cid, req.model or "grok", token)
            except Exception as e:
                yield _chunk(cid, req.model or "grok", f"\n[Error: {e}]", finish="stop")
                yield "data: [DONE]\n\n"
                return
            yield _chunk(cid, req.model or "grok", finish="stop")
            yield "data: [DONE]\n\n"
        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        try:
            content = backend.complete_add_response(
                conv_id, req.message,
                model_id=model_id,
                disable_search=req.disable_search,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))
        return JSONResponse(_full(cid, req.model or "grok", content))


@app.post("/grok/conversations/{conv_id}/share", dependencies=[Depends(verify_key)])
def grok_share_conversation(conv_id: str, resp_id: Optional[str] = None, share_publicly: bool = True):
    """
    Genera un link de compartir para una conversación.
    Si resp_id no se indica, usa la última respuesta del asistente.
    Retorna share_token y share_url.
    """
    try:
        return backend.share_conversation(conv_id, resp_id=resp_id, share_publicly=share_publicly)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ── /grok/files ───────────────────────────────────────────────────────────────
@app.post("/grok/files", dependencies=[Depends(verify_key)])
def grok_upload_file(req: FileUploadRequest):
    """
    Sube (o registra) un archivo en el storage de Grok.
    Retorna file_id y storage_path que pueden usarse en mensajes futuros.

    Tipos soportados: txt, py, json, html, css, csv, svg, xlsx, pptx, docx, zip, mp3.
    Para imágenes (jpg/png) enviar content_b64 con los bytes reales.
    content_b64: contenido del archivo en base64 (opcional para archivos de texto pequeños).
    """
    content = None
    if req.content_b64:
        try:
            content = base64.b64decode(req.content_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="content_b64 no es base64 válido")
    try:
        return backend.upload_file(req.filename, content=content, mime_type=req.mime_type)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ── /grok/memory ──────────────────────────────────────────────────────────────
@app.delete("/grok/memory", dependencies=[Depends(verify_key)])
def grok_delete_memory(conv_id: Optional[str] = None):
    """
    Elimina la memoria de Grok asociada a una conversación específica,
    o la memoria global si no se indica conv_id.
    """
    try:
        return backend.delete_memory(conv_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ── /grok/suggest ─────────────────────────────────────────────────────────────
@app.get("/grok/suggest", dependencies=[Depends(verify_key)])
def grok_suggest(
    q: str,
    types: list[str] = Query(
        default=["search_completion", "quick_answer", "stock", "grok_completion"]
    ),
):
    """
    Sugerencias en tiempo real para un query (search completions, quick answers, stocks).
    """
    try:
        return {"suggestions": backend.stream_suggestions_collect(q, types)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ── /auth/login ───────────────────────────────────────────────────────────────
@app.post("/auth/login")
def login(req: LoginRequest):
    """Login con email y contraseña. Retorna session_token."""
    try:
        token = auth.login_email_password(req.email, req.password)
    except RuntimeError as e:
        raise HTTPException(status_code=401, detail=str(e))
    return {"session_token": token, "hint": "Setear como GROK_SESSION_TOKEN en el .env"}


# ── /auth/otp/send ────────────────────────────────────────────────────────────
@app.post("/auth/otp/send")
def otp_send(req: OtpSendRequest):
    """
    Envía OTP al email. Para cuentas sin contraseña (Twitter/Google OAuth).
    """
    try:
        auth.otp_send(req.email)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "message": f"OTP enviado a {req.email}"}


# ── /auth/otp/verify ──────────────────────────────────────────────────────────
@app.post("/auth/otp/verify")
def otp_verify(req: OtpVerifyRequest):
    """
    Verifica OTP y retorna session_token.
    Acepta '938-612' o '938612'.
    """
    try:
        token = auth.otp_verify(req.email, req.code)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "session_token": token,
        "hint": "Guardá este token como GROK_SESSION_TOKEN en el .env",
    }
