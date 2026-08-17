"""
POST /v1/chat/completions  (mode: search)
POST /perplexity/search

Uses Perplexity's native SSE search endpoint (/rest/sse/perplexity_ask).
Returns real web-search-grounded text responses (not Gemini Live audio).

Free tier behavior: any model_api_name is accepted but silently downgraded
to "turbo" on free accounts. display_model in response shows what ran.
"""
from __future__ import annotations

import json
import uuid
from typing import AsyncGenerator, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from auth import require_api_key
from config import settings, PPLX_HEADERS
from curl_cffi.requests import AsyncSession

router = APIRouter()

_SEARCH_URL = f"{settings.pplx_base}/rest/sse/perplexity_ask"

SEARCH_MODES = Literal["search", "research", "study", "browser_agent"]


def _headers() -> dict:
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


async def _search_stream(
    query: str,
    model_api_name: str,
    search_mode: str,
    frontend_uuid: str,
) -> AsyncGenerator[bytes, None]:
    body = {
        "query_str": query,
        "params": {
            "query_mode": "COPILOT",
            "search_mode": search_mode,
            "model_api_name": model_api_name,
            "search_mode_supports_reasoning": False,
            "frontend_uuid": frontend_uuid,
            "version": "2.9",
        },
    }
    async with AsyncSession(impersonate="chrome120") as s:
        async with s.stream(
            "POST",
            _SEARCH_URL,
            headers=_headers(),
            json=body,
            timeout=60,
        ) as resp:
            if resp.status_code != 200:
                err = {"error": {"message": f"Perplexity returned HTTP {resp.status_code}", "type": "upstream_error"}}
                yield f"data: {json.dumps(err)}\n\ndata: [DONE]\n\n".encode()
                return

            async for raw in resp.aiter_lines():
                # curl_cffi entrega bytes, no str: decodificar antes de tocar el texto.
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8", "replace")
                raw = raw.strip()
                if not raw or not raw.startswith("data:"):
                    continue
                try:
                    evt = json.loads(raw[5:].strip())
                except Exception:
                    continue

                # Stream incremental chunks
                blocks = evt.get("blocks", [])
                for block in blocks:
                    mb = block.get("markdown_block") or {}
                    for chunk in mb.get("chunks", []):
                        if chunk:
                            delta = {
                                "id": frontend_uuid,
                                "object": "chat.completion.chunk",
                                "model": evt.get("display_model", model_api_name),
                                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(delta)}\n\n".encode()

                if evt.get("final_sse_message"):
                    stop_delta = {
                        "id": frontend_uuid,
                        "object": "chat.completion.chunk",
                        "model": evt.get("display_model", model_api_name),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "_pplx": {
                            "display_model": evt.get("display_model"),
                            "user_selected_model": evt.get("user_selected_model"),
                            "search_mode": evt.get("search_mode"),
                            "status": evt.get("status"),
                        },
                    }
                    yield f"data: {json.dumps(stop_delta)}\n\ndata: [DONE]\n\n".encode()
                    return


async def _search_sync(
    query: str,
    model_api_name: str,
    search_mode: str,
    frontend_uuid: str,
) -> dict:
    body = {
        "query_str": query,
        "params": {
            "query_mode": "COPILOT",
            "search_mode": search_mode,
            "model_api_name": model_api_name,
            "search_mode_supports_reasoning": False,
            "frontend_uuid": frontend_uuid,
            "version": "2.9",
        },
    }
    async with AsyncSession(impersonate="chrome120") as s:
        r = await s.post(
            _SEARCH_URL,
            headers=_headers(),
            json=body,
            timeout=60,
        )
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=f"Perplexity returned HTTP {r.status_code}")

        display_model = model_api_name
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
                answer = _extract_answer(evt.get("blocks", []))
                return {
                    "answer": answer,
                    "display_model": display_model,
                    "user_selected_model": evt.get("user_selected_model"),
                    "search_mode": evt.get("search_mode"),
                    "status": evt.get("status"),
                }
    raise HTTPException(status_code=502, detail="No final_sse_message received")


# ── Native Perplexity Search endpoint ────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., description="Query to search (plain text, no messages array needed)")
    model: str = Field(
        "turbo",
        description=(
            "Model to use. **Free tier silently falls back to `turbo` for all models.**\n\n"
            "| Model | Provider |\n|---|---|\n"
            "| `turbo` | Perplexity (free) |\n"
            "| `gpt4o`, `gpt41`, `gpt5`, `o4mini` | OpenAI → turbo on free |\n"
            "| `claude46sonnet`, `claude45haiku` | Anthropic → turbo on free |\n"
            "| `gemini25pro`, `gemini35flash` | Google → turbo on free |\n"
            "| `grok4`, `grok46medium` | XAI → turbo on free |"
        ),
    )
    search_mode: SEARCH_MODES = Field(
        "search",
        description=(
            "Search mode:\n"
            "- `search` — standard web search (✓ free)\n"
            "- `research` — deep research (requires Pro/Max)\n"
            "- `study` — study mode (requires Pro)\n"
            "- `browser_agent` — browser agent (requires Max)"
        ),
    )
    stream: bool = Field(False, description="Stream response as SSE delta chunks")


@router.post(
    "/search",
    summary="Perplexity Web Search (native)",
    description="""
Direct access to Perplexity's main search SSE endpoint (`/rest/sse/perplexity_ask`).

Returns **web-search-grounded text** responses. Unlike `/v1/chat/completions` (Gemini Live),
this uses Perplexity's actual search AI with real-time web indexing.

**Free tier:** any model is accepted but silently downgraded to `turbo`.
The `display_model` field in the response shows what actually ran.

**Source APK:** `pvp.java` (request builder), `es0.java` (SSE parser), `fs0.java` (endpoint constant)
""",
    tags=["Perplexity Native"],
)
async def perplexity_search(
    body: SearchRequest,
    _=Depends(require_api_key),
):
    fid = str(uuid.uuid4())
    if body.stream:
        return StreamingResponse(
            _search_stream(body.query, body.model, body.search_mode, fid),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )
    return await _search_sync(body.query, body.model, body.search_mode, fid)


# ── OpenAI-compatible search completions ─────────────────────────────────────

class ChatMsg(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class SearchChatRequest(BaseModel):
    messages: list[ChatMsg] = Field(
        ...,
        description=(
            "✓ Soportado. El **último** mensaje `user` se usa como query de búsqueda.\n"
            "Mensajes `system` y `assistant` anteriores se ignoran (Perplexity search no tiene historial multi-turn en este endpoint)."
        ),
    )
    model: str = Field(
        "turbo",
        description=(
            "~ Parcial. Se pasa como `model_api_name` pero en **free tier siempre corre `turbo`**.\n\n"
            "Usa modelos del catálogo de 124 modelos de Perplexity (ver `/v1/models`)."
        ),
    )
    search_mode: SEARCH_MODES = Field(
        "search",
        description="Modo de búsqueda: `search` (free), `research`/`study`/`browser_agent` (Pro/Max).",
    )
    stream: bool = Field(False, description="✓ Streaming SSE.")
    temperature: float | None = Field(None, description="✗ No soportado.")
    max_tokens: int | None = Field(None, description="✗ No soportado.")
    top_p: float | None = Field(None, description="✗ No soportado.")
    tools: list | None = Field(None, description="✗ No soportado.")
    n: int | None = Field(None, description="✗ No soportado. Siempre 1 resultado.")

    model_config = {"extra": "allow"}


@router.post(
    "/chat/completions/search",
    summary="Chat Completions — Search mode (OpenAI-compatible)",
    description="""
**Alternativa** a `/v1/chat/completions` (Gemini Live): usa el motor de búsqueda web
real de Perplexity en lugar de Gemini Live.

**Ventajas sobre Gemini Live:**
- Respuesta de texto puro (sin audio → transcripción)
- Búsqueda web real con fuentes indexadas
- Menor latencia
- Sin errores de transcripción

**Limitaciones:**
- En free tier: modelo siempre cae a `turbo`
- No soporta historial multi-turn (solo el último mensaje `user`)
- No soporta `temperature`, `max_tokens`, `tools`

**Source APK:** `pvp.java` + `fs0.java` + `es0.java`
""",
    tags=["OpenAI — Chat"],
)
async def search_chat_completions(
    body: SearchChatRequest,
    _=Depends(require_api_key),
):
    # Extraer el último mensaje user como query
    user_msgs = [m for m in body.messages if m.role == "user"]
    if not user_msgs:
        raise HTTPException(status_code=400, detail="Se requiere al menos un mensaje con rol 'user'.")
    query = user_msgs[-1].content
    fid = str(uuid.uuid4())

    if body.stream:
        return StreamingResponse(
            _search_stream(query, body.model, body.search_mode, fid),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    result = await _search_sync(query, body.model, body.search_mode, fid)
    import time
    return {
        "id": f"chatcmpl-{fid[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": result["display_model"],
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["answer"]},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1,
                  "_note": "Token counts unavailable from Perplexity search SSE."},
        "_pplx": {
            "display_model": result["display_model"],
            "user_selected_model": result["user_selected_model"],
            "search_mode": result["search_mode"],
            "status": result["status"],
        },
    }
