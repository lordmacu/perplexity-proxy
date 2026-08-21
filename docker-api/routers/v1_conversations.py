"""
/v1/conversations — historial de hilos nativo de Perplexity.

Sin almacenamiento local: los hilos viven en los servidores de Perplexity. El
`_THREAD_CACHE` de `v1_chat.py` es otra cosa — una caché en memoria para
continuar un hilo dentro del mismo proceso, no un listado.

El protocolo se tomó del APK decompilado, no de la colección de Postman del
repo, que se equivoca en dos puntos (dice que el listado es `GET`, y que
`export` recibe `thread_id`). Las declaraciones Retrofit reales son:

  jkp.java:76   @POST /rest/thread/list_ask_threads
                @Query source, version, include_entity_relations
                @Body  dkk  -> descriptor bkk.java
                returns List<zll>  -> descriptor xll.java

  cjp.java:15   @GET  /rest/thread/{backend_uuid_or_slug}
                @Query with_parent_info, with_schematized_response,
                       supported_block_use_cases, source, version
                returns edk  -> descriptor cdk.java

La forma de respuesta replica la de mistral-proxy (`api/routes/conversations.py`)
a propósito: el gateway debe leer una sola forma para los cinco proxies.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from curl_cffi.requests import AsyncSession
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from auth import require_api_key
from config import settings, PPLX_HEADERS

router = APIRouter(prefix="/v1/conversations", tags=["OpenAI — Conversations"])

BASE = settings.pplx_base

# El cliente Android manda estos dos en cada llamada; se replican para que la
# petición sea indistinguible de la suya.
_SOURCE = "android"
_APK_VERSION = "2.95.0"

# Tope de páginas que recorre la búsqueda por id. 100 x 20 = 20 000 hilos, muy
# por encima de cualquier cuenta real; existe para que un backend que nunca
# devuelva una página corta no deje la petición girando para siempre.
_MAX_LOOKUP_PAGES = 20


# ── Forma de respuesta (idéntica a mistral-proxy) ─────────────────────────────

class ConversationItem(BaseModel):
    id: str
    title: Optional[str] = None
    generated_title: Optional[str] = None
    updated_at: Optional[str] = None
    pinned: bool = False
    project_id: Optional[str] = None


class ConversationList(BaseModel):
    object: str = "list"
    data: list[ConversationItem]
    next_cursor: Optional[str] = None


class ConversationMessage(BaseModel):
    role: str
    content: str
    id: Optional[str] = None


class ConversationMessages(BaseModel):
    object: str = "list"
    conversation_id: str
    data: list[ConversationMessage]
    next_cursor: Optional[str] = None


# ── Transporte ────────────────────────────────────────────────────────────────

def _headers() -> dict:
    return {
        **PPLX_HEADERS,
        "Cookie": f"__Secure-next-auth.session-token={settings.perplexity_session}",
    }


async def _call(method: str, path: str, *, params: dict, json_body: Optional[dict] = None) -> Any:
    """Una llamada nativa, con el error upstream propagado tal cual.

    A diferencia de `routers/perplexity.py`, que devuelve `{"status", "body"}`
    para que el cliente vea la respuesta cruda, acá un fallo upstream tiene que
    ser un fallo HTTP: estas rutas prometen un esquema, y devolver 200 con un
    cuerpo de error adentro rompería a cualquiera que confíe en él.
    """
    async with AsyncSession() as s:
        r = await s.request(
            method, f"{BASE}{path}", headers=_headers(), params=params,
            json=json_body, impersonate="chrome120", timeout=20,
        )
    if r.status_code != 200:
        raise HTTPException(
            status_code=502 if r.status_code >= 500 else r.status_code,
            detail=f"perplexity {path}: {r.text[:200]}",
        )
    try:
        return r.json()
    except Exception:
        raise HTTPException(status_code=502, detail=f"perplexity {path}: respuesta no-JSON")


def _thread_to_item(t: dict) -> ConversationItem:
    """Mapea un `zll` (xll.java) a la forma estándar.

    `title` en Perplexity ya viene generado a partir de la consulta salvo que el
    usuario lo renombre, y la respuesta no distingue un caso del otro — por eso
    `generated_title` queda en None en lugar de duplicar `title` y afirmar algo
    que el backend no dice.
    """
    return ConversationItem(
        id=t.get("uuid") or "",
        title=t.get("title"),
        generated_title=None,
        updated_at=t.get("last_query_datetime"),
        pinned=bool(t.get("is_pinned") or False),
        project_id=None,
    )


async def _list_threads(
    *, limit: int, offset: int, show_archived: bool, search: Optional[str],
) -> list[dict]:
    """Una página del listado.

    `limit` y `offset` son reales y están medidos: dos páginas contiguas de 5
    devolvieron uuids disjuntos. No aparecían en la primera lectura del
    descriptor porque jadx los dejó como referencias a constantes
    (`MapboxMap.QFE_LIMIT` / `QFE_OFFSET` en bkk.java:16-17) en vez de
    literales -- sin ellos este endpoint devolvía siempre los mismos 20 hilos
    y juraba que eran todos.
    """
    body: dict = {
        "ascending": False,
        "limit": limit,
        "offset": offset,
        "show_archived": show_archived,
    }
    if search:
        body["search_term"] = search
    data = await _call(
        "POST", "/rest/thread/list_ask_threads",
        params={
            "source": _SOURCE,
            "version": _APK_VERSION,
            "include_entity_relations": "false",
        },
        json_body=body,
    )
    return data if isinstance(data, list) else []


def _parse_cursor(cursor: Optional[str]) -> int:
    """El cursor es el `offset` de la próxima página, como string.

    Opaco para quien llama, igual que en mistral-proxy: si mañana el backend
    cambia a un token de verdad, cambia esta función y nadie más se entera.
    """
    if not cursor:
        return 0
    try:
        value = int(cursor)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"cursor inválido: {cursor}")
    if value < 0:
        raise HTTPException(status_code=400, detail=f"cursor inválido: {cursor}")
    return value


# ── Rutas ─────────────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=ConversationList,
    summary="Listar conversaciones",
    description="""
Lista los hilos de la cuenta, paginado.

**Backend:** `POST /rest/thread/list_ask_threads` (jkp.java:76 del APK v2.95.0),
con `limit`/`offset` en el cuerpo (bkk.java:16-17).

`next_cursor` viene con el offset de la página siguiente, o `null` si esta es la
última. Se pasa tal cual en `cursor` para seguir.
""",
)
async def list_conversations(
    limit: int = Query(20, ge=1, le=100),
    cursor: Optional[str] = Query(None, description="`next_cursor` de la respuesta anterior"),
    search: Optional[str] = Query(None, description="Filtra por `search_term` upstream"),
    show_archived: bool = Query(False),
    _=Depends(require_api_key),
):
    offset = _parse_cursor(cursor)
    threads = await _list_threads(
        limit=limit, offset=offset, show_archived=show_archived, search=search,
    )
    items = [_thread_to_item(t) for t in threads if isinstance(t, dict)]

    # `has_next_page` viene por hilo, no por página; se toma del primero. Si la
    # página volvió vacía o incompleta ya no hay más, sin importar qué diga.
    more = bool(threads and len(threads) >= limit
                and isinstance(threads[0], dict) and threads[0].get("has_next_page"))
    return ConversationList(
        data=items,
        next_cursor=str(offset + limit) if more else None,
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationItem,
    summary="Metadata de una conversación",
    description="""
**Backend:** `POST /rest/thread/list_ask_threads`, paginando hasta encontrar el
`uuid`.

Deliberadamente NO usa `GET /rest/thread/{backend_uuid_or_slug}`: esa ruta
devuelve `{entries, background_entries, has_next_page, next_cursor,
thread_metadata}` (cdk.java) y su `thread_metadata` solo contiene `crons`
(nml.java) — no trae título, ni fecha, ni estado de pin. Los únicos campos que
esta respuesta promete viven en el listado.
""",
)
async def get_conversation(conversation_id: str, _=Depends(require_api_key)):
    page_size = 100
    for page in range(_MAX_LOOKUP_PAGES):
        threads = await _list_threads(
            limit=page_size, offset=page * page_size, show_archived=True, search=None,
        )
        for t in threads:
            if isinstance(t, dict) and t.get("uuid") == conversation_id:
                return _thread_to_item(t)
        if len(threads) < page_size:
            break
    raise HTTPException(status_code=404, detail=f"conversación no encontrada: {conversation_id}")


def _extract_answer(text: Any) -> str:
    """Saca el texto del asistente del campo `text` de una entry.

    Medido contra la cuenta real, no inferido: `text` es un STRING con JSON
    adentro -- una lista de pasos. El paso `FINAL` tiene `content.answer`, que a
    su vez es OTRO string con JSON: `{"answer": "...", "chunks": [...]}`. Dos
    capas de codificación, y ninguna documentada en el APK, porque el cliente
    Android nunca lee este campo (usa el stream SSE en vivo).

    Distinto de `_extract_answer` en `v1_chat.py`: aquel lee los `blocks` del
    SSE, que es otra forma. Mismo nombre, formatos distintos -- no es duplicado.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        steps = json.loads(text)
    except json.JSONDecodeError:
        return ""
    if not isinstance(steps, list):
        return ""

    for step in reversed(steps):
        if not isinstance(step, dict) or step.get("step_type") != "FINAL":
            continue
        content = step.get("content")
        if not isinstance(content, dict):
            continue
        answer = content.get("answer")
        if not isinstance(answer, str):
            continue
        try:
            inner = json.loads(answer)
        except json.JSONDecodeError:
            return answer.strip()
        if isinstance(inner, dict):
            plain = inner.get("answer")
            if isinstance(plain, str) and plain.strip():
                return plain.strip()
            chunks = inner.get("chunks")
            if isinstance(chunks, list):
                joined = "".join(str(c) for c in chunks).strip()
                if joined:
                    return joined
        return answer.strip()
    return ""


def _entry_to_messages(entry: dict) -> list[ConversationMessage]:
    """Convierte una entry en el par usuario/asistente.

    Las entries no tienen esquema estático -- `szb.java` las deserializa como
    `Map<String, JsonElement>` y el cliente las interpreta en runtime -- así que
    los nombres de campo (`query_str`, `text`) salen de medir la respuesta real,
    y una entry de la que no se pueda sacar texto se omite en vez de producir un
    mensaje vacío.
    """
    out: list[ConversationMessage] = []
    entry_id = entry.get("backend_uuid") or entry.get("uuid")

    question = entry.get("query_str")
    if isinstance(question, str) and question.strip():
        out.append(ConversationMessage(role="user", content=question.strip(), id=entry_id))

    answer = _extract_answer(entry.get("text"))
    if answer:
        out.append(ConversationMessage(role="assistant", content=answer, id=entry_id))

    return out


@router.get(
    "/{conversation_id}/messages",
    response_model=ConversationMessages,
    summary="Mensajes de una conversación",
    description="""
**Backend:** `GET /rest/thread/{backend_uuid_or_slug}` (cjp.java:15 del APK v2.95.0).

Devuelve los pares consulta/respuesta del hilo. Las entries upstream no tienen
esquema estático — el propio cliente Android las lee como un mapa JSON libre
(`szb.java`) — así que el mapeo prueba los nombres conocidos y descarta lo que
no reconoce en vez de inventar contenido.
""",
)
async def get_messages(conversation_id: str, _=Depends(require_api_key)):
    data = await _call(
        "GET", f"/rest/thread/{conversation_id}",
        params={
            "with_parent_info": "true",
            "with_schematized_response": "false",
            "source": _SOURCE,
            "version": _APK_VERSION,
        },
    )
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="perplexity: respuesta de hilo inesperada")

    messages: list[ConversationMessage] = []
    for entry in data.get("entries") or []:
        if isinstance(entry, dict):
            messages.extend(_entry_to_messages(entry))

    return ConversationMessages(
        conversation_id=conversation_id,
        data=messages,
        next_cursor=data.get("next_cursor") or None,
    )
