from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

import asyncio
import logging

from token_manager import TokenManager
from auth_watchdog import watchdog_loop
from routers import v1_chat, v1_audio, v1_models, perplexity, v1_search, v1_auth

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.token_manager = TokenManager()
    watchdog_task = asyncio.create_task(watchdog_loop(), name="auth_watchdog")
    yield
    watchdog_task.cancel()
    try:
        await watchdog_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Perplexity Proxy API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_chat.router,       prefix="/v1",          tags=["OpenAI — Chat"])
app.include_router(v1_search.router,     prefix="/v1",          tags=["OpenAI — Chat"])
app.include_router(v1_audio.router,      prefix="/v1",          tags=["OpenAI — Audio"])
app.include_router(v1_models.router,     prefix="/v1",          tags=["OpenAI — Models"])
app.include_router(perplexity.router,    prefix="/perplexity",  tags=["Perplexity Native"])
app.include_router(v1_search.router,     prefix="/perplexity",  tags=["Perplexity Native"])
app.include_router(v1_auth.router,       prefix="/perplexity",  tags=["Auth"])


@app.get("/health", tags=["Meta"], summary="Health check")
async def health():
    return {"status": "ok"}


@app.get("/", tags=["Meta"], summary="Info del proxy")
async def root():
    return {
        "name": "Perplexity Proxy API",
        "version": "1.0.0",
        "docs": "/docs",
        "source": "APK RE v2.95.0",
        "openai_compatible": ["/v1/chat/completions", "/v1/chat/completions/search", "/v1/audio/speech", "/v1/models"],
        "perplexity_native": [
            "/perplexity/models/config",
            "/perplexity/models/modes",
            "/perplexity/tokens/gemini",
            "/perplexity/tokens/soniox",
            "/perplexity/tokens/ai-coustics",
            "/perplexity/session/v2",
            "/perplexity/session/v4",
            "/perplexity/realtime/execute-tools",
            "/perplexity/realtime/create-entry",
            "/perplexity/sdk/config",
            "/perplexity/auth/exchange",
            "/perplexity/tts",
            "/perplexity/search",
        ],
    }


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title="Perplexity Proxy API",
        version="1.0.0",
        description="""
## Proxy OpenAI-compatible para Perplexity

Expone los endpoints internos del APK de Perplexity v2.95.0 como una API OpenAI-compatible.

---

### Compatibilidad OpenAI

| Endpoint | Compatible | Backend |
|---|---|---|
| Endpoint | Compatible | Backend |
|---|---|---|
| `POST /v1/chat/completions` | ✓ Parcial | Gemini 3.1 Flash Live Preview |
| `POST /v1/chat/completions/search` | ✓ Mejor | Perplexity Search SSE (web-grounded) |
| `POST /v1/audio/speech` | ✓ Parcial | Perplexity TTS SSE (8 voces) |
| `GET /v1/models` | ✓ | Dinámico (124 modelos) |
| `POST /v1/audio/transcriptions` | ✗ No implementado | Requiere Soniox WS |
| `POST /v1/embeddings` | ✗ No disponible | — |
| `POST /v1/images/generations` | ✗ No disponible | — |

---

### Recomendación: usar Search en lugar de Gemini Live

`/v1/chat/completions/search` es **mejor** que `/v1/chat/completions` para la mayoría de casos:
- Respuesta de texto puro (sin audio → transcripción)
- Búsqueda web real con fuentes indexadas
- Menor latencia

### Free tier: todos los modelos caen a `turbo`

En free tier, **cualquier `model`** se acepta pero Perplexity lo degrada silenciosamente a `turbo`.
El campo `_pplx.display_model` en la respuesta muestra qué modelo corrió realmente.

Modelos testeados (todos → `turbo` en free): `gpt4o`, `gpt5`, `claude46sonnet`, `gemini25pro`, `grok4`, `r1`...

---

### ¿Qué NO funciona igual que OpenAI?

- **Chat completions:** la respuesta pasa por Gemini Live (audio → transcripción), no es un modelo de texto puro.
  Puede haber errores de transcripción menores. No soporta `temperature`, `max_tokens`, `tools`.
- **TTS:** siempre devuelve MP3. No soporta `speed` ni otros `response_format`.
- **Token counts:** siempre devuelven `-1` (Gemini Live constrained no expone uso).

---

### Autenticación del proxy

Si `PROXY_API_KEY` está configurada: `Authorization: Bearer <tu_key>`.
Si no está configurada: no se requiere auth.

---

### Fuente

Reverse engineering del APK Android de Perplexity v2.95.0.
Archivos clave: `fvi.java`, `f91.java`, `g3j.java`, `xvi.java`, `j0g.java`, `libmultimodal_uniffi.so`.
        """,
        routes=app.routes,
        tags=[
            {
                "name": "OpenAI — Chat",
                "description": "Endpoints compatibles con la API de Chat de OpenAI. Backend: Gemini Live vía tokens Perplexity.",
            },
            {
                "name": "OpenAI — Audio",
                "description": "Endpoints compatibles con la API de Audio de OpenAI. Backend: TTS SSE de Perplexity.",
            },
            {
                "name": "OpenAI — Models",
                "description": "Listado de modelos disponibles en este proxy.",
            },
            {
                "name": "Perplexity Native",
                "description": (
                    "Endpoints nativos de Perplexity descubiertos via RE del APK. "
                    "Proxy directo sin transformación OpenAI. "
                    "Incluye: tokens (Gemini, Soniox, AI Coustics), sesiones de voz (v2/v4), "
                    "execute-tools, create-entry, SDK config, auth exchange, TTS raw."
                ),
            },
            {
                "name": "Meta",
                "description": "Endpoints de infraestructura: health check, info del proxy.",
            },
        ],
    )
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi
