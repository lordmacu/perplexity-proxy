from __future__ import annotations

import time
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from auth import require_api_key
from config import settings, PPLX_HEADERS
from curl_cffi.requests import AsyncSession

router = APIRouter()

MODELS_CONFIG_URL = f"{settings.pplx_base}/rest/models/config/v2"


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: str = "list"
    data: list[Model]


def _headers():
    return {
        **PPLX_HEADERS,
        "Cookie": f"__Secure-next-auth.session-token={settings.perplexity_session}",
    }


@router.get(
    "/models",
    response_model=ModelList,
    summary="Listar modelos disponibles (dinámico)",
    description="""
Devuelve **todos** los modelos disponibles en Perplexity, obtenidos dinámicamente
de `/rest/models/config/v2`.

**100+ modelos** de múltiples providers:

| Provider | Ejemplos |
|---|---|
| `PERPLEXITY` | turbo, pplx_pro, pplx_alpha |
| `OPENAI` | gpt4o, gpt5, gpt56_sol, o3pro |
| `ANTHROPIC` | claude46sonnet, claude45opus, pplx_asi_fable_5 |
| `GOOGLE` | gemini35flash, gemini31pro, veo31 |
| `XAI` | grok4, grok46medium |
| `MOONSHOT_AI` | kimik3thinking, kimik26instant |
| `NVIDIA` | nv_nemotron_3_ultra |
| `SONAR` | pplx_sonar_internal_testing |
| `FIREWORKS` | pplx_asi_kimi, pplx_asi_qwen |

Incluye también los modelos propios de voice proxy:
- `perplexity-gemini-live`, `perplexity-gemini-live-charon`, etc.
- `perplexity-tts`

**Fuente APK:** `rp2.java` (interface Retrofit) / `ank.java` (response model)
""",
)
async def list_models(_=Depends(require_api_key)):
    ts = int(time.time())
    models: list[Model] = []

    try:
        async with AsyncSession() as s:
            r = await s.get(
                MODELS_CONFIG_URL,
                headers=_headers(),
                impersonate="chrome120",
                timeout=10,
            )
            data = r.json()
            for model_id, info in data.get("models", {}).items():
                provider = (info.get("provider") or "PERPLEXITY").lower()
                models.append(Model(id=model_id, created=ts, owned_by=provider))
    except Exception:
        pass

    # Añadir modelos propios del proxy (siempre presentes)
    proxy_models = [
        Model(id="perplexity-gemini-live",        created=ts, owned_by="perplexity-proxy"),
        Model(id="perplexity-gemini-live-charon",  created=ts, owned_by="perplexity-proxy"),
        Model(id="perplexity-gemini-live-fenrir",  created=ts, owned_by="perplexity-proxy"),
        Model(id="perplexity-gemini-live-zephyr",  created=ts, owned_by="perplexity-proxy"),
        Model(id="perplexity-tts",                 created=ts, owned_by="perplexity-proxy"),
    ]
    # evitar duplicados
    existing_ids = {m.id for m in models}
    for m in proxy_models:
        if m.id not in existing_ids:
            models.append(m)

    return ModelList(data=models)
