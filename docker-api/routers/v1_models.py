"""
GET /v1/models — lista modelos disponibles.

Estrategia de dos capas:
  1. Modelos dinámicos de /rest/models/config/v2 (modelos premium/experimentales).
  2. Modelos estáticos del APK (p37.java) — siempre presentes aunque config/v2 falle.
     Incluye "turbo", que es el modelo default gratuito hardcodeado en la app
     y por eso NO aparece en config/v2.

Source APK: p37.java (lista estática) / rp2.java + ank.java (config/v2 dinámica).
"""
from __future__ import annotations

import time
from fastapi import APIRouter, Depends
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


# Modelos hardcodeados en el APK (p37.java) que config/v2 no devuelve.
# Formato: (model_api_name, provider)
_APK_STATIC_MODELS: list[tuple[str, str]] = [
    # ── Free tier ──────────────────────────────────────────────────
    ("turbo",                   "perplexity"),   # default gratuito — el que falta
    # ── Perplexity ─────────────────────────────────────────────────
    ("pplx_pro",                "perplexity"),   # Best (Pro)
    ("pplx_pro_upgraded",       "perplexity"),   # Best upgraded
    ("experimental",            "perplexity"),   # Sonar 2
    ("pplx_alpha",              "perplexity"),   # Research / DeepResearch
    ("pplx_beta",               "perplexity"),   # Labs / DeepSearch
    # ── OpenAI ─────────────────────────────────────────────────────
    ("gpt56_terra",             "openai"),
    ("gpt56_terra_thinking",    "openai"),
    ("gpt56_sol",               "openai"),
    ("gpt56_sol_thinking",      "openai"),
    # ── Google ─────────────────────────────────────────────────────
    ("gemini31pro_high",        "google"),       # Gemini 3.1 Pro (con reasoning)
    # ── Anthropic ──────────────────────────────────────────────────
    ("claude50sonnet",          "anthropic"),
    ("claude50sonnetthinking",  "anthropic"),
    ("claude50opus",            "anthropic"),
    ("claude50opusthinking",    "anthropic"),
    # ── ZAI / GLM ──────────────────────────────────────────────────
    ("glm_5_2",                 "zai"),
    # ── Kimi / MoonshotAI ──────────────────────────────────────────
    ("kimik3",                  "kimi"),
    ("kimik3thinking",          "kimi"),
    # ── xAI / Grok ─────────────────────────────────────────────────
    ("grok45low",               "xai"),
    ("grok45medium",            "xai"),
    # ── NVIDIA ─────────────────────────────────────────────────────
    ("nv_nemotron_3_ultra",     "nvidia"),
    # ── Computer Use (pplx_asi*) ───────────────────────────────────
    ("pplx_asi",                "perplexity"),
    ("pplx_asi_gpt54",          "openai"),
    ("pplx_asi_gpt_55",         "openai"),
    ("pplx_asi_gpt_56_sol",     "openai"),
    ("pplx_asi_fable_5",        "anthropic"),
    ("pplx_asi_opus",           "anthropic"),
    ("pplx_asi_opus_fast",      "anthropic"),
    ("pplx_asi_opus_thinking",  "anthropic"),
    ("pplx_asi_opus_4_7",       "anthropic"),
    ("pplx_asi_opus_4_8",       "anthropic"),
    ("pplx_asi_sonnet",         "anthropic"),
    ("pplx_asi_sonnet_thinking","anthropic"),
    ("pplx_asi_kimi",           "kimi"),
    ("pplx_asi_kimi_k3",        "kimi"),
    ("pplx_asi_qwen",           "perplexity"),
    ("pplx_asi_glm",            "perplexity"),
    ("pplx_asi_grok_45",        "xai"),
    ("pplx_asi_beta",           "perplexity"),
]

# Modelos propios del proxy (Gemini Live + TTS)
_PROXY_MODELS: list[tuple[str, str]] = [
    ("perplexity-gemini-live",         "perplexity-proxy"),
    ("perplexity-gemini-live-charon",  "perplexity-proxy"),
    ("perplexity-gemini-live-fenrir",  "perplexity-proxy"),
    ("perplexity-gemini-live-zephyr",  "perplexity-proxy"),
    ("perplexity-tts",                 "perplexity-proxy"),
]


@router.get(
    "/models",
    response_model=ModelList,
    summary="Listar modelos disponibles",
    description="""
Devuelve todos los modelos de Perplexity disponibles.

**Dos fuentes combinadas:**
1. `/rest/models/config/v2` — modelos dinámicos (~100+ modelos premium/experimentales).
2. `p37.java` (APK estática) — modelos siempre presentes, incluyendo **`turbo`**
   (el modelo default gratuito que NO aparece en config/v2).

**Modelos propios del proxy:** `perplexity-gemini-live*`, `perplexity-tts`.

**Source APK:** `p37.java` (static list) | `rp2.java` + `ank.java` (config/v2).
""",
)
async def list_models(_=Depends(require_api_key)):
    ts = int(time.time())
    models: list[Model] = []

    # 1. Modelos dinámicos de config/v2
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
                provider = (info.get("provider") or "perplexity").lower()
                models.append(Model(id=model_id, created=ts, owned_by=provider))
    except Exception:
        pass

    # 2. Modelos estáticos del APK (incluyendo turbo) — añadir solo si no vinieron en config/v2
    existing_ids = {m.id for m in models}
    for model_id, provider in _APK_STATIC_MODELS:
        if model_id not in existing_ids:
            models.append(Model(id=model_id, created=ts, owned_by=provider))
            existing_ids.add(model_id)

    # 3. Modelos propios del proxy
    for model_id, provider in _PROXY_MODELS:
        if model_id not in existing_ids:
            models.append(Model(id=model_id, created=ts, owned_by=provider))
            existing_ids.add(model_id)

    return ModelList(data=models)
