"""
Perplexity chat — sin login ni registro.
Protocolo: REST SSE nativo del app Android (ingeniería inversa del APK 2.95.0).

Requiere: pip install curl_cffi

Uso:
  python perplexity_chat.py                                    # chat interactivo
  python perplexity_chat.py "¿Qué es la IA?"                  # pregunta única
  python perplexity_chat.py "pregunta" scholar                 # foco de búsqueda
  python perplexity_chat.py "pregunta" internet concise turbo  # foco / modo / modelo

Parámetros de generación del servidor (no controlables desde el cliente):
  temperature, top_p, top_k, max_tokens — el servidor los ignora.
  El modelo se controla con 'model_preference' y el estilo con 'mode'.
"""

import sys
import json
import uuid
import re
import os
import getpass
from pathlib import Path
from typing import Generator, Optional

from curl_cffi import requests


BASE_URL   = "https://www.perplexity.ai"
API_URL    = f"{BASE_URL}/rest/sse/perplexity_ask?version=2.17&source=android"
TOKEN_FILE = Path.home() / ".perplexity_session"

# ── Modelos disponibles (extraídos de p37.java del APK) ──────────────────────
# Formato: "model_preference" que va en el body → funciona sin suscripción
MODELS = {
    # Perplexity propios
    "default":       "turbo",               # Perplexity (gratis, más rápido)
    "turbo":         "turbo",
    "best":          "pplx_pro",            # Best (Pro)
    "pplx_pro":      "pplx_pro",
    "experimental":  "experimental",        # Sonar 2
    "research":      "pplx_alpha",          # Deep Research
    "labs":          "pplx_beta",           # Labs (deeper research)
    # OpenAI
    "gpt56_sol":         "gpt56_sol",       # GPT-5.6 Sol (más nuevo)
    "gpt56_sol_think":   "gpt56_sol_thinking",
    "gpt56_terra":       "gpt56_terra",     # GPT-5.6 Terra
    "gpt56_terra_think": "gpt56_terra_thinking",
    # Anthropic
    "sonnet5":       "claude50sonnet",      # Claude Sonnet 5
    "sonnet5_think": "claude50sonnetthinking",
    "opus5":         "claude50opus",        # Claude Opus 5
    "opus5_think":   "claude50opusthinking",
    # Google
    "gemini31":      "gemini31pro_high",    # Gemini 3.1 Pro (reasoning)
    # xAI
    "grok45":        "grok45low",           # Grok 4.5
    "grok45_think":  "grok45medium",        # Grok 4.5 Thinking
    # Kimi / Moonshot
    "kimi3":         "kimik3",              # Kimi K3
    "kimi3_think":   "kimik3thinking",
    # Otros
    "glm52":         "glm_5_2",             # GLM-5.2 (ZAI)
    "nemotron":      "nv_nemotron_3_ultra", # Nvidia Nemotron 3 Ultra
}

# ── Fuentes de búsqueda ───────────────────────────────────────────────────────
SOURCES = {
    "internet": ["web"],
    "web":      ["web"],
    "scholar":  ["scholar"],    # papers académicos
    "youtube":  ["youtube"],
    "reddit":   ["reddit"],
    "writing":  [],             # sin búsqueda — modo escritura
}

MODES = ["concise", "copilot"]


def _build_session():
    s = requests.Session()
    s.impersonate = "chrome120"
    return s


def _build_params(query, device_id, last_backend_uuid, search_focus, mode,
                  model_key, sources_param, streaming):
    return {
        "source":                          "android",
        "version":                         "2.17",
        "frontend_uuid":                   str(uuid.uuid4()),
        "last_backend_uuid":               last_backend_uuid,
        "android_device_id":               device_id,
        "mode":                            mode,
        "is_related_query":                last_backend_uuid is not None,
        "is_voice_to_voice":               False,
        "timezone":                        "America/Bogota",
        "language":                        "es",
        "is_incognito":                    False,
        "use_schematized_api":             True,
        "send_back_text_in_streaming_api": streaming,
        "supported_block_use_cases":       ["ANSWER", "SOURCES", "IMAGE", "VIDEO"],
        "sources":                         sources_param,
        "model_preference":                model_key,
    }


# ── Headers del app Android (lk0.java + wh2.java del APK) ───────────────────
def _android_headers(device_id: str) -> dict:
    return {
        "X-App-Version":    "2.95.0",
        "X-Client-Version": "2.95.0",
        "X-Client-Name":    "Perplexity-Android",
        "X-Client-Env":     "production",
        "X-App-ApiClient":  "android",
        "X-App-ApiVersion": "2.17",
        "X-Device-ID":      device_id,
        "Accept-Language":  "es-ES",
        "Content-Type":     "application/json",
        "Accept":           "text/event-stream",
        "User-Agent":       "okhttp/4.12.0",
    }


def _parse_sse(raw: bytes):
    """Parsea stream SSE → lista de {type, data}."""
    text = raw.decode("utf-8", errors="replace")
    events = []
    for chunk in re.split(r"\r\n\r\n", text):
        ev = {}
        for line in chunk.split("\r\n"):
            if line.startswith("event:"):
                ev["type"] = line[6:].strip()
            elif line.startswith("data:"):
                ev["data"] = line[5:].strip()
        if "type" in ev and "data" in ev:
            events.append(ev)
    return events


def _extract_result(final_data, model_key, search_focus):
    """Extrae texto y fuentes del evento SSE final."""
    text = ""
    sources = []
    for block in final_data.get("blocks", []):
        usage = block.get("intended_usage", "")
        if "markdown_block" in block:
            mb = block["markdown_block"]
            block_text = "".join(c for c in mb.get("chunks", []) if isinstance(c, str))
            if "ask_text_0_markdown" in usage and block_text:
                text = block_text
            elif not text and block_text and "ask_text" in usage:
                text = block_text
        elif "web_result_block" in block:
            for src in block["web_result_block"].get("web_results", []):
                url = src.get("url", "")
                if url and not any(s["url"] == url for s in sources):
                    sources.append({
                        "name":    src.get("name", url),
                        "url":     url,
                        "snippet": src.get("snippet", ""),
                    })
    return {
        "text":          text.strip(),
        "sources":       sources,
        "backend_uuid":  final_data.get("backend_uuid", ""),
        "display_model": final_data.get("display_model", model_key),
        "search_focus":  final_data.get("search_focus", search_focus),
    }


def chat(
    query: str,
    device_id: str = None,
    last_backend_uuid: str = None,
    search_focus: str = "internet",
    mode: str = "concise",
    model: str = "turbo",
) -> dict:
    """
    Envía una pregunta a Perplexity y devuelve la respuesta completa.

    Returns:
        {
          "text":          str,   # respuesta en markdown
          "sources":       list,  # [{name, url, snippet}, ...]
          "backend_uuid":  str,   # pasar como last_backend_uuid al siguiente turno
          "device_id":     str,   # mantener para follow-ups
          "display_model": str,
          "search_focus":  str,
        }
    """
    if device_id is None:
        device_id = str(uuid.uuid4())
    model_key = MODELS.get(model, model)
    sources_param = SOURCES.get(search_focus, ["web"])

    session = _build_session()
    body = {
        "query_str": query,
        "params": _build_params(query, device_id, last_backend_uuid,
                                search_focus, mode, model_key, sources_param,
                                streaming=False),
    }
    r = session.post(API_URL, json=body,
                     headers=_android_headers(device_id), timeout=120)
    r.raise_for_status()

    raw = r.content
    if not raw[:6] == b"event:":
        try:
            err = json.loads(raw)
            raise RuntimeError(err.get("text") or err.get("error_message") or str(err))
        except json.JSONDecodeError:
            raise RuntimeError(f"Respuesta inesperada: {raw[:200]}")

    final_data = None
    for ev in _parse_sse(raw):
        if ev["type"] == "message":
            try:
                d = json.loads(ev["data"])
                if d.get("final_sse_message"):
                    final_data = d
                    break
            except json.JSONDecodeError:
                pass

    if final_data is None:
        raise RuntimeError("No se recibió mensaje final del servidor.")

    result = _extract_result(final_data, model_key, search_focus)
    result["device_id"] = device_id
    return result


def stream_chat(
    query: str,
    device_id: str = None,
    last_backend_uuid: str = None,
    search_focus: str = "internet",
    mode: str = "concise",
    model: str = "turbo",
) -> Generator:
    """
    Versión streaming: yield de chunks de texto a medida que llegan del servidor.
    El último elemento yielded es el dict de resultado completo (con sources, etc).

    Ejemplo:
        for item in stream_chat("¿Qué es Python?"):
            if isinstance(item, str):
                print(item, end="", flush=True)  # chunk incremental
            else:
                print()  # item es el dict final con sources
    """
    if device_id is None:
        device_id = str(uuid.uuid4())
    model_key = MODELS.get(model, model)
    sources_param = SOURCES.get(search_focus, ["web"])

    session = _build_session()
    body = {
        "query_str": query,
        "params": _build_params(query, device_id, last_backend_uuid,
                                search_focus, mode, model_key, sources_param,
                                streaming=True),
    }
    r = session.post(API_URL, json=body,
                     headers=_android_headers(device_id), timeout=120,
                     stream=True)
    r.raise_for_status()

    buf = ""
    final_data = None

    for chunk in r.iter_content(chunk_size=None):
        if not chunk:
            continue
        buf += chunk.decode("utf-8", errors="replace")
        # Procesar eventos completos (terminados en \r\n\r\n)
        while "\r\n\r\n" in buf:
            seg, buf = buf.split("\r\n\r\n", 1)
            ev_type, data_str = None, ""
            for line in seg.split("\r\n"):
                if line.startswith("event:"):
                    ev_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_str = line[5:].strip()
            if ev_type != "message" or not data_str:
                continue
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if data.get("final_sse_message"):
                final_data = data
                continue

            # Chunks de texto incremental — cada evento trae solo el delta nuevo.
            # Usamos ask_text_0_markdown y saltamos ask_text para evitar duplicados.
            for block in data.get("blocks", []):
                usage = block.get("intended_usage", "")
                if "ask_text_0_markdown" in usage and "markdown_block" in block:
                    delta = "".join(
                        c for c in block["markdown_block"].get("chunks", [])
                        if isinstance(c, str)
                    )
                    if delta:
                        yield delta

    if final_data is None:
        raise RuntimeError("No se recibió mensaje final del servidor.")

    result = _extract_result(final_data, model_key, search_focus)
    result["device_id"] = device_id
    yield result


def chat_interactive(
    search_focus: str = "internet",
    mode: str = "concise",
    model: str = "turbo",
) -> None:
    """Sesión de chat interactiva con memoria de hilo."""
    device_id         = str(uuid.uuid4())
    last_backend_uuid = None

    model_key = MODELS.get(model, model)
    print(f"\n{'='*62}")
    print("  Perplexity — protocolo Android nativo (sin login)")
    print(f"  Modelo: {model_key}  |  Foco: {search_focus}  |  Modo: {mode}")
    print("  Comandos: 'nuevo'=nuevo hilo  'modelos'=ver lista  'salir'=salir")
    print(f"{'='*62}\n")

    while True:
        try:
            query = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo...")
            break

        if not query:
            continue

        if query.lower() in ("salir", "exit", "quit"):
            print("¡Hasta luego!")
            break

        if query.lower() in ("nuevo", "new"):
            device_id         = str(uuid.uuid4())
            last_backend_uuid = None
            print(f"[Nuevo hilo: {device_id[:8]}...]\n")
            continue

        if query.lower() == "modelos":
            print("\nModelos disponibles:")
            for alias, key in MODELS.items():
                print(f"  {alias:<20} → {key}")
            print()
            continue

        print("\nPerplexity: ", end="", flush=True)
        try:
            result = None
            for item in stream_chat(
                query,
                device_id=device_id,
                last_backend_uuid=last_backend_uuid,
                search_focus=search_focus,
                mode=mode,
                model=model,
            ):
                if isinstance(item, str):
                    print(item, end="", flush=True)
                else:
                    result = item
            if result is None:
                print("(Sin respuesta)")
            else:
                last_backend_uuid = result["backend_uuid"]
                if not result["text"]:
                    print("(Sin respuesta)")
                if result["sources"]:
                    print(f"\n\nFuentes [{result['search_focus']}] — {result['display_model']}:")
                    for i, s in enumerate(result["sources"][:5], 1):
                        print(f"  [{i}] {s['name']}")
                        print(f"      {s['url']}")

        except Exception as e:
            print(f"[Error: {e}]")

        print()


def chat_once(
    query: str,
    search_focus: str = "internet",
    mode: str = "concise",
    model: str = "turbo",
) -> None:
    result = None
    first_chunk = True
    for item in stream_chat(query, search_focus=search_focus, mode=mode, model=model):
        if isinstance(item, str):
            if first_chunk:
                print(f"[{MODELS.get(model, model)}]")
                print("-" * 60)
                first_chunk = False
            print(item, end="", flush=True)
        else:
            result = item
    print()
    if result and result["sources"]:
        print(f"\nFuentes ({len(result['sources'])}) [{result['search_focus']}]:")
        for i, s in enumerate(result["sources"][:5], 1):
            print(f"  [{i}] {s['name']}")
            print(f"      {s['url']}")


# ── Autenticación (email + OTP) ───────────────────────────────────────────────

def _load_token() -> Optional[str]:
    """Lee el session token guardado en disco."""
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip() or None
    return None


def _save_token(token: str) -> None:
    TOKEN_FILE.write_text(token)
    TOKEN_FILE.chmod(0o600)


def login(email: str, otp: Optional[str] = None) -> str:
    """
    Autenticación en dos pasos (email + OTP numérico).

    Paso 1 — envía el OTP al email.
    Paso 2 — canjea el OTP por un session token y lo guarda en ~/.perplexity_session.

    Args:
        email: dirección de correo de la cuenta Perplexity.
        otp:   código de 6 dígitos. Si es None, lo solicita por consola.

    Returns:
        El session token (nextauth_session_token).
    """
    s = _build_session()
    headers = {
        "Content-Type":     "application/json",
        "X-App-Version":    "2.95.0",
        "X-Client-Version": "2.95.0",
        "X-Client-Name":    "Perplexity-Android",
        "X-Client-Env":     "production",
        "User-Agent":       "okhttp/4.12.0",
    }

    # Paso 1: solicitar OTP
    r1 = s.post(
        f"{BASE_URL}/api/auth/signin-email",
        json={"email": email, "useNumericOtp": True},
        headers=headers,
        timeout=15,
    )
    r1.raise_for_status()

    if otp is None:
        otp = getpass.getpass(f"OTP enviado a {email}: ").strip()

    # Paso 2: canjear OTP → perplexity JWT
    r2 = s.post(
        f"{BASE_URL}/api/auth/signin-otp",
        json={"email": email, "otp": otp},
        headers=headers,
        timeout=15,
    )
    r2.raise_for_status()
    pplx_jwt = r2.json().get("token") or r2.json().get("access_token")
    if not pplx_jwt:
        raise RuntimeError(f"No se obtuvo JWT: {r2.text}")

    # Paso 3: intercambiar JWT → session token de NextAuth
    r3 = s.post(
        f"{BASE_URL}/rest/auth/exchange_perplexity_jwt",
        json={"perplexity_jwt": pplx_jwt},
        headers=headers,
        timeout=15,
    )
    r3.raise_for_status()
    session_token = r3.json().get("token")
    if not session_token:
        raise RuntimeError(f"No se obtuvo session token: {r3.text}")

    _save_token(session_token)
    return session_token


def _authed_headers(device_id: Optional[str] = None) -> dict:
    """Headers con cookie de sesión autenticada. Requiere token guardado."""
    token = _load_token()
    if not token:
        raise RuntimeError("No hay sesión activa. Usa login() primero.")
    did = device_id or str(uuid.uuid4())
    return {
        **_android_headers(did),
        "Cookie": (
            f"__Secure-next-auth.session-token={token}; "
            f"next-auth.session-token={token}"
        ),
    }


# ── Utilidades de datos en tiempo real ───────────────────────────────────────

def _api_session():
    s = requests.Session()
    s.impersonate = "chrome120"
    return s

def _api_headers(device_id=None):
    return {
        "X-App-Version":    "2.95.0",
        "X-Client-Version": "2.95.0",
        "X-Client-Name":    "Perplexity-Android",
        "X-Client-Env":     "production",
        "X-App-ApiClient":  "android",
        "X-App-ApiVersion": "2.17",
        "X-Device-ID":      device_id or str(uuid.uuid4()),
        "Accept":           "application/json",
        "User-Agent":       "okhttp/4.12.0",
    }


def get_quote(symbol: str) -> dict:
    """
    Cotización en tiempo real de acciones o criptomonedas.

    Símbolos:
      Acciones:  AAPL, TSLA, GOOGL, NVDA, MSFT, META, SPY, QQQ ...
      Crypto:    BTCUSD, ETHUSD, SOLUSD (sin guión)
      Forex:     via get_exchange_rate()

    Returns:
        {
          "symbol", "name", "price", "change", "percentChange",
          "marketCap", "volume", "high", "low",
          "isCrypto", "isMarketOpen", "afterHoursPrice",
          "exchangeTimezone", "currency",
          ...  (todos los campos del servidor)
        }
    """
    s = _api_session()
    r = s.get(
        f"{BASE_URL}/rest/finance/quote/{symbol}",
        headers=_api_headers(),
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def get_exchange_rate(
    source: str,
    target: str,
    amount: float = 1.0,
    with_ui_hints: bool = False,
) -> dict:
    """
    Tipo de cambio entre monedas.

    Args:
        source:        Código ISO de la moneda origen (ej: "USD", "EUR", "COP")
        target:        Código ISO de la moneda destino
        amount:        Monto a convertir (default 1.0)
        with_ui_hints: Incluir metadatos de UI (banderas, nombre completo)

    Returns:
        {
          "symbol", "exchange_rate", "converted_amount",
          "source_currency", "target_currency",
          "source_currency_name", "target_currency_name",
          "source_currency_symbol", "target_currency_symbol",
          "type",  ...
        }
    """
    s = _api_session()
    r = s.get(
        f"{BASE_URL}/rest/finance/currency-exchange",
        headers=_api_headers(),
        params={
            "source_currency": source,
            "target_currency": target,
            "amount":          amount,
            "with_ui_hints":   str(with_ui_hints).lower(),
        },
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def get_rate_limit_status() -> dict:
    """
    Estado del rate limit para la sesión actual (sin login).

    Returns:
        {
          "free_queries": {"available": bool, ...},
          "modes": {
            "pro_search": {"available": bool, "remaining_detail": {...}},
            "research":   ...,
            "labs":       ...,
            "agentic_research": ...,
          },
          "sources": {
            "scholar": ..., "web": ..., "gcal": ..., "github_mcp_direct": ...,
            ...  (todos los sources premium)
          }
        }
    """
    s = _api_session()
    r = s.get(
        f"{BASE_URL}/rest/rate-limit/status",
        headers=_api_headers(),
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def get_user_settings() -> dict:
    """
    Límites y configuración de la cuenta anónima.

    Returns dict con: pages_limit, upload_limit, daily_attachment_limit,
    default_model, subscription_tier, connector_limits, etc.
    """
    s = _api_session()
    r = s.get(
        f"{BASE_URL}/rest/user/settings",
        headers=_api_headers(),
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def save_profile(
    bio: Optional[str] = None,
    occupation: Optional[str] = None,
    response_language: Optional[str] = None,
    use_memory: Optional[bool] = None,
    use_search_history: Optional[bool] = None,
    answer_styling_preferences: Optional[dict] = None,
) -> dict:
    """
    Actualiza el perfil de IA del usuario autenticado.

    Requiere sesión activa (ejecuta login() antes).
    Solo los campos no-None se envían al servidor.

    Campos válidos (de RemoteUpdateUserAiProfileRequest / vtl.java):
      bio                        str   — descripción personal
      occupation                 str   — profesión
      response_language          str   — idioma de respuesta (ej: "es", "en")
      use_memory                 bool  — activar memoria de conversaciones
      use_search_history         bool  — activar historial de búsqueda
      answer_styling_preferences dict  — preferencias de estilo de respuesta

    Returns:
        La respuesta del servidor (normalmente {"status": "ok"} o el perfil).
    """
    profile = {}
    if bio is not None:
        profile["bio"] = bio
    if occupation is not None:
        profile["occupation"] = occupation
    if response_language is not None:
        profile["response_language"] = response_language
    if use_memory is not None:
        profile["use_memory"] = use_memory
    if use_search_history is not None:
        profile["use_search_history"] = use_search_history
    if answer_styling_preferences is not None:
        profile["answer_styling_preferences"] = answer_styling_preferences

    if not profile:
        raise ValueError("Debe especificar al menos un campo del perfil.")

    s = _build_session()
    r = s.post(
        f"{BASE_URL}/rest/user/save_user_ai_profile",
        json={"action": "save_profile", "updated_profile": profile},
        headers=_authed_headers(),
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


# ── Text-to-Speech ────────────────────────────────────────────────────────────

TTS_URL     = f"{BASE_URL}/rest/sse/audio/text_to_speech"
SUGGEST_URL = "https://suggest.perplexity.ai/suggest"

# Voces disponibles (j0g.java del APK). Preset = nombre que acepta el servidor.
TTS_VOICES = {
    "kyrin": "Kyrin-mp3",   # cedar   — voz por defecto
    "velox": "Velox-mp3",   # marin
    "tylis": "Tylis-mp3",   # alloy
    "torma": "Torma-mp3",   # ash
    "mylva": "Mylva-mp3",   # ballad
    "syla":  "Syla-mp3",    # coral
    "gravo": "Gravo-mp3",   # echo
    "solva": "Solva-mp3",   # sage
}


def tts(
    text: str,
    voice: str = "kyrin",
    output_path: Optional[str] = None,
) -> bytes:
    """
    Convierte texto a voz usando el endpoint SSE de Perplexity.
    No requiere autenticación.

    Args:
        text:        Texto a sintetizar (cualquier idioma).
        voice:       Nombre de voz: kyrin (def), velox, tylis, torma,
                     mylva, syla, gravo, solva.
        output_path: Si se provee, guarda el MP3 en esa ruta.

    Returns:
        Bytes del audio MP3.
    """
    preset = TTS_VOICES.get(voice.lower())
    if preset is None:
        # Aceptar también el formato "Kyrin-mp3" directamente
        preset = voice if voice.endswith("-mp3") else f"{voice.capitalize()}-mp3"

    device_id = str(uuid.uuid4())
    s = _build_session()
    body = {
        "uuid":       str(uuid.uuid4()),
        "text":       text,
        "preset":     preset,
        "source":     "assistant",
        "visitor_id": device_id,
    }
    hdrs = _android_headers(device_id)
    hdrs["Accept"] = "text/event-stream"

    r = s.post(TTS_URL, json=body, headers=hdrs, timeout=60, stream=True)
    r.raise_for_status()

    import base64
    parts = []
    buf = ""
    for chunk in r.iter_content(chunk_size=None):
        buf += chunk.decode("utf-8", errors="replace")
        while "\r\n\r\n" in buf:
            seg, buf = buf.split("\r\n\r\n", 1)
            data_str = ""
            for line in seg.split("\r\n"):
                if line.startswith("data:"):
                    data_str = line[5:].strip()
            if not data_str:
                continue
            try:
                d = json.loads(data_str)
                if "data" in d and d["data"]:
                    # Cada chunk es base64 independiente → decodificar por separado
                    parts.append(base64.b64decode(d["data"] + "=="))
                elif "error_code" in d:
                    raise RuntimeError(d.get("message") or d["error_code"])
            except (json.JSONDecodeError, KeyError):
                pass

    if not parts:
        raise RuntimeError("No se recibió audio del servidor.")

    audio = b"".join(parts)

    if output_path:
        with open(output_path, "wb") as f:
            f.write(audio)

    return audio


def suggest(query: str, limit: int = 5) -> list:
    """
    Autocompletado de queries (suggest.perplexity.ai).
    No requiere autenticación.

    Args:
        query: texto parcial a completar.
        limit: máximo de sugerencias a devolver.

    Returns:
        Lista de strings con las queries sugeridas.
    """
    s = _build_session()
    r = s.get(
        SUGGEST_URL,
        params={"q": query, "version": "2.17", "source": "mobile"},
        headers=_android_headers(str(uuid.uuid4())),
        timeout=10,
    )
    r.raise_for_status()
    items = r.json().get("suggests", [])
    return [item["text"] for item in items[:limit] if "text" in item]


def search_images(
    query: str,
    limit: int = 20,
    device_id: Optional[str] = None,
) -> list:
    """
    Busca imágenes para una consulta usando el motor de Perplexity.

    Args:
        query: Consulta de búsqueda (ej: "gatos persas").
        limit: Máximo de imágenes a devolver (por defecto 20).

    Returns:
        Lista de dicts con campos: image, thumbnail, name, url, caption,
        image_width, image_height, original_image_url.
    """
    if device_id is None:
        device_id = str(uuid.uuid4())

    session = _build_session()
    body = {
        "query_str": query,
        "params": _build_params(
            query, device_id, None, "internet", "concise", "turbo", ["web"],
            streaming=False,
        ),
    }
    r = session.post(API_URL, json=body,
                     headers=_android_headers(device_id), timeout=60)
    r.raise_for_status()

    media = []
    seen = set()
    for ev in _parse_sse(r.content):
        if ev["type"] == "message":
            try:
                d = json.loads(ev["data"])
                for item in d.get("media_items", []):
                    if item.get("medium") == "image":
                        url = item.get("image") or item.get("original_image_url", "")
                        if url and url not in seen:
                            seen.add(url)
                            media.append(item)
            except (json.JSONDecodeError, KeyError):
                pass

    return media[:limit]


def search_videos(
    query: str,
    limit: int = 20,
    device_id: Optional[str] = None,
) -> list:
    """
    Busca videos para una consulta usando el motor de Perplexity.

    Args:
        query: Consulta de búsqueda (ej: "tutorial Python").
        limit: Máximo de videos a devolver (por defecto 20).

    Returns:
        Lista de dicts con campos: image (thumbnail), name, url, caption,
        thumbnail, original_image_url.
    """
    if device_id is None:
        device_id = str(uuid.uuid4())

    session = _build_session()
    body = {
        "query_str": query,
        "params": _build_params(
            query, device_id, None, "internet", "concise", "turbo", ["web"],
            streaming=False,
        ),
    }
    r = session.post(API_URL, json=body,
                     headers=_android_headers(device_id), timeout=60)
    r.raise_for_status()

    media = []
    seen = set()
    for ev in _parse_sse(r.content):
        if ev["type"] == "message":
            try:
                d = json.loads(ev["data"])
                for item in d.get("media_items", []):
                    if item.get("medium") == "video":
                        url = item.get("url", "")
                        if url and url not in seen:
                            seen.add(url)
                            media.append(item)
            except (json.JSONDecodeError, KeyError):
                pass

    return media[:limit]


def get_more_images(
    backend_uuid: str,
    read_write_token: str,
    page: int = 1,
    limit: int = 20,
) -> list:
    """
    Carga más imágenes para un hilo existente (paginación).

    Requiere el backend_uuid y read_write_token de un turno previo.

    Returns:
        Lista de dicts con los mismos campos que search_images().
    """
    s = _build_session()
    r = s.post(
        f"{BASE_URL}/rest/media/search-images-for-entry",
        json={
            "entry_uuid":      backend_uuid,
            "read_write_token": read_write_token,
            "page":            page,
            "limit":           limit,
        },
        headers=_authed_headers(),
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("media_items", [])


DISCOVER_URL    = f"{BASE_URL}/rest/discover"
DIGEST_GQL = {
    "home":     f"{BASE_URL}/api/graphql/MobileDigestHomeV2RelayQuery/ab8f0a8d45dc3d3091972ea30fae064edb095504a7f7df21d8982c0e7624939f",
    "by_date":  f"{BASE_URL}/api/graphql/MobileDigestArtifactFeedByDateV2RelayQuery/f1e02a071d8178d85570cda7d25dcde7814701a06d3f3da5abdb5326b467fdf9",
    "timeline": f"{BASE_URL}/api/graphql/MobileDigestArtifactTimelineV2RelayQuery/876c233a4f928a4ce6ad31952ed56414475f7bd9921cf6e4349cb99987f7927c",
    "detail":   f"{BASE_URL}/api/graphql/MobileDigestArtifactDetailV2RelayQuery/32b4a3812c144a7595685e8b5c9d511e7e2eb1f0d65a61c2d17769dbbaa195e6",
    "mark_read":   f"{BASE_URL}/api/graphql/MobileDigestMarkArtifactRead/568d677568dd50c1db03a13213b605f529e61888ea655cde1aaf7344c9fa8fa5",
    "mark_unread": f"{BASE_URL}/api/graphql/MobileDigestMarkArtifactUnread/95edcf2b60288e38821a00147c8483914e046e109b5a500d645777425192f87d",
}
DISCOVER_TOPICS = {
    "tech":          "9be812cb-6120-41b7-bc9e-993200db6cfc",
    "tecnologia":    "9be812cb-6120-41b7-bc9e-993200db6cfc",
    "business":      "e46915e3-9d25-4f85-9e43-e8a9d834729f",
    "negocios":      "e46915e3-9d25-4f85-9e43-e8a9d834729f",
    "arts":          "f00846db-e258-4455-9e9f-74b12cfbf06d",
    "culture":       "f00846db-e258-4455-9e9f-74b12cfbf06d",
    "arte":          "f00846db-e258-4455-9e9f-74b12cfbf06d",
    "sports":        "c69c15fd-559e-4714-b82c-481a89b82a94",
    "deportes":      "c69c15fd-559e-4714-b82c-481a89b82a94",
    "entertainment": "0130fdd1-4017-43ac-91ae-59ace446130f",
    "entretenimiento": "0130fdd1-4017-43ac-91ae-59ace446130f",
}


def get_news_topics() -> list:
    """
    Devuelve los temas disponibles en el feed de Discover.
    No requiere autenticación.

    Returns:
        Lista de dicts: [{id, name, translated_name, icon}, ...]
    """
    s = _build_session()
    r = s.get(
        f"{DISCOVER_URL}/topics",
        params={"version": "2.17"},
        headers=_android_headers(str(uuid.uuid4())),
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("all_topics", [])


def news(
    topics: Optional[list] = None,
    limit: int = 10,
    next_token: Optional[str] = None,
) -> dict:
    """
    Obtiene el feed de noticias/Discover de Perplexity.
    No requiere autenticación.

    Args:
        topics: Lista de IDs o nombres de temas (tech, business, sports, arts,
                entertainment). Si es None devuelve todos.
        limit:  Número de artículos a devolver (por defecto 10).
        next_token: Token para paginación (viene de una respuesta previa).

    Returns:
        {
          "items":       list de artículos,
          "next_token":  str (para la siguiente página),
          "feed_uuid":   str,
        }

    Cada artículo tiene: title, summary, url, domain_name, published_timestamp,
    featured_images, sources, context_uuid, audio_url, is_live.
    """
    # Resolver nombres de temas a IDs
    topic_ids = []
    if topics:
        for t in topics:
            key = t.lower().strip()
            if key in DISCOVER_TOPICS:
                topic_ids.append(DISCOVER_TOPICS[key])
            else:
                topic_ids.append(t)  # Asumir que ya es un ID

    s = _build_session()
    params = [("offset", 0), ("limit", limit), ("version", "2.17")]
    for tid in topic_ids:
        params.append(("topic", tid))
    if next_token:
        params.append(("next_token", next_token))

    r = s.get(
        f"{DISCOVER_URL}/feed",
        params=params,
        headers=_android_headers(str(uuid.uuid4())),
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    return {
        "items":      data.get("items", []),
        "next_token": data.get("next_token"),
        "feed_uuid":  data.get("feed_uuid"),
    }


def get_article(slug: str) -> dict:
    """
    Texto completo + fuentes de un artículo de Discover.
    No requiere autenticación.

    Args:
        slug: slug del artículo (del campo 'slug' del feed) o URL completa
              https://www.perplexity.ai/page/<slug>

    Returns:
        {
          "title":        str,
          "query":        str,
          "text":         str,   # respuesta markdown completa
          "sources":      list,  # [{name, url, snippet}, ...]
          "backend_uuid": str,
          "context_uuid": str,
          "related_queries": list,
        }
    """
    # Aceptar URL completa o solo el slug
    slug = slug.replace(f"{BASE_URL}/page/", "").strip("/")

    s = _build_session()
    r = s.get(
        f"{BASE_URL}/p/api/v1/article/{slug}",
        headers=_android_headers(str(uuid.uuid4())),
        timeout=20,
    )
    r.raise_for_status()

    entries = r.json().get("entries", [])
    if not entries:
        raise RuntimeError("Artículo no encontrado o sin contenido.")
    e = entries[0]

    # Extraer texto de blocks (mismo formato que SSE normal)
    text = ""
    sources = []
    for block in e.get("blocks", []):
        usage = block.get("intended_usage", "")
        if "markdown_block" in block:
            mb = block["markdown_block"]
            bt = "".join(c for c in mb.get("chunks", []) if isinstance(c, str))
            if "ask_text" in usage and bt:
                text = text + bt if text else bt
        elif "web_result_block" in block:
            for src in block["web_result_block"].get("web_results", []):
                url = src.get("url", "")
                if url and not any(s2["url"] == url for s2 in sources):
                    sources.append({
                        "name":    src.get("name", url),
                        "url":     url,
                        "snippet": src.get("snippet", ""),
                    })

    return {
        "title":           e.get("thread_title") or e.get("query_str", ""),
        "query":           e.get("query_str", ""),
        "text":            text.strip(),
        "sources":         sources,
        "backend_uuid":    e.get("backend_uuid", ""),
        "context_uuid":    e.get("context_uuid", ""),
        "related_queries": e.get("related_queries", []),
    }


def list_threads(
    search_term: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    show_archived: bool = False,
) -> list:
    """
    Lista o busca los hilos de chat del usuario.
    Requiere autenticación.

    Args:
        search_term: Texto a buscar en los títulos de los hilos.
        limit:       Máximo de resultados (por defecto 20).
        offset:      Para paginación.
        show_archived: Incluir hilos archivados.

    Returns:
        Lista de dicts: [{context_uuid, title, query_str, mode,
                          last_query_datetime, search_focus}, ...]
    """
    body = {
        "ascending":     False,
        "limit":         limit,
        "offset":        offset,
        "show_archived": show_archived,
    }
    if search_term:
        body["search_term"] = search_term

    s = _build_session()
    r = s.post(
        f"{BASE_URL}/rest/thread/list_ask_threads",
        params={"source": "android", "version": "2.17", "include_entity_relations": False},
        json=body,
        headers=_authed_headers(),
        timeout=20,
    )
    r.raise_for_status()
    threads = r.json()
    if not isinstance(threads, list):
        threads = threads.get("threads", [])
    return threads


def like_article(context_uuid: str, undo: bool = False) -> None:
    """
    Da like (o lo deshace) a un artículo del feed de Discover.
    Requiere autenticación.

    Args:
        context_uuid: campo 'context_uuid' del artículo (del feed de news).
        undo:         True para quitar el like (DELETE en vez de PUT).
    """
    s = _build_session()
    url = f"{DISCOVER_URL}/feed-items/{context_uuid}/feedback/like"
    h = _authed_headers()
    r = s.delete(url, params={"version": "2.17"}, headers=h, timeout=15) if undo \
        else s.put(url, params={"version": "2.17"}, headers=h, timeout=15)
    r.raise_for_status()


def dislike_article(context_uuid: str, undo: bool = False) -> None:
    """
    Da dislike (o lo deshace) a un artículo del feed de Discover.
    Requiere autenticación.

    Args:
        context_uuid: campo 'context_uuid' del artículo.
        undo:         True para quitar el dislike.
    """
    s = _build_session()
    url = f"{DISCOVER_URL}/feed-items/{context_uuid}/feedback/dislike"
    h = _authed_headers()
    r = s.delete(url, params={"version": "2.17"}, headers=h, timeout=15) if undo \
        else s.put(url, params={"version": "2.17"}, headers=h, timeout=15)
    r.raise_for_status()


def get_digest(
    date: Optional[str] = None,
    timeline_first: int = 10,
) -> dict:
    """
    Obtiene el Digest diario personalizado de Perplexity.
    Requiere autenticación + fuentes propias conectadas (email/calendar).
    Si no hay fuentes configuradas devuelve listas vacías.

    Args:
        date:           Fecha en formato YYYY-MM-DD. Por defecto hoy.
        timeline_first: Máximo de artículos en el timeline.

    Returns:
        {
          "timeline":      list,  # artículos recientes
          "feed_by_date":  list,  # artículos del día especificado
          "date":          str,
        }
    """
    import datetime
    if date is None:
        date = datetime.date.today().isoformat()

    s = _build_session()
    r = s.post(
        DIGEST_GQL["home"],
        json={"date": date, "includeDateFeed": True, "timelineFirst": timeline_first},
        headers={**_authed_headers(), "Content-Type": "application/json"},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()

    scope = (data.get("viewer") or {}).get("knowledgeContext", {}).get("scopeByKind", {})
    timeline_edges = scope.get("artifactTimeline", {}).get("edges", [])
    feed_edges     = scope.get("artifactFeedByDate", {}).get("artifactsConnection", {}).get("edges", [])

    def _extract_edges(edges):
        items = []
        for edge in edges:
            node = edge.get("node") or edge
            items.append(node)
        return items

    return {
        "date":         date,
        "timeline":     _extract_edges(timeline_edges),
        "feed_by_date": _extract_edges(feed_edges),
    }


def get_related_queries(backend_uuid: str) -> list:
    """
    Preguntas relacionadas sugeridas para un hilo de conversación.

    Requiere sesión activa y el backend_uuid del último turno.

    Returns:
        Lista de strings con las preguntas sugeridas.
    """
    s = _build_session()
    r = s.get(
        f"{BASE_URL}/rest/sse/related-queries/{backend_uuid}",
        headers=_authed_headers(),
        timeout=15,
    )
    r.raise_for_status()
    # El endpoint devuelve formato SSE (event: message\ndata: {...})
    text = r.text
    for line in text.splitlines():
        if line.startswith("data:"):
            try:
                data = json.loads(line[5:].strip())
                queries = data.get("related_queries", [])
                if queries:
                    return queries
            except json.JSONDecodeError:
                pass
    return []


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        # Comandos de datos financieros
        if cmd == "quote" and len(sys.argv) > 2:
            q = get_quote(sys.argv[2])
            chg = q.get("change") or 0
            pct = q.get("changesPercentage") or q.get("percentChange24h") or 0
            sign = "+" if chg >= 0 else ""
            cur = q.get("currency") or "$"
            print(f"{q.get('name')} ({q.get('symbol')})")
            print(f"  Precio:  {cur} {q.get('price')}")
            print(f"  Cambio:  {sign}{chg} ({sign}{pct:.2f}%)")
            if q.get("marketCap"):
                print(f"  Mkt Cap: ${q.get('marketCap'):,.0f}")
            if not q.get("isMarketOpen") and q.get("afterHoursPrice"):
                print(f"  After-h: {cur} {q.get('afterHoursPrice')}")
        elif cmd == "fx" and len(sys.argv) > 3:
            src, tgt = sys.argv[2].upper(), sys.argv[3].upper()
            amt = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
            r = get_exchange_rate(src, tgt, amt)
            print(f"{amt} {src} = {r.get('converted_amount')} {tgt}")
            print(f"  Tasa: 1 {src} = {r.get('exchange_rate')} {tgt}")
        elif cmd == "limits":
            s = get_rate_limit_status()
            fq = s["free_queries"]
            print(f"Free queries: {'✓' if fq['available'] else '✗'}")
            print("Modos premium:")
            for mode, info in s.get("modes", {}).items():
                rem = info.get("remaining_detail", {}).get("remaining", "?")
                avail = "✓" if info["available"] else f"✗ (remaining: {rem})"
                print(f"  {mode:<20} {avail}")
        elif cmd == "login" and len(sys.argv) > 2:
            # python perplexity_chat.py login email@example.com [otp]
            email = sys.argv[2]
            otp   = sys.argv[3] if len(sys.argv) > 3 else None
            try:
                tok = login(email, otp)
                print(f"✓ Sesión iniciada. Token guardado en {TOKEN_FILE}")
                print(f"  {tok[:40]}...")
            except Exception as e:
                print(f"✗ Error: {e}")
        elif cmd == "profile":
            # python perplexity_chat.py profile bio "..." occupation "..." lang es
            # Argumentos: pares clave valor
            args = sys.argv[2:]
            kwargs: dict = {}
            it = iter(args)
            for k in it:
                v = next(it, None)
                if v is None:
                    break
                if k == "bio":
                    kwargs["bio"] = v
                elif k in ("occupation", "job"):
                    kwargs["occupation"] = v
                elif k in ("lang", "language", "response_language"):
                    kwargs["response_language"] = v
                elif k == "memory":
                    kwargs["use_memory"] = v.lower() in ("1", "true", "yes", "sí")
                elif k == "history":
                    kwargs["use_search_history"] = v.lower() in ("1", "true", "yes", "sí")
            if not kwargs:
                print("Uso: profile bio '...' occupation '...' lang es memory true")
            else:
                try:
                    result = save_profile(**kwargs)
                    print(f"✓ Perfil actualizado: {result}")
                except Exception as e:
                    print(f"✗ Error: {e}")
        elif cmd == "tts" and len(sys.argv) > 2:
            # python perplexity_chat.py tts "texto" [voz] [archivo.mp3]
            # python perplexity_chat.py tts voces          ← lista voces
            if sys.argv[2] == "voces":
                print("Voces disponibles:")
                for alias, preset in TTS_VOICES.items():
                    print(f"  {alias:<8} ({preset})")
            else:
                text  = sys.argv[2]
                voice = sys.argv[3] if len(sys.argv) > 3 and not sys.argv[3].endswith(".mp3") else "kyrin"
                # Si el 3er arg termina en .mp3, es el archivo destino
                out   = next((a for a in sys.argv[3:] if a.endswith(".mp3")), None)
                if out is None:
                    # Generar nombre automático
                    slug = re.sub(r"[^a-z0-9]", "_", text[:30].lower()).strip("_")
                    out  = f"{slug}_{voice}.mp3"
                try:
                    audio = tts(text, voice=voice, output_path=out)
                    kb = len(audio) / 1024
                    print(f"✓ {kb:.0f} KB → {out}")
                    # Reproducir automáticamente en macOS
                    if sys.platform == "darwin":
                        os.system(f'afplay "{out}" &')
                except Exception as e:
                    print(f"✗ Error: {e}")
        elif cmd == "suggest" and len(sys.argv) > 2:
            # python perplexity_chat.py suggest "texto parcial" [limit]
            query = sys.argv[2]
            limit = int(sys.argv[3]) if len(sys.argv) > 3 else 8
            try:
                results = suggest(query, limit=limit)
                for s2 in results:
                    print(s2)
            except Exception as e:
                print(f"✗ Error: {e}")
        elif cmd == "related" and len(sys.argv) > 2:
            # python perplexity_chat.py related <backend_uuid>
            try:
                queries = get_related_queries(sys.argv[2])
                print("Preguntas relacionadas:")
                for i, q in enumerate(queries, 1):
                    print(f"  {i}. {q}")
            except Exception as e:
                print(f"✗ Error: {e}")
        elif cmd == "news":
            # python perplexity_chat.py news [topic] [limit]
            # python perplexity_chat.py news topics      ← lista de temas
            args = sys.argv[2:]
            if args and args[0] == "topics":
                try:
                    topics_list = get_news_topics()
                    print("Temas disponibles:")
                    for t in topics_list:
                        ename = t.get("name", "")
                        tname = t.get("translated_name") or ""
                        tid   = t.get("id", "")
                        print(f"  {ename:<20} ({tname})  → {tid}")
                    print("\nUso: news tech 5 | news business | news sports 10")
                except Exception as e:
                    print(f"✗ Error: {e}")
            else:
                # Separar temas de límite numérico
                topic_args, lim = [], 10
                for a in args:
                    if a.isdigit():
                        lim = int(a)
                    else:
                        topic_args.append(a)
                try:
                    result = news(topics=topic_args or None, limit=lim)
                    items = result["items"]
                    label = ", ".join(topic_args) if topic_args else "todos los temas"
                    print(f"📰  {len(items)} artículos — {label}:\n")
                    for i, item in enumerate(items, 1):
                        title   = item.get("title") or item.get("short_title", "")
                        domain  = item.get("domain_name") or item.get("author_username", "")
                        ts      = (item.get("published_timestamp") or "")[:10]
                        slug    = item.get("slug", "")
                        url     = f"{BASE_URL}/page/{slug}" if slug else ""
                        summary = (item.get("summary") or item.get("description") or "").strip()
                        live    = " 🔴 EN VIVO" if item.get("is_live") else ""
                        print(f"  [{i}] {title}{live}")
                        if domain or ts:
                            print(f"       {domain}  {ts}")
                        if summary:
                            short = summary[:120].rsplit(" ", 1)[0] + "…" if len(summary) > 120 else summary
                            print(f"       {short}")
                        if url:
                            print(f"       {url}")
                except Exception as e:
                    print(f"✗ Error: {e}")
        elif cmd == "article" and len(sys.argv) > 2:
            # python perplexity_chat.py article <slug-o-url>
            slug = sys.argv[2]
            try:
                a = get_article(slug)
                print(f"📄  {a['title']}\n")
                if a["text"]:
                    print(a["text"])
                if a["sources"]:
                    print(f"\nFuentes ({len(a['sources'])}):")
                    for i, src in enumerate(a["sources"][:8], 1):
                        print(f"  [{i}] {src['name']}")
                        print(f"       {src['url']}")
                if a["related_queries"]:
                    print(f"\nRelacionadas:")
                    for q2 in a["related_queries"][:5]:
                        print(f"  • {q2}")
                print(f"\nbackend_uuid: {a['backend_uuid']}")
                print(f"context_uuid: {a['context_uuid']}")
            except Exception as e:
                print(f"✗ Error: {e}")
        elif cmd == "threads":
            # python perplexity_chat.py threads [search_term] [limit]
            args = sys.argv[2:]
            search = None
            lim = 10
            for a in args:
                if a.isdigit():
                    lim = int(a)
                else:
                    search = a
            try:
                threads = list_threads(search_term=search, limit=lim)
                label = f'"{search}"' if search else "recientes"
                print(f"💬  {len(threads)} hilos {label}:\n")
                for t2 in threads:
                    title = (t2.get("thread_title") or t2.get("query_str") or "")[:70]
                    dt    = (t2.get("last_query_datetime") or "")[:10]
                    mode  = t2.get("mode", "")
                    ctx   = t2.get("context_uuid") or t2.get("uuid", "")
                    print(f"  {title}")
                    print(f"   {dt}  [{mode}]  {ctx}")
            except Exception as e:
                print(f"✗ Error: {e}")
        elif cmd in ("like", "dislike") and len(sys.argv) > 2:
            # python perplexity_chat.py like <context_uuid> [undo]
            # python perplexity_chat.py dislike <context_uuid> [undo]
            ctx  = sys.argv[2]
            undo = len(sys.argv) > 3 and sys.argv[3] == "undo"
            try:
                if cmd == "like":
                    like_article(ctx, undo=undo)
                    print(f"✓ {'Like quitado' if undo else 'Like'} → {ctx}")
                else:
                    dislike_article(ctx, undo=undo)
                    print(f"✓ {'Dislike quitado' if undo else 'Dislike'} → {ctx}")
            except Exception as e:
                print(f"✗ Error: {e}")
        elif cmd == "digest":
            # python perplexity_chat.py digest [YYYY-MM-DD]
            date_arg = sys.argv[2] if len(sys.argv) > 2 else None
            try:
                d = get_digest(date=date_arg)
                print(f"📋  Digest del {d['date']}")
                tl = d["timeline"]
                fd = d["feed_by_date"]
                if not tl and not fd:
                    print("\n  (vacío — el Digest requiere fuentes propias conectadas en la cuenta)")
                    print("  Conecta tu email o calendario en la app para ver contenido aquí.")
                else:
                    if tl:
                        print(f"\nTimeline ({len(tl)}):")
                        for item in tl[:5]:
                            t2 = item.get("title") or item.get("id", "")
                            print(f"  • {t2}")
                    if fd:
                        print(f"\nFeed del día ({len(fd)}):")
                        for item in fd[:5]:
                            t2 = item.get("title") or item.get("id", "")
                            print(f"  • {t2}")
            except Exception as e:
                print(f"✗ Error: {e}")
        elif cmd == "images" and len(sys.argv) > 2:
            # python perplexity_chat.py images "query" [limit]
            query = sys.argv[2]
            limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            try:
                results = search_images(query, limit=limit)
                print(f"🖼  {len(results)} imágenes para \"{query}\":\n")
                for i, img in enumerate(results, 1):
                    caption = img.get("caption") or img.get("name", "")
                    src_url = img.get("url", "")
                    img_url = img.get("image") or img.get("original_image_url", "")
                    w = img.get("image_width", "")
                    h = img.get("image_height", "")
                    print(f"  [{i}] {caption[:60]}")
                    if w and h:
                        print(f"       {w}×{h}  {img_url}")
                    else:
                        print(f"       {img_url}")
                    print(f"       Fuente: {src_url}")
            except Exception as e:
                print(f"✗ Error: {e}")
        elif cmd == "videos" and len(sys.argv) > 2:
            # python perplexity_chat.py videos "query" [limit]
            query = sys.argv[2]
            limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            try:
                results = search_videos(query, limit=limit)
                if not results:
                    print(f"No se encontraron videos para \"{query}\".")
                else:
                    print(f"🎬  {len(results)} videos para \"{query}\":\n")
                    for i, v in enumerate(results, 1):
                        title = v.get("name") or v.get("caption", "")
                        url   = v.get("url", "")
                        thumb = v.get("thumbnail") or v.get("image", "")
                        print(f"  [{i}] {title[:60]}")
                        print(f"       {url}")
                        if thumb:
                            print(f"       Thumbnail: {thumb}")
            except Exception as e:
                print(f"✗ Error: {e}")
        else:
            q      = cmd
            focus  = sys.argv[2] if len(sys.argv) > 2 else "internet"
            m      = sys.argv[3] if len(sys.argv) > 3 else "concise"
            mdl    = sys.argv[4] if len(sys.argv) > 4 else "turbo"
            chat_once(q, search_focus=focus, mode=m, model=mdl)
    else:
        chat_interactive()
