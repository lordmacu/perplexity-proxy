"""
Cliente WebSocket de Soniox — el motor de speech-to-text que usa Perplexity.

IMPORTANTE, y distinto del resto del proxy: este módulo NO habla con
Perplexity. Habla con Soniox, un tercero. Lo único que aporta Perplexity es la
credencial: `POST /rest/realtime/v1/transcription/soniox-api-key` devuelve un
`apiKey` de vida corta, emitido contra la cuenta de Perplexity. La transcripción
en sí, su cuota y su disponibilidad son de Soniox.

Protocolo tomado del APK v2.95.0 decompilado:

  e3o.java:23    wss://stt-rt.soniox.com/transcribe-websocket
  e3o.java:20-25 model="stt-rt-v4", audio_format="pcm_s16le", sample_rate=16000
  a4o.java:286   el frame de configuración se manda como TEXTO al abrir,
                 antes de cualquier audio; max_endpoint_delay_ms=2000 cuando
                 enable_endpoint_detection está activo
  x2o.java       campos del frame de configuración y cuáles son opcionales
  o3o.java       cada token: text, start_ms, end_ms, confidence, is_final,
                 speaker, language
  k3o.java       cada mensaje recibido: tokens[], finished,
                 final_audio_proc_ms, total_audio_proc_ms

DOS COSAS NO SALEN DEL APK y están marcadas donde se usan:

  1. `audio_format="auto"`. El APK solo prueba `pcm_s16le`, porque transmite el
     micrófono en crudo y ahí no hay contenedor que detectar. Para un archivo
     subido (mp3, wav, m4a) hace falta que Soniox detecte el formato. `"auto"`
     es comportamiento documentado de Soniox, no algo leído de este APK.
  2. El fin de audio se señala con un frame de texto vacío. También es
     protocolo de Soniox, no del APK — la app transmite en vivo y nunca
     termina el audio, solo cierra el socket.

Mientras esas dos no estén medidas contra el servicio real, `capabilities.py`
NO debe reportar `audio_transcription: true`. El contrato dice qué logra una
petición hoy, no qué implementa este archivo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import websockets
from curl_cffi.requests import AsyncSession

from config import settings, PPLX_HEADERS

WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
TOKEN_URL = "/rest/realtime/v1/transcription/soniox-api-key"

MODEL = "stt-rt-v4"
NATIVE_AUDIO_FORMAT = "pcm_s16le"
NATIVE_SAMPLE_RATE = 16000
NATIVE_NUM_CHANNELS = 1
ENDPOINT_DELAY_MS = 2000

# e3o.java:14 — los 59 idiomas que el cliente acepta como `language_hints`.
SUPPORTED_LANGUAGES = frozenset({
    "af", "sq", "ar", "az", "eu", "be", "bn", "bs", "bg", "ca", "zh", "hr",
    "cs", "da", "nl", "en", "et", "fi", "fr", "gl", "de", "el", "gu", "he",
    "hi", "hu", "id", "it", "ja", "kn", "kk", "ko", "lv", "lt", "mk", "ms",
    "ml", "mr", "no", "fa", "pl", "pt", "pa", "ro", "ru", "sr", "sk", "sl",
    "es", "sw", "sv", "tl", "ta", "te", "th", "tr", "uk", "ur", "vi", "cy",
})


class SonioxError(RuntimeError):
    """Falla de Soniox o de la emisión de su credencial."""


@dataclass
class Transcript:
    text: str
    language: str | None = None
    tokens: list[dict] = field(default_factory=list)


async def fetch_api_key(timezone: str = "America/Bogota", version: str = "2.95.0") -> str:
    """Pide a Perplexity una credencial de Soniox de vida corta."""
    async with AsyncSession() as s:
        r = await s.post(
            f"{settings.pplx_base}{TOKEN_URL}",
            headers={
                **PPLX_HEADERS,
                "Cookie": f"__Secure-next-auth.session-token={settings.perplexity_session}",
            },
            json={"source": "android", "timezone": timezone, "version": version},
            impersonate="chrome120",
            timeout=20,
        )
    if r.status_code != 200:
        raise SonioxError(f"perplexity {TOKEN_URL}: {r.status_code} {r.text[:200]}")
    try:
        payload = r.json()
    except Exception:
        raise SonioxError(f"perplexity {TOKEN_URL}: respuesta no-JSON")
    # `api_key`, medido: la respuesta real es
    # {"api_key": "temp:...", "expires_at": "..."}. El docstring de
    # `routers/perplexity.py` decía `apiKey` y estaba equivocado.
    key = payload.get("api_key") or payload.get("apiKey")
    if not key:
        raise SonioxError(f"perplexity {TOKEN_URL}: sin api_key en la respuesta")
    return key


def build_config(
    api_key: str,
    *,
    audio_format: str,
    language: str | None = None,
    sample_rate: int | None = None,
    num_channels: int | None = None,
    context: str | None = None,
) -> dict:
    """El frame de configuración, en el orden y con los tipos de x2o.java.

    `sample_rate` y `num_channels` solo se mandan para PCM crudo: con un
    contenedor (`auto`) los trae el archivo, y mandarlos a mano sería afirmar
    algo del audio que no sabemos.
    """
    hints: list[str] = []
    if language:
        base = language.split("-")[0].lower()
        if base in SUPPORTED_LANGUAGES:
            hints.append(base)

    cfg: dict = {
        "api_key": api_key,
        "model": MODEL,
        "audio_format": audio_format,
        "enable_endpoint_detection": True,
        "language_hints": hints,
        "max_endpoint_delay_ms": ENDPOINT_DELAY_MS,
    }
    if audio_format == NATIVE_AUDIO_FORMAT:
        cfg["sample_rate"] = sample_rate or NATIVE_SAMPLE_RATE
        cfg["num_channels"] = num_channels or NATIVE_NUM_CHANNELS
    if context:
        cfg["context"] = context
    return cfg


def _is_marker(text: str) -> bool:
    """`<end>` y compañía son marcadores de control, no habla.

    Medido: con `enable_endpoint_detection` activo, Soniox cierra la
    transcripción con un token final cuyo texto es literalmente `<end>`.
    Concatenarlo dejaba `"...transcripción.<end>"` en la respuesta.
    """
    stripped = text.strip()
    return len(stripped) > 2 and stripped.startswith("<") and stripped.endswith(">")


def _collect(messages: list[dict]) -> Transcript:
    """Junta los tokens finales en un texto.

    Solo cuentan los `is_final`: los no finales son hipótesis que Soniox
    reemplaza en mensajes posteriores, así que incluirlos duplicaría palabras.

    `language` casi siempre queda en None y eso es correcto: aunque `o3o.java`
    declara `language` y `speaker` por token, `stt-rt-v4` con esta configuración
    solo devuelve `text`, `start_ms`, `end_ms`, `confidence` e `is_final`
    (medido). Se reporta lo que Soniox detecta, no el `language` que pidió quien
    llama -- eso sería devolver la pregunta como si fuera la respuesta.
    """
    tokens: list[dict] = []
    for m in messages:
        for t in m.get("tokens") or []:
            if isinstance(t, dict):
                tokens.append(t)

    final = [t for t in tokens if t.get("is_final")]
    text = "".join(
        t.get("text", "") for t in final
        if isinstance(t.get("text"), str) and not _is_marker(t["text"])
    ).strip()
    language = next((t.get("language") for t in final if t.get("language")), None)
    return Transcript(text=text, language=language, tokens=tokens)


async def transcribe(
    audio: bytes,
    *,
    api_key: str,
    audio_format: str = "auto",
    language: str | None = None,
    sample_rate: int | None = None,
    num_channels: int | None = None,
    context: str | None = None,
    chunk_size: int = 16000,
    timeout: int = 120,
) -> Transcript:
    """Transcribe `audio` completo por el WebSocket de Soniox.

    El servicio es de streaming en vivo; acá se usa en lote — se manda el
    archivo entero en trozos y se espera `finished`. Por eso el bucle de lectura
    corta con `finished`, no con el cierre del socket: Soniox sigue mandando
    tokens finales después del último byte de audio, y cortar antes perdería el
    final de la transcripción.
    """
    cfg = build_config(
        api_key,
        audio_format=audio_format,
        language=language,
        sample_rate=sample_rate,
        num_channels=num_channels,
        context=context,
    )

    messages: list[dict] = []
    try:
        async with websockets.connect(WS_URL, open_timeout=20, close_timeout=10) as ws:
            await ws.send(json.dumps(cfg))

            for i in range(0, len(audio), chunk_size):
                await ws.send(audio[i:i + chunk_size])

            # Frame de texto vacío = fin del audio (protocolo de Soniox; ver la
            # nota 2 del docstring del módulo -- no está probado por el APK).
            await ws.send("")

            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict):
                    continue
                if msg.get("error_code") or msg.get("error_message"):
                    raise SonioxError(
                        f"soniox: {msg.get('error_code')} {msg.get('error_message')}"
                    )
                messages.append(msg)
                if msg.get("finished"):
                    break
    except SonioxError:
        raise
    except Exception as exc:
        raise SonioxError(f"soniox websocket: {type(exc).__name__}: {exc}") from exc

    return _collect(messages)
