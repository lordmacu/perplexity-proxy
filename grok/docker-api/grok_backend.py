"""
Grok gRPC backend con rotación automática entre los modelos de 999/hora.
"""
from __future__ import annotations
import os, uuid, re, threading, time
from typing import Iterator, Optional
import grpc

import grok_api_pb2 as ga
import grok_api_pb2_grpc as gg

# ── Catálogo de modelos con sus límites ───────────────────────────────────────
#
# Todos los modelos de 999/hora son Grok 4.5 bajo el capó.
# Los límites son por ventana de 1 hora, independientes entre sí.
#
MODELS_CATALOG = {
    # model_id                         : {"rate": N/h, "window_h": H, "notes": "..."}
    "grok-3":                           {"rate": 30,  "window_h": 24, "notes": "standard grok-3 alias (Grok 4.5 real)"},
    "grok-4":                           {"rate": 7,   "window_h": 24, "notes": "Grok 4 (7 req/24h)"},
    "grok-3-mini-companion":            {"rate": 10,  "window_h": 1,  "notes": "companion variant"},
    "grok-4-1-non-thinking-companion":  {"rate": 10,  "window_h": 1,  "notes": "companion non-thinking"},
    # 999/hora — imagen/plugin agents que también aceptan texto
    "imagine-agent-mode":               {"rate": 999, "window_h": 1,  "notes": "Grok Imagine agent (producción)"},
    "imagine-agent-mode-dev":           {"rate": 999, "window_h": 1,  "notes": "Imagine dev (tiene web_search activo)"},
    "imagine-agent-mode-grok-4-5":      {"rate": 999, "window_h": 1,  "notes": "Imagine variant Grok 4.5"},
    "grok-plugins-4p6-excel":           {"rate": 999, "window_h": 1,  "notes": "Excel plugin agent"},
    "grok-plugins-4p6-word":            {"rate": 999, "window_h": 1,  "notes": "Word plugin agent"},
    "grok-plugins-4p6-docs":            {"rate": 999, "window_h": 1,  "notes": "Google Docs plugin agent"},
    "grok-plugins-4p6-sheets":          {"rate": 999, "window_h": 1,  "notes": "Google Sheets plugin agent"},
    "grok-plugins-4p6-powerpoint":      {"rate": 999, "window_h": 1,  "notes": "PowerPoint plugin agent"},
    "grok-plugins-4p6-outlook":         {"rate": 999, "window_h": 1,  "notes": "Outlook plugin agent"},
    "grok-plugins-4p6-slides":          {"rate": 999, "window_h": 1,  "notes": "Google Slides plugin agent"},
    "grok-plugins-4p5-excel":           {"rate": 999, "window_h": 1,  "notes": "Excel plugin agent (4p5)"},
    "grok-plugins-4p5-word":            {"rate": 999, "window_h": 1,  "notes": "Word plugin agent (4p5)"},
    "grok-plugins-4p5-powerpoint":      {"rate": 999, "window_h": 1,  "notes": "PowerPoint plugin agent (4p5)"},
    "grok-plugins-4p5-outlook":         {"rate": 999, "window_h": 1,  "notes": "Outlook plugin agent (4p5)"},
    "grok-plugins-4p5-docs":            {"rate": 999, "window_h": 1,  "notes": "Google Docs plugin agent (4p5)"},
    "grok-plugins-4p5-sheets":          {"rate": 999, "window_h": 1,  "notes": "Google Sheets plugin agent (4p5)"},
    "grok-plugins-4p5-slides":          {"rate": 999, "window_h": 1,  "notes": "Google Slides plugin agent (4p5)"},
    # Modelos premium (8 req / 4h — SuperGrok restringido)
    "grok-420":                         {"rate": 8,   "window_h": 4,  "notes": "Grok 4.2 premium (8 req/4h)"},
    "grok-4-1-thinking-1129":           {"rate": 8,   "window_h": 4,  "notes": "Grok 4.1 thinking — build nov-29 (8 req/4h)"},
}

# Pool de rotación para chat: todos los de 999/hora
HIGH_RATE_POOL = [m for m, info in MODELS_CATALOG.items() if info["rate"] == 999]

# Pool de rotación para generación de imágenes Aurora (imagine-agent-mode)
IMAGE_POOL = [m for m in MODELS_CATALOG if m.startswith("imagine-agent-mode")]

# Pool de visión (vision): modelos que ven imágenes vía f6 y tienen 999/h
# Confirmado: grok-plugins-4p6-* y grok-plugins-4p5-* ven imágenes + 999/h
# imagine-agent-mode* NO ven imágenes vía f6 (usan @asset interno)
VISION_POOL = [m for m in MODELS_CATALOG if m.startswith("grok-plugins-")]

# Aliases públicos (OpenAI names → modelo interno)
# None = usa rotación del HIGH_RATE_POOL
MODEL_ALIASES: dict[str, Optional[str]] = {
    "auto":           None,   # rotación round-robin explícita
    "grok-4.5":       None,
    "grok":           None,
    "gpt-4o":         None,
    "gpt-4":          None,
    "gpt-4-turbo":    None,
    "gpt-3.5-turbo":  None,
    "claude-3-opus":  None,
    "claude-3-sonnet":None,
    "grok-3":         "grok-3",
}

# Alias de imagen especial
IMAGE_ALIASES = {"auto", "auto-image"}

# Alias de visión (OpenAI-compatible vision aliases)
VISION_ALIASES = {"gpt-4-vision-preview", "gpt-4o-vision", "auto-vision"}

# ── Rotación de modelos ───────────────────────────────────────────────────────
class _ModelRotator:
    """
    Distribuye requests en round-robin entre todos los modelos del pool.
    Cada llamada a next() avanza el índice → los 13 modelos de 999/hora
    se usan de forma pareja desde el primer request, no solo cuando uno
    se agota. Capacidad total: 13 × 999 = ~13.000 req/hora.
    """
    def __init__(self, pool: list[str]):
        self._pool = pool
        self._idx  = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        """Retorna el siguiente modelo en round-robin (avanza siempre)."""
        with self._lock:
            model = self._pool[self._idx]
            self._idx = (self._idx + 1) % len(self._pool)
            return model

    def current(self) -> str:
        return self._pool[self._idx]

    # alias para compatibilidad con el código de rotación en errores
    def advance(self) -> str:
        return self.next()


_rotator        = _ModelRotator(HIGH_RATE_POOL)
_image_rotator  = _ModelRotator(IMAGE_POOL)
_vision_rotator = _ModelRotator(VISION_POOL)


def resolve_model(requested: Optional[str]) -> str:
    # Modelo interno directo → usarlo tal cual
    if requested and requested in MODELS_CATALOG:
        return requested
    # Alias de visión → pool de plugins (999/h con visión)
    if requested and requested in VISION_ALIASES:
        return _vision_rotator.next()
    # Alias OpenAI → mapear o round-robin
    if requested and requested in MODEL_ALIASES:
        mapped = MODEL_ALIASES[requested]
        return mapped if mapped else _rotator.next()
    # Cualquier otra cosa → round-robin
    return _rotator.next()


def resolve_vision_model(requested: Optional[str]) -> str:
    """
    Igual que resolve_model pero cuando hay imágenes adjuntas:
    si el modelo pedido no tiene visión confirmada, usa el VISION_POOL.
    Modelos con visión confirmada: grok-3, grok-3-mini-companion,
    grok-4-1-non-thinking-companion, todos los grok-plugins-*.
    imagine-agent-mode* NO tienen visión vía f6.
    """
    VISION_CAPABLE = {"grok-3", "grok-3-mini-companion", "grok-4-1-non-thinking-companion"}
    if requested and (requested in VISION_CAPABLE or requested.startswith("grok-plugins-")):
        return requested
    if requested and requested in VISION_ALIASES:
        return _vision_rotator.next()
    # Para aliases genéricos (auto, gpt-4o, etc.) con imágenes → VISION_POOL
    return _vision_rotator.next()


# ── Filtro de artefactos XML y ruido interno ──────────────────────────────────
_THINKING = re.compile(r'Thinking about your request\s*')


def _strip_xai(text: str) -> str:
    text = re.sub(r'<xai:[^>]*>.*?</xai:[^>]*>', '', text, flags=re.DOTALL)
    text = re.sub(r'<xai:[^/][^>]*/>', '', text, flags=re.DOTALL)
    text = re.sub(r'</xai:[^>]*>', '', text)
    text = re.sub(r'<!\[CDATA\[.*?\]\]>', '', text, flags=re.DOTALL)
    text = _THINKING.sub('', text)
    return text


# ── Proto helpers (raw serialization for services sin stubs compilados) ───────

def _varint(v: int) -> bytes:
    r = []
    while v > 0x7F:
        r.append((v & 0x7F) | 0x80); v >>= 7
    r.append(v)
    return bytes(r)

def _str_field(tag: int, s: str) -> bytes:
    b = s.encode()
    return bytes([(tag << 3) | 2]) + _varint(len(b)) + b

def _bool_field(tag: int, v: bool) -> bytes:
    return bytes([(tag << 3) | 0, 1 if v else 0])

def _int_field(tag: int, v: int) -> bytes:
    return bytes([(tag << 3) | 0]) + _varint(v)

def _nested_field(tag: int, data: bytes) -> bytes:
    return bytes([(tag << 3) | 2]) + _varint(len(data)) + data

# Decoder genérico de campos protobuf (wire-level)
def _decode_varint(data: bytes, i: int):
    v = 0; shift = 0
    while i < len(data):
        b = data[i]; i += 1
        v |= (b & 0x7F) << shift; shift += 7
        if not (b & 0x80): return v, i
    return v, i

def _decode_proto(data: bytes) -> dict:
    """Parsea bytes proto a {field_tag: [(wire_type, value), ...]}."""
    i = 0; fields: dict = {}
    while i < len(data):
        try:
            tag_wire, i = _decode_varint(data, i)
            tag = tag_wire >> 3; wire = tag_wire & 0x7
            if wire == 0:
                v, i = _decode_varint(data, i)
                fields.setdefault(tag, []).append(('int', v))
            elif wire == 2:
                l, i = _decode_varint(data, i)
                val = data[i:i+l]; i += l
                try:    fields.setdefault(tag, []).append(('str', val.decode('utf-8')))
                except: fields.setdefault(tag, []).append(('raw', val))
            elif wire == 5: fields.setdefault(tag, []).append(('f32', data[i:i+4])); i+=4
            elif wire == 1: fields.setdefault(tag, []).append(('f64', data[i:i+8])); i+=8
            else: break
        except Exception:
            break
    return fields

def _first_str(fields: dict, tag: int) -> str:
    """Retorna el primer string en fields[tag], o '' si no existe."""
    for kind, val in fields.get(tag, []):
        if kind == 'str': return val
    return ''

def _first_int(fields: dict, tag: int) -> int:
    """Retorna el primer int en fields[tag], o 0 si no existe."""
    for kind, val in fields.get(tag, []):
        if kind == 'int': return val
    return 0

def _first_raw(fields: dict, tag: int) -> bytes:
    for kind, val in fields.get(tag, []):
        if kind == 'raw': return val
        if kind == 'str': return val.encode()
    return b''

def _ts_to_iso(nested_bytes: bytes) -> Optional[str]:
    """Convierte nested Google Timestamp bytes a ISO 8601 string."""
    try:
        import datetime
        f = _decode_proto(nested_bytes)
        secs = _first_int(f, 1); nanos = _first_int(f, 2)
        dt = datetime.datetime.utcfromtimestamp(secs)
        return dt.isoformat() + 'Z'
    except Exception:
        return None


_GROK_RENDER_RE = re.compile(
    r'<grok:render\s[^>]*card_id="([^"]+)"[^>]*>.*?'
    r'<argument\s+name="file_path">([^<]+)</argument>.*?</grok:render>',
    re.DOTALL,
)


class _StreamCleaner:
    """
    Limpia el stream token a token. Elimina:
      - <xai:*>...</xai:*> (tool usage cards, etc.)
      - <grok:render>...</grok:render> (file render cards)
      - textos de status como "Thinking about your request", "Creating the..."
    Almacena los (card_id, file_path) de cada <grok:render> en self.renders.
    """

    _OPEN_PREFIXES = ('<xai:', '<grok:')
    _CLOSE_PREFIXES = ('</xai:', '</grok:')

    def __init__(self):
        self._buf     = ""
        self._raw     = ""   # raw token accumulation for grok:render extraction
        self._inside  = 0
        self.renders: list[tuple[str, str]] = []  # (card_id, file_path)

    def feed(self, token: str) -> str:
        out = []
        self._buf += token
        self._raw += token

        while self._buf:
            if self._inside > 0:
                close = min(
                    (self._buf.find(p) for p in self._CLOSE_PREFIXES if self._buf.find(p) >= 0),
                    default=-1,
                )
                if close == -1:
                    break
                end = self._buf.find('>', close)
                if end == -1:
                    break
                # Before discarding, check for completed grok:render blocks
                segment = self._buf[:end + 1]
                for m in _GROK_RENDER_RE.finditer(segment):
                    self.renders.append((m.group(1), m.group(2)))
                self._inside -= 1
                self._buf = self._buf[end + 1:]
            else:
                open_pos = min(
                    (self._buf.find(p) for p in self._OPEN_PREFIXES if self._buf.find(p) >= 0),
                    default=-1,
                )
                if open_pos == -1:
                    out.append(self._buf)
                    self._buf = ""
                    break
                if open_pos > 0:
                    out.append(self._buf[:open_pos])
                close_pos = self._buf.find('>', open_pos)
                if close_pos == -1:
                    self._buf = self._buf[open_pos:]
                    break
                self_closing = self._buf[close_pos - 1] == '/'
                self._buf = self._buf[close_pos + 1:]
                if not self_closing:
                    self._inside += 1

        text = "".join(out)
        text = re.sub(r'</(?:xai|grok):[^>]*>', '', text)
        text = _THINKING.sub('', text)
        return text

    def flush(self) -> str:
        # Extract grok:render blocks from the full raw stream
        for m in _GROK_RENDER_RE.finditer(self._raw):
            self.renders.append((m.group(1), m.group(2)))
        rest = _strip_xai(self._buf)
        rest = re.sub(r'</?grok:[^>]*>', '', rest, flags=re.DOTALL)
        self._buf    = ""
        self._raw    = ""
        self._inside = 0
        return rest


def _list_files(conversation_id: str, timeout: int = 10) -> list[dict]:
    """
    Llama grok_api_v2.FilesService/ListFiles.
    Retorna lista de dicts {filename, path, size, mime_type}.
    Usado como fallback cuando el plugin model no emite <grok:render>.
    """
    try:
        ch = get_channel()
        call_fn = ch.unary_unary(
            '/grok_api_v2.FilesService/ListFiles',
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x,
        )
        req_bytes = _str_field(1, conversation_id)
        resp_bytes = call_fn(req_bytes, metadata=_make_meta(), timeout=timeout)

        def _parse_file_entry(data: bytes) -> dict:
            pos = 0; result = {}
            while pos < len(data):
                b = data[pos]; pos += 1
                fnum = b >> 3; wt = b & 7
                if wt == 2:
                    ln = 0; sh = 0
                    while True:
                        b2 = data[pos]; pos += 1
                        ln |= (b2 & 0x7F) << sh; sh += 7
                        if not (b2 & 0x80): break
                    val = data[pos:pos+ln]
                    result[fnum] = val.decode('utf-8', errors='replace')
                    pos += ln
                elif wt == 0:
                    v = 0; sh = 0
                    while True:
                        b2 = data[pos]; pos += 1
                        v |= (b2 & 0x7F) << sh; sh += 7
                        if not (b2 & 0x80): break
                    result[fnum] = v
                else:
                    break
            return result

        # Parse outer response: field 1 = repeated file entries (nested), field 2 = dir
        files = []
        pos = 0
        while pos < len(resp_bytes):
            b = resp_bytes[pos]; pos += 1
            fnum = b >> 3; wt = b & 7
            if wt == 2:
                ln = 0; sh = 0
                while True:
                    b2 = resp_bytes[pos]; pos += 1
                    ln |= (b2 & 0x7F) << sh; sh += 7
                    if not (b2 & 0x80): break
                val = resp_bytes[pos:pos+ln]
                if fnum == 1:
                    entry = _parse_file_entry(val)
                    files.append({
                        "filename":  entry.get(1, ""),
                        "path":      entry.get(2, ""),
                        "size":      entry.get(4, 0),
                        "mime_type": entry.get(5, ""),
                    })
                pos += ln
            elif wt == 0:
                while resp_bytes[pos] & 0x80: pos += 1
                pos += 1
            else:
                break
        return files
    except Exception:
        return []


def _resolve_file_url(conversation_id: str, file_path: str, timeout: int = 15) -> Optional[str]:
    """
    Llama GetFileContentV2 en grok_api_v2.FilesService para obtener la URL
    de descarga de un archivo generado por un plugin model.
    Retorna download_signed_url o signed_url, o None si falla.
    """
    try:
        ch = get_channel()
        call_fn = ch.unary_unary(
            '/grok_api_v2.FilesService/GetFileContentV2',
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x,
        )
        # Server needs just the filename, not the full workdir path
        fname = file_path.rsplit('/', 1)[-1]
        req_bytes = _str_field(1, conversation_id) + _str_field(2, fname) + _bool_field(4, True)
        resp_bytes = call_fn(req_bytes, metadata=_make_meta(), timeout=timeout)

        # Parse GetFileContentResponse: field 5=signed_url, 6=download_signed_url
        def _parse_strings(data: bytes) -> dict[int, str]:
            result = {}; pos = 0
            while pos < len(data):
                b = data[pos]; pos += 1
                field_num = b >> 3; wire_type = b & 7
                if wire_type == 2:
                    length = 0; shift = 0
                    while True:
                        b2 = data[pos]; pos += 1
                        length |= (b2 & 0x7F) << shift; shift += 7
                        if not (b2 & 0x80): break
                    result[field_num] = data[pos:pos+length].decode('utf-8', errors='replace')
                    pos += length
                elif wire_type == 0:
                    while data[pos] & 0x80: pos += 1
                    pos += 1
                else:
                    break
            return result

        fields = _parse_strings(resp_bytes)
        return fields.get(6) or fields.get(5) or fields.get(1)
    except Exception:
        return None


# ── Canal gRPC (singleton) ────────────────────────────────────────────────────
_channel: Optional[grpc.Channel] = None
_channel_lock = threading.Lock()


def get_channel() -> grpc.Channel:
    global _channel
    if _channel is None:
        with _channel_lock:
            if _channel is None:
                _channel = grpc.secure_channel(
                    "grok.com:443",
                    grpc.ssl_channel_credentials(),
                    options=[
                        ("grpc.keepalive_time_ms", 30_000),
                        ("grpc.keepalive_timeout_ms", 10_000),
                        ("grpc.keepalive_permit_without_calls", 1),
                    ],
                )
    return _channel


def _session_token() -> str:
    token = os.environ.get("GROK_SESSION_TOKEN", "")
    if not token:
        raise RuntimeError("GROK_SESSION_TOKEN no está configurado")
    return token


def _make_meta() -> list:
    token = _session_token()
    return [
        ("user-agent",       "Grok/1.2.22 (Android; arm64-v8a)"),
        ("x-app-name",       "Grok Android"),
        ("x-app-version",    "1.2.22"),
        ("x-app-language",   "en-US"),
        ("x-xai-request-id", str(uuid.uuid4())),
        ("cookie",           f"sso={token}; sso-rw={token}"),
    ]


# ── Conversión de mensajes OpenAI → prompt ────────────────────────────────────
def extract_images_from_messages(messages: list[dict]) -> list[tuple[bytes, str]]:
    """
    Extrae imágenes base64 de mensajes OpenAI con content tipo lista.
    Retorna lista de (image_bytes, mime_type).
    Soporta: data:image/...;base64,<data> y URLs de datos sin prefijo MIME.
    """
    import base64 as _b64
    images: list[tuple[bytes, str]] = []
    for m in messages:
        content = m.get("content", "")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image_url":
                continue
            url = part.get("image_url", {}).get("url", "")
            if not url.startswith("data:"):
                continue
            # data:image/png;base64,XXXX
            try:
                header, data = url.split(",", 1)
                mime = header.split(":")[1].split(";")[0] if ":" in header else "image/jpeg"
                img_bytes = _b64.b64decode(data)
                images.append((img_bytes, mime))
            except Exception:
                continue
    return images


def messages_to_prompt(
    messages: list[dict],
    tools: Optional[list] = None,
) -> tuple[str, Optional[str]]:
    """
    Convierte el array messages de OpenAI a (prompt, system).
    - Los mensajes "system" se extraen como custom_instructions nativo (field 19).
    - Los tools se inyectan también en el system.
    Retorna (prompt_str, system_str_or_None).
    """
    import json as _json

    system_parts: list[str] = []
    conv_parts:   list[str] = []

    # Recolectar system messages
    for m in messages:
        role    = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
        if role == "system":
            system_parts.append(content)

    # Tools como contexto de sistema
    if tools:
        tools_desc = _json.dumps(tools, indent=2)
        system_parts.append(
            "Available Tools — when you want to call a tool respond with a JSON block:\n"
            '```tool_call\n{"name": "<tool_name>", "arguments": {<args>}}\n```\n\n'
            f"Tools:\n{tools_desc}"
        )

    # Construir el prompt de conversación (sin los system)
    non_system = [m for m in messages if m.get("role") != "system"]
    for m in non_system:
        role    = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
        if role == "tool":
            conv_parts.append(f"[Tool Result - {m.get('name', 'tool')}]\n{content}")
        elif role == "assistant":
            conv_parts.append(f"[Assistant]\n{content}")
        else:
            conv_parts.append(f"[User]\n{content}")

    # Un único mensaje de usuario sin extras → enviarlo directo
    if len(non_system) == 1 and non_system[0].get("role") == "user" and not tools and not system_parts:
        raw_content = non_system[0].get("content", "")
        if isinstance(raw_content, list):
            raw_content = " ".join(p.get("text", "") for p in raw_content if p.get("type") == "text")
        return raw_content, None

    prompt = "\n\n".join(conv_parts)
    system = "\n\n".join(system_parts) if system_parts else None
    return prompt, system


# ── Tool call parser (básico) ─────────────────────────────────────────────────
_TOOL_BLOCK = re.compile(
    r'```tool_call\s*\n(\{.*?\})\s*\n```', re.DOTALL
)


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """
    Extrae bloques ```tool_call``` del texto.
    Devuelve (texto_sin_tool_calls, lista_de_tool_calls).
    """
    import json as _json
    tool_calls = []
    def replace(m):
        try:
            payload = _json.loads(m.group(1))
            tool_calls.append({
                "id":       f"call_{uuid.uuid4().hex[:8]}",
                "type":     "function",
                "function": {
                    "name":      payload.get("name", "unknown"),
                    "arguments": _json.dumps(payload.get("arguments", {})),
                },
            })
        except Exception:
            pass
        return ""

    clean = _TOOL_BLOCK.sub(replace, text).strip()
    return clean, tool_calls


# ── Stream de tokens ──────────────────────────────────────────────────────────
def _build_chat_request(
    prompt: str,
    model_id: str,
    is_reasoning: bool = False,
    disable_search: bool = True,
    system: Optional[str] = None,
    personality: Optional[str] = None,
    image_file_ids: Optional[list[str]] = None,
) -> bytes:
    """
    Construye CreateConversationAndRespondRequest como bytes proto raw.
    Cuando hay image_file_ids usa raw encoding (f6=repeated string).
    De lo contrario usa los stubs compilados para compatibilidad.
    """
    if image_file_ids:
        req = (
            _str_field(4, model_id)
            + _str_field(5, prompt)
            + _bool_field(8, disable_search)
        )
        if is_reasoning:
            req += _bool_field(32, True)
        if system:
            req += _str_field(19, system)
        if personality:
            req += _str_field(20, personality)
        for fid in image_file_ids:
            req += _str_field(6, fid)
        return req
    else:
        kwargs: dict = dict(
            model_name=model_id,
            message=prompt,
            disable_search=disable_search,
            is_reasoning=is_reasoning,
        )
        if system:
            kwargs["custom_instructions"] = system
        if personality:
            kwargs["custom_personality"] = personality
        return ga.CreateConversationAndRespondRequest(**kwargs).SerializeToString()


def _stream_raw_chat(
    req_bytes: bytes,
    timeout: int = 120,
    use_raw: bool = False,
):
    """Itera sobre chunks del stream CreateConversationAndRespond."""
    ch = get_channel()
    if use_raw:
        fn = ch.unary_stream(
            "/grok_api.Chat/CreateConversationAndRespond",
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x,
        )
        for raw in fn(req_bytes, metadata=_make_meta(), timeout=timeout):
            yield raw, True   # (data, is_raw)
    else:
        stub = gg.ChatStub(ch)
        req  = ga.CreateConversationAndRespondRequest.FromString(req_bytes)
        for chunk in stub.CreateConversationAndRespond(
            req, metadata=_make_meta(), timeout=timeout
        ):
            yield chunk, False


def stream_chat(
    prompt: str,
    model_id: str,
    is_reasoning: bool = False,
    disable_search: bool = True,
    timeout: int = 120,
    system: Optional[str] = None,
    personality: Optional[str] = None,
    image_file_ids: Optional[list[str]] = None,
) -> Iterator[str]:
    """
    Genera tokens limpios de texto. Rota modelo si RESOURCE_EXHAUSTED.
    system         → custom_instructions (field 19, system prompt nativo)
    personality    → custom_personality (field 20)
    image_file_ids → IDs de imágenes previamente subidas (field 6, file_attachments)
    """
    use_raw = bool(image_file_ids)

    for _ in range(len(HIGH_RATE_POOL) + 1):
        req_bytes = _build_chat_request(
            prompt, model_id, is_reasoning, disable_search,
            system, personality, image_file_ids,
        )
        cleaner = _StreamCleaner()
        try:
            conv_id: Optional[str] = None
            for chunk, is_raw in _stream_raw_chat(req_bytes, timeout, use_raw):
                if is_raw:
                    f = _decode_proto(chunk)
                    # conv_info en f2
                    if conv_id is None:
                        for kind, val in f.get(2, []):
                            data = val.encode('latin-1') if kind == 'str' else (val if kind == 'raw' else b'')
                            inner = _decode_proto(data)
                            cid = _first_str(inner, 1)
                            if cid: conv_id = cid
                    # token en f1.f2
                    for kind, val in f.get(1, []):
                        data = val.encode('latin-1') if kind == 'str' else (val if kind == 'raw' else b'')
                        inner = _decode_proto(data)
                        tok = _first_str(inner, 2)
                        if tok:
                            out = cleaner.feed(tok)
                            if out:
                                yield out
                else:
                    if conv_id is None:
                        cinfo = chunk.conversation_info
                        if cinfo and cinfo.conversation_id:
                            conv_id = cinfo.conversation_id
                    ar = chunk.add_response
                    if ar.token:
                        out = cleaner.feed(ar.token)
                        if out:
                            yield out

            tail = cleaner.flush()
            if tail:
                yield tail
            # Resolve generated files (plugin models)
            if conv_id:
                if cleaner.renders:
                    for _card_id, file_path in cleaner.renders:
                        url = _resolve_file_url(conv_id, file_path)
                        fname = file_path.rsplit('/', 1)[-1]
                        if url:
                            yield f"\n\n[Descargar: **{fname}**]({url})"
                        else:
                            yield f"\n\n*Archivo generado: `{fname}` (URL no disponible)*"
                elif model_id.startswith("grok-plugins-"):
                    files = _list_files(conv_id)
                    for f in files:
                        fname = f["filename"]
                        if fname:
                            url = _resolve_file_url(conv_id, fname)
                            if url:
                                size_kb = f["size"] / 1024
                                yield f"\n\n[Descargar: **{fname}**]({url}) ·  {size_kb:.1f} KB · {f['mime_type']}"
            return

        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                model_id = _rotator.advance()
                continue
            raise


def complete_chat(
    prompt: str,
    model_id: str,
    is_reasoning: bool = False,
    disable_search: bool = True,
    timeout: int = 120,
    system: Optional[str] = None,
    personality: Optional[str] = None,
    image_file_ids: Optional[list[str]] = None,
) -> str:
    return "".join(stream_chat(
        prompt, model_id, is_reasoning, disable_search,
        timeout, system, personality, image_file_ids,
    ))


def generate_image(
    prompt: str,
    model_id: Optional[str] = None,
    timeout: int = 180,
) -> list[dict]:
    """
    Genera imágenes usando los modelos Imagine de Grok (Aurora).
    Rota entre los 3 imagine-agent variants si uno está rate-limited.
    Retorna lista de dicts con {url, image_id, asset_id, progress, moderated}.
    Lanza RuntimeError solo si TODOS los imagine models están rate-limited.
    """
    ch   = get_channel()
    stub = gg.ChatStub(ch)

    # "auto" o None → rotar entre los 3 imagine-agent
    # modelo específico del pool → usarlo directamente
    if not model_id or model_id in IMAGE_ALIASES or model_id not in IMAGE_POOL:
        pool = IMAGE_POOL
    else:
        pool = [model_id]

    last_error = None
    for attempt, mid in enumerate(pool):
        req = ga.CreateConversationAndRespondRequest(
            model_name=mid,
            message=prompt,
            disable_search=True,
        )

        images   = []
        text_buf = []

        try:
            for chunk in stub.CreateConversationAndRespond(
                req, metadata=_make_meta(), timeout=timeout
            ):
                ar = chunk.add_response
                if ar.token:
                    text_buf.append(ar.token)
                img = ar.streaming_image_generation_response
                if img.image_url:
                    images.append({
                        "url":       img.image_url,
                        "image_id":  img.image_id,
                        "asset_id":  img.asset_id,
                        "progress":  img.progress,
                        "moderated": img.moderated,
                        "r_rated":   img.r_rated,
                        "model":     mid,
                    })
        except grpc.RpcError as e:
            last_error = str(e.details())
            continue

        full_text = "".join(text_buf).lower()
        if not images and any(kw in full_text for kw in ["rate limit", "try again", "reached the", "limit"]):
            last_error = f"{mid}: image generation rate limited"
            continue   # intentar con el siguiente imagine model

        return images  # éxito (puede ser lista vacía si la respuesta fue solo texto)

    raise RuntimeError(
        f"Image generation rate limited on all {len(pool)} imagine models. "
        f"Last error: {last_error}"
    )


# ── Grok-native helpers ───────────────────────────────────────────────────────

def list_skills(locale: str = "en-US") -> list[dict]:
    """
    Llama grok_api.Skills/ListSkills.
    Usa decoder raw porque el proto compilado tiene field numbers incorrectos.
    Layout real del server (reverse-engineered):
      f1 = skill_type (int: 1=docx, 3=pdf, 4=pptx, 5=skill-creator, 7=xlsx)
      f2 = skill_id   (string)
      f3 = description (string)
      f4 = icon_name  (string — e.g. "file-text", "wand")
      f5 = display_name (string)
    """
    raw = _raw_unary("/grok_api.Skills/ListSkills", _str_field(1, locale), timeout=10)
    f   = _decode_proto(raw)
    skills = []
    for kind, val in f.get(1, []):
        data  = val.encode('latin-1') if kind == 'str' else (val if kind == 'raw' else b'')
        inner = _decode_proto(data)
        skills.append({
            "skill_id":     _first_str(inner, 2),
            "display_name": _first_str(inner, 5),
            "description":  _first_str(inner, 3),
            "icon_name":    _first_str(inner, 4),
            "skill_type":   _first_int(inner, 1),
        })
    return skills


def get_user_settings() -> dict:
    """Llama grok_api.Settings/GetUserSettings."""
    ch   = get_channel()
    stub = gg.SettingsStub(ch)
    resp = stub.GetUserSettings(ga.EmptyRequest(), metadata=_make_meta(), timeout=10)
    return {
        "exclude_from_training":            resp.exclude_from_training if resp.HasField("exclude_from_training") else None,
        "allow_x_personalization":          resp.allow_x_personalization if resp.HasField("allow_x_personalization") else None,
        "enable_memory":                    resp.enable_memory if resp.HasField("enable_memory") else None,
        "allow_share_indexing":             resp.allow_share_indexing if resp.HasField("allow_share_indexing") else None,
        "allow_companion_notifications":    resp.allow_companion_notifications if resp.HasField("allow_companion_notifications") else None,
        "allow_auto_share":                 resp.allow_auto_share if resp.HasField("allow_auto_share") else None,
        "allow_grok_finished_notification": resp.allow_grok_finished_notification if resp.HasField("allow_grok_finished_notification") else None,
    }


def list_modes(locale: str = "en-US") -> dict:
    """Llama grok_api.Models/ListModes."""
    ch   = get_channel()
    stub = gg.ModelsStub(ch)
    resp = stub.ListModes(
        ga.ListModesRequest(locale=locale),
        metadata=_make_meta(), timeout=10,
    )
    return {
        "modes": [
            {
                "mode_id":      m.mode_id,
                "display_name": m.display_name,
                "description":  m.description,
                "icon_url":     m.icon_url,
            }
            for m in resp.modes
        ],
        "default_mode_id": resp.default_mode_id,
    }


def list_voices() -> dict:
    """Llama grok_api.Voice/ListTopVoices y ListVoices."""
    ch   = get_channel()
    stub = gg.VoiceStub(ch)
    meta = _make_meta()
    top = stub.ListTopVoices(ga.ListTopVoicesRequest(), metadata=meta, timeout=10)
    all_ = stub.ListVoices(ga.ListVoicesRequest(), metadata=meta, timeout=10)
    return {
        "top_voices": [
            {"voice_id": v.voice_id, "display_name": v.display_name,
             "description": v.description, "preview_url": v.preview_url}
            for v in top.voices
        ],
        "voices": [
            {"voice_id": v.voice_id, "display_name": v.display_name,
             "description": v.description}
            for v in all_.voices
        ],
        "next_cursor": all_.next_cursor or None,
    }


def get_rate_limit_single(model_id: str, kind: int = 0) -> dict:
    """Consulta rate limits de un modelo específico."""
    ch   = get_channel()
    meta = _make_meta()
    rl   = ch.unary_unary(
        "/grok_api.RateLimits/GetRateLimits",
        request_serializer=ga.GetRateLimitsRequest.SerializeToString,
        response_deserializer=ga.GetRateLimitsResponse.FromString,
    )
    r = rl(ga.GetRateLimitsRequest(model_name=model_id, request_kind=kind),
           metadata=meta, timeout=5)
    info = MODELS_CATALOG.get(model_id, {})
    return {
        "model":          model_id,
        "notes":          info.get("notes", ""),
        "rate_per_hour":  info.get("rate"),
        "window_hours":   info.get("window_h"),
        "remaining":      r.remaining_queries,
        "total":          r.total_queries,
        "window_seconds": r.window_size_seconds,
        "wait_time_seconds": r.wait_time_seconds if r.HasField("wait_time_seconds") else None,
    }


def stream_suggestions_collect(query: str, types: list[str]) -> list[dict]:
    """
    Llama grok_api.SuggestionService/StreamSuggestions.

    Estado real del servicio (verificado):
    - GROK_COMPLETION (type=4): funciona, devuelve 5 autocompletions
    - SEARCH_COMPLETION (type=1): deployado pero siempre vacío
    - QUICK_ANSWER (type=2): deployado pero siempre vacío
    - STOCK (type=3): la combinación con resultados reales causa error de
      deserialización en el cliente → excluido por defecto

    Usa GROK_COMPLETION si el caller pide "stock" o "search_completion"
    (no tienen efecto real), pero sí los incluye en los accepted_types para
    que el servidor eventualmente los active.
    """
    TYPE_MAP = {
        "search_completion": 1,
        "quick_answer":      2,
        "grok_completion":   4,
        # stock (3) omitido — causa deserialization crash cuando AAPL + todos los tipos
    }
    requested = [TYPE_MAP[t] for t in types if t in TYPE_MAP]
    if not requested:
        requested = [4]  # solo GROK_COMPLETION por defecto

    accepted = [ga.AcceptedSuggestionType(type=t, max_items=5) for t in requested]
    ch   = get_channel()
    stub = gg.SuggestionServiceStub(ch)
    req  = ga.StreamSuggestionsRequest(query=query, accepted_types=accepted)

    results = []
    try:
        for chunk in stub.StreamSuggestions(req, metadata=_make_meta(), timeout=10):
            kind = chunk.WhichOneof("suggestion")
            if kind == "search_completion":
                results.append({"type": "search_completion", "text": chunk.search_completion.text})
            elif kind == "quick_answer":
                results.append({"type": "quick_answer", "text": chunk.quick_answer.text})
            elif kind == "stock":
                results.append({"type": "stock", "ticker": chunk.stock.ticker, "name": chunk.stock.name})
            elif kind == "grok_completion":
                results.append({"type": "grok_completion", "text": chunk.grok_completion.text})
    except grpc.RpcError:
        pass  # deserialization errors on stock responses are swallowed
    return results


# ── Rate limits en tiempo real ────────────────────────────────────────────────
# ── Chat management (raw gRPC — sin stubs compilados) ────────────────────────

def _raw_unary(path: str, req: bytes, timeout: int = 10) -> bytes:
    ch = get_channel()
    fn = ch.unary_unary(path, request_serializer=lambda x: x, response_deserializer=lambda x: x)
    return fn(req, metadata=_make_meta(), timeout=timeout)

def _raw_stream(path: str, req: bytes, timeout: int = 120):
    ch = get_channel()
    fn = ch.unary_stream(path, request_serializer=lambda x: x, response_deserializer=lambda x: x)
    return fn(req, metadata=_make_meta(), timeout=timeout)


def _parse_conversation(f: dict) -> dict:
    """Parsea campos de un mensaje Conversation proto."""
    created = updated = None
    for kind, val in f.get(3, []):
        if kind == 'raw': created = _ts_to_iso(val)
        elif kind == 'str': created = _ts_to_iso(val.encode('latin-1'))
    for kind, val in f.get(4, []):
        if kind == 'raw': updated = _ts_to_iso(val)
        elif kind == 'str': updated = _ts_to_iso(val.encode('latin-1'))
    return {
        "conversation_id": _first_str(f, 1),
        "title":           _first_str(f, 2),
        "created_at":      created,
        "updated_at":      updated,
    }


def list_conversations(limit: int = 20, cursor: str = "") -> dict:
    """Llama grok_api.Chat/ListConversations.
    ListConversationsRequest: f1=page_size(int), f2=page_token(str cursor).
    """
    req = b''
    if limit:  req += _int_field(1, limit)
    if cursor: req += _str_field(2, cursor)
    raw = _raw_unary("/grok_api.Chat/ListConversations", req)
    f   = _decode_proto(raw)
    convs = []
    for kind, val in f.get(1, []):
        data = val.encode('latin-1') if kind == 'str' else (val if kind == 'raw' else b'')
        inner = _decode_proto(data)
        convs.append(_parse_conversation(inner))
    next_cursor = _first_str(f, 2)
    return {"conversations": convs, "next_cursor": next_cursor or None}


def get_conversation(conv_id: str) -> dict:
    """Llama grok_api.Chat/GetConversation."""
    raw = _raw_unary("/grok_api.Chat/GetConversation", _str_field(1, conv_id))
    f   = _decode_proto(raw)
    return _parse_conversation(f)


def delete_conversation(conv_id: str) -> dict:
    """Llama grok_api.Chat/DeleteConversation."""
    _raw_unary("/grok_api.Chat/DeleteConversation", _str_field(1, conv_id))
    return {"deleted": True, "conversation_id": conv_id}


def update_conversation(conv_id: str, title: str) -> dict:
    """Llama grok_api.Chat/UpdateConversation (renombrar)."""
    req = _str_field(1, conv_id) + _str_field(2, title)
    raw = _raw_unary("/grok_api.Chat/UpdateConversation", req)
    f   = _decode_proto(raw)
    return {
        "conversation_id": _first_str(f, 1) or conv_id,
        "title":           _first_str(f, 2) or title,
    }


def list_responses(conv_id: str) -> list[dict]:
    """
    Llama grok_api.Chat/ListResponses.
    Retorna historial completo de mensajes con IDs, roles y texto.
    """
    raw  = _raw_unary("/grok_api.Chat/ListResponses", _str_field(1, conv_id))
    f    = _decode_proto(raw)
    msgs = []
    for kind, val in f.get(1, []):
        data  = val.encode('latin-1') if kind == 'str' else (val if kind == 'raw' else b'')
        inner = _decode_proto(data)
        msgs.append({
            "response_id": _first_str(inner, 1),
            "text":        _first_str(inner, 2),
            "role":        _first_str(inner, 3).lower() or "unknown",
            "parent_id":   _first_str(inner, 5) or None,
            "model":       _first_str(inner, 42) or None,
        })
    return msgs


def stream_add_response(
    conv_id: str,
    message: str,
    model_id: Optional[str] = None,
    disable_search: bool = True,
    timeout: int = 120,
) -> Iterator[str]:
    """
    Llama grok_api.Chat/AddResponse (streaming).
    Añade un mensaje a una conversación existente y retorna tokens limpios.

    Proto AddResponseRequest (reverse-engineered):
      f1 = conversation_id
      f2 = message (texto del usuario)
      f3 = model_name
      f5 = new_human_message_id (UUID generado por el cliente)
      f8 = disable_search (bool)
    """
    mid = model_id or resolve_model(None)
    new_msg_id = str(uuid.uuid4())
    req = (
        _str_field(1, conv_id)
        + _str_field(2, message)
        + _str_field(3, mid)
        + _str_field(5, new_msg_id)
        + _bool_field(8, disable_search)
    )
    cleaner = _StreamCleaner()
    for raw in _raw_stream("/grok_api.Chat/AddResponse", req, timeout=timeout):
        chunk_fields = _decode_proto(raw)
        token = _first_str(chunk_fields, 2)
        if token:
            out = cleaner.feed(token)
            if out:
                yield out
    tail = cleaner.flush()
    if tail:
        yield tail


def complete_add_response(
    conv_id: str,
    message: str,
    model_id: Optional[str] = None,
    disable_search: bool = True,
) -> str:
    return "".join(stream_add_response(conv_id, message, model_id, disable_search))


def share_conversation(conv_id: str, resp_id: Optional[str] = None, share_publicly: bool = True) -> dict:
    """
    Llama grok_api.Chat/ShareConversation.
    Si resp_id es None, toma el último response del historial.
    Retorna share_token (formato: base64('shard-N')_<uuid>)
    y la URL de share inferida.
    """
    if not resp_id:
        msgs = list_responses(conv_id)
        asst = [m for m in msgs if m["role"] in ("assistant",)]
        if not asst:
            raise RuntimeError("No hay respuestas de asistente en la conversación")
        resp_id = asst[-1]["response_id"]

    req = _str_field(1, conv_id) + _str_field(2, resp_id) + _bool_field(5, share_publicly)
    raw = _raw_unary("/grok_api.Chat/ShareConversation", req, timeout=10)
    f   = _decode_proto(raw)
    token = _first_str(f, 1)
    return {
        "share_token": token,
        "share_url":   f"https://grok.com/share/{token}" if token else None,
    }


def upload_file(filename: str, content: Optional[bytes] = None,
                mime_type: Optional[str] = None) -> dict:
    """
    Llama grok_api.Chat/UploadFile.
    Reverse-engineered UploadFileRequest:
      f1 = file_name  (string — el servidor infiere MIME del extension)
      f2 = file_mime_type (string — override de MIME type)
      f3 = content    (bytes — contenido del archivo)
      f4 = make_public (bool)
    Tipos de archivo soportados (confirmados):
      texto/código: txt, py, json, html, css, csv, svg
      office:       xlsx, pptx, docx
      otros:        zip, mp3
    Imágenes (jpg/png/gif) requieren content real para parsear.

    Retorna dict con file_id, mime_type, storage_path, created_at.
    """
    req = _str_field(1, filename)
    if mime_type:
        req += _str_field(2, mime_type)
    if content:
        req += bytes([(3 << 3) | 2]) + _varint(len(content)) + content
    req += _bool_field(4, True)

    raw = _raw_unary("/grok_api.Chat/UploadFile", req, timeout=15)
    f   = _decode_proto(raw)

    created_at = None
    for kind, val in f.get(6, []):
        data = val if kind == 'raw' else (val.encode('latin-1') if kind == 'str' else b'')
        created_at = _ts_to_iso(data)

    return {
        "file_id":      _first_str(f, 1),
        "mime_type":    _first_str(f, 2),
        "storage_path": _first_str(f, 4),
        "created_at":   created_at,
    }


def delete_memory(conv_id: Optional[str] = None) -> dict:
    """
    Llama grok_api.Chat/DeleteMemory.
    Elimina memoria asociada a la conversación (o global si conv_id=None).
    """
    req = _str_field(1, conv_id) if conv_id else b''
    _raw_unary("/grok_api.Chat/DeleteMemory", req, timeout=10)
    return {"deleted": True}


def get_rate_limits_live(models: Optional[list[str]] = None) -> dict[str, dict]:
    """Consulta los rate limits actuales de los modelos indicados."""
    if models is None:
        models = list(MODELS_CATALOG.keys())
    try:
        ch   = get_channel()
        meta = _make_meta()
        rl   = ch.unary_unary(
            "/grok_api.RateLimits/GetRateLimits",
            request_serializer=ga.GetRateLimitsRequest.SerializeToString,
            response_deserializer=ga.GetRateLimitsResponse.FromString,
        )
        result = {}
        for m in models:
            try:
                r = rl(
                    ga.GetRateLimitsRequest(model_name=m, request_kind=0),
                    metadata=meta, timeout=5,
                )
                result[m] = {
                    "remaining":      r.remaining_queries,
                    "total":          r.total_queries,
                    "window_seconds": r.window_size_seconds,
                }
            except Exception:
                result[m] = {}
        return result
    except Exception:
        return {}
