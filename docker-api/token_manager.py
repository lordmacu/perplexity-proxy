"""
TokenManager — gestiona el token auth_tokens/... de Perplexity → Gemini Live.

Reglas:
- El token expira en ~24h (campo expires_at del servidor).
- Es de un solo uso por conexión WebSocket.
- Se cachea en disco para sobrevivir reinicios del contenedor.
- Refresco automático cuando caduca o fue consumido.
"""


from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from curl_cffi.requests import AsyncSession

from config import settings, PPLX_HEADERS

TOKEN_ENDPOINT = f"{settings.pplx_base}/rest/realtime/v1/transcription/gemini-api-key"


class TokenManager:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._api_key: str | None = None
        self._expires_at: datetime | None = None
        self._used: bool = True  # fuerza fetch en el primer get_token()

    def _cookie_header(self) -> dict:
        return {
            **PPLX_HEADERS,
            "Cookie": f"__Secure-next-auth.session-token={settings.perplexity_session}",
        }

    def _load_cache(self):
        try:
            data = json.loads(Path(settings.token_cache_path).read_text())
            self._api_key   = data["api_key"]
            self._expires_at = datetime.fromisoformat(data["expires_at"])
            self._used       = data.get("used", False)
        except Exception:
            pass

    def _save_cache(self):
        try:
            Path(settings.token_cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(settings.token_cache_path).write_text(json.dumps({
                "api_key":    self._api_key,
                "expires_at": self._expires_at.isoformat() if self._expires_at else None,
                "used":       self._used,
            }, indent=2))
        except Exception:
            pass

    def _is_valid(self) -> bool:
        if self._used or not self._api_key or not self._expires_at:
            return False
        exp = self._expires_at
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) < exp - timedelta(seconds=60)

    async def get_token(self) -> str:
        async with self._lock:
            self._load_cache()
            if self._is_valid():
                return self._api_key

            async with AsyncSession() as session:
                r = await session.post(
                    TOKEN_ENDPOINT,
                    headers=self._cookie_header(),
                    json={"source": "android", "timezone": "America/Bogota", "version": "2.95.0"},
                    impersonate="chrome120",
                    timeout=15,
                )
                r.raise_for_status()
                data = r.json()

            self._api_key    = data["api_key"]
            self._expires_at = datetime.fromisoformat(data["expires_at"])
            self._used       = False
            self._save_cache()
            return self._api_key

    def mark_used(self):
        self._used = True
        self._save_cache()
