"""
AuthWatchdog — verifica periódicamente que el session token de Perplexity
siga vivo. Si no, ejecuta el flujo OTP automático para reautenticar.

Corre como tarea asyncio en background, arranca en el lifespan de FastAPI.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path

import httpx
from curl_cffi.requests import AsyncSession

from config import settings

logger = logging.getLogger("auth_watchdog")

SESSION_CHECK_URL = "https://www.perplexity.ai/api/auth/session"
SIGNIN_URL        = "https://www.perplexity.ai/api/auth/signin-email"
VERIFY_OTP_URL    = "https://www.perplexity.ai/api/auth/signin-otp"

PPLX_HEADERS = {
    "User-Agent": "Ask-App/android",
    "x-client-name": "perplexity-android",
    "x-client-version": "2.9.5",
    "Content-Type": "application/json",
}


def _cookie_header() -> dict:
    return {**PPLX_HEADERS, "Cookie": f"__Secure-next-auth.session-token={settings.perplexity_session}"}


async def check_session() -> bool:
    """Devuelve True si el session token actual es válido."""
    if not settings.perplexity_session or settings.perplexity_session == "your_session_token_here":
        return False
    try:
        async with AsyncSession(impersonate="chrome120") as s:
            r = await s.get(SESSION_CHECK_URL, headers=_cookie_header(), timeout=10)
        if r.status_code != 200:
            logger.warning(f"Session check HTTP {r.status_code}")
            return False
        data = r.json()
        return bool(data.get("user") or data.get("accessToken"))
    except Exception as e:
        logger.warning(f"Session check failed: {e}")
        return False  # no disparar re-login por error de red


async def request_otp(email: str) -> bool:
    """Pide a Perplexity que envíe un OTP al email."""
    try:
        async with AsyncSession(impersonate="chrome120") as s:
            r = await s.post(
                SIGNIN_URL,
                json={"email": email, "useNumericOtp": True},
                headers=PPLX_HEADERS,
                timeout=15,
            )
        ok = r.status_code in (200, 201)
        logger.info(f"OTP request → {r.status_code}: {r.text[:80]}")
        return ok
    except Exception as e:
        logger.error(f"OTP request failed: {e}")
        return False


async def poll_otp(email: str, timeout: int = 90) -> str | None:
    """Espera hasta timeout segundos a que el Worker deposite el OTP en KV."""
    if not settings.otp_worker_url or not settings.otp_secret:
        logger.error("OTP_WORKER_URL / OTP_SECRET no configurados — no se puede hacer auto-login")
        return None

    deadline = time.time() + timeout
    async with httpx.AsyncClient(timeout=10) as client:
        while time.time() < deadline:
            await asyncio.sleep(4)
            try:
                r = await client.get(
                    f"{settings.otp_worker_url}/otp",
                    params={"email": email},
                    headers={"x-otp-secret": settings.otp_secret},
                )
                data = r.json()
                if data.get("status") == "ready":
                    return data["otp"]
            except Exception as e:
                logger.warning(f"Poll OTP error: {e}")
    return None


async def verify_otp(email: str, otp: str) -> str | None:
    """Verifica el OTP y devuelve el nuevo session token, o None si falla."""
    try:
        async with AsyncSession(impersonate="chrome120") as s:
            r = await s.post(
                VERIFY_OTP_URL,
                json={"email": email, "otp": otp},
                headers=PPLX_HEADERS,
                timeout=15,
            )

        if r.status_code not in (200, 201):
            logger.error(f"OTP verify failed: {r.status_code} {r.text[:200]}")
            return None

        for h_name, h_val in r.headers.items():
            if h_name.lower() == "set-cookie" and "__Secure-next-auth.session-token" in h_val:
                m = re.search(r"__Secure-next-auth\.session-token=([^;]+)", h_val)
                if m:
                    return m.group(1)

        # Perplexity devuelve el token en el CUERPO, no siempre como Set-Cookie:
        # {"is_new_user":false,"status":"success","token":"eyJhbGciOiJkaXIi..."}
        # Verificado el 2026-08-17: el OTP se pedia, el Worker lo capturaba y la
        # verificacion respondia 200 con el token adentro, pero se descartaba por
        # buscar solo la cookie -- el re-login fallaba con todo lo demas correcto.
        try:
            data = r.json()
        except Exception:
            data = None
        if isinstance(data, dict):
            tok = data.get("token") or data.get("session_token") or data.get("sessionToken")
            if tok:
                logger.info("OTP verify: token tomado del cuerpo (sin Set-Cookie)")
                return tok

        logger.error(f"OTP verify: sin token ni en cookie ni en cuerpo. Body: {r.text[:200]}")
        return None
    except Exception as e:
        logger.error(f"OTP verify exception: {e}")
        return None


def update_session_token(new_token: str):
    """Actualiza el token en memoria y en el .env."""
    settings.perplexity_session = new_token

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        text = env_path.read_text()
        text = re.sub(
            r"^PERPLEXITY_SESSION=.*$",
            f"PERPLEXITY_SESSION={new_token}",
            text,
            flags=re.MULTILINE,
        )
        env_path.write_text(text)
        logger.info("PERPLEXITY_SESSION actualizado en .env")


async def auto_relogin() -> bool:
    """Ejecuta el flujo completo OTP para reautenticar. Devuelve True si tuvo éxito."""
    email = settings.perplexity_email
    if not email:
        logger.error("PERPLEXITY_EMAIL no configurado — no se puede hacer auto-login")
        return False

    logger.info(f"Iniciando auto-login OTP para {email}")

    if not await request_otp(email):
        logger.error("Falló el envío del OTP")
        return False

    otp = await poll_otp(email)
    if not otp:
        logger.error("OTP no llegó en el tiempo límite")
        return False

    logger.info(f"OTP recibido: {otp}")
    new_token = await verify_otp(email, otp)
    if not new_token:
        logger.error("Verificación del OTP falló")
        return False

    update_session_token(new_token)
    logger.info("Re-autenticación exitosa — nuevo session token guardado")
    return True


async def watchdog_loop():
    """Loop que corre cada SESSION_CHECK_INTERVAL segundos."""
    interval = settings.session_check_interval
    logger.info(f"AuthWatchdog iniciado — chequeando cada {interval}s")

    # Primer chequeo al arrancar (con delay de 10s para que el server esté listo)
    await asyncio.sleep(10)

    while True:
        try:
            valid = await check_session()
            if valid:
                logger.info("Session token OK")
            else:
                logger.warning("Session token inválido o expirado — iniciando re-login")
                success = await auto_relogin()
                if not success:
                    logger.error("Re-login fallido — se reintentará en el próximo ciclo")
        except asyncio.CancelledError:
            logger.info("AuthWatchdog detenido")
            break
        except Exception as e:
            logger.error(f"AuthWatchdog error inesperado: {e}")

        await asyncio.sleep(interval)
