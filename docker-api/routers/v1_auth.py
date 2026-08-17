import asyncio
import re
import time
from pathlib import Path
from typing import Optional

import httpx
from curl_cffi.requests import AsyncSession
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import settings

router = APIRouter()

ENV_PATH = Path(__file__).parent.parent / ".env"

PPLX_BASE = "https://www.perplexity.ai"

PPLX_HEADERS = {
    "User-Agent": "Ask-App/android",
    "x-client-name": "perplexity-android",
    "x-client-version": "2.9.5",
    "Content-Type": "application/json",
}


class RequestOtpBody(BaseModel):
    email: str


class VerifyOtpBody(BaseModel):
    email: str
    otp: str


@router.post("/auth/request-otp", summary="Pedir OTP por email a Perplexity")
async def request_otp(body: RequestOtpBody):
    """
    Llama a POST /api/auth/signin-email con useNumericOtp=true.
    Perplexity envía un código de 6 dígitos al email.
    """
    async with AsyncSession(impersonate="chrome120") as s:
        r = await s.post(
            f"{PPLX_BASE}/api/auth/signin-email",
            json={"email": body.email, "useNumericOtp": True},
            headers=PPLX_HEADERS,
            timeout=15,
        )

    if r.status_code not in (200, 201):
        raise HTTPException(status_code=r.status_code, detail=r.text[:300])

    return {"status": "otp_sent", "email": body.email, "message": "Revisa tu email"}


@router.get("/auth/poll-otp", summary="Consultar si llegó el OTP (via Cloudflare Worker)")
async def poll_otp(email: str):
    """
    Consulta el KV del Worker de Cloudflare para ver si el OTP ya llegó.
    Devuelve status=pending mientras no llega, status=ready con el otp cuando sí.
    """
    if not settings.otp_worker_url or not settings.otp_secret:
        raise HTTPException(status_code=503, detail="OTP_WORKER_URL / OTP_SECRET no configurados")

    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            f"{settings.otp_worker_url}/otp",
            params={"email": email},
            headers={"x-otp-secret": settings.otp_secret},
        )

    data = r.json()
    return data


@router.post("/auth/verify-otp", summary="Verificar OTP y obtener session cookie")
async def verify_otp(body: VerifyOtpBody):
    """
    Llama a POST /api/auth/signin-otp con email + otp.
    Si es correcto, Perplexity devuelve la session cookie en Set-Cookie.
    """
    async with AsyncSession(impersonate="chrome120") as s:
        r = await s.post(
            f"{PPLX_BASE}/api/auth/signin-otp",
            json={"email": body.email, "otp": body.otp},
            headers=PPLX_HEADERS,
            timeout=15,
        )

    # Extraer la cookie de sesión de la respuesta
    session_cookie = None
    for h_name, h_val in r.headers.items():
        if h_name.lower() == "set-cookie" and "__Secure-next-auth.session-token" in h_val:
            m = re.search(r"__Secure-next-auth\.session-token=([^;]+)", h_val)
            if m:
                session_cookie = m.group(1)
                break

    if r.status_code not in (200, 201):
        raise HTTPException(status_code=r.status_code, detail=r.text[:300])

    # Perplexity devuelve el token en el CUERPO, no siempre como Set-Cookie:
    # {"is_new_user":false,"status":"success","token":"eyJhbGciOiJkaXIi..."}
    # Sin esto el endpoint respondia "OTP incorrecto o expirado" con un OTP
    # perfectamente valido -- un diagnostico que manda a buscar donde no es.
    if not session_cookie:
        try:
            data = r.json()
        except Exception:
            data = None
        if isinstance(data, dict):
            session_cookie = (data.get("token") or data.get("session_token")
                              or data.get("sessionToken"))

    if not session_cookie:
        return {
            "status": "error",
            "message": "OTP incorrecto o expirado, o la respuesta no traia token",
            "raw_status": r.status_code,
            "body": r.text[:200],
        }

    return {
        "status": "authenticated",
        "session_token": session_cookie,
        "hint": "Agregá PERPLEXITY_SESSION=<session_token> en el .env del proxy",
    }


@router.post("/auth/auto-login", summary="Flujo completo: request OTP → esperar → verify → session")
async def auto_login(body: RequestOtpBody):
    """
    Flujo totalmente automático si el Email Worker está configurado:
    1. Pide OTP a Perplexity
    2. Espera hasta 90s polling el Worker cada 3s
    3. Verifica el OTP
    4. Devuelve la session cookie
    """
    if not settings.otp_worker_url or not settings.otp_secret:
        raise HTTPException(
            status_code=503,
            detail="OTP_WORKER_URL y OTP_SECRET requeridos para auto-login",
        )

    # Paso 1: pedir OTP
    await request_otp(body)

    # Paso 2: polling hasta 90s
    deadline = time.time() + 90
    otp = None
    while time.time() < deadline:
        await asyncio.sleep(3)
        poll = await poll_otp(body.email)
        if poll.get("status") == "ready":
            otp = poll["otp"]
            break

    if not otp:
        raise HTTPException(status_code=408, detail="Timeout: el OTP no llegó en 90 segundos")

    # Paso 3: verificar y guardar token
    result = await verify_otp(VerifyOtpBody(email=body.email, otp=otp))

    # Actualizar session en memoria y .env si login fue exitoso
    if result.get("status") == "authenticated":
        new_token = result["session_token"]
        settings.perplexity_session = new_token
        if ENV_PATH.exists():
            text = ENV_PATH.read_text()
            text = re.sub(
                r"^PERPLEXITY_SESSION=.*$",
                f"PERPLEXITY_SESSION={new_token}",
                text,
                flags=re.MULTILINE,
            )
            ENV_PATH.write_text(text)

    return result


class AuthConfigBody(BaseModel):
    email: Optional[str] = None
    check_interval: Optional[int] = None


@router.put("/auth/config", summary="Cambiar email de cuenta y/o intervalo de chequeo en runtime")
async def update_auth_config(body: AuthConfigBody):
    """
    Actualiza la configuración de auth en memoria y en .env.
    - email: nuevo email para auto-login OTP (cambia de cuenta)
    - check_interval: segundos entre chequeos del watchdog
    """
    changes = {}

    if body.email is not None:
        settings.perplexity_email = body.email
        changes["perplexity_email"] = body.email

    if body.check_interval is not None:
        settings.session_check_interval = body.check_interval
        changes["session_check_interval"] = body.check_interval

    # Persistir en .env
    if changes and ENV_PATH.exists():
        text = ENV_PATH.read_text()
        if "email" in changes:
            text = re.sub(
                r"^PERPLEXITY_EMAIL=.*$",
                f"PERPLEXITY_EMAIL={changes['perplexity_email']}",
                text,
                flags=re.MULTILINE,
            )
        if "session_check_interval" in changes:
            text = re.sub(
                r"^SESSION_CHECK_INTERVAL=.*$",
                f"SESSION_CHECK_INTERVAL={changes['session_check_interval']}",
                text,
                flags=re.MULTILINE,
            )
        ENV_PATH.write_text(text)

    return {"status": "updated", "changes": changes}


@router.get("/auth/status", summary="Estado actual del session token y configuración")
async def auth_status():
    """Muestra si el token está configurado, el email activo y el intervalo del watchdog."""
    from auth_watchdog import check_session
    valid = await check_session()
    return {
        "session_configured": bool(settings.perplexity_session and settings.perplexity_session != "your_session_token_here"),
        "session_valid": valid,
        "email": settings.perplexity_email or "(no configurado)",
        "check_interval_seconds": settings.session_check_interval,
        "worker_configured": bool(settings.otp_worker_url and settings.otp_secret),
    }
