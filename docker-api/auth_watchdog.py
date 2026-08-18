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
from datetime import datetime, timezone
from pathlib import Path

import httpx
from curl_cffi.requests import AsyncSession

from config import settings

logger = logging.getLogger("auth_watchdog")

# Vencimiento de la sesion actual, tal como lo reporta Perplexity en
# /api/auth/session. Se guarda aparte para no cambiarle la firma a
# check_session(), que tambien usa el endpoint /auth/status.
_vence_en: datetime | None = None

SESSION_CHECK_URL = "https://www.perplexity.ai/api/auth/session"
SIGNIN_URL        = "https://www.perplexity.ai/api/auth/signin-email"
VERIFY_OTP_URL    = "https://www.perplexity.ai/api/auth/signin-otp"

PPLX_HEADERS = {
    "User-Agent": "Ask-App/android",
    "x-client-name": "perplexity-android",
    "x-client-version": "2.9.5",
    "Content-Type": "application/json",
}


def _parsear_vencimiento(valor) -> datetime | None:
    """Convierte el `expires` de Perplexity a datetime, o None si no se entiende.

    Viene como ISO-8601 con Z y nanosegundos ("2026-09-16T23:28:19.936104938Z"),
    que fromisoformat no acepta: hay que recortar la fraccion a microsegundos.
    """
    if not isinstance(valor, str) or not valor:
        return None
    try:
        v = valor.replace("Z", "+00:00")
        if "." in v:
            cabeza, resto = v.split(".", 1)
            frac = "".join(ch for ch in resto if ch.isdigit())[:6]
            signo = resto[len(frac):] if len(resto) > len(frac) else ""
            for s in ("+", "-"):
                if s in resto:
                    signo = resto[resto.index(s):]
                    break
            v = f"{cabeza}.{frac}{signo or '+00:00'}"
        d = datetime.fromisoformat(v)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception as e:
        logger.warning("Sesion: no se pudo interpretar expires=%r (%s)", valor, e)
        return None


def _cookie_header() -> dict:
    return {**PPLX_HEADERS, "Cookie": f"__Secure-next-auth.session-token={settings.perplexity_session}"}


async def check_session() -> bool:
    """Devuelve True si el session token actual es válido.

    Loguea el MOTIVO de cada respuesta: un "invalido o expirado" sin causa
    obliga a adivinar entre token ausente, token rechazado y red caida, que
    son tres problemas distintos con tres arreglos distintos.
    """
    tok = settings.perplexity_session
    if not tok or tok == "your_session_token_here":
        logger.info("Sesion: no hay token configurado (PERPLEXITY_SESSION vacio)")
        return False
    logger.debug("Sesion: probando token de %d chars", len(tok))
    try:
        async with AsyncSession(impersonate="chrome120") as s:
            r = await s.get(SESSION_CHECK_URL, headers=_cookie_header(), timeout=10)
        if r.status_code != 200:
            logger.warning("Sesion: rechazada con HTTP %s — cuerpo: %s",
                           r.status_code, r.text[:120])
            return False
        data = r.json()
        ok = bool(data.get("user") or data.get("accessToken"))
        if not ok:
            # 200 pero sin identidad: el token existe y no sirve. Sin las claves
            # que SI vinieron es imposible saber si cambio el contrato.
            logger.warning("Sesion: HTTP 200 sin user ni accessToken — claves: %s",
                           sorted(data.keys()) if isinstance(data, dict) else type(data).__name__)
        else:
            global _vence_en
            _vence_en = _parsear_vencimiento(data.get("expires"))
            if _vence_en:
                faltan = (_vence_en - datetime.now(timezone.utc)).total_seconds()
                logger.info("Sesion: valida, vence en %.1f dias (%s)",
                            faltan / 86400, data.get("expires"))
            else:
                logger.info("Sesion: valida (sin fecha de vencimiento en la respuesta)")
        return ok
    except Exception as e:
        logger.warning("Sesion: no se pudo verificar (%s: %s) — se asume valida "
                       "para no disparar un re-login por un problema de red",
                       type(e).__name__, str(e)[:100])
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
        if ok:
            logger.info("OTP: pedido enviado a %s", email)
        elif r.status_code == 429:
            # El rate limit de Perplexity es la causa mas comun de que el
            # re-login falle, y es TRANSITORIO: distinguirlo de un fallo real
            # evita salir a buscar un problema que no existe.
            espera = r.headers.get("retry-after") or r.headers.get("Retry-After")
            logger.warning(
                "OTP: Perplexity limito el pedido (429). Es transitorio; "
                "reintento en el proximo ciclo (%ss)%s",
                settings.session_check_interval,
                f", el servidor sugiere esperar {espera}s" if espera else "",
            )
        else:
            logger.error("OTP: el pedido fallo con HTTP %s — cuerpo: %s",
                         r.status_code, r.text[:120])
        return ok
    except Exception as e:
        logger.error("OTP: el pedido fallo (%s: %s)", type(e).__name__, str(e)[:120])
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
                logger.warning("OTP: error consultando el Worker (%s: %s)",
                               type(e).__name__, str(e)[:100])
    # Sin este log, un OTP que nunca llega es indistinguible de un Worker caido
    # o de un email que Mailgun no entrego: tres causas, un mismo silencio.
    logger.error("OTP: no llego en %ss. Revisar (1) que Mailgun entregue a %s, "
                 "(2) que el Worker %s este vivo, (3) que OTP_SECRET coincida "
                 "entre proxy y Worker", timeout, email, settings.otp_worker_url)
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
                logger.info("OTP: token obtenido del CUERPO (%d chars), sin Set-Cookie",
                            len(tok))
                return tok
            # Logueamos las CLAVES, nunca los valores: si Perplexity renombra el
            # campo, esto lo dice en una linea en vez de costar una sesion de
            # sondeo a ciegas.
            logger.error("OTP: la respuesta no traia token. Claves recibidas: %s",
                         sorted(data.keys()))
            return None

        logger.error("OTP: sin token ni en cookie ni en cuerpo, y el cuerpo no es JSON. "
                     "Primeros bytes: %s", r.text[:150])
        return None
    except Exception as e:
        logger.error(f"OTP verify exception: {e}")
        return None


def update_session_token(new_token: str):
    """Actualiza el token en memoria, en el cache persistente y en el .env."""
    settings.perplexity_session = new_token

    # Cache persistente: es lo unico que sobrevive a un redeploy del contenedor.
    # El .env de abajo vive dentro de la imagen y se pierde en cada despliegue.
    try:
        cache = Path(settings.session_cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(new_token)
        logger.info("Sesion: token guardado en el cache persistente (%s)", cache)
    except Exception as e:
        logger.warning("Sesion: no se pudo escribir el cache %s (%s: %s) — el token "
                       "sobrevive en memoria pero NO a un reinicio",
                       settings.session_cache_path, type(e).__name__, str(e)[:80])

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

    # Enmascarado: es una credencial de un solo uso y los logs de un contenedor
    # se leen, se copian y se pegan en un chat. Los ultimos 2 digitos alcanzan
    # para correlacionar con el email si hace falta.
    logger.info("OTP: recibido del Worker (termina en %s)", otp[-2:] if len(otp) >= 2 else "??")
    new_token = await verify_otp(email, otp)
    if not new_token:
        logger.error("Verificación del OTP falló")
        return False

    update_session_token(new_token)
    logger.info("Sesion: re-login OK, token nuevo de %d chars guardado", len(new_token))
    return True


def cargar_token_del_cache() -> bool:
    """Levanta el token del cache persistente si el entorno no trae uno usable.

    Orden deliberado: el ENTORNO gana sobre el cache. Si el operador puso un
    PERPLEXITY_SESSION a mano en Coolify, esa es una decision explicita y no la
    puede pisar un archivo viejo del volumen. El cache existe para el caso
    contrario: sobrevivir a un redeploy cuando nadie toco nada.
    """
    actual = settings.perplexity_session
    if actual and actual != "your_session_token_here":
        return False
    try:
        cache = Path(settings.session_cache_path)
        if not cache.exists():
            return False
        tok = cache.read_text().strip()
        if not tok:
            return False
        settings.perplexity_session = tok
        logger.info("Sesion: token recuperado del cache persistente (%d chars) — "
                    "no hace falta un OTP nuevo", len(tok))
        return True
    except Exception as e:
        logger.warning("Sesion: no se pudo leer el cache %s (%s: %s)",
                       settings.session_cache_path, type(e).__name__, str(e)[:80])
        return False


def _dentro_del_margen() -> bool:
    """True si la sesion vence dentro del margen de renovacion."""
    if _vence_en is None:
        return False
    faltan = (_vence_en - datetime.now(timezone.utc)).total_seconds()
    return faltan <= settings.session_renew_margin_s


def _cuanto_dormir(intervalo_por_defecto: int) -> float:
    """Segundos hasta el proximo chequeo, derivados del vencimiento real.

    Sondear cada 30 min para descubrir algo que la propia respuesta ya dice
    (la sesion dura ~30 dias) son ~1400 consultas al mes para usar una. Se
    duerme hasta un dia antes de vencer.

    El TOPE existe igual: una sesion se puede revocar antes de tiempo (cambio de
    clave, cierre desde otro dispositivo), y dormir 30 dias de un tiron dejaria
    eso sin detectar hasta el final. Con el tope, lo peor es enterarse 12 h tarde.
    """
    if _vence_en is None:
        return intervalo_por_defecto
    faltan = (_vence_en - datetime.now(timezone.utc)).total_seconds() - settings.session_renew_margin_s
    dormir = max(60.0, min(float(settings.session_check_max_s), faltan))
    if faltan > settings.session_check_max_s:
        logger.info("Sesion: vence en %.1f dias; proximo chequeo en %.1f h (tope)",
                    (_vence_en - datetime.now(timezone.utc)).total_seconds() / 86400,
                    dormir / 3600)
    else:
        logger.info("Sesion: cerca del vencimiento, proximo chequeo en %.1f min", dormir / 60)
    return dormir


async def watchdog_loop():
    """Loop que corre cada SESSION_CHECK_INTERVAL segundos."""
    interval = settings.session_check_interval
    logger.info(f"AuthWatchdog iniciado — chequeando cada {interval}s")

    # Primer chequeo al arrancar (con delay de 10s para que el server esté listo)
    await asyncio.sleep(10)

    while True:
        try:
            valid = await check_session()
            if valid and _dentro_del_margen():
                # Valida pero por vencer: se renueva AHORA, con la sesion todavia
                # viva. Esperar a que caduque significa una ventana en la que el
                # proxy no sirve, y encima deja el re-login a merced del rate
                # limit del OTP justo cuando ya no hay sesion de respaldo.
                logger.info("Sesion: entra en el margen de renovacion — re-login preventivo")
                if not await auto_relogin():
                    logger.warning("Sesion: el re-login preventivo fallo; la sesion actual "
                                   "sigue siendo valida, se reintenta en el proximo ciclo")
            elif valid:
                pass  # el log del proximo chequeo lo emite _cuanto_dormir()
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

        await asyncio.sleep(_cuanto_dormir(interval))
