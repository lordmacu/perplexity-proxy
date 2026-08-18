from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    perplexity_session: str = ""
    proxy_api_key: str = ""
    default_voice: str = "Aoede"
    token_cache_path: str = "/app/cache/gemini_token.json"
    # El token de sesion se cachea en el MISMO volumen que el de gemini.
    # Sin esto, un redeploy pierde el token y dispara un OTP nuevo -- y el
    # rate limit de Perplexity convierte eso en un bloqueo de horas. La
    # sesion dura 30 dias (verificado 2026-08-17), asi que perderla por un
    # reinicio es puro desperdicio.
    session_cache_path: str = "/app/cache/session_token.txt"
    # La sesion trae su propia fecha de vencimiento (~30 dias), asi que el
    # watchdog duerme HASTA ella en vez de sondear a ciegas cada media hora.
    session_renew_margin_s: int = 86400   # renovar 1 dia antes de vencer
    session_check_max_s: int = 43200      # pero nunca dormir mas de 12 h de una
    pplx_base: str = "https://www.perplexity.ai"
    gemini_ws: str = (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1alpha.GenerativeService"
        ".BidiGenerateContentConstrained"
    )
    gemini_model: str = "models/gemini-3.1-flash-live-preview"
    otp_worker_url: str = ""
    otp_secret: str = ""
    mailgun_key: str = ""
    perplexity_email: str = ""          # email para auto-login OTP
    session_check_interval: int = 1800  # segundos entre chequeos (default 30 min)
    # Interruptor del re-login automatico por OTP. Encendido por defecto para no
    # cambiar el comportamiento existente; ponerlo en false cuando Perplexity este
    # rate-limitando los OTP (reintentar solo empeora el bloqueo) o cuando se quiera
    # correr el proxy en modo anonimo, que no necesita sesion.
    auth_watchdog: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()

PPLX_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "okhttp/4.12.0",
}

# OpenAI voice name → Perplexity TTS preset
VOICE_MAP: dict[str, str] = {
    "alloy":   "Tylis-mp3",
    "echo":    "Gravo-mp3",
    "ash":     "Torma-mp3",
    "ballad":  "Mylva-mp3",
    "coral":   "Syla-mp3",
    "sage":    "Solva-mp3",
    "cedar":   "Kyrin-mp3",
    "marin":   "Velox-mp3",
    # fallbacks for unsupported OAI voices
    "fable":   "Kyrin-mp3",
    "onyx":    "Gravo-mp3",
    "nova":    "Syla-mp3",
    "shimmer": "Velox-mp3",
}

# Gemini voice names (for /v1/chat/completions x-voice extension)
GEMINI_VOICES = {"Aoede", "Charon", "Fenrir", "Zephyr"}
