"""
Gemini Live via Perplexity — cliente completo con:
  - TokenManager: refresca automáticamente el token (caduca ~24h, uso único por sesión)
  - Entrada de micrófono en tiempo real (PCM 16kHz)
  - Salida de audio en tiempo real (PCM 24kHz → altavoz)
  - Transcripción de lo que dice Gemini

Uso:
  python3 gemini_live_client.py
  python3 gemini_live_client.py --voice Charon --system "Eres un chef experto."
"""

import asyncio
import argparse
import base64
import json
import struct
import time
import threading
import queue
from datetime import datetime, timezone
from pathlib import Path

import sounddevice as sd
import numpy as np
from curl_cffi import requests as cffi_requests
import websockets

# ─── Constantes ───────────────────────────────────────────────────────────────

SESSION_FILE   = Path.home() / ".perplexity_session"
TOKEN_CACHE    = Path.home() / ".perplexity_gemini_token.json"
TOKEN_ENDPOINT = "https://www.perplexity.ai/rest/realtime/v1/transcription/gemini-api-key"
GEMINI_WS      = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1alpha.GenerativeService"
    ".BidiGenerateContentConstrained"
)
MODEL          = "models/gemini-3.1-flash-live-preview"

# Audio: entrada 16kHz, salida 24kHz, mono, PCM int16
IN_RATE        = 16000
OUT_RATE       = 24000
CHUNK_MS       = 100   # ms de audio por chunk de entrada
IN_CHUNK       = IN_RATE * CHUNK_MS // 1000   # 1600 muestras


# ─── Token Manager ────────────────────────────────────────────────────────────

class TokenManager:
    """
    Gestiona el token auth_tokens/... de Perplexity → Google Gemini.

    Reglas de refresco:
    - El token caduca en ~24h (expires_at del servidor).
    - Es de un solo uso: una vez usada la conexión WS, marcar como usado.
    - Al pedir token: si está en caché, no caduca y no fue usado → reutilizar.
    - Persiste en disco para sobrevivir reinicios del proceso.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._api_key = None
        self._expires_at = None
        self._used: bool = True  # fuerza fetch inicial

    def _load_cache(self):
        try:
            data = json.loads(TOKEN_CACHE.read_text())
            self._api_key   = data["api_key"]
            self._expires_at = datetime.fromisoformat(data["expires_at"])
            self._used      = data.get("used", False)
        except Exception:
            pass

    def _save_cache(self):
        try:
            TOKEN_CACHE.write_text(json.dumps({
                "api_key":    self._api_key,
                "expires_at": self._expires_at.isoformat() if self._expires_at else None,
                "used":       self._used,
            }, indent=2))
        except Exception:
            pass

    def _is_valid(self) -> bool:
        if self._used or not self._api_key or not self._expires_at:
            return False
        # margen de 60s para evitar expiración durante la conexión
        return datetime.now(timezone.utc) < self._expires_at.replace(
            tzinfo=timezone.utc if self._expires_at.tzinfo is None else self._expires_at.tzinfo
        ) - __import__('datetime').timedelta(seconds=60)

    async def get_token(self) -> str:
        async with self._lock:
            self._load_cache()
            if self._is_valid():
                print(f"[Token] Reutilizando caché (expira {self._expires_at.strftime('%H:%M:%S')})")
                return self._api_key

            print("[Token] Solicitando nuevo token a Perplexity...")
            pplx_session = SESSION_FILE.read_text().strip()
            r = cffi_requests.post(
                TOKEN_ENDPOINT,
                headers={
                    "Cookie":       f"__Secure-next-auth.session-token={pplx_session}",
                    "Content-Type": "application/json",
                    "User-Agent":   "okhttp/4.12.0",
                },
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

            print(f"[Token] Nuevo token obtenido. Expira: {self._expires_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            return self._api_key

    def mark_used(self):
        self._used = True
        self._save_cache()


# ─── Audio helpers ────────────────────────────────────────────────────────────

def pcm_bytes_to_np(data: bytes) -> np.ndarray:
    """PCM int16 little-endian → float32 [-1, 1]"""
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def np_to_pcm_bytes(arr: np.ndarray) -> bytes:
    """float32 → PCM int16 little-endian"""
    return (arr * 32767).clip(-32767, 32767).astype("<i2").tobytes()


def save_wav(path: str, pcm_data: bytes, sample_rate: int):
    with open(path, "wb") as f:
        n = len(pcm_data)
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n))
        f.write(b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", n))
        f.write(pcm_data)


# ─── Gemini Live Session ──────────────────────────────────────────────────────

class GeminiLiveSession:
    def __init__(self, token_manager: TokenManager, voice: str = "Aoede", system_prompt: str = ""):
        self.tokens       = token_manager
        self.voice        = voice
        self.system_prompt = system_prompt or "Responde siempre en español de forma concisa y natural."

        self._mic_queue: asyncio.Queue = asyncio.Queue()
        self._play_queue: queue.Queue  = queue.Queue()
        self._stop        = False
        self._ws          = None

    # ── Mic capture (hilo separado) ──────────────────────────────────────────

    def _mic_callback(self, indata, frames, time_info, status):
        """Llamado por sounddevice en hilo de audio — pone PCM en la cola."""
        if self._stop:
            return
        pcm = np_to_pcm_bytes(indata[:, 0])
        # Poner en el event loop del hilo principal
        asyncio.get_event_loop().call_soon_threadsafe(
            self._mic_queue.put_nowait, pcm
        )

    # ── Speaker playback (hilo separado) ────────────────────────────────────

    def _playback_thread(self):
        """Lee PCM de _play_queue y lo reproduce en tiempo real."""
        with sd.OutputStream(samplerate=OUT_RATE, channels=1, dtype="int16") as stream:
            while not self._stop:
                try:
                    chunk = self._play_queue.get(timeout=0.1)
                    arr = np.frombuffer(chunk, dtype="<i2")
                    stream.write(arr)
                except queue.Empty:
                    continue

    # ── Enviar audio al WS ───────────────────────────────────────────────────

    async def _send_audio_loop(self):
        """Lee de _mic_queue y envía chunks a Gemini."""
        while not self._stop:
            try:
                pcm = await asyncio.wait_for(self._mic_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if self._ws:
                msg = {
                    "realtime_input": {
                        "media_chunks": [{
                            "mime_type": "audio/pcm;rate=16000",
                            "data": base64.b64encode(pcm).decode(),
                        }]
                    }
                }
                try:
                    await self._ws.send(json.dumps(msg))
                except Exception:
                    break

    # ── Recibir respuestas del WS ────────────────────────────────────────────

    async def _recv_loop(self):
        """Recibe mensajes de Gemini y despacha audio/texto."""
        full_transcript = ""
        while not self._stop:
            try:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=5)
            except asyncio.TimeoutError:
                continue
            except Exception:
                break

            msg = json.loads(raw)
            sc = msg.get("serverContent", {})

            if "modelTurn" in sc:
                for part in sc["modelTurn"].get("parts", []):
                    if "inlineData" in part:
                        audio = base64.b64decode(part["inlineData"]["data"])
                        self._play_queue.put(audio)

            if "outputTranscription" in sc:
                text = sc["outputTranscription"].get("text", "")
                if text:
                    print(f"\r\033[KGemini: {text}", end="", flush=True)
                    full_transcript += text

            if sc.get("turnComplete") and full_transcript:
                print()  # nueva línea al terminar el turno
                full_transcript = ""

            if "inputTranscription" in sc:
                text = sc["inputTranscription"].get("text", "")
                if text:
                    print(f"\r\033[KTú: {text}", end="", flush=True)

    # ── Sesión principal ─────────────────────────────────────────────────────

    async def run(self):
        api_key = await self.tokens.get_token()
        url = f"{GEMINI_WS}?access_token={api_key}"

        print(f"\n[*] Conectando a Gemini Live (voz: {self.voice})...")

        try:
            async with websockets.connect(url, open_timeout=15) as ws:
                self._ws = ws

                # Setup
                await ws.send(json.dumps({
                    "setup": {
                        "model": MODEL,
                        "generation_config": {
                            "response_modalities": ["AUDIO"],
                            "speech_config": {
                                "voice_config": {
                                    "prebuilt_voice_config": {"voice_name": self.voice}
                                }
                            },
                        },
                        "system_instruction": {
                            "parts": [{"text": self.system_prompt}]
                        },
                        "input_audio_transcription": {},   # pedir transcripción de entrada
                        "output_audio_transcription": {},  # pedir transcripción de salida
                    }
                }))

                # Esperar setupComplete
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                    if "setupComplete" in msg:
                        break

                self.tokens.mark_used()  # token consumido, próxima sesión pide uno nuevo
                print("[*] Conectado. Habla ahora (Ctrl+C para salir).\n")

                # Iniciar hilo de reproducción
                pb_thread = threading.Thread(target=self._playback_thread, daemon=True)
                pb_thread.start()

                # Iniciar captura de micrófono
                with sd.InputStream(
                    samplerate=IN_RATE,
                    channels=1,
                    dtype="float32",
                    blocksize=IN_CHUNK,
                    callback=self._mic_callback,
                ):
                    # Correr send + recv en paralelo
                    await asyncio.gather(
                        self._send_audio_loop(),
                        self._recv_loop(),
                    )

        except KeyboardInterrupt:
            print("\n[*] Sesión terminada por el usuario.")
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"\n[!] Conexión cerrada: {e}")
            # Si el token expiró o fue revocado, el TokenManager pedirá uno nuevo la próxima vez
        finally:
            self._stop = True
            self._ws = None


# ─── Modo texto (para probar sin micrófono) ──────────────────────────────────

class GeminiTextSession:
    """Versión simplificada: entrada por teclado, sin audio de entrada."""

    def __init__(self, token_manager: TokenManager, voice: str = "Aoede", system_prompt: str = ""):
        self.tokens        = token_manager
        self.voice         = voice
        self.system_prompt = system_prompt or "Responde siempre en español de forma concisa y natural."

    async def run(self, prompts=None):
        api_key = await self.tokens.get_token()
        url = f"{GEMINI_WS}?access_token={api_key}"
        interactive = prompts is None

        print(f"\n[*] Conectando a Gemini Live (voz: {self.voice}, modo texto)...")

        async with websockets.connect(url, open_timeout=15) as ws:
            await ws.send(json.dumps({
                "setup": {
                    "model": MODEL,
                    "generation_config": {
                        "response_modalities": ["AUDIO"],
                        "speech_config": {
                            "voice_config": {
                                "prebuilt_voice_config": {"voice_name": self.voice}
                            }
                        },
                    },
                    "system_instruction": {"parts": [{"text": self.system_prompt}]},
                    "output_audio_transcription": {},
                }
            }))

            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if "setupComplete" in msg:
                    break

            self.tokens.mark_used()
            print("[*] Conectado.\n")

            all_audio = []

            if interactive:
                prompts_iter = iter([])  # se leerá de stdin
            else:
                prompts_iter = iter(prompts)

            while True:
                if interactive:
                    try:
                        text = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: input("Tú: ")
                        )
                    except EOFError:
                        break
                    if text.lower() in ("salir", "exit", "q"):
                        break
                else:
                    try:
                        text = next(prompts_iter)
                    except StopIteration:
                        break
                    print(f"Tú: {text}")

                await ws.send(json.dumps({
                    "client_content": {
                        "turns": [{"role": "user", "parts": [{"text": text}]}],
                        "turn_complete": True,
                    }
                }))

                transcript = ""
                audio_bytes = b""
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=15)
                    except asyncio.TimeoutError:
                        break
                    msg = json.loads(raw)
                    sc = msg.get("serverContent", {})

                    if "modelTurn" in sc:
                        for part in sc["modelTurn"].get("parts", []):
                            if "inlineData" in part:
                                audio_bytes += base64.b64decode(part["inlineData"]["data"])

                    if "outputTranscription" in sc:
                        transcript += sc["outputTranscription"].get("text", "")

                    if sc.get("turnComplete"):
                        break

                print(f"Gemini [{self.voice}]: {transcript.strip()}")
                print(f"  ({len(audio_bytes):,} bytes PCM, {len(audio_bytes)/48000:.1f}s)\n")
                all_audio.append(audio_bytes)

                # Reproducir el audio si sounddevice está disponible
                try:
                    arr = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0
                    sd.play(arr, samplerate=OUT_RATE, blocking=False)
                except Exception:
                    pass

        print(f"[*] Sesión terminada. Total turnos: {len(all_audio)}")


# ─── Entrypoint ───────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice",  default="Aoede",
                        choices=["Aoede", "Charon", "Fenrir", "Zephyr"],
                        help="Voz de Gemini")
    parser.add_argument("--system", default="",
                        help="System prompt personalizado")
    parser.add_argument("--text",   action="store_true",
                        help="Modo texto (sin micrófono, entrada por teclado)")
    args = parser.parse_args()

    tokens = TokenManager()

    if args.text:
        session = GeminiTextSession(tokens, voice=args.voice, system_prompt=args.system)
        await session.run()
    else:
        session = GeminiLiveSession(tokens, voice=args.voice, system_prompt=args.system)
        await session.run()


if __name__ == "__main__":
    asyncio.run(main())
