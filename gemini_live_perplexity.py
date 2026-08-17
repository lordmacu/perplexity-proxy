"""
Gemini Live via Perplexity auth_tokens
- Obtiene token de /rest/realtime/v1/transcription/gemini-api-key
- Conecta a BidiGenerateContentConstrained con access_token=auth_tokens/...
- Modelo: models/gemini-3.1-flash-live-preview
"""

import asyncio
import json
import base64
import sys
from curl_cffi import requests as cffi_requests
import websockets

SESSION_FILE = "/Users/cristian/.perplexity_session"
GEMINI_WS = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1alpha.GenerativeService"
    ".BidiGenerateContentConstrained"
)
MODEL = "models/gemini-3.1-flash-live-preview"


def get_gemini_token(pplx_session: str) -> dict:
    r = cffi_requests.post(
        "https://www.perplexity.ai/rest/realtime/v1/transcription/gemini-api-key",
        headers={
            "Cookie": f"__Secure-next-auth.session-token={pplx_session}",
            "Content-Type": "application/json",
            "User-Agent": "okhttp/4.12.0",
        },
        json={"source": "android", "timezone": "America/Bogota", "version": "2.95.0"},
        impersonate="chrome120",
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


async def gemini_chat(prompts: list[str], voice: str = "Aoede"):
    pplx_session = open(SESSION_FILE).read().strip()

    print(f"[*] Obteniendo token Gemini de Perplexity...")
    token_data = get_gemini_token(pplx_session)
    api_key = token_data["api_key"]
    expires_at = token_data.get("expiresAt", "desconocido")
    print(f"[*] Token: {api_key[:50]}... (expira: {expires_at})")

    url = f"{GEMINI_WS}?access_token={api_key}"

    async with websockets.connect(url, open_timeout=15) as ws:
        # Setup
        setup = {
            "setup": {
                "model": MODEL,
                "generation_config": {
                    "response_modalities": ["AUDIO"],
                    "speech_config": {
                        "voice_config": {
                            "prebuilt_voice_config": {"voice_name": voice}
                        }
                    },
                },
                "system_instruction": {
                    "parts": [{"text": "Eres un asistente amigable. Responde en español de forma concisa."}]
                },
            }
        }
        await ws.send(json.dumps(setup))

        # Esperar setupComplete
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            if "setupComplete" in msg:
                print(f"[*] Setup completo. Iniciando conversación...")
                break

        total_audio_bytes = 0

        for i, prompt in enumerate(prompts):
            print(f"\n[Turno {i+1}] Usuario: {prompt}")

            await ws.send(json.dumps({
                "client_content": {
                    "turns": [{"role": "user", "parts": [{"text": prompt}]}],
                    "turn_complete": True,
                }
            }))

            # Recopilar respuesta completa
            audio_chunks = []
            full_transcript = ""
            turn_done = False

            while not turn_done:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=15)
                except asyncio.TimeoutError:
                    print("  [timeout esperando respuesta]")
                    break

                msg = json.loads(raw)

                if "serverContent" in msg:
                    sc = msg["serverContent"]

                    if "modelTurn" in sc:
                        for part in sc["modelTurn"].get("parts", []):
                            if "inlineData" in part:
                                chunk = base64.b64decode(part["inlineData"]["data"])
                                audio_chunks.append(chunk)
                            if "text" in part:
                                full_transcript += part["text"]

                    if "outputTranscription" in sc:
                        full_transcript += sc["outputTranscription"].get("text", "")

                    if sc.get("turnComplete"):
                        turn_done = True

                elif "sessionResumptionUpdate" in msg:
                    handle = msg["sessionResumptionUpdate"].get("newHandle", "")
                    print(f"  [session handle: {handle[:20]}...]")

            audio_data = b"".join(audio_chunks)
            total_audio_bytes += len(audio_data)
            print(f"  Gemini: {full_transcript.strip() or '(sin texto)'}")
            print(f"  Audio: {len(audio_data):,} bytes PCM @ 24kHz ({len(audio_data)/48000:.1f}s)")

            # Guardar audio del primer turno como WAV para verificar
            if i == 0 and audio_data:
                wav_path = "/tmp/gemini_response.wav"
                import struct
                with open(wav_path, "wb") as f:
                    # WAV header para PCM 16-bit mono 24kHz
                    data_size = len(audio_data)
                    f.write(b"RIFF")
                    f.write(struct.pack("<I", 36 + data_size))
                    f.write(b"WAVE")
                    f.write(b"fmt ")
                    f.write(struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16))
                    f.write(b"data")
                    f.write(struct.pack("<I", data_size))
                    f.write(audio_data)
                print(f"  Guardado en: {wav_path}")

        print(f"\n[*] Total audio recibido: {total_audio_bytes:,} bytes")


if __name__ == "__main__":
    prompts = [
        "Hola, ¿cómo estás?",
        "¿Cuál es la capital de Francia?",
        "Hasta luego.",
    ]
    asyncio.run(gemini_chat(prompts, voice="Aoede"))
