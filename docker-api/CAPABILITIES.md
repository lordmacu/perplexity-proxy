# Capacidades de perplexity-proxy

Este documento acompaña al contrato que publica `GET /health`
(llm-libre, `docs/superpowers/specs/2026-08-20-proxy-capability-contract-design.md`).
El código vive en [`capabilities.py`](capabilities.py); acá está el **porqué** de
cada booleano.

**La regla:** un booleano dice qué **lograría** una petición mandada ahora, ya
resuelta contra la cuenta — no qué implementa este código. Y no sigue al
medidor: una cuota agotada es un 429 con cooldown, que se recupera solo, y nunca
apaga una capacidad.

## La tabla

| Capacidad | Valor | Condición | Por qué |
|---|---|---|---|
| `chat` | ✅ | con sesión | `routers/v1_chat.py` sirve el camino síncrono contra el backend de búsqueda de Perplexity. |
| `streaming` | ✅ | con sesión | El mismo router responde `text/event-stream` cuando `stream: true`. |
| `tools` | ❌ (siempre) | — | El campo `tools` de `ChatRequest` está documentado `✗ No soportado` y nadie lo lee; ningún camino emite `tool_calls`. La emulación por inyección de prompt vive en el **gateway** (`emulates_tools` de llm-libre) — reportar `true` acá sería atribuirse trabajo ajeno. |
| `vision` | ❌ (siempre) | — | `v1_chat.py` trata `content` como texto; una parte `image_url` se descarta antes de llegar al backend. |
| `images` | ❌ (siempre) | — | No existe ruta `/v1/images/generations`. |
| `audio_speech` | ✅ | con sesión | `routers/v1_audio.py` llama a `/rest/sse/audio/text_to_speech` y devuelve MP3 real. 8 voces de Perplexity mapeadas a 12 nombres de OpenAI. |
| `audio_transcription` | ✅ | con sesión | **Nuevo, y medido.** Ver la sección de Soniox abajo — con la advertencia de que la transcripción **no es de Perplexity**. |
| `translate` | ❌ (siempre) | — | No existe ruta `/v1/translate`. |
| `search` | ✅ | con sesión | `routers/v1_search.py` está montado, y buscar es lo que Perplexity hace. |
| `files` | ❌ (siempre) | — | No existe ruta `/v1/files*`. |
| `conversations` | ✅ | con sesión | **Nuevo, y medido.** Ver la sección de conversaciones abajo. |

Ninguno de los cinco `❌` se arregla con credenciales: los cinco necesitan código
nuevo, no una cuenta mejor.

## Conversaciones (`/v1/conversations`)

Los hilos viven en los servidores de Perplexity — no hay almacenamiento local.
(El `_THREAD_CACHE` de `v1_chat.py` es otra cosa: una caché en memoria para
continuar un hilo dentro del mismo proceso, no un listado.)

El protocolo salió del **APK decompilado**, no de `perplexity_postman.json`, que
se equivoca en dos puntos comprobados:

- dice que el listado es `GET`; las anotaciones Retrofit muestran `POST` con
  cuerpo (`jkp.java:76`, 18 métodos con esa anotación, 15 de ellos con `@Body`);
- dice que `export` recibe `thread_id`; recibe `thread_uuid` + `format`
  (`q5k.java`) y devuelve `{file_content_64, filename}` (`n5k.java`) — un
  archivo en base64, no una conversación. Por eso `export` **no** se usa acá.

| Ruta nuestra | Backend | Declaración |
|---|---|---|
| `GET /v1/conversations` | `POST /rest/thread/list_ask_threads` | `jkp.java:76`, cuerpo `bkk.java`, respuesta `xll.java` |
| `GET /v1/conversations/{id}` | el mismo listado, paginando hasta el `uuid` | — |
| `GET /v1/conversations/{id}/messages` | `GET /rest/thread/{backend_uuid_or_slug}` | `cjp.java:15`, respuesta `cdk.java` |

**Por qué el detalle no usa la ruta de detalle:** `GET /rest/thread/{uuid}`
devuelve `{entries, background_entries, has_next_page, next_cursor,
thread_metadata}`, y ese `thread_metadata` solo contiene `crons` (`nml.java`).
No trae título, ni fecha, ni estado de pin — los únicos campos que promete
`ConversationItem` están en el listado.

**Paginación, y por qué casi se pierde.** El cuerpo lleva `limit` y `offset`
(`bkk.java:16-17`), y funcionan: dos páginas contiguas de 5 devolvieron uuids
disjuntos. Casi se me escapan — jadx los dejó como referencias a constantes
(`MapboxMap.QFE_LIMIT` / `QFE_OFFSET`) en vez de literales, así que la primera
lectura del descriptor solo mostró cinco de los siete campos. Sin ellos el
endpoint devolvía siempre los mismos 20 hilos **y afirmaba que eran todos**:
`has_next_page` venía `true` en los 20. `next_cursor` es el offset de la página
siguiente, opaco para quien llama.

**Dos capas de JSON.** Las entries no tienen esquema estático: `szb.java` las
deserializa como `Map<String, JsonElement>` y el cliente Android las interpreta
en runtime. Midiendo contra un hilo real, la respuesta del asistente está en
`text`, que es un **string** con una lista de pasos adentro; el paso `FINAL`
tiene `content.answer`, que es **otro** string con `{"answer": ..., "chunks":
[...]}`. `_extract_answer` desenvuelve las dos.

La forma de respuesta replica la de **mistral-proxy**
(`api/routes/conversations.py`) a propósito: el gateway debe leer una sola forma
para los cinco proxies.

**Medido el 2026-08-20** contra la cuenta real: 20 hilos listados, y el par
usuario/asistente correctamente extraído de uno de ellos.

## Transcripción (`/v1/audio/transcriptions`)

**Esto es distinto de todo lo demás en este proxy: la transcripción no la hace
Perplexity, la hace Soniox** — un tercero, con su propia cuota y su propia
disponibilidad. Perplexity solo emite la credencial de vida corta
(`POST /rest/realtime/v1/transcription/soniox-api-key`).

Consecuencia operativa: si Soniox se cae o deja de honrar estos tokens,
`audio_transcription` es el booleano que hay que poner en `false`, y **nada en
el estado de Perplexity lo avisaría**.

Protocolo del APK v2.95.0:

| Qué | Dónde |
|---|---|
| `wss://stt-rt.soniox.com/transcribe-websocket` | `e3o.java:23` |
| `model=stt-rt-v4`, `audio_format=pcm_s16le`, `sample_rate=16000` | `e3o.java:20-25` |
| El frame de config va como **texto**, al abrir, antes del audio | `a4o.java:286` |
| `max_endpoint_delay_ms=2000` con endpoint detection activo | `a4o.java:286` |
| Campos del frame de config y cuáles son opcionales | `x2o.java` |
| Cada token: `text, start_ms, end_ms, confidence, is_final, speaker, language` | `o3o.java` |
| Cada mensaje: `tokens[], finished, final_audio_proc_ms, total_audio_proc_ms` | `k3o.java` |

**Dos cosas NO salen del APK**, y están marcadas en `soniox.py` donde se usan:

1. `audio_format="auto"`. El APK solo prueba `pcm_s16le`, porque transmite el
   micrófono en crudo y ahí no hay contenedor que detectar. Para un archivo
   subido hace falta que Soniox lo detecte. **Medido: funciona.**
2. El fin del audio se señala con un frame de texto vacío. La app transmite en
   vivo y nunca termina el audio, solo cierra el socket. **Medido: funciona.**

**Medido el 2026-08-20:** un WAV de ~5 s en español volvió como
`"Hola, esto es una prueba de transcripción."` por **ambos** caminos (`auto` y
`pcm_s16le`).

Dos cosas que solo aparecieron midiendo, y que el código ahora maneja:

- el token de Perplexity viene en `api_key`, no en `apiKey` — la documentación
  del propio proxy lo decía mal, y se corrigió;
- con endpoint detection, Soniox cierra con un token final cuyo texto es
  literalmente `<end>`; concatenarlo dejaba `"...transcripción.<end>"`.

`language` se reporta como `null` casi siempre, y es correcto: aunque `o3o.java`
declara `language` y `speaker` por token, `stt-rt-v4` con esta configuración
solo devuelve `text`, `start_ms`, `end_ms`, `confidence` e `is_final`. Se
reporta lo que Soniox detecta, no el `language` que pidió quien llama.

## `GET /health`

```json
{
  "status": "ok",
  "version": "1.0.0",
  "contract": 1,
  "provider": "perplexity",
  "auth": {"mode": "account", "plan": null,
           "subscription_active": false, "expires_at": null},
  "capabilities": { ... los once booleanos ... }
}
```

Sin autenticación y **sin llamar al vendor** (spec §3.1): es el objetivo del
health sweep del gateway y también el healthcheck del contenedor, y los dos
tienen que responder aunque Perplexity esté caído. `status: "ok"` se mantuvo
porque era toda la respuesta antes del contrato.

`plan` es `null` a propósito. Perplexity **sí** vende Pro, pero saber qué plan
tiene esta sesión exige una llamada al vendor, y `/health` la tiene prohibida.
Poner `"free"` o `"pro"` sin preguntar sería exactamente la clase de mentira que
este contrato existe para terminar.

## Lo que falta (§3.4)

La spec exige que un endpoint cuya capacidad es `false` responda **`501 Not
Implemented`**, no `404`. Este proxy **todavía no**: `/v1/images/generations`,
`/v1/translate` y `/v1/files*` simplemente no existen como rutas, así que
FastAPI devuelve su `404` genérico. Los booleanos de `/health` ya son correctos;
el código de estado que vería un cliente real todavía no cumple la letra de
§3.4. Cerrarlo es trabajo aparte.
