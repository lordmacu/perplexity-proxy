"""What this proxy can actually do right now.

Spec: the proxy capability contract, llm-libre
docs/superpowers/specs/2026-08-20-proxy-capability-contract-design.md

THE RULE: a boolean says what a request sent right now would ACHIEVE, not what
this codebase implements. Where the two differ, the endpoint is the liar and
this module is the correction (spec 3.2).

Where the rule STOPS: a boolean tracks entitlement, not the meter. A quota
running out is a 429 the gateway already handles with a cooldown and recovers
from on its own; it must never flip a capability off. The dividing line is
durability -- if a fresh request tomorrow would still be refused for the same
reason, it belongs in the boolean.

Perplexity sells Pro, but this proxy cannot know whether the session behind it
has one without asking the vendor, and `GET /health` must answer with no vendor
call at all (spec 3.1). So `snapshot()` is a local read of one thing --
`PERPLEXITY_SESSION` -- and `auth_block()` reports `plan: None` rather than
guessing. Same shape as deepseek's, and for the same reason: a placeholder here
would be the class of lie this contract exists to end.

UNLIKE deepseek's module, most of what follows WAS measured against the live
account on 2026-08-20, not just read off the code. `conversations` and
`audio_transcription` were exercised end to end before their booleans were set
to True; `effective()` says per capability which ones those are.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import settings

REQUIRED_CAPABILITIES = (
    "chat", "streaming", "tools", "vision", "images",
    "audio_speech", "audio_transcription", "translate",
    "search", "files", "conversations",
)


@dataclass(frozen=True)
class SessionState:
    mode: str          # "account" | "anonymous"


def snapshot() -> SessionState:
    """Read local credentials. No lock, no cache, no vendor call.

    `PERPLEXITY_SESSION` is the whole story here: every upstream call in this
    proxy sends it as the `__Secure-next-auth.session-token` cookie, and none
    of them work without it. There is no token file to fall behind the env var
    the way deepseek has, so there is nothing else to check.

    A present-but-expired session still reads as "account". That is deliberate:
    expiry is recoverable -- `auth_watchdog` re-logs in on its own -- and the
    contract must not flip capabilities off for a condition that heals itself
    (spec 3.2). A revoked account would keep reporting `account` too; that case
    is rare enough not to justify a vendor call on every health sweep.
    """
    session = (settings.perplexity_session or "").strip()
    return SessionState(mode="account" if session else "anonymous")


def auth_block(state: SessionState) -> dict:
    """The contract's informational `auth` block.

    `plan` is None on purpose, not unknown-by-omission: Perplexity DOES sell
    Pro, but resolving which plan this session holds needs a vendor call, and
    `/health` is forbidden one (spec 3.1). Reporting "free" or "pro" without
    asking would be a guess dressed as a fact.
    """
    return {"mode": state.mode, "plan": None,
            "subscription_active": False, "expires_at": None}


def effective(state: SessionState) -> dict:
    """The eleven booleans, as of 2026-08-20.

    MEASURED against the live account (not merely read off the code):
      `conversations` -- True. `GET /v1/conversations` returned 20 real
        threads; `GET /v1/conversations/{id}/messages` returned the
        user/assistant pair for one of them, correctly decoded through the
        two layers of JSON that Perplexity stores an answer behind. See
        routers/v1_conversations.py.
      `audio_transcription` -- True. A 5-second Spanish WAV came back as
        "Hola, esto es una prueba de transcripción." over BOTH paths --
        `audio_format="auto"` for the container and the APK's native
        `pcm_s16le`. Note what this capability actually depends on: the
        transcription is SONIOX's, not Perplexity's. Perplexity only issues
        the short-lived credential. If Soniox goes down or stops honouring
        these tokens, this boolean is the one that has to go False, and
        nothing in Perplexity's own status would tell us.

    READ OFF THE CODE (no live call made while writing this):
      `chat` / `streaming` -- True with an account. routers/v1_chat.py serves
        both the sync and the `text/event-stream` path.
      `search` -- True. routers/v1_search.py is mounted, and search is what
        Perplexity is.
      `audio_speech` -- True. routers/v1_audio.py POSTs to
        `/rest/sse/audio/text_to_speech` and returns real MP3 bytes.
      `tools` -- False. The `tools` field on ChatRequest is documented
        "✗ No soportado" and nothing reads it; no code path emits
        `tool_calls`. Prompt-injection emulation lives in the gateway
        (`emulates_tools`), and claiming it here would take credit for the
        gateway's work.
      `vision` -- False. v1_chat.py handles `content` as text only; an
        `image_url` part is dropped before it reaches the backend.
      `images` -- False. No `/v1/images/generations` route.
      `translate` -- False. No `/v1/translate` route.
      `files` -- False. No `/v1/files*` route.

    Everything False above needs new code, not new credentials -- no account
    upgrade flips any of them.
    """
    live = state.mode == "account"
    return {
        "chat":                live,
        "streaming":           live,
        "tools":               False,
        "vision":              False,
        "images":              False,
        "audio_speech":        live,
        "audio_transcription": live,
        "translate":           False,
        "search":              live,
        "files":               False,
        "conversations":       live,
    }
