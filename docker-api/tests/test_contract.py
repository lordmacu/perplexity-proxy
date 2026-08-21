"""Tests del contrato de capacidades y de los dos mapeos nuevos.

Todo acá es puro: nada toca la red. Las formas que se afirman NO son
inventadas -- se midieron contra la cuenta real el 2026-08-20, y los
comentarios dicen de dónde salió cada una.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import capabilities  # noqa: E402
import soniox  # noqa: E402
from routers import v1_conversations as conv  # noqa: E402


# ── El contrato ───────────────────────────────────────────────────────────────

def test_effective_declares_exactly_the_eleven_capabilities():
    keys = set(capabilities.effective(capabilities.SessionState(mode="account")))
    assert keys == set(capabilities.REQUIRED_CAPABILITIES)


def test_every_capability_is_a_bool():
    caps = capabilities.effective(capabilities.SessionState(mode="account"))
    assert all(isinstance(v, bool) for v in caps.values())


def test_anonymous_loses_every_account_capability():
    """Sin sesión no hay cookie, y sin cookie ninguna llamada upstream sirve."""
    anon = capabilities.effective(capabilities.SessionState(mode="anonymous"))
    assert not any(anon.values())


def test_account_gains_exactly_the_measured_six():
    acct = capabilities.effective(capabilities.SessionState(mode="account"))
    assert {k for k, v in acct.items() if v} == {
        "chat", "streaming", "audio_speech", "audio_transcription",
        "search", "conversations",
    }


def test_auth_block_reports_plan_as_none_not_a_guess():
    """Resolver el plan exige llamar al vendor, y /health lo tiene prohibido."""
    block = capabilities.auth_block(capabilities.SessionState(mode="account"))
    assert block == {"mode": "account", "plan": None,
                     "subscription_active": False, "expires_at": None}


# ── Conversaciones: mapeo del listado ─────────────────────────────────────────

def test_thread_to_item_maps_the_measured_field_names():
    """Nombres tomados de xll.java y confirmados contra la respuesta real."""
    item = conv._thread_to_item({
        "uuid": "a19f9a8b-aacb-40f9-a333-840c3ba48eb4",
        "title": "ping",
        "last_query_datetime": "2026-08-21T03:02:24.632322",
        "is_pinned": True,
    })
    assert item.id == "a19f9a8b-aacb-40f9-a333-840c3ba48eb4"
    assert item.title == "ping"
    assert item.updated_at == "2026-08-21T03:02:24.632322"
    assert item.pinned is True


def test_thread_to_item_does_not_invent_a_generated_title():
    """La respuesta no distingue título automático de renombrado por el usuario."""
    assert conv._thread_to_item({"uuid": "x", "title": "t"}).generated_title is None


def test_thread_to_item_survives_a_thread_with_only_a_uuid():
    item = conv._thread_to_item({"uuid": "x"})
    assert item.id == "x" and item.title is None and item.pinned is False


# ── Conversaciones: las dos capas de JSON de una respuesta ────────────────────

_REAL_TEXT = json.dumps([
    {"step_type": "INITIAL_QUERY", "content": {"goal_id": None, "query": "ping"}, "uuid": ""},
    {"step_type": "FINAL",
     "content": {"goal_id": None,
                 "answer": json.dumps({"answer": "Pong.", "chunks": ["Pong."]})},
     "uuid": ""},
])


def test_extract_answer_unwraps_both_json_layers():
    """Forma exacta medida contra el hilo real: string JSON dentro de string JSON."""
    assert conv._extract_answer(_REAL_TEXT) == "Pong."


def test_extract_answer_falls_back_to_chunks():
    text = json.dumps([{"step_type": "FINAL",
                        "content": {"answer": json.dumps({"chunks": ["Ho", "la"]})}}])
    assert conv._extract_answer(text) == "Hola"


def test_extract_answer_takes_the_last_final_step():
    text = json.dumps([
        {"step_type": "FINAL", "content": {"answer": json.dumps({"answer": "vieja"})}},
        {"step_type": "FINAL", "content": {"answer": json.dumps({"answer": "nueva"})}},
    ])
    assert conv._extract_answer(text) == "nueva"


@pytest.mark.parametrize("bad", [None, "", "   ", "no-json", "[", json.dumps({"a": 1}),
                                 json.dumps([{"step_type": "INITIAL_QUERY"}])])
def test_extract_answer_returns_empty_instead_of_raising(bad):
    """Una entry ilegible se descarta; nunca tumba el endpoint entero."""
    assert conv._extract_answer(bad) == ""


def test_entry_to_messages_produces_the_user_assistant_pair():
    msgs = conv._entry_to_messages({
        "backend_uuid": "a19f9a8b", "query_str": "ping", "text": _REAL_TEXT,
    })
    assert [(m.role, m.content) for m in msgs] == [("user", "ping"), ("assistant", "Pong.")]
    assert all(m.id == "a19f9a8b" for m in msgs)


def test_entry_to_messages_omits_an_answerless_entry():
    """Un hilo a medio responder devuelve la pregunta, no un mensaje vacío."""
    msgs = conv._entry_to_messages({"query_str": "ping", "text": ""})
    assert [(m.role, m.content) for m in msgs] == [("user", "ping")]


def test_entry_to_messages_of_an_empty_entry_is_empty():
    assert conv._entry_to_messages({}) == []


# ── Soniox ────────────────────────────────────────────────────────────────────

def test_config_carries_the_apk_values():
    """e3o.java: model stt-rt-v4, pcm_s16le, 16000 Hz, 1 canal, 2000 ms."""
    cfg = soniox.build_config("k", audio_format="pcm_s16le", language="es")
    assert cfg["model"] == "stt-rt-v4"
    assert cfg["sample_rate"] == 16000
    assert cfg["num_channels"] == 1
    assert cfg["max_endpoint_delay_ms"] == 2000
    assert cfg["language_hints"] == ["es"]


def test_config_omits_pcm_fields_for_a_container():
    """Con `auto` los trae el archivo; mandarlos sería afirmar algo del audio."""
    cfg = soniox.build_config("k", audio_format="auto")
    assert "sample_rate" not in cfg and "num_channels" not in cfg


def test_config_drops_a_language_soniox_does_not_support():
    assert soniox.build_config("k", audio_format="auto", language="klingon")["language_hints"] == []


def test_config_normalises_a_regional_tag():
    assert soniox.build_config("k", audio_format="auto", language="es-CO")["language_hints"] == ["es"]


def test_collect_strips_the_end_marker():
    """Medido: con endpoint detection, el último token final es `<end>`."""
    msgs = [{"tokens": [{"text": "Hola", "is_final": True},
                        {"text": "<end>", "is_final": True}]}]
    assert soniox._collect(msgs).text == "Hola"


def test_collect_ignores_non_final_hypotheses():
    """Los no finales son hipótesis que Soniox reemplaza; sumarlos duplicaría."""
    msgs = [{"tokens": [{"text": "Ho", "is_final": True},
                        {"text": "la mund", "is_final": False}]},
            {"tokens": [{"text": "la", "is_final": True}]}]
    assert soniox._collect(msgs).text == "Hola"


def test_collect_reports_no_language_when_soniox_sends_none():
    """stt-rt-v4 no puebla `language`; se reporta eso, no el hint del que llama."""
    assert soniox._collect([{"tokens": [{"text": "x", "is_final": True}]}]).language is None


# ── Paginación ────────────────────────────────────────────────────────────────

def test_cursor_absent_starts_at_the_beginning():
    assert conv._parse_cursor(None) == 0
    assert conv._parse_cursor("") == 0


def test_cursor_is_the_offset_of_the_next_page():
    assert conv._parse_cursor("40") == 40


@pytest.mark.parametrize("bad", ["abc", "-1", "1.5", "0x10"])
def test_a_malformed_cursor_is_a_400_not_a_crash(bad):
    with pytest.raises(Exception) as exc:
        conv._parse_cursor(bad)
    assert getattr(exc.value, "status_code", None) == 400


# ── El gate de §3.4 ───────────────────────────────────────────────────────────

def test_a_false_capability_answers_501_not_404():
    """404 es indistinguible de un error de ruteo, y 503 hace que el gateway
    reintente algo que nunca iba a funcionar en esta configuración."""
    from fastapi.testclient import TestClient
    import main

    client = TestClient(main.app)
    for method, path in (("post", "/v1/images/generations"),
                         ("post", "/v1/translate"),
                         ("post", "/v1/files"),
                         ("get", "/v1/files"),
                         ("get", "/v1/files/abc"),
                         ("delete", "/v1/files/abc")):
        assert getattr(client, method)(path).status_code == 501, path


def test_the_gate_names_the_capability_and_where_to_look():
    from fastapi.testclient import TestClient
    import main

    detail = TestClient(main.app).post("/v1/translate").json()["detail"]
    assert "translate" in detail and "/health" in detail


def test_require_passes_for_a_capability_this_proxy_has(monkeypatch):
    monkeypatch.setattr(capabilities, "snapshot",
                        lambda: capabilities.SessionState(mode="account"))
    capabilities.require("conversations")   # no raise


def test_require_refuses_when_the_session_is_gone(monkeypatch):
    """Sin sesión no hay cookie, y sin cookie el endpoint no puede lograr nada:
    501 es la respuesta honesta, no un 500 más adelante."""
    import pytest as _pytest
    from fastapi import HTTPException

    monkeypatch.setattr(capabilities, "snapshot",
                        lambda: capabilities.SessionState(mode="anonymous"))
    with _pytest.raises(HTTPException) as exc:
        capabilities.require("conversations")
    assert exc.value.status_code == 501
