"""
backend.py — Interfaccia LLM MODEL-AGNOSTIC per il supervisore (Fase 4).

Un solo metodo: decide(context) -> dict conforme a DECISION_JSON_SCHEMA.
Swappare il modello = cambiare la classe di backend (o il nome modello in
Ollama). Questo rende BANALE l'ablation sulla dimensione richiesta dalla tesi:
    Qwen2.5 0.5B / 1.5B / 3B / 7B  (+ Haiku come tetto)

Constrained decoding: i backend passano DECISION_JSON_SCHEMA al modello, cosi'
l'output e' sempre un JSON valido — anche per i modelli piccoli. Il modello
sceglie QUALE azione, non il formato.

Backend inclusi:
    MockBackend    — nessun modello, regole stub. Fa girare lo scheletro su
                     qualunque macchina (nessuna dipendenza). Default per i test.
    OllamaBackend  — modello locale via Ollama HTTP (Qwen2.5-3B raccomandato).
                     Riproducibile, offline: l'argomento di credibilita' tesi.
    AnthropicBackend — Claude Haiku 4.5 (braccio di confronto qualita', opzionale).
"""
from __future__ import annotations
import json
import urllib.request
from abc import ABC, abstractmethod

from .actions import Action, DECISION_JSON_SCHEMA


class LLMBackend(ABC):
    """Contratto minimo: dato un contesto, ritorna un dict conforme allo schema."""

    name: str = "abstract"

    @abstractmethod
    def decide(self, context: dict, system_prompt: str, user_prompt: str,
               schema: dict | None = None) -> dict:
        # schema=None → DECISION_JSON_SCHEMA (retrocompatibile con M1/M2/M3).
        # Un altro schema (es. escalation) consente output strutturati diversi.
        ...


# ── Mock: nessun modello, fa girare lo scheletro ovunque ─────────────────────────
class MockBackend(LLMBackend):
    """
    Backend deterministico senza LLM. Regola stub: se il drop rate della finestra
    supera una soglia, suggerisce di forzare la compressione (override); altrimenti
    endorse. Serve a: (a) far girare lo scheletro/test senza modelli installati,
    (b) essere la baseline "nessun ragionamento" nell'ablation.
    """
    name = "mock"

    def __init__(self, drop_threshold: float = 0.15):
        self.drop_threshold = drop_threshold

    def decide(self, context: dict, system_prompt: str, user_prompt: str,
               schema: dict | None = None) -> dict:
        m = context.get("metrics", {})
        drop = float(m.get("drop_rate", 0.0))
        if drop > self.drop_threshold:
            return {
                "action": Action.OVERRIDE_STATE.value,
                "target_state": 3,
                "hold_seconds": 30.0,
                "justification": (f"[mock] drop rate {drop:.2f} > soglia "
                                  f"{self.drop_threshold}: forzo INCREMENTAL per drenare la coda."),
            }
        return {"action": Action.ENDORSE.value,
                "justification": "[mock] metriche nella norma: nessun intervento."}


# ── Ollama: modello locale (primario) ────────────────────────────────────────────
class OllamaBackend(LLMBackend):
    """
    Modello locale via Ollama (http://localhost:11434). Usa `format` = schema JSON
    per il constrained decoding. Modello raccomandato: 'qwen2.5:3b'.
    temperature=0 + seed → deterministico per la riproducibilita' della tesi.

    Prerequisito:  ollama pull qwen2.5:3b
    """
    name = "ollama"

    def __init__(self, model: str = "qwen2.5:3b",
                 host: str = "http://localhost:11434", seed: int = 0,
                 timeout: float = 30.0):
        self.model = model
        self.host = host.rstrip("/")
        self.seed = seed
        self.timeout = timeout

    def decide(self, context: dict, system_prompt: str, user_prompt: str,
               schema: dict | None = None) -> dict:
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema or DECISION_JSON_SCHEMA,   # constrained decoding
            "options": {"temperature": 0.0, "seed": self.seed},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read())
        # Ollama ritorna il JSON conforme nel campo message.content (stringa JSON).
        return json.loads(body["message"]["content"])


# ── Anthropic: Claude Haiku (tetto di confronto, opzionale) ──────────────────────
class AnthropicBackend(LLMBackend):
    """
    Braccio di confronto 'capacita' del supervisore'. Usa l'SDK anthropic se
    installato + ANTHROPIC_API_KEY. Structured output via output_config.format.
    Non necessario per il core della tesi (rompe offline/riproducibilita'): usato
    solo come tetto qualitativo nell'ablation.
    """
    name = "anthropic"

    def __init__(self, model: str = "claude-haiku-4-5"):
        self.model = model

    def decide(self, context: dict, system_prompt: str, user_prompt: str,
               schema: dict | None = None) -> dict:
        import anthropic  # import lazy: dipendenza opzionale
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={"format": {"type": "json_schema",
                                      "schema": schema or DECISION_JSON_SCHEMA}},
        )
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)
