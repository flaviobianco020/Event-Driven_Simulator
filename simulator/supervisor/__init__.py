"""
supervisor/ — Fase 4: supervisore LLM sul percorso lento (System 2).

Architettura ibrida a due livelli:
  * PERCORSO VELOCE (Fase 3, intatto): actor MAPPO, ogni 1s, deterministico.
  * PERCORSO LENTO (qui): supervisore LLM, ~30s, monitora/spiega/corregge OOD.

Lo scheletro e' DISACCOPPIATO dal motore (nessuna modifica a core.py) e vive su
un branch dedicato: additivo e scartabile.

Modello raccomandato: SLM locale (Qwen2.5-3B via Ollama) + ablation sulla
dimensione (0.5/1.5/3/7B, + Haiku come tetto). Constrained decoding garantisce
output valido anche coi modelli piccoli.
"""
from .actions import Action, Decision, DECISION_JSON_SCHEMA
from .backend import LLMBackend, MockBackend, OllamaBackend, AnthropicBackend
from .guardrail import Guardrail, GuardrailVerdict
from .controller import SupervisorController, SupervisorLog

__all__ = [
    "Action", "Decision", "DECISION_JSON_SCHEMA",
    "LLMBackend", "MockBackend", "OllamaBackend", "AnthropicBackend",
    "Guardrail", "GuardrailVerdict",
    "SupervisorController", "SupervisorLog",
]
