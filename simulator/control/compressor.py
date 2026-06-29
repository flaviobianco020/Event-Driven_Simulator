"""
CompressionEngine — Phase 2 packet-size reduction model.

Source: eFRAC paper (Abate, Sacco, Fiore, Esposito) Table 1 + ablation.
All ratios = compressed_size / original_size.

Compression Fusion (paper §3.2): Delta and Incremental stack on top of HC as
base layer — the ratios below are all-inclusive (already include HC savings).

Traffic-class mapping (priority_level):
  0 = CONTROL   ~100 B  fixed-format messages
  1 = TELEMETRY ~250 B  structured sensor data (MQTT-like)
  2 = VIDEO    ~1450 B  binary ISR / media frames
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from ..network.congestion import CongestionState

if TYPE_CHECKING:
    from ..traffic.packet import Packet

# (CongestionState, priority_level) → compressed_size / original_size
#
# HEADER_COMPRESSION:
#   IP/UDP header 28 B → compact 4 B (saves 24 B flat, format-agnostic).
#   Paper: 1.02× on 987 B CoT — negligible for large, significant for small.
#   ratio = (size - 24) / size  (clamped per priority below)
#
# DELTA_COMPRESSION (HC + XOR + zlib):
#   Paper: ~987 B → ~675 B (1.5×) on CoT XML with high temporal redundancy.
#   CONTROL: fixed-format → XOR produces near-all-zeros → good zlib ratio.
#   VIDEO: paper measured ~1.5× on binary large payloads.
#
# INCREMENTAL_COMPRESSION (HC + semantic field-level diff):
#   Paper: 987 B → 163 B avg (6.1×), 70 B best (14.1×) on CoT XML.
#   Only effective on structured/XML data (TELEMETRY).
#   VIDEO is binary → parser cannot extract fields → falls back to Delta ratio.
#
# DROP_LOW_PRIORITY:
#   priority > 0 packets are dropped by the core before reaching the compressor.
#   priority = 0 (CONTROL) gets INCREMENTAL compression (paper: keepalive only).
_RATIOS: dict[tuple[CongestionState, int], float] = {
    # ── NORMAL ──────────────────────────────────────────────────────────────────
    (CongestionState.NORMAL, 0): 1.00,
    (CongestionState.NORMAL, 1): 1.00,
    (CongestionState.NORMAL, 2): 1.00,

    # ── HEADER_COMPRESSION ──────────────────────────────────────────────────────
    # Saves exactly 24 B (28 B IP/UDP → 4 B compact header).
    # Paper §3.2: "fixed saving of 24 B per IPv4/UDP packet … format-agnostic"
    (CongestionState.HEADER_COMPRESSION, 0): 0.760,  # (100-24)/100
    (CongestionState.HEADER_COMPRESSION, 1): 0.904,  # (250-24)/250
    (CongestionState.HEADER_COMPRESSION, 2): 0.983,  # (1450-24)/1450

    # ── DELTA_COMPRESSION ───────────────────────────────────────────────────────
    # HC + XOR + zlib. Paper Table 1: 1.5× on CoT XML payloads.
    # CONTROL: small fixed messages → XOR near-zeros → zlib ≈ 0.55
    # TELEMETRY: sensor telemetry with periodic updates → 0.50
    # VIDEO: large binary; paper ablation shows ~1.5× → 0.667
    (CongestionState.DELTA_COMPRESSION, 0): 0.550,
    (CongestionState.DELTA_COMPRESSION, 1): 0.500,
    (CongestionState.DELTA_COMPRESSION, 2): 0.667,

    # ── INCREMENTAL_COMPRESSION ─────────────────────────────────────────────────
    # HC + semantic field-level diff. Paper Table 1: 6.1× avg on CoT XML.
    # CONTROL: small structured → modest improvement over Delta → 0.50
    # TELEMETRY: XML-like structured → significant semantic savings → 0.25 (~4×)
    # VIDEO: binary, parser cannot extract fields → falls back to Delta → 0.667
    (CongestionState.INCREMENTAL_COMPRESSION, 0): 0.500,
    (CongestionState.INCREMENTAL_COMPRESSION, 1): 0.250,
    (CongestionState.INCREMENTAL_COMPRESSION, 2): 0.667,

    # ── DROP_LOW_PRIORITY ────────────────────────────────────────────────────────
    # priority > 0 dropped before reaching compressor (handled in core.py).
    # priority = 0 (CONTROL): INCREMENTAL ratio applied (paper: keepalive-only).
    (CongestionState.DROP_LOW_PRIORITY, 0): 0.500,
    (CongestionState.DROP_LOW_PRIORITY, 1): 1.000,   # never reached (dropped)
    (CongestionState.DROP_LOW_PRIORITY, 2): 1.000,   # never reached (dropped)
}


class CompressionEngine:
    """
    Apply per-state, per-traffic-class packet size reduction (Phase 2).

    Sets pkt.compressed_size in-place without touching pkt.size (original).
    The service-time model in QueueManager uses compressed_size so smaller
    packets drain the queue faster — the key congestion-relief mechanism.
    """

    def compress(self, pkt: "Packet", state: CongestionState) -> None:
        ratio = _RATIOS.get((state, pkt.priority), 1.0)
        pkt.compressed_size = max(1, int(pkt.size * ratio))

    @staticmethod
    def ratio_for(state: CongestionState, priority: int) -> float:
        return _RATIOS.get((state, priority), 1.0)
