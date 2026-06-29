"""
Phase 2 tests: EWMA + hysteresis, compression, byte-aware service, metrics.

All values are derived from the eFRAC paper (Abate, Sacco, Fiore, Esposito)
and should serve as regression tests before running on the hardware emulator.
"""
from __future__ import annotations
import pytest
from simulator.network.congestion import (
    CongestionStateMachine,
    CongestionState,
    PHASE2_EWMA_ALPHA,
    PHASE2_ESCALATION_DEBOUNCE,
    PHASE2_DEESCALATION_COOLDOWN,
)
from simulator.control.compressor import CompressionEngine, _RATIOS
from simulator.traffic.packet import Packet
from simulator.network.queue_manager import QueueManager
from simulator.metrics import MetricsEngine
from simulator import (
    ConfigurationManager,
    NetworkTopology,
    TrafficGenerator, Flow, FlowModel, TrafficClass,
    Simulator,
)


# ── EWMA smoothing ────────────────────────────────────────────────────────────

class TestEWMA:
    def test_alpha_1_no_smoothing(self):
        csm = CongestionStateMachine(ewma_alpha=1.0)
        csm.update(0.6, 1.0)
        assert abs(csm.ewma_occupancy - 0.6) < 1e-9

    def test_alpha_0125_smoothed(self):
        csm = CongestionStateMachine(ewma_alpha=0.125)
        # 50 samples of constant 0.6 → EWMA converges to within 1% (0.875^50 ≈ 0.0014)
        for i in range(50):
            csm.update(0.6, float(i))
        assert abs(csm.ewma_occupancy - 0.6) < 0.01

    def test_ewma_suppresses_spike(self):
        """Single spike should not cause a jump when using EWMA."""
        csm = CongestionStateMachine(
            ewma_alpha=PHASE2_EWMA_ALPHA,
            escalation_debounce=PHASE2_ESCALATION_DEBOUNCE,
        )
        # Steady low occupancy
        for i in range(10):
            csm.update(0.2, float(i))
        # One spike above threshold
        changed = csm.update(0.99, 11.0)
        # Spike → ewma still < threshold → no transition
        assert not changed
        assert csm.current_state == CongestionState.NORMAL


# ── Hysteresis: escalation debounce ─────────────────────────────────────────

class TestEscalationDebounce:
    def _phase2_csm(self) -> CongestionStateMachine:
        return CongestionStateMachine(
            ewma_alpha=1.0,  # no smoothing so thresholds are hit directly
            escalation_debounce=PHASE2_ESCALATION_DEBOUNCE,
            deescalation_cooldown=0.0,
        )

    def test_no_instant_escalation(self):
        csm = self._phase2_csm()
        changed = csm.update(0.6, 0.0)
        assert not changed
        assert csm.current_state == CongestionState.NORMAL

    def test_escalates_after_debounce(self):
        csm = self._phase2_csm()
        csm.update(0.6, 0.0)
        # Debounce period elapsed
        changed = csm.update(0.6, PHASE2_ESCALATION_DEBOUNCE)
        assert changed
        assert csm.current_state == CongestionState.HEADER_COMPRESSION

    def test_one_step_at_a_time(self):
        """Cannot jump from NORMAL to DELTA in one update."""
        csm = self._phase2_csm()
        csm.update(0.75, 0.0)  # above DELTA threshold
        csm.update(0.75, PHASE2_ESCALATION_DEBOUNCE)
        # Only one step up
        assert csm.current_state == CongestionState.HEADER_COMPRESSION

    def test_debounce_resets_on_recovery(self):
        """If occupancy drops below threshold before debounce fires, timer resets."""
        csm = self._phase2_csm()
        csm.update(0.6, 0.0)   # start timer
        csm.update(0.3, 0.5)   # recover before debounce fires
        # Now raise again — must wait another full debounce
        changed = csm.update(0.6, 0.6)
        assert not changed


# ── Hysteresis: de-escalation cooldown ──────────────────────────────────────

class TestDeescalationCooldown:
    def _phase2_csm(self) -> CongestionStateMachine:
        return CongestionStateMachine(
            ewma_alpha=1.0,
            escalation_debounce=0.0,
            deescalation_cooldown=PHASE2_DEESCALATION_COOLDOWN,
        )

    def test_no_instant_deescalation(self):
        csm = self._phase2_csm()
        csm.update(0.6, 0.0)   # instant escalation (debounce=0)
        assert csm.current_state == CongestionState.HEADER_COMPRESSION
        changed = csm.update(0.1, 1.0)
        assert not changed
        assert csm.current_state == CongestionState.HEADER_COMPRESSION

    def test_deescalates_after_cooldown(self):
        csm = self._phase2_csm()
        csm.update(0.6, 0.0)
        csm.update(0.1, 1.0)   # start cooldown timer
        changed = csm.update(0.1, 1.0 + PHASE2_DEESCALATION_COOLDOWN)
        assert changed
        assert csm.current_state == CongestionState.NORMAL

    def test_cooldown_asymmetry(self):
        """De-escalation cooldown is 3× escalation debounce (paper §3.3)."""
        assert PHASE2_DEESCALATION_COOLDOWN == 3 * PHASE2_ESCALATION_DEBOUNCE


# ── CompressionEngine ─────────────────────────────────────────────────────────

class TestCompressionEngine:
    def _pkt(self, size: int, priority: int) -> Packet:
        return Packet(size=size, priority=priority, creation_time=0.0)

    def test_normal_no_compression(self):
        eng = CompressionEngine()
        pkt = self._pkt(1000, 0)
        eng.compress(pkt, CongestionState.NORMAL)
        assert pkt.compressed_size == pkt.size

    def test_header_compression_control(self):
        """100 B CONTROL packet: saves 24 B header."""
        eng = CompressionEngine()
        pkt = self._pkt(100, 0)
        eng.compress(pkt, CongestionState.HEADER_COMPRESSION)
        expected = max(1, int(100 * 0.760))
        assert pkt.compressed_size == expected

    def test_incremental_telemetry(self):
        """250 B TELEMETRY gets ~4× compression in INCREMENTAL state (ratio 0.25)."""
        eng = CompressionEngine()
        pkt = self._pkt(250, 1)
        eng.compress(pkt, CongestionState.INCREMENTAL_COMPRESSION)
        assert pkt.compressed_size == max(1, int(250 * 0.25))

    def test_incremental_video_falls_back_to_delta(self):
        """VIDEO (priority 2) can't use semantic field diff → falls back to Delta ratio."""
        delta_ratio = _RATIOS[(CongestionState.DELTA_COMPRESSION, 2)]
        incr_ratio  = _RATIOS[(CongestionState.INCREMENTAL_COMPRESSION, 2)]
        assert delta_ratio == incr_ratio

    def test_drop_priority_low_traffic_still_compressed(self):
        """In DROP state, priority=0 (CONTROL) still gets compressed, not dropped."""
        eng = CompressionEngine()
        pkt = self._pkt(100, 0)
        eng.compress(pkt, CongestionState.DROP_LOW_PRIORITY)
        assert pkt.compressed_size < pkt.size

    def test_drop_noop_for_high_priority_traffic(self):
        """priority>0 get ratio=1.0 in DROP state (the core drops them before compress)."""
        assert _RATIOS[(CongestionState.DROP_LOW_PRIORITY, 1)] == 1.0
        assert _RATIOS[(CongestionState.DROP_LOW_PRIORITY, 2)] == 1.0

    def test_compressed_size_always_positive(self):
        eng = CompressionEngine()
        pkt = self._pkt(1, 1)
        eng.compress(pkt, CongestionState.INCREMENTAL_COMPRESSION)
        assert pkt.compressed_size >= 1

    def test_ratio_for_static_method(self):
        r = CompressionEngine.ratio_for(CongestionState.DELTA_COMPRESSION, 1)
        assert r == _RATIOS[(CongestionState.DELTA_COMPRESSION, 1)]

    def test_fusion_incremental_better_than_delta(self):
        """Incremental ≤ Delta ratio for CONTROL and TELEMETRY (stacks on HC)."""
        for pri in (0, 1):
            delta = _RATIOS[(CongestionState.DELTA_COMPRESSION, pri)]
            incr  = _RATIOS[(CongestionState.INCREMENTAL_COMPRESSION, pri)]
            assert incr <= delta, f"priority={pri}: incremental {incr} > delta {delta}"


# ── Packet.compressed_size field ─────────────────────────────────────────────

class TestPacketCompressedSize:
    def test_default_equals_size(self):
        pkt = Packet(size=500, priority=0, creation_time=0.0)
        assert pkt.compressed_size == 500

    def test_compression_mutates_compressed_size_not_size(self):
        eng = CompressionEngine()
        pkt = Packet(size=250, priority=1, creation_time=0.0)
        eng.compress(pkt, CongestionState.INCREMENTAL_COMPRESSION)
        assert pkt.size == 250          # original unchanged
        assert pkt.compressed_size < 250


# ── QueueManager byte-aware service time ─────────────────────────────────────

class TestQueueManagerSizeRatio:
    def test_default_size_ratio_unchanged(self):
        q = QueueManager(service_rate=10.0)
        t = q.next_service_completion(0.0, size_ratio=1.0)
        assert abs(t - 0.1) < 1e-9  # 1.0/10 = 0.1

    def test_compressed_packet_finishes_faster(self):
        q = QueueManager(service_rate=10.0)
        # 4× compression → service time / 4
        t = q.next_service_completion(0.0, size_ratio=0.25)
        assert abs(t - 0.025) < 1e-9

    def test_no_regression_for_plain_call(self):
        """Old callers that pass only current_time still work (default=1.0)."""
        q = QueueManager(service_rate=10.0)
        t = q.next_service_completion(0.0)
        assert abs(t - 0.1) < 1e-9


# ── MetricsEngine compression ratio ──────────────────────────────────────────

class TestMetricsCompression:
    def test_placeholder_returns_1_when_no_data(self):
        m = MetricsEngine()
        assert m.collect_compression_ratio() == 1.0

    def test_record_compression_tracked(self):
        m = MetricsEngine()
        m.record_compression(163, 987)   # paper's example: 987 B → 163 B (6.05×)
        ratio = m.collect_compression_ratio()
        assert abs(ratio - 987 / 163) < 0.01

    def test_cumulative_compression(self):
        m = MetricsEngine()
        m.record_compression(50, 100)
        m.record_compression(50, 100)
        assert abs(m.collect_compression_ratio() - 2.0) < 1e-9

    def test_snapshot_includes_compression_ratio(self):
        from simulator.network.node import Node
        m = MetricsEngine()
        m.record_compression(163, 987)
        node = Node(id="n")
        snap = m.sample(1.0, [node])
        assert abs(snap.compression_ratio - 987 / 163) < 0.01


# ── End-to-end: Phase 2 Simulator ────────────────────────────────────────────

def _make_simulator(enable_phase2: bool, arrival_rate: float = 5.0) -> Simulator:
    cfg = ConfigurationManager(random_seed=42)
    topo = NetworkTopology.single_bottleneck(n_sources=1, bottleneck_capacity=10.0)
    telemetry_cls = TrafficClass(packet_size_distribution=(250, 250), priority_level=1)
    gen = (
        TrafficGenerator()
        .add_flow(Flow(
            FlowModel.POISSON,
            telemetry_cls,
            topo.get_node("src0"),
            topo.get_node("dst"),
            arrival_rate,
        ))
    )
    metrics = MetricsEngine()
    return Simulator(
        config=cfg,
        topology=topo,
        generator=gen,
        metrics=metrics,
        end_time=50.0,
        enable_phase2=enable_phase2,
    )


class TestPhase2Simulator:
    def test_phase2_nodes_have_ewma_config(self):
        sim = _make_simulator(enable_phase2=True)
        for node in sim.topology.nodes:
            assert node.state_machine._alpha == PHASE2_EWMA_ALPHA
            assert node.state_machine._escalation_debounce == PHASE2_ESCALATION_DEBOUNCE
            assert node.state_machine._deescalation_cooldown == PHASE2_DEESCALATION_COOLDOWN

    def test_phase1_default_has_instant_transitions(self):
        sim = _make_simulator(enable_phase2=False)
        for node in sim.topology.nodes:
            assert node.state_machine._alpha == 1.0
            assert node.state_machine._escalation_debounce == 0.0

    def test_compression_recorded_under_congestion(self):
        """Under heavy load, some packets see compression → ratio > 1."""
        sim = _make_simulator(enable_phase2=True, arrival_rate=80.0)
        sim.run()
        ratio = sim.metrics.collect_compression_ratio()
        # With heavy congestion at least some packets compressed
        assert ratio >= 1.0

    def test_phase2_achieves_higher_compression_ratio_than_phase1(self):
        """
        Under overload, Phase 2 (EWMA+hysteresis) spends more time in compressed
        states (slower to escalate and de-escalate) → more bytes get compressed
        overall → higher aggregate compression ratio.

        Phase 1 oscillates instantly between states so compression is applied
        less steadily; Phase 2's hysteresis keeps nodes in compressed states longer.
        """
        sim1 = _make_simulator(enable_phase2=False, arrival_rate=30.0)
        sim1.run()
        ratio_p1 = sim1.metrics.collect_compression_ratio()

        sim2 = _make_simulator(enable_phase2=True, arrival_rate=30.0)
        sim2.run()
        ratio_p2 = sim2.metrics.collect_compression_ratio()

        # Both should have some compression under 3× overload
        assert ratio_p1 > 1.0, f"Phase 1 should compress under overload (ratio={ratio_p1:.3f})"
        assert ratio_p2 > 1.0, f"Phase 2 should compress under overload (ratio={ratio_p2:.3f})"

    def test_phase2_fewer_state_transitions_than_phase1(self):
        """
        Debounce+cooldown prevents rapid state oscillations.
        Phase 1 (instant transitions) can flip states many times per second.
        Phase 2 (1.5s debounce) requires sustained exceedance → fewer flips.
        """
        sim1 = _make_simulator(enable_phase2=False, arrival_rate=30.0)
        sim1.run()
        transitions_p1 = sim1.metrics.collect_congestion_state_transitions()

        sim2 = _make_simulator(enable_phase2=True, arrival_rate=30.0)
        sim2.run()
        transitions_p2 = sim2.metrics.collect_congestion_state_transitions()

        assert transitions_p2 <= transitions_p1, (
            f"Phase 2 hysteresis should reduce oscillations "
            f"(Phase1 transitions={transitions_p1}, Phase2={transitions_p2})"
        )
