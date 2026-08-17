"""Audit: why does throughput collapse with PID/Kalman trackers at easy cells?

Hypothesis to test: at beta=10 deg / mag=0.5x (the easiest cell in the
envelope), the link is far above sensitivity at all times. A no-op tracker
should produce identical throughput to no tracker. PID/Kalman should produce
~equal throughput because the offsets they're correcting are negligible.

If PID/Kalman still tank throughput at this cell, one of three things is
true:
  (1) Each TRACKER_TICK perturbs current_pointing enough to drop rx_power
      below sensitivity briefly, causing real LINK_LOST events.
  (2) The pointing churn triggers ALIGNMENT_END / ALIGNMENT_START events that
      reset the data session even without true outage.
  (3) The data_session bookkeeping doesn't accurately reflect the link state
      (so reported "outage_duration" is wrong).

This script runs a single instrumented trial per tracker, logs:
  - TRACKER_TICK count
  - ALIGNMENT_START/END count
  - LINK_ESTABLISHED/LINK_LOST count + timestamps
  - rx_power statistics
  - data_session throughput trace
and prints a clean root-cause diagnostic.

Run from the repo root:
    python tools/audit_tracker_throughput.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from owdesmp.core.event_queue import EventQueue  # noqa: E402
from owdesmp.core.simulator import Simulator  # noqa: E402

POSE_CSV_ABS = (REPO_ROOT / "drone data" / "hover" / "pose_deviation.csv").resolve()


def make_scenario(tracker_type: str, beta_deg: float, mag_mult: float,
                  duration_s: float, seed: int) -> dict:
    """Minimal version of the envelope experiment's build_scenario."""
    pos_b_list = [100.0, 0.0, 0.0]
    csv_path = str(POSE_CSV_ABS)

    proto = {
        "beacon_period_s": 0.001512,
        "beacon_phase_s": "random",
        "packet_duration_s": 0.000404,
        "processing_delay_s": 0.0001,
        "handshake_timeout_s": 0.02,
        "keepalive_period_s": 0.001,
        "link_timeout_s": 0.005,
    }
    base_terminal = {
        "beam_divergence_deg": beta_deg,
        "max_angular_speed_deg_s": 1800.0,
        "range_m": 1000.0,
        "tx_power_dBm": 20.0,
        "receiver_aperture_m": 0.08,
        "receiver_sensitivity_dBm": -35.0,
    }

    if mag_mult > 0:
        ta = dict(base_terminal, pointing_jitter={
            "type": "csv_playback", "csv_path": csv_path,
            "time_offset_s": 0.0, "mag_mult": mag_mult,
        })
        tb = dict(base_terminal, pointing_jitter={
            "type": "csv_playback", "csv_path": csv_path,
            "time_offset_s": 80.0, "mag_mult": mag_mult,
        })
        pa = {"model": "vibration_csv", "initial_m": [0.0, 0.0, 0.0],
              "csv_path": csv_path, "time_offset_s": 0.0, "mag_mult": mag_mult}
        pb = {"model": "vibration_csv", "initial_m": pos_b_list,
              "csv_path": csv_path, "time_offset_s": 80.0, "mag_mult": mag_mult}
    else:
        ta = dict(base_terminal); tb = dict(base_terminal)
        pa = {"model": "fixed", "initial_m": [0.0, 0.0, 0.0]}
        pb = {"model": "fixed", "initial_m": pos_b_list}

    if tracker_type != "none":
        cfg = {
            "type": tracker_type, "max_correction_rad": 0.003,
            "tick_interval_s": 0.001,
        }
        if tracker_type == "pid":
            cfg.update({"Kp": 0.3, "Ki": 0.01, "Kd": 0.05})
        elif tracker_type == "kalman":
            cfg.update({"process_noise": 1e-4, "measurement_noise": 1e-3})
        ta["beam_tracker"] = dict(cfg)
        tb["beam_tracker"] = dict(cfg)

    return {
        "sim_name": f"audit_{tracker_type}",
        "random_seed": seed,
        "simulation": {
            "duration_s": duration_s,
            "mobility_update_period_s": 0.001,
            "beam_sample_interval_s": 0.0001,
        },
        "termination_condition": {"type": "fixed_duration"},
        "channel": {"mode": "threshold", "wavelength_nm": 1550,
                    "sigma_dBkm": 0.0, "Cn2": 1e-20},
        "data_session": {
            "src": "A", "dst": "B", "bytes_to_transfer": 1.0e15,
            "tick_interval_s": 0.001, "capacity_bps": 1.0e9,
            "noise_floor_dBm": -65.0, "bandwidth_hz": 1.0e9,
            "max_data_rate_bps": 1.0e10,
            "tcp": {"mss_bytes": 1500, "init_cwnd_mss": 1},
        },
        "neighbor_knowledge": {
            "pairs": [
                {"src": "A", "dst": "B", "estimated_position_m": pos_b_list,
                 "uncertainty_radius_m": 1.0},
                {"src": "B", "dst": "A", "estimated_position_m": [0.0, 0.0, 0.0],
                 "uncertainty_radius_m": 1.0},
            ],
        },
        "nodes": [
            {"id": "A", "start_time_s": 0.0, "position": pa,
             "optical_terminal": ta, "initial_pointing": [1.0, 0.0, 0.0],
             "scanner": {"type": "beta", "alpha": 0.7616, "beta": 0.7616},
             "protocol": dict(proto)},
            {"id": "B", "start_time_s": 0.0, "position": pb,
             "optical_terminal": tb, "initial_pointing": [-1.0, 0.0, 0.0],
             "scanner": {"type": "beta", "alpha": 0.7616, "beta": 0.7616},
             "protocol": dict(proto)},
        ],
    }


def run_instrumented(scenario: dict, seed: int) -> dict:
    """Run a single trial with full event-timeline instrumentation."""
    sim = Simulator(scenario=scenario, run_id=0, seed=seed)

    # Instrument the queue: every pop logs (time, type) to a list.
    timeline: list[tuple[float, str]] = []
    real_pop = sim.queue.pop

    def logging_pop():
        ev = real_pop()
        timeline.append((ev.time, ev.type))
        return ev

    sim.queue.pop = logging_pop  # type: ignore[assignment]

    # Instrument rx_power samples at each data tick (channel evaluation).
    rx_samples: list[tuple[float, float]] = []
    real_on_data = sim._on_data_tick

    def logging_on_data(ev):
        if sim.data_session is not None:
            a, b = sim.data_session.src, sim.data_session.dst
            if sim.channel.mode != "disabled":
                p = sim.channel.received_power_dBm(
                    sim.nodes[a], sim.nodes[b], sim.clock,
                    sim.rng.stream("channel") if sim.channel.mode == "stochastic" else None,
                )
                rx_samples.append((sim.clock, p))
        real_on_data(ev)

    sim._on_data_tick = logging_on_data  # type: ignore[assignment]

    t0 = time.perf_counter()
    result = sim.run()
    wall = time.perf_counter() - t0

    # Event timeline -> per-type counts and key event timestamps.
    counts = Counter(t for _, t in timeline)
    key_events = [
        (t, k) for t, k in timeline
        if k in ("LINK_ESTABLISHED", "LINK_LOST",
                 "ALIGNMENT_START", "ALIGNMENT_END",
                 "DATA_SESSION_START", "DATA_SESSION_END")
    ]

    ds = sim.data_session
    trace = list(ds.throughput_trace) if ds is not None else []
    bytes_acked = ds.tcp.total_bytes_acked if ds is not None else 0.0
    outage_count = ds.outage_count if ds is not None else 0
    total_outage = ds.total_outage_duration_s if ds is not None else 0.0

    rx_arr = np.array([p for _, p in rx_samples]) if rx_samples else np.array([])

    summary = {
        "tracker": scenario["sim_name"].replace("audit_", ""),
        "wall_s": round(wall, 3),
        "events_processed": int(result["events_processed"]),
        "event_counts": dict(counts),
        "first_link_time_s": result.get("first_link_time"),
        "links_lost": int(result.get("links_lost", 0)),
        "key_events_first_20": key_events[:20],
        "key_events_total": len(key_events),
        "data_session": {
            "outage_count": int(outage_count),
            "total_outage_duration_s": round(float(total_outage), 4),
            "trace_pts": len(trace),
            "bytes_acked": float(bytes_acked),
            "effective_mbps": (
                float(bytes_acked) * 8 / 1e6
                / max(1e-9, sim.duration - (result.get("first_link_time") or 0))
            ),
        },
        "rx_power_stats": {
            "n_samples": int(rx_arr.size),
            "mean_dBm": float(rx_arr.mean()) if rx_arr.size else None,
            "std_dBm": float(rx_arr.std()) if rx_arr.size else None,
            "min_dBm": float(rx_arr.min()) if rx_arr.size else None,
            "p5_dBm": float(np.percentile(rx_arr, 5)) if rx_arr.size else None,
            "p95_dBm": float(np.percentile(rx_arr, 95)) if rx_arr.size else None,
            "sensitivity_dBm": -35.0,
            "frac_below_sensitivity": (
                float((rx_arr < -35.0).mean()) if rx_arr.size else None
            ),
        },
    }
    return summary


def diagnose(none_s: dict, pid_s: dict, kalman_s: dict) -> str:
    """Print a root-cause assessment."""
    out = []
    out.append("=" * 78)
    out.append("ROOT-CAUSE DIAGNOSIS")
    out.append("=" * 78)

    none_mbps = none_s["data_session"]["effective_mbps"]
    pid_mbps = pid_s["data_session"]["effective_mbps"]
    kal_mbps = kalman_s["data_session"]["effective_mbps"]

    out.append(f"Throughput:  none={none_mbps:.2f}  pid={pid_mbps:.2f}  "
               f"kalman={kal_mbps:.2f}  (Mbps)")
    out.append(f"Reported outages: none={none_s['links_lost']}  "
               f"pid={pid_s['links_lost']}  kalman={kalman_s['links_lost']}")
    out.append(f"Reported outage_s: "
               f"none={none_s['data_session']['total_outage_duration_s']}  "
               f"pid={pid_s['data_session']['total_outage_duration_s']}  "
               f"kalman={kalman_s['data_session']['total_outage_duration_s']}")
    out.append("")

    for tag, s in [("none", none_s), ("pid", pid_s), ("kalman", kalman_s)]:
        rx = s["rx_power_stats"]
        out.append(f"[{tag}] rx_power dBm: "
                   f"mean={rx['mean_dBm']:.1f}  std={rx['std_dBm']:.2f}  "
                   f"min={rx['min_dBm']:.1f}  p5={rx['p5_dBm']:.1f}  "
                   f"frac<sens={rx['frac_below_sensitivity']:.3f}")
    out.append("")

    for tag, s in [("none", none_s), ("pid", pid_s), ("kalman", kalman_s)]:
        ec = s["event_counts"]
        out.append(f"[{tag}] events: "
                   f"TRACKER_TICK={ec.get('TRACKER_TICK', 0)}  "
                   f"ALIGNMENT_END={ec.get('ALIGNMENT_END', 0)}  "
                   f"ALIGNMENT_START={ec.get('ALIGNMENT_START', 0)}  "
                   f"LINK_LOST={ec.get('LINK_LOST', 0)}  "
                   f"LINK_ESTABLISHED={ec.get('LINK_ESTABLISHED', 0)}  "
                   f"DATA_TICK={ec.get('DATA_TICK', 0)}")
    out.append("")

    out.append("VERDICT:")
    pid_drop = (none_mbps - pid_mbps) / max(1e-9, none_mbps)
    kal_drop = (none_mbps - kal_mbps) / max(1e-9, none_mbps)
    if max(pid_drop, kal_drop) < 0.05:
        out.append("  No collapse observed at this operating point. Either")
        out.append("  the bug was already fixed or the original ARCC run was")
        out.append("  using older buggy tracker code (see TODO at end).")
        return "\n".join(out)

    # Determine which hypothesis fits.
    pid_outage_explains = (
        pid_s["data_session"]["total_outage_duration_s"] / 4.985
        if 4.985 > 0 else 0
    )
    if pid_drop > 0.5 and pid_s["data_session"]["total_outage_duration_s"] > 0.2:
        out.append("  H1 LIKELY: real LINK_LOST events. The tracker's per-tick")
        out.append("  pointing perturbation IS large enough to drop rx_power")
        out.append("  below sensitivity transiently.")
    elif pid_drop > 0.5 and pid_s["rx_power_stats"]["frac_below_sensitivity"] < 0.05:
        out.append("  H2 LIKELY: ALIGNMENT churn without true outage. rx_power")
        out.append("  rarely drops below sensitivity, yet throughput is dead.")
        out.append("  ALIGNMENT_END/START count vs. LINK_LOST count is the tell:")
        out.append("  if ALIGNMENT_END >> LINK_LOST, the alignment gate is")
        out.append("  killing in-flight packets faster than the link timeout")
        out.append("  fires.")
    elif (pid_s["data_session"]["total_outage_duration_s"] < 0.1
          and pid_drop > 0.5):
        out.append("  H3 LIKELY: data_session bookkeeping disagrees with truth.")
        out.append("  Reported outage_duration is tiny but throughput is dead.")
        out.append("  Likely cause: data session ticks against a healthy")
        out.append("  channel but receives near-zero ACK windows due to")
        out.append("  perturbation-driven PER spikes (TCP sees the channel as")
        out.append("  high-PER and refuses to grow cwnd past 1 MSS).")

    return "\n".join(out)


def main():
    print("=" * 78)
    print("TRACKER THROUGHPUT AUDIT")
    print("=" * 78)
    print("Cell under audit: beta=10.0 deg, mag=0.5x, duration=5 s")
    print("This is the easiest cell in the envelope — rx_power should remain")
    print("well above sensitivity at all times.")
    print()

    BETA = 10.0
    MAG = 0.5
    DURATION = 5.0
    SEED = 20260525

    summaries = {}
    for tr in ("none", "pid", "kalman"):
        print(f"\n--- running {tr} ---")
        sc = make_scenario(tr, BETA, MAG, DURATION, SEED)
        s = run_instrumented(sc, SEED)
        summaries[tr] = s
        ds = s["data_session"]
        print(f"  wall={s['wall_s']:.2f}s  events={s['events_processed']:,}")
        print(f"  bytes_acked={ds['bytes_acked']:.3e}  "
              f"throughput={ds['effective_mbps']:.2f} Mbps")
        print(f"  outage_count={ds['outage_count']}  "
              f"outage_dur={ds['total_outage_duration_s']:.4f}s")

    print()
    print(diagnose(summaries["none"], summaries["pid"], summaries["kalman"]))

    # Save full JSON for offline inspection.
    out_path = REPO_ROOT / "tools" / "audit_tracker_throughput_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\nFull instrumentation written to {out_path}")


if __name__ == "__main__":
    main()
