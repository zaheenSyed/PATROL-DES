"""Scanner comparison vs beam divergence, WIDE-OFFSET geometry, static vs
measured UAV vibration.

Purpose
-------
Resolve the apparent contradiction between:
  - exp_scanner_compare_vs_beta.py  (Beta wins at every beta, wide search,
                                     random offset, NO vibration)
  - exp_scanner_vibration_journal.py (compact centered search where the
                                     deterministic scans win statically, then
                                     collapse under vibration)

by running ONE geometry that keeps Beta dominant in the static case AND shows a
legible vibration penalty on top. Both conditions share the same wide-offset
search, so the static curve reproduces the disc_time_vs_beta trend (Beta lowest
at every beta) and the measured curve sits above it (everyone slower, penalty
strongest at narrow beta).

Geometry / sweep (matches exp_scanner_compare_vs_beta.py)
---------------------------------------------------------
    d = 100 m, search half-angle = 20 deg  ->  search radius ~= 36.4 m
    A's estimate of B = cone center; true B randomly offset in the cone
    beta_deg in {3, 5, 10, 15, 20, 25}
    omega = omega_ideal(beta) = beta / (sqrt(2) * C_handshake)
    scanners {beta, spiral, raster}
    condition {static (mag 0.0), measured (mag 1.0)}
    channel threshold, tx 35 dBm, sens -35 dBm  (same as the compare script)

Vibration is applied exactly as in the journal vibration experiment: measured
6-DOF pose deviations (drone data/hover/pose_deviation.csv) drive the optical
pointing jitter, decorrelated per node by an 80 s time offset.

Modes
-----
    --quick : num_pos=5,  trials_per_pos=5   (~150 trials/condition, < 1 min)
    --mid   : num_pos=10, trials_per_pos=10  (~600/condition, few min)
    (full)  : num_pos=50, trials_per_pos=50  (~2500/condition, cluster)

Usage
-----
    python experiments/exp_scanner_vib_wide_offset.py --quick --workers 8
"""

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    __import__("os").environ.setdefault(_v, "1")

# Locate the OWDES-MP package root by walking up from this script until a folder
# containing the `owdesmp` package is found, so the script works regardless of
# where it is placed (experiments/, or journal_results/scanner compare/). This
# also makes REPO_ROOT-relative paths below (the pose CSV, results dir) resolve
# correctly. Falls back to two levels up.
_HERE = Path(__file__).resolve().parent
REPO_ROOT = next((p for p in [_HERE, *_HERE.parents] if (p / "owdesmp").is_dir()),
                 _HERE.parent.parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from owdesmp.core.simulator import Simulator  # noqa: E402

POSE_CSV_ABS = (REPO_ROOT / "drone data" / "hover" / "pose_deviation.csv").resolve()

# -- protocol / handshake timing (identical to both source scripts) ----------
PACKET_TX_S = 304e-6
PROC_S = 100e-6
C_HANDSHAKE_S = 3.0 * (PACKET_TX_S + 2 * PROC_S)   # 1.512 ms

# -- sweep configuration -----------------------------------------------------
BETA_DEGS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
SCANNERS = ["beta", "spiral", "raster"]
CONDITIONS = [("static", 0.0), ("measured", 5.0)]   # add ("severe", 2.0) if wanted
DIST_M = 100.0
SEARCH_DEG = 20.0
TX_POWER_DBM = 35.0
SENSITIVITY_DBM = -35.0
DURATION_S = 120.0


def calculate_omega_ideal(beta_rad: float) -> float:
    return beta_rad / (math.sqrt(2.0) * C_HANDSHAKE_S)


def _stable_seed(*parts) -> int:
    import hashlib
    h = hashlib.blake2b(repr(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % (2 ** 31)


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def generate_n2_positions(n_pos, dist_m, search_angle_rad, rng):
    """Sample Node-B positions inside a cone of half-angle search_angle around
    +Z at depth ~dist_m. Identical to exp_scanner_compare_vs_beta.py."""
    cos_min = math.cos(search_angle_rad)
    out = []
    for _ in range(n_pos):
        cos_theta = rng.uniform(cos_min, 1.0)
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        phi = rng.uniform(0.0, 2.0 * math.pi)
        direction = np.array(
            [sin_theta * math.cos(phi), sin_theta * math.sin(phi), cos_theta]
        )
        r = dist_m * rng.uniform(0.9, 1.1)
        out.append(r * direction)
    return out


def _scanner_cfg(name, flags=None):
    """Scanner config, parameterized by a `flags` dict (all default False):
      full_coverage  - spiral/raster guarantee >=1 full pass of the disk at wide beams
      perimeter_start- enter from a random circumference point (no centre head-start)
      center_lead_in - random circumference entry -> dead-head to centre -> centre-out
      dense_turns    - spiral pitch = beam_r/(2*pi) (~50% overlap; reliable tiling)
      dense_rows     - raster rows spaced by beam_r (~50% overlap; >=3 rows)
    Beta is a randomized perimeter walk and unaffected by any flag."""
    f = flags or {}
    if name == "beta":
        return {"type": "beta", "alpha": 0.41485, "beta": 0.41485}
    if name == "spiral":
        return {"type": "spiral", "samples_per_turn": 60,
                "full_coverage": bool(f.get("full_coverage")),
                "perimeter_start": bool(f.get("perimeter_start")),
                "center_lead_in": bool(f.get("center_lead_in")),
                "dense_turns": bool(f.get("dense_turns"))}
    if name == "raster":
        return {"type": "raster", "step_along": 0.9,
                "full_coverage": bool(f.get("full_coverage")),
                "perimeter_start": bool(f.get("perimeter_start")),
                "center_lead_in": bool(f.get("center_lead_in")),
                "dense_rows": bool(f.get("dense_rows"))}
    raise ValueError(f"unknown scanner {name!r}")


def build_scenario(scanner_type, beta_deg, omega_rad_s, dist_m,
                   search_angle_rad, pos_b, mag_mult, seed,
                   duration_s=DURATION_S, scanner_flags=None):
    """Wide-offset geometry (A's estimate = cone center, B offset in the cone)
    with optional measured-vibration pointing jitter (mag_mult > 0).
    scanner_flags: dict of coverage/start options (see _scanner_cfg)."""
    omega_deg_s = math.degrees(omega_rad_s)
    search_radius_m = dist_m * math.tan(search_angle_rad)
    pos_b_list = [float(pos_b[0]), float(pos_b[1]), float(pos_b[2])]
    a_estimate_of_b = [0.0, 0.0, float(dist_m)]

    proto = {
        "beacon_period_s": 0.001512,
        "beacon_phase_s": "random",
        "packet_duration_s": PACKET_TX_S + PROC_S,
        "processing_delay_s": PROC_S,
        "handshake_timeout_s": 0.02,
    }
    terminal_base = {
        "beam_divergence_deg": beta_deg,
        "max_angular_speed_deg_s": omega_deg_s,
        "range_m": 100000.0,
        "tx_power_dBm": TX_POWER_DBM,
        "receiver_aperture_m": 0.08,
        "receiver_sensitivity_dBm": SENSITIVITY_DBM,
    }
    if mag_mult > 0.0:
        csv_path = str(POSE_CSV_ABS)
        terminal_a = dict(terminal_base, pointing_jitter={
            "type": "csv_playback", "csv_path": csv_path,
            "time_offset_s": 0.0, "mag_mult": float(mag_mult)})
        terminal_b = dict(terminal_base, pointing_jitter={
            "type": "csv_playback", "csv_path": csv_path,
            "time_offset_s": 80.0, "mag_mult": float(mag_mult)})
    else:
        terminal_a = dict(terminal_base)
        terminal_b = dict(terminal_base)

    scanner_cfg = _scanner_cfg(scanner_type, scanner_flags)
    return {
        "sim_name": f"svwo_{scanner_type}_b{beta_deg:g}_m{mag_mult:g}",
        "random_seed": seed,
        "simulation": {
            "duration_s": duration_s,
            # 1 ms tick so the measured-vibration jitter actually re-samples;
            # for the static (no-jitter) case a FixedPosition node schedules no
            # mobility ticks, so static behaviour is identical to the compare
            # script regardless of this value.
            "mobility_update_period_s": 0.001,
            "beam_sample_interval_s": 0.0001,
        },
        "termination_condition": {"type": "first_link_discovered"},
        "channel": {
            "mode": "threshold",
            "wavelength_nm": 1550,
            "sigma_dBkm": 0.1,
        },
        "nodes": [
            {"id": "A", "start_time_s": 0.0,
             "position": {"model": "fixed", "initial_m": [0.0, 0.0, 0.0]},
             "optical_terminal": terminal_a,
             "scanner": dict(scanner_cfg),
             "protocol": dict(proto)},
            {"id": "B", "start_time_s": 0.0,
             "position": {"model": "fixed", "initial_m": pos_b_list},
             "optical_terminal": terminal_b,
             "scanner": dict(scanner_cfg),
             "protocol": dict(proto)},
        ],
        "neighbor_knowledge": {
            "pairs": [
                {"src": "A", "dst": "B",
                 "estimated_position_m": a_estimate_of_b,
                 "uncertainty_radius_m": search_radius_m},
                {"src": "B", "dst": "A",
                 "estimated_position_m": [0.0, 0.0, 0.0],
                 "uncertainty_radius_m": search_radius_m},
            ]
        },
    }


def run_single_trial(packed):
    (scanner_type, beta_deg, omega_rad_s, dist_m, search_angle_rad,
     pos_b, mag_mult, condition, seed, pos_idx, trial, scanner_flags) = packed
    scenario = build_scenario(scanner_type, beta_deg, omega_rad_s, dist_m,
                              search_angle_rad, pos_b, mag_mult, seed,
                              scanner_flags=scanner_flags)
    sim = Simulator(scenario=scenario, run_id=0, seed=seed)
    r = sim.run()
    t_disc = float(r.get("first_link_time", DURATION_S))
    success = bool(r.get("success", False)) and t_disc < DURATION_S
    return {
        "scanner": scanner_type, "beta_deg": beta_deg, "condition": condition,
        "pos_idx": pos_idx, "trial": trial,
        "t_disc_s": t_disc if success else DURATION_S,
        "discovered": int(success),
    }


def aggregate(results):
    buckets = defaultdict(list)
    for r in results:
        buckets[(r["scanner"], r["beta_deg"], r["condition"])].append(r)
    agg = []
    for (sc, b, cond), rs in sorted(buckets.items()):
        succ = [r["t_disc_s"] for r in rs if r["discovered"]]
        n = len(rs)
        rate = sum(r["discovered"] for r in rs) / n if n else 0.0
        mean_t = statistics.mean(succ) if succ else DURATION_S
        std_t = statistics.pstdev(succ) if len(succ) > 1 else 0.0
        agg.append({
            "scanner": sc, "beta_deg": b, "condition": cond,
            "n_trials": n, "success_rate": rate,
            "mean_time_s": mean_t, "std_time_s": std_t,
        })
    return agg


def save_csv(results, agg, out_dir):
    with open(out_dir / "scanner_vib_wide_offset_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    with open(out_dir / "scanner_vib_wide_offset_agg.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader()
        w.writerows(agg)


def plot(agg, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[plot] matplotlib unavailable ({e}); skipping figure")
        return
    color = {"beta": "C0", "spiral": "C1", "raster": "C2"}
    cond_style = {"static": "-", "measured": "--"}
    cond_label = {"static": "static", "measured": r"measured ($1\times$)"}

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for sc in SCANNERS:
        for cond, _ in CONDITIONS:
            rows = sorted(
                [a for a in agg if a["scanner"] == sc and a["condition"] == cond],
                key=lambda a: a["beta_deg"])
            if not rows:
                continue
            xs = [a["beta_deg"] for a in rows]
            ys = [a["mean_time_s"] for a in rows]
            ax.plot(xs, ys, cond_style[cond], color=color[sc], marker="o",
                    ms=4, lw=1.6,
                    label=f"{sc.capitalize()} ({cond_label[cond]})")
    ax.set_yscale("log")
    ax.set_xlabel(r"Beam divergence $\beta$ (deg)")
    ax.set_ylabel("Mean discovery time (s)")
    ax.set_title(
        f"Wide-offset search (d={DIST_M:g} m, search={SEARCH_DEG:g}$^\\circ$), "
        f"static vs measured vibration")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig_wide_offset_static_vs_measured.{ext}", dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--mid", action="store_true")
    p.add_argument("--workers", type=int, default=cpu_count())
    p.add_argument("--serial", action="store_true",
                   help="run trials in-process (no multiprocessing); use when "
                        "Pool is blocked by the OS/sandbox")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--faircoverage", action="store_true",
                   help="spiral/raster guarantee >=1 full pass of the search "
                        "disk at wide beams (fixes the non-monotonic wide-beam "
                        "trend); beta is unaffected")
    p.add_argument("--perimeter-start", action="store_true",
                   help="spiral/raster enter from a random point on the search "
                        "circumference (spiral winds circumference->centre; "
                        "raster basis randomly rotated) instead of the centre, "
                        "so they get no centre head-start over Beta")
    p.add_argument("--center-lead-in", action="store_true",
                   help="spiral/raster enter from a random circumference point, "
                        "dead-head to the centre, then run normal centre-out "
                        "coverage (>=1 revolution); all three scanners pay an "
                        "equal perimeter->centre lead-in. Implies full coverage.")
    p.add_argument("--dense-turns", action="store_true",
                   help="spiral pitch = beam_r/(2*pi) (~50%% turn overlap) for "
                        "reliable disk tiling at narrow beams")
    p.add_argument("--dense-rows", action="store_true",
                   help="raster rows spaced by beam_r (~50%% overlap, >=3 rows) "
                        "for reliable disk tiling at narrow beams")
    p.add_argument("--nshards", type=int, default=1,
                   help="split the sweep into N interleaved shards; run each in "
                        "its own process with --shard i. Lets you parallelize by "
                        "launching N serial processes when multiprocessing.Pool "
                        "is unavailable (e.g. Windows Store Python spawn).")
    p.add_argument("--shard", type=int, default=0,
                   help="which shard [0, nshards) this process runs")
    args = p.parse_args()

    if not POSE_CSV_ABS.exists():
        sys.exit(f"pose_deviation.csv not found at {POSE_CSV_ABS}")

    if args.quick:
        num_pos, trials_per_pos = 5, 5
    elif args.mid:
        num_pos, trials_per_pos = 10, 10
    else:
        num_pos, trials_per_pos = 25, 10

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # All scanner coverage/start options in one dict, threaded to every trial.
    scanner_flags = {
        "full_coverage": bool(args.faircoverage),
        "perimeter_start": bool(args.perimeter_start),
        "center_lead_in": bool(args.center_lead_in),
        "dense_turns": bool(args.dense_turns),
        "dense_rows": bool(args.dense_rows),
    }
    tags = {"full_coverage": "_faircov", "perimeter_start": "_perimstart",
            "center_lead_in": "_leadin", "dense_turns": "_denseturns",
            "dense_rows": "_denserows"}
    suffix = "".join(t for k, t in tags.items() if scanner_flags[k])
    out_dir = Path(args.output_dir or REPO_ROOT / "results" / f"scanner_vib_wide_offset_{ts}{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")
    print(f"scanner flags: {', '.join(k for k, v in scanner_flags.items() if v) or 'none (legacy)'}")

    search_angle_rad = math.radians(SEARCH_DEG)
    pos_rng = random.Random(_stable_seed("positions", DIST_M, SEARCH_DEG, num_pos))
    positions = generate_n2_positions(num_pos, DIST_M, search_angle_rad, pos_rng)

    (out_dir / "axes.json").write_text(json.dumps({
        "beta_deg": BETA_DEGS, "scanners": SCANNERS,
        "conditions": [{"label": c, "mag_mult": m} for c, m in CONDITIONS],
        "dist_m": DIST_M, "search_deg": SEARCH_DEG,
        "search_radius_m": DIST_M * math.tan(search_angle_rad),
        "tx_power_dBm": TX_POWER_DBM, "sensitivity_dBm": SENSITIVITY_DBM,
        "duration_s": DURATION_S, "num_pos": num_pos,
        "trials_per_pos": trials_per_pos,
        "scanner_flags": scanner_flags,
        "omega_deg_s_per_beta": {
            f"beta_{int(b)}": math.degrees(calculate_omega_ideal(math.radians(b)))
            for b in BETA_DEGS},
    }, indent=2))

    tasks = []
    for sc in SCANNERS:
        for b in BETA_DEGS:
            omega = calculate_omega_ideal(math.radians(b))
            for cond, mag in CONDITIONS:
                for pi, pos_b in enumerate(positions):
                    for tr in range(trials_per_pos):
                        seed = _stable_seed("svwo", sc, b, cond, pi, tr)
                        tasks.append((sc, b, omega, DIST_M, search_angle_rad,
                                      pos_b, mag, cond, seed, pi, tr,
                                      scanner_flags))

    # Interleaved sharding: each shard takes every nshards-th task. Positions and
    # seeds are built identically in every shard (deterministic), so the union of
    # all shards == the full sweep exactly once. Merge the shard raw CSVs after.
    if args.nshards > 1:
        tasks = tasks[args.shard::args.nshards]
        print(f"[{_ts()}] shard {args.shard}/{args.nshards}: {len(tasks)} trials")

    # Progress bar (tqdm if available, else a lightweight fallback that prints
    # every ~2% so cluster stdout/log still shows live progress).
    def _pbar(iterable, total, desc):
        try:
            from tqdm import tqdm
            return tqdm(iterable, total=total, desc=desc, unit="trial",
                        smoothing=0.05, mininterval=1.0)
        except Exception:
            def gen():
                step = max(1, total // 50)
                for i, x in enumerate(iterable, 1):
                    if i % step == 0 or i == total:
                        el = time.time() - t0
                        rate = i / el if el > 0 else 0.0
                        eta = (total - i) / rate if rate > 0 else float("inf")
                        print(f"  [{_ts()}] {desc}: {i}/{total} "
                              f"({100*i/total:5.1f}%) {rate:5.1f} trial/s "
                              f"ETA {eta:6.0f}s", flush=True)
                    yield x
            return gen()

    t0 = time.time()
    tag = (f"shard {args.shard}/{args.nshards}" if args.nshards > 1
           else ("serial" if args.serial else f"{args.workers} workers"))
    if args.serial or args.nshards > 1:
        print(f"[{_ts()}] {len(tasks)} trials ({tag})")
        results = [r for r in _pbar((run_single_trial(t) for t in tasks),
                                    len(tasks), tag)]
    else:
        print(f"[{_ts()}] {len(tasks)} trials on {args.workers} workers")
        try:
            with Pool(args.workers) as pool:
                results = list(_pbar(
                    pool.imap_unordered(run_single_trial, tasks, chunksize=8),
                    len(tasks), tag))
        except (PermissionError, OSError) as e:
            print(f"[{_ts()}] Pool unavailable ({e}); falling back to serial")
            results = [r for r in _pbar((run_single_trial(t) for t in tasks),
                                        len(tasks), "serial")]
    print(f"[{_ts()}] done in {time.time()-t0:.1f}s")

    agg = aggregate(results)
    save_csv(results, agg, out_dir)
    if args.nshards == 1:
        plot(agg, out_dir)

    # Console summary: penalty ratio per scanner x beta.
    print("\nmean discovery time (s)  [static -> measured, penalty x]")
    by = {(a["scanner"], a["beta_deg"], a["condition"]): a for a in agg}
    for sc in SCANNERS:
        for b in BETA_DEGS:
            s = by.get((sc, b, "static"))
            m = by.get((sc, b, "measured"))
            if s and m and s["mean_time_s"] > 0:
                ratio = m["mean_time_s"] / s["mean_time_s"]
                print(f"  {sc:6s} beta={b:5.1f}: "
                      f"{s['mean_time_s']:8.4f} -> {m['mean_time_s']:8.4f} "
                      f"({ratio:5.1f}x)  succ {s['success_rate']:.2f}/{m['success_rate']:.2f}")
    print(f"\nwrote: {out_dir}")


if __name__ == "__main__":
    main()
