"""Two-node scanner comparison vs beam divergence (beta / FOV).

Companion to exp_beta_speed_curve.py. Where that script holds the scanner
fixed (Beta walk) and sweeps omega/omega_ideal, this script does the dual:
holds omega = omega_ideal(beta) (the legacy-recommended operating point)
and sweeps beam divergence beta across {3, 5, 10, 15, 20, 25} deg, running
Beta-walk, Spiral, and Raster scanners side-by-side on the SAME randomly
sampled Node-B positions per beta.

Why fix omega = omega_ideal(beta):
  omega_ideal is calibrated to the Beta scanner's dwell statistics
  (beta / (sqrt(2) * C_handshake)). For the comparison to be fair to the
  Beta walk, we pin all three scanners at that rate so the Beta plot
  matches its operating point on exp_beta_speed_curve.py; Spiral/Raster
  are deterministic-coverage and benefit from any reasonable omega within
  the linear-coverage regime.

Plots:
  pdisc_vs_beta.png       - success probability vs beta (3 curves)
  disc_time_vs_beta.png   - mean discovery time +/- 1 sigma vs beta (3 curves)
  events_vs_beta.png      - mean events_processed vs beta (control overhead)

CSV outputs include a `scanner` column so downstream notebooks can join
against exp_beta_speed_curve outputs.

Quick mode runs ~225 trials in well under a minute; mid is ~1800; full
mirrors the exp_beta_speed_curve.py num_pos=50, trials_per_pos=50 size.
"""

import argparse
import csv
import hashlib
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np


# -- Supercomputer / Slurm tips ---------------------------------------------
# srun --cpus-per-task=$N python experiments/exp_scanner_compare_vs_beta.py \
#      --workers $SLURM_CPUS_PER_TASK
# Prevent BLAS oversubscription per worker (set automatically below):
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def _stable_seed(*parts) -> int:
    h = hashlib.blake2b(repr(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % (2 ** 31)


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from owdesmp.core.simulator import Simulator  # noqa: E402


# -- Protocol / handshake timing constants (match exp_beta_speed_curve.py) ---
PACKET_TX_S = 304e-6
PROC_S = 100e-6
TAU_PACKET_S = PACKET_TX_S + 2 * PROC_S
C_HANDSHAKE_S = 3.0 * TAU_PACKET_S

SCANNER_TYPES = ("beta", "spiral", "raster")
SCANNER_COLORS = {"beta": "#1f77b4", "spiral": "#d62728", "raster": "#2ca02c"}


def calculate_omega_ideal(beta_rad: float) -> float:
    return beta_rad / (math.sqrt(2.0) * C_HANDSHAKE_S)


def generate_n2_positions(n_pos: int, dist_m: float, search_angle_rad: float, rng: random.Random):
    """Sample n_pos Node-B positions within a cone of half-angle search_angle
    around +Z, around depth dist_m. Identical to exp_beta_speed_curve.py
    so both experiments draw from the same distribution."""
    cos_min = math.cos(search_angle_rad)
    positions = []
    for _ in range(n_pos):
        cos_theta = rng.uniform(cos_min, 1.0)
        sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        phi = rng.uniform(0.0, 2.0 * math.pi)
        direction = np.array(
            [sin_theta * math.cos(phi), sin_theta * math.sin(phi), cos_theta]
        )
        r = dist_m * rng.uniform(0.9, 1.1)
        positions.append(r * direction)
    return positions


def _scanner_cfg(scanner_type: str) -> dict:
    if scanner_type == "beta":
        return {"type": "beta", "alpha": 0.41485, "beta": 0.41485}
    if scanner_type == "spiral":
        return {"type": "spiral", "samples_per_turn": 60}
    if scanner_type == "raster":
        return {"type": "raster", "step_along": 0.9}
    raise ValueError(f"unknown scanner type {scanner_type!r}")


def build_scenario(
    scanner_type: str,
    beta_deg: float,
    omega_rad_s: float,
    dist_m: float,
    search_angle_rad: float,
    pos_b,
    tx_power_dbm: float,
    sensitivity_dbm: float,
    duration_s: float,
    seed: int,
) -> dict:
    """Build a per-trial scenario, parameterized by scanner.

    Geometry/protocol are identical to exp_beta_speed_curve.build_scenario;
    the only knob is the scanner type. A's estimate of B is the cone center
    so the scanner has to hunt, matching the legacy exp-7 setup.
    """
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
    scanner_cfg = _scanner_cfg(scanner_type)
    terminal_cfg = {
        "beam_divergence_deg": beta_deg,
        "max_angular_speed_deg_s": omega_deg_s,
        "range_m": 100000.0,
        "tx_power_dBm": tx_power_dbm,
        "receiver_aperture_m": 0.08,
        "receiver_sensitivity_dBm": sensitivity_dbm,
    }

    return {
        "sim_name": f"scanner_cmp_{scanner_type}_b{beta_deg:g}",
        "random_seed": seed,
        "simulation": {
            "duration_s": duration_s,
            "mobility_update_period_s": 0.0,
            "beam_sample_interval_s": 0.0001,
        },
        "termination_condition": {"type": "first_link_discovered"},
        "channel": {
            "mode": "threshold",
            "wavelength_nm": 1550,
            "sigma_dBkm": 0.1,
        },
        "nodes": [
            {
                "id": "A",
                "start_time_s": 0.0,
                "position": {"model": "fixed", "initial_m": [0.0, 0.0, 0.0]},
                "optical_terminal": dict(terminal_cfg),
                "scanner": dict(scanner_cfg),
                "protocol": dict(proto),
            },
            {
                "id": "B",
                "start_time_s": 0.0,
                "position": {"model": "fixed", "initial_m": pos_b_list},
                "optical_terminal": dict(terminal_cfg),
                "scanner": dict(scanner_cfg),
                "protocol": dict(proto),
            },
        ],
        "neighbor_knowledge": {
            "pairs": [
                {
                    "src": "A",
                    "dst": "B",
                    "estimated_position_m": a_estimate_of_b,
                    "uncertainty_radius_m": search_radius_m,
                },
                {
                    "src": "B",
                    "dst": "A",
                    "estimated_position_m": [0.0, 0.0, 0.0],
                    "uncertainty_radius_m": search_radius_m,
                },
            ]
        },
    }


def run_single_trial(packed):
    (
        scanner_type,
        beta_deg,
        omega_rad_s,
        dist_m,
        search_angle_rad,
        pos_b,
        tx_power_dbm,
        sensitivity_dbm,
        duration_s,
        seed,
        pos_idx,
        trial,
    ) = packed
    scenario = build_scenario(
        scanner_type, beta_deg, omega_rad_s, dist_m, search_angle_rad, pos_b,
        tx_power_dbm, sensitivity_dbm, duration_s, seed,
    )
    sim = Simulator(scenario=scenario, run_id=0, seed=seed)
    r = sim.run()
    return {
        "scanner": scanner_type,
        "beta_deg": beta_deg,
        "omega_rad_s": omega_rad_s,
        "pos_idx": pos_idx,
        "trial": trial,
        "discovery_time_s": float(r["first_link_time"]),
        "success": bool(r["success"]),
        "events_processed": int(r["events_processed"]),
    }


def aggregate(results):
    buckets = defaultdict(lambda: {"times": [], "succ": [], "events": []})
    for r in results:
        k = (r["scanner"], r["beta_deg"])
        buckets[k]["times"].append(r["discovery_time_s"])
        buckets[k]["succ"].append(1 if r["success"] else 0)
        buckets[k]["events"].append(r["events_processed"])

    agg = []
    for (scanner, beta_deg), v in buckets.items():
        times = v["times"]
        agg.append({
            "scanner": scanner,
            "beta_deg": beta_deg,
            "mean_time_s": statistics.fmean(times),
            "std_time_s": statistics.pstdev(times) if len(times) > 1 else 0.0,
            "success_rate": statistics.fmean(v["succ"]),
            "mean_events": statistics.fmean(v["events"]),
            "n_trials": len(times),
        })
    agg.sort(key=lambda d: (d["scanner"], d["beta_deg"]))
    return agg


def save_csv(results, agg, out_dir: Path):
    raw_path = out_dir / "results_raw.csv"
    fields = ["scanner", "beta_deg", "omega_rad_s", "pos_idx", "trial",
              "discovery_time_s", "success", "events_processed"]
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"  wrote {raw_path}")

    agg_path = out_dir / "results_agg.csv"
    fields = ["scanner", "beta_deg", "mean_time_s", "std_time_s",
              "success_rate", "mean_events", "n_trials"]
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in agg:
            w.writerow(r)
    print(f"  wrote {agg_path}")


def plot_results(agg, out_dir: Path, dist_m: float, search_deg: float, duration_s: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scanners = [s for s in SCANNER_TYPES if any(r["scanner"] == s for r in agg)]

    # --- P(discovery) vs beta ----------------------------------------------
    plt.figure(figsize=(11, 6))
    for s in scanners:
        rows = [r for r in agg if r["scanner"] == s]
        rows.sort(key=lambda r: r["beta_deg"])
        x = [r["beta_deg"] for r in rows]
        y = [r["success_rate"] for r in rows]
        plt.plot(x, y, marker="o", linewidth=2, label=s, color=SCANNER_COLORS.get(s))
    plt.xlabel("Beam divergence beta (deg)")
    plt.ylabel("Success probability P(discovery)")
    plt.title(f"Two-node scanner comparison: P(discovery) vs beta  "
              f"(dist={dist_m:g} m, search={search_deg:g} deg, "
              f"omega=omega_ideal, max={duration_s:g} s)")
    plt.ylim(0, 1.05)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "pdisc_vs_beta.png"
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"  wrote {p}")

    # --- Mean discovery time vs beta ---------------------------------------
    plt.figure(figsize=(11, 6))
    for s in scanners:
        rows = [r for r in agg if r["scanner"] == s]
        rows.sort(key=lambda r: r["beta_deg"])
        x = [r["beta_deg"] for r in rows]
        y = [r["mean_time_s"] for r in rows]
        e = [r["std_time_s"] for r in rows]
        plt.errorbar(x, y, yerr=e, marker="o", capsize=4, linewidth=2,
                     label=s, color=SCANNER_COLORS.get(s))
    plt.axhline(duration_s, color="grey", linestyle="--", alpha=0.5,
                label=f"timeout ({duration_s:g} s)")
    plt.xlabel("Beam divergence beta (deg)")
    plt.ylabel("Mean discovery time (s)")
    plt.title(f"Two-node scanner comparison: discovery time vs beta  "
              f"(dist={dist_m:g} m, search={search_deg:g} deg, omega=omega_ideal)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "disc_time_vs_beta.png"
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"  wrote {p}")

    # --- Events processed vs beta (control overhead proxy) ----------------
    plt.figure(figsize=(11, 6))
    for s in scanners:
        rows = [r for r in agg if r["scanner"] == s]
        rows.sort(key=lambda r: r["beta_deg"])
        x = [r["beta_deg"] for r in rows]
        y = [r["mean_events"] for r in rows]
        plt.plot(x, y, marker="o", linewidth=2, label=s, color=SCANNER_COLORS.get(s))
    plt.xlabel("Beam divergence beta (deg)")
    plt.ylabel("Mean simulator events per trial")
    plt.yscale("log")
    plt.title("Two-node scanner comparison: control overhead vs beta")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    p = out_dir / "events_vs_beta.png"
    plt.savefig(p, dpi=200)
    plt.close()
    print(f"  wrote {p}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true",
                   help="3 betas x 3 scanners x 5 pos x 5 trials (~225 trials)")
    p.add_argument("--mid", action="store_true",
                   help="6 betas x 3 scanners x 10 pos x 10 trials (~1800 trials)")
    p.add_argument("--workers", type=int, default=cpu_count())
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dist", type=float, default=100.0)
    p.add_argument("--search-deg", type=float, default=20.0)
    p.add_argument("--tx-power-dbm", type=float, default=35.0)
    p.add_argument("--sensitivity-dbm", type=float, default=-35.0)
    p.add_argument("--duration", type=float, default=60.0)
    args = p.parse_args()

    full_betas = [3, 5, 10, 15, 20, 25]

    if args.quick:
        beta_vals = [5, 10, 20]
        num_pos = 5
        num_trials_per_pos = 5
    elif args.mid:
        beta_vals = full_betas
        num_pos = 10
        num_trials_per_pos = 10
    else:
        beta_vals = full_betas
        num_pos = 50
        num_trials_per_pos = 50

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or REPO_ROOT / "results" / f"scanner_vs_beta_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")
    print(f"config: dist={args.dist} m, search={args.search_deg} deg, "
          f"tx={args.tx_power_dbm} dBm, sens={args.sensitivity_dbm} dBm, "
          f"duration={args.duration} s")

    # Same position draws shared across scanners so all three see the same
    # realizations of B at each beta. Position sample is independent of beta
    # (geometry is fixed; only beam footprint changes with beta).
    pos_rng = random.Random(42)
    search_angle_rad = math.radians(args.search_deg)
    positions = generate_n2_positions(num_pos, args.dist, search_angle_rad, pos_rng)

    tasks = []
    for beta_deg in beta_vals:
        omega_rad_s = calculate_omega_ideal(math.radians(beta_deg))
        for scanner_type in SCANNER_TYPES:
            for pos_idx, pos_b in enumerate(positions):
                for trial in range(num_trials_per_pos):
                    seed = _stable_seed("scanner_cmp", scanner_type, beta_deg,
                                        pos_idx, trial)
                    tasks.append((
                        scanner_type, beta_deg, omega_rad_s, args.dist,
                        search_angle_rad, pos_b, args.tx_power_dbm,
                        args.sensitivity_dbm, args.duration, seed,
                        pos_idx, trial,
                    ))
    # LPT scheduling: smallest beta = smallest beam footprint = slowest trial
    # (Spiral/Raster have N(search/footprint)^2 coverage). Run those first.
    tasks.sort(key=lambda t: t[1])
    print(f"total trials: {len(tasks)}")
    t_start = time.perf_counter()

    results = []
    if args.workers > 1 and len(tasks) > args.workers:
        chunk = max(1, min(64, len(tasks) // (args.workers * 8)))
        with Pool(processes=args.workers) as pool:
            report_every = max(1, len(tasks) // 20)
            for i, r in enumerate(pool.imap_unordered(run_single_trial, tasks, chunksize=chunk)):
                results.append(r)
                if (i + 1) % report_every == 0 or (i + 1) == len(tasks):
                    elapsed = time.perf_counter() - t_start
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (len(tasks) - (i + 1)) / rate if rate > 0 else float("inf")
                    print(f"  [{_ts()}] {i+1}/{len(tasks)} ({100*(i+1)/len(tasks):5.1f}%) "
                          f"elapsed={elapsed:6.1f}s ETA={eta:6.1f}s")
    else:
        for i, task in enumerate(tasks):
            results.append(run_single_trial(task))
            if (i + 1) % max(1, len(tasks) // 10) == 0 or (i + 1) == len(tasks):
                print(f"  [{_ts()}] {i+1}/{len(tasks)}")

    agg = aggregate(results)
    save_csv(results, agg, out_dir)
    plot_results(agg, out_dir, args.dist, args.search_deg, args.duration)

    print("\nsummary (scanner x beta -> mean discovery / success):")
    print(f"  {'scanner':<8} {'beta [deg]':>10} {'mean_t [s]':>11} {'success':>8} {'events':>10} {'n':>5}")
    for r in agg:
        print(f"  {r['scanner']:<8} {r['beta_deg']:>10.1f} {r['mean_time_s']:>11.3f} "
              f"{r['success_rate']:>8.1%} {r['mean_events']:>10.0f} {r['n_trials']:>5d}")


if __name__ == "__main__":
    main()
