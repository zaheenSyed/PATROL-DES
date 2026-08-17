"""Scanner vibration robustness: discovery time vs beam divergence under
multiple vibration magnitudes for Beta-walk, Spiral, and Raster scanners.

Sweep axes
----------
    beta_deg     {1, 3, 5, 10}
    scanner      {beta, spiral, raster}
    condition    {static, mild, measured, severe}   (mag_mult 0, 0.5, 1.0, 2.0)

Fixed geometry (matches the arcc spiral vibration experiment so the static
baseline is clean and t_vib/t_static is measured against 100% discovery):
    d = 100 m, scan_radius = 5 m, omega = omega_ideal(beta), threshold channel,
    target AT the peer estimate (NO random disk offset).

Why no offset: a ~1 deg beam cannot sweep a 5 m offset disk inside the time
budget, so any random offset collapses spiral/raster static discovery to
~10-30% at narrow beta -- every penalty ratio then divides by noise. With the
target at the estimate all scanners discover 100% statically and the vibration
penalty isolates the disturbance cleanly.

Produces four figures:
    1. Discovery time vs β — all scanners × vibration on one plot
    2. Bar chart — vibration magnitude effect grouped by scanner
    3. Bar chart — discovery probability by scanner × condition
    4. Penalty heatmap — scanner × β, colored by t_vib / t_static

Quick = 30 trials/cell   (~1440 trials, ~2 min on 8 cores)
Mid   = 100 trials/cell  (~4800, ~8 min)
Full  = 500 trials/cell  (~24000, ~40 min)
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from owdesmp.core.simulator import Simulator  # noqa: E402
from owdesmp.plotting.journal import (  # noqa: E402
    apply_journal_style, bootstrap_ci, figsize_1col, figsize_2col,
    save_journal, WONG_PALETTE,
)

POSE_CSV_ABS = (REPO_ROOT / "drone data" / "hover" / "pose_deviation.csv").resolve()

# ---- Sweep axes ----------------------------------------------------------
BETA_VALUES = [1.0, 3.0, 5.0, 10.0]
SCANNERS = ["beta", "spiral", "raster"]
CONDITIONS = [
    ("static",   0.0),
    ("mild",     0.5),
    ("measured", 1.0),
    ("severe",   2.0),
]

# ---- Omega = omega_ideal(beta): the legacy-recommended operating point, the
#      same rate used by the disc_time_vs_beta baseline figure. -------------
PACKET_TX_S = 304e-6
PROC_S = 100e-6
C_HANDSHAKE_S = 3.0 * (PACKET_TX_S + 2 * PROC_S)
OMEGA_MULT = 1.0
BETA_RADIUS_MODE = "uniform_disk"

DIST_M = 100.0
UNCERTAINTY_RADIUS_M = 5.0
DURATION_S = 120.0

# ---- Visual style --------------------------------------------------------
SCANNER_COLOR = {
    "beta":   WONG_PALETTE[5],   # blue
    "spiral": WONG_PALETTE[6],   # vermilion
    "raster": WONG_PALETTE[3],   # bluish green
}
SCANNER_MARKER = {"beta": "o", "spiral": "s", "raster": "D"}
SCANNER_LABEL = {"beta": "Beta", "spiral": "Spiral", "raster": "Raster"}

CONDITION_LINESTYLE = {
    "static":   "-",
    "mild":     "--",
    "measured": "-.",
    "severe":   ":",
}
CONDITION_LABEL = {
    "static":   "Static",
    "mild":     r"Mild (0.5$\times$)",
    "measured": r"Measured (1$\times$)",
    "severe":   r"Severe (2$\times$)",
}


def _omega_deg_s(beta_deg: float) -> float:
    beta_rad = math.radians(beta_deg)
    return math.degrees(OMEGA_MULT * beta_rad / (math.sqrt(2.0) * C_HANDSHAKE_S))


def _stable_seed(*parts) -> int:
    h = hashlib.blake2b(repr(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") % (2 ** 31)


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _scanner_cfg(name: str) -> dict:
    if name == "spiral":
        return {"type": "spiral", "samples_per_turn": 60}
    if name == "raster":
        return {"type": "raster", "step_along": 0.9}
    if name == "beta":
        return {"type": "beta", "alpha": 0.41485, "beta": 0.41485,
                "radius_mode": BETA_RADIUS_MODE}
    raise ValueError(f"unknown scanner: {name}")


def build_scenario(beta_deg, scanner_name, mag_mult, seed,
                   duration_s=DURATION_S):
    csv_path = str(POSE_CSV_ABS)
    proto = {
        "beacon_period_s": 0.001512,
        "beacon_phase_s": "random",
        "packet_duration_s": 0.000404,
        "processing_delay_s": 0.0001,
        "handshake_timeout_s": 0.02,
    }
    terminal_base = {
        "beam_divergence_deg": beta_deg,
        "max_angular_speed_deg_s": _omega_deg_s(beta_deg),
        "range_m": 1000.0,
        "tx_power_dBm": 20.0,
        "receiver_aperture_m": 0.08,
        "receiver_sensitivity_dBm": -35.0,
    }
    if mag_mult > 0.0:
        terminal_a = dict(terminal_base, pointing_jitter={
            "type": "csv_playback", "csv_path": csv_path,
            "time_offset_s": 0.0, "mag_mult": float(mag_mult)})
        terminal_b = dict(terminal_base, pointing_jitter={
            "type": "csv_playback", "csv_path": csv_path,
            "time_offset_s": 80.0, "mag_mult": float(mag_mult)})
    else:
        terminal_a = dict(terminal_base)
        terminal_b = dict(terminal_base)

    scanner = _scanner_cfg(scanner_name)

    pos_b_est = [DIST_M, 0.0, 0.0]
    pos_a_est = [0.0, 0.0, 0.0]
    # No disk offset: each node sits at its estimated position, so the target
    # lands at the scanner's first dwell. The scanner still sweeps the 5 m
    # uncertainty disk -- under vibration the beam is knocked off the centred
    # target and must re-acquire, which is exactly the disturbance we measure.
    pos_b_actual = list(pos_b_est)
    pos_a_actual = list(pos_a_est)

    return {
        "sim_name": f"svj_b{beta_deg:g}_{scanner_name}_m{mag_mult:g}",
        "random_seed": seed,
        "simulation": {
            "duration_s": duration_s,
            "mobility_update_period_s": 0.001,
            "beam_sample_interval_s": 0.0001,
        },
        "termination_condition": {"type": "first_link_discovered"},
        "channel": {
            "mode": "threshold", "wavelength_nm": 1550,
            "sigma_dBkm": 0.0, "Cn2": 1e-20,
        },
        "neighbor_knowledge": {
            "pairs": [
                {"src": "A", "dst": "B", "estimated_position_m": pos_b_est,
                 "uncertainty_radius_m": UNCERTAINTY_RADIUS_M},
                {"src": "B", "dst": "A", "estimated_position_m": pos_a_est,
                 "uncertainty_radius_m": UNCERTAINTY_RADIUS_M},
            ],
        },
        "nodes": [
            {"id": "A", "start_time_s": 0.0,
             "position": {"model": "fixed", "initial_m": pos_a_actual},
             "optical_terminal": terminal_a,
             "initial_pointing": [1.0, 0.0, 0.0],
             "scanner": dict(scanner),
             "protocol": dict(proto)},
            {"id": "B", "start_time_s": 0.0,
             "position": {"model": "fixed", "initial_m": pos_b_actual},
             "optical_terminal": terminal_b,
             "initial_pointing": [-1.0, 0.0, 0.0],
             "scanner": dict(scanner),
             "protocol": dict(proto)},
        ],
    }


def run_single_trial(packed):
    beta_deg, scanner_name, mag_mult, condition, seed, trial = packed
    scenario = build_scenario(beta_deg, scanner_name, mag_mult, seed)
    sim = Simulator(scenario=scenario, run_id=0, seed=seed)
    r = sim.run()
    t_disc = float(r.get("first_link_time", DURATION_S))
    return {
        "beta_deg": beta_deg, "scanner": scanner_name,
        "mag_mult": mag_mult, "condition": condition, "trial": trial,
        "first_link_time_s": t_disc,
        "discovered": int(t_disc < DURATION_S),
    }


def aggregate(results):
    buckets = defaultdict(lambda: {"t": [], "ok": []})
    for r in results:
        k = (r["beta_deg"], r["scanner"], r["condition"])
        buckets[k]["t"].append(r["first_link_time_s"])
        buckets[k]["ok"].append(r["discovered"])
    rows = []
    for (beta, sc, cond), d in buckets.items():
        succ = [t for t, ok in zip(d["t"], d["ok"]) if ok]
        if succ:
            mean, lo, hi = bootstrap_ci(succ)
            med = float(np.median(succ))
        else:
            mean, lo, hi = float("nan"), float("nan"), float("nan")
            med = float("nan")
        rows.append({
            "beta_deg": beta, "scanner": sc, "condition": cond,
            "n_trials": len(d["t"]),
            "discovery_rate": float(np.mean(d["ok"])),
            "mean_t_disc_s": mean, "ci_lo_s": lo, "ci_hi_s": hi,
            "median_t_disc_s": med,
            "mean_t_disc_all_s": float(np.mean(d["t"])),
        })
    rows.sort(key=lambda r: (r["condition"], r["scanner"], r["beta_deg"]))
    return rows


def save_csv(results, agg, out_dir):
    rp = out_dir / "scanner_vib_journal_raw.csv"
    with open(rp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); [w.writerow(r) for r in results]
    print(f"  wrote {rp}")
    ap = out_dir / "scanner_vib_journal_agg.csv"
    with open(ap, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader(); [w.writerow(r) for r in agg]
    print(f"  wrote {ap}")


def _row(agg, beta, sc, cond):
    return next((r for r in agg if r["beta_deg"] == beta
                 and r["scanner"] == sc and r["condition"] == cond), None)


# ======================================================================
# FIGURE 1 — Discovery time vs β, all scanners × all vibration levels
# ======================================================================
def plot_discovery_time_vs_beta(agg, out_dir):
    import matplotlib.pyplot as plt
    apply_journal_style()
    fig, ax = plt.subplots(figsize=figsize_2col(aspect=0.45))
    betas_sorted = sorted({r["beta_deg"] for r in agg})

    for sc in SCANNERS:
        for cond, _ in CONDITIONS:
            ys, los, his = [], [], []
            for b in betas_sorted:
                row = _row(agg, b, sc, cond)
                ys.append(row["mean_t_disc_s"] if row else np.nan)
                los.append(row["ci_lo_s"] if row else np.nan)
                his.append(row["ci_hi_s"] if row else np.nan)
            ys = np.array(ys, dtype=float)
            yerr_lo = ys - np.array(los, dtype=float)
            yerr_hi = np.array(his, dtype=float) - ys
            label = f"{SCANNER_LABEL[sc]} – {CONDITION_LABEL[cond]}"
            ax.errorbar(betas_sorted, ys * 1000,
                        yerr=[yerr_lo * 1000, yerr_hi * 1000],
                        marker=SCANNER_MARKER[sc], linewidth=1.1,
                        markersize=4, capsize=2,
                        color=SCANNER_COLOR[sc],
                        linestyle=CONDITION_LINESTYLE[cond],
                        label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Beam divergence $\beta$ (deg)")
    ax.set_ylabel("Mean discovery time (ms)")
    ax.set_xticks(BETA_VALUES)
    ax.set_xticklabels([f"{b:g}" for b in BETA_VALUES])
    ax.set_title(
        f"Discovery time: all scanners × vibration levels\n"
        f"$d={DIST_M:g}$ m, search $r={UNCERTAINTY_RADIUS_M:g}$ m",
        loc="left", fontsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=6, ncol=1, columnspacing=0.8, handlelength=2.0)
    fig.tight_layout()
    p, _ = save_journal(fig, out_dir / "fig1_discovery_time_vs_beta")
    plt.close(fig)
    print(f"  wrote {p}")


# ======================================================================
# FIGURE 2 — Grouped bar chart: vibration magnitude effect
# ======================================================================
def plot_magnitude_bars(agg, out_dir):
    import matplotlib.pyplot as plt
    apply_journal_style()
    betas_sorted = sorted({r["beta_deg"] for r in agg})
    cond_labels = [c for c, _ in CONDITIONS]
    n_conds = len(cond_labels)
    n_betas = len(betas_sorted)

    fig, axes = plt.subplots(1, len(SCANNERS), figsize=figsize_2col(aspect=0.42),
                             sharey=True, gridspec_kw=dict(wspace=0.08))

    # Color palette for vibration levels: increasing saturation
    cond_colors = [WONG_PALETTE[5], WONG_PALETTE[2], WONG_PALETTE[1], WONG_PALETTE[6]]

    bar_width = 0.18
    x = np.arange(n_betas)

    for ax_idx, sc in enumerate(SCANNERS):
        ax = axes[ax_idx]
        for ci, cond in enumerate(cond_labels):
            vals = []
            for b in betas_sorted:
                row = _row(agg, b, sc, cond)
                vals.append(row["mean_t_disc_s"] * 1000 if row and not math.isnan(row["mean_t_disc_s"]) else 0)
            offset = (ci - (n_conds - 1) / 2) * bar_width
            bars = ax.bar(x + offset, vals, bar_width * 0.9,
                          color=cond_colors[ci], label=CONDITION_LABEL[cond],
                          edgecolor="white", linewidth=0.3)
            # Add value labels on tall bars
            for bar_obj, v in zip(bars, vals):
                if v > 0:
                    fontsize = 4.5
                    ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                            bar_obj.get_height(), f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                            ha="center", va="bottom", fontsize=fontsize, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{b:g}°" for b in betas_sorted])
        ax.set_xlabel(r"$\beta$ (deg)")
        ax.set_title(SCANNER_LABEL[sc], fontsize=8)
        ax.set_yscale("log")
        if ax_idx == 0:
            ax.set_ylabel("Mean discovery time (ms)")

    axes[-1].legend(loc="upper right", fontsize=5, title="Vibration",
                    title_fontsize=5.5)
    fig.suptitle("Effect of vibration magnitude on discovery time",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    p, _ = save_journal(fig, out_dir / "fig2_magnitude_bars")
    plt.close(fig)
    print(f"  wrote {p}")


# ======================================================================
# FIGURE 3 — Grouped bar chart: discovery probability
# ======================================================================
def plot_probability_bars(agg, out_dir):
    import matplotlib.pyplot as plt
    apply_journal_style()
    betas_sorted = sorted({r["beta_deg"] for r in agg})
    cond_labels = [c for c, _ in CONDITIONS]
    n_conds = len(cond_labels)
    n_betas = len(betas_sorted)

    cond_colors = [WONG_PALETTE[5], WONG_PALETTE[2], WONG_PALETTE[1], WONG_PALETTE[6]]

    fig, axes = plt.subplots(1, len(SCANNERS), figsize=figsize_2col(aspect=0.42),
                             sharey=True, gridspec_kw=dict(wspace=0.08))

    bar_width = 0.18
    x = np.arange(n_betas)

    for ax_idx, sc in enumerate(SCANNERS):
        ax = axes[ax_idx]
        for ci, cond in enumerate(cond_labels):
            vals = []
            for b in betas_sorted:
                row = _row(agg, b, sc, cond)
                vals.append(row["discovery_rate"] * 100 if row else 0)
            offset = (ci - (n_conds - 1) / 2) * bar_width
            bars = ax.bar(x + offset, vals, bar_width * 0.9,
                          color=cond_colors[ci], label=CONDITION_LABEL[cond],
                          edgecolor="white", linewidth=0.3)
            # Annotate <100% bars
            for bar_obj, v in zip(bars, vals):
                if 0 < v < 100:
                    ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                            bar_obj.get_height() + 0.5,
                            f"{v:.0f}%", ha="center", va="bottom",
                            fontsize=4.5, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{b:g}°" for b in betas_sorted])
        ax.set_xlabel(r"$\beta$ (deg)")
        ax.set_title(SCANNER_LABEL[sc], fontsize=8)
        ax.set_ylim(0, 115)
        ax.axhline(100, color="grey", linestyle=":", linewidth=0.5, alpha=0.5)
        if ax_idx == 0:
            ax.set_ylabel("Discovery probability (%)")

    axes[-1].legend(loc="lower right", fontsize=5, title="Vibration",
                    title_fontsize=5.5)
    fig.suptitle("Discovery probability by scanner and vibration level",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    p, _ = save_journal(fig, out_dir / "fig3_probability_bars")
    plt.close(fig)
    print(f"  wrote {p}")


# ======================================================================
# FIGURE 4 — Penalty heatmap: scanner × β for each vibration level
# ======================================================================
def plot_penalty_heatmap(agg, out_dir):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    apply_journal_style()

    betas_sorted = sorted({r["beta_deg"] for r in agg})
    vib_conds = [c for c, m in CONDITIONS if m > 0]  # exclude static
    n_panels = len(vib_conds)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize_2col(aspect=0.38),
                             gridspec_kw=dict(wspace=0.35))
    if n_panels == 1:
        axes = [axes]

    # Compute all penalty ratios for shared colorbar
    all_ratios = []
    for cond in vib_conds:
        for sc in SCANNERS:
            for b in betas_sorted:
                r_s = _row(agg, b, sc, "static")
                r_v = _row(agg, b, sc, cond)
                if (r_s and r_v and not math.isnan(r_s["mean_t_disc_s"])
                        and not math.isnan(r_v["mean_t_disc_s"])
                        and r_s["mean_t_disc_s"] > 0):
                    all_ratios.append(r_v["mean_t_disc_s"] / r_s["mean_t_disc_s"])

    vmin = max(0.5, min(all_ratios)) if all_ratios else 0.5
    vmax = max(all_ratios) * 1.2 if all_ratios else 100.0

    panel_labels = ["(a)", "(b)", "(c)"]
    last_im = None
    for pi, (cond, ax) in enumerate(zip(vib_conds, axes)):
        mat = np.full((len(SCANNERS), len(betas_sorted)), np.nan)
        for si, sc in enumerate(SCANNERS):
            for bi, b in enumerate(betas_sorted):
                r_s = _row(agg, b, sc, "static")
                r_v = _row(agg, b, sc, cond)
                if (r_s and r_v and not math.isnan(r_s["mean_t_disc_s"])
                        and not math.isnan(r_v["mean_t_disc_s"])
                        and r_s["mean_t_disc_s"] > 0):
                    mat[si, bi] = r_v["mean_t_disc_s"] / r_s["mean_t_disc_s"]

        mat_masked = np.ma.masked_invalid(mat)
        last_im = ax.imshow(mat_masked, aspect="auto", origin="lower",
                            cmap="YlOrRd",
                            norm=LogNorm(vmin=vmin, vmax=vmax))

        # Annotate cells
        for si in range(len(SCANNERS)):
            for bi in range(len(betas_sorted)):
                v = mat[si, bi]
                if not np.isnan(v):
                    txt = f"{v:.1f}×" if v < 100 else f"{v:.0f}×"
                    color = "white" if v > vmax * 0.3 else "black"
                    ax.text(bi, si, txt, ha="center", va="center",
                            fontsize=6, fontweight="bold", color=color)

        ax.set_xticks(range(len(betas_sorted)))
        ax.set_xticklabels([f"{b:g}°" for b in betas_sorted])
        ax.set_yticks(range(len(SCANNERS)))
        ax.set_yticklabels([SCANNER_LABEL[s] for s in SCANNERS])
        ax.set_xlabel(r"$\beta$ (deg)")
        if pi == 0:
            ax.set_ylabel("Scanner")
        ax.set_title(f"{panel_labels[pi]} {CONDITION_LABEL[cond]}",
                     loc="left", fontsize=7.5)
        ax.grid(False)

    cbar = fig.colorbar(last_im, ax=axes if isinstance(axes, list) else axes.ravel().tolist(),
                        fraction=0.04, pad=0.04, shrink=0.85)
    cbar.set_label(r"Penalty ratio $t_\mathrm{vib}\,/\,t_\mathrm{static}$",
                   fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        f"Vibration penalty: $d={DIST_M:g}$ m, search $r={UNCERTAINTY_RADIUS_M:g}$ m",
        fontsize=9, y=1.04)
    fig.tight_layout()
    p, _ = save_journal(fig, out_dir / "fig4_penalty_heatmap")
    plt.close(fig)
    print(f"  wrote {p}")


# ======================================================================
# Main
# ======================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--smoke", action="store_true",
                   help="5 trials/cell (~240 trials) — pipeline check only")
    p.add_argument("--quick", action="store_true",
                   help="30 trials/cell (~1440 trials)")
    p.add_argument("--mid", action="store_true",
                   help="100 trials/cell (~4800)")
    p.add_argument("--workers", type=int, default=cpu_count())
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    if not POSE_CSV_ABS.exists():
        sys.exit(f"pose_deviation.csv not found at {POSE_CSV_ABS}")

    trials = (5 if args.smoke else
              30 if args.quick else
              100 if args.mid else 500)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or REPO_ROOT / "results" / f"scanner_vib_journal_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")
    print(f"axes: betas={BETA_VALUES}, scanners={SCANNERS}")
    print(f"conditions: {[(c, m) for c, m in CONDITIONS]}, trials/cell={trials}")
    n_cells = len(BETA_VALUES) * len(SCANNERS) * len(CONDITIONS)
    print(f"total cells: {n_cells}, total trials: {n_cells * trials}")

    (out_dir / "axes.json").write_text(json.dumps({
        "beta_deg": BETA_VALUES, "scanners": SCANNERS,
        "conditions": [{"label": c, "mag_mult": m} for c, m in CONDITIONS],
        "trials_per_cell": trials, "dist_m": DIST_M,
        "uncertainty_radius_m": UNCERTAINTY_RADIUS_M,
        "duration_s": DURATION_S,
        "omega_mult": OMEGA_MULT,
        "omega_deg_s_per_beta": {f"beta_{b:g}": _omega_deg_s(b)
                                  for b in BETA_VALUES},
        "beta_radius_mode": BETA_RADIUS_MODE,
        "position_offset": "none (target at peer estimate)",
    }, indent=2))

    tasks = []
    for beta in BETA_VALUES:
        for sc in SCANNERS:
            for cond, mag in CONDITIONS:
                for trial in range(trials):
                    seed = _stable_seed("scanner_vib_journal", beta, sc, cond, trial)
                    tasks.append((beta, sc, mag, cond, seed, trial))
    # LPT: narrow beta + vibration cells dominate runtime; run those first.
    tasks.sort(key=lambda t: (t[0], -t[2]))
    print(f"total trial tasks: {len(tasks)}")

    t0 = time.perf_counter()
    results = []
    if args.workers > 1 and len(tasks) > args.workers:
        chunk = max(1, min(64, len(tasks) // (args.workers * 8)))
        with Pool(processes=args.workers, maxtasksperchild=20) as pool:
            for i, r in enumerate(pool.imap_unordered(run_single_trial,
                                                       tasks, chunksize=chunk)):
                results.append(r)
                if (i + 1) % max(1, len(tasks) // 10) == 0:
                    elapsed = time.perf_counter() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta = (len(tasks) - (i + 1)) / rate if rate > 0 else float("inf")
                    print(f"  [{_ts()}] {i+1}/{len(tasks)} "
                          f"elapsed={elapsed:5.1f}s ETA={eta:5.1f}s")
    else:
        for i, t in enumerate(tasks):
            results.append(run_single_trial(t))
            if (i + 1) % max(1, len(tasks) // 5) == 0:
                print(f"  [{_ts()}] {i+1}/{len(tasks)}")

    elapsed = time.perf_counter() - t0
    print(f"\nsimulation complete: {len(results)} trials in {elapsed:.1f}s")

    agg = aggregate(results)
    save_csv(results, agg, out_dir)

    print("\ngenerating figures ...")
    plot_discovery_time_vs_beta(agg, out_dir)
    plot_magnitude_bars(agg, out_dir)
    plot_probability_bars(agg, out_dir)
    plot_penalty_heatmap(agg, out_dir)

    print(f"\nsummary:")
    print(f"  {'beta':>5} {'scanner':<8} {'cond':<10} {'disc%':>6} "
          f"{'mean ms':>10} {'median ms':>10}")
    for r in agg:
        v = r["mean_t_disc_s"]
        m = r["median_t_disc_s"]
        v_str = f"{v*1000:>10.2f}" if not math.isnan(v) else "        TO"
        m_str = f"{m*1000:>10.2f}" if not math.isnan(m) else "        TO"
        print(f"  {r['beta_deg']:>5.1f} {r['scanner']:<8} {r['condition']:<10} "
              f"{r['discovery_rate']*100:>5.1f}% {v_str} {m_str}")


if __name__ == "__main__":
    main()
