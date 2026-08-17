"""Plotting helpers for the Phase-A/B/C link-maintenance experiments.

These functions read the CSV outputs from:
  experiments/exp_link_availability_vs_fov.py    (Exp 2)
  experiments/exp_throughput_laser_vs_vlc.py     (Exp 3)
  experiments/exp_tracker_comparison.py          (Exp 4)

and re-render the figures in a journal-grade style, using the OWDES-MP
plotting/journal.py palette and sizing. Each function takes a results
directory (the same dir the experiment script wrote `results_raw.csv` and
`results_agg.csv` to) and writes PDF + PNG figures next to those CSVs.

Schemas (must match the experiments):
  Exp 2  results_agg.csv:
    beta_deg, mag_mult, mean_availability, std_availability, mean_outages,
    mean_first_link_s, success_rate, n_trials
  Exp 3  results_agg.csv:
    link_type, mag_mult, mean_mbps, std_mbps, mean_outages, mean_bytes,
    n_trials
  Exp 4  results_agg.csv:
    tracker, mag_mult, mean_availability, std_availability, mean_outages,
    mean_first_link_s, n_trials

CLI:
    python -m owdesmp.plotting.link_maintenance <results_dir> [--kind 2|3|4]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from owdesmp.plotting.journal import (
    COL_2,
    WONG_PALETTE,
    apply_journal_style,
    figsize_1col,
    figsize_2col,
    save_journal,
)


TRACKER_COLORS = {
    "none": WONG_PALETTE[0],
    "pid": WONG_PALETTE[2],
    "kalman": WONG_PALETTE[3],
    "rl": WONG_PALETTE[6],
}
LINKTYPE_COLORS = {"laser": WONG_PALETTE[2], "vlc": WONG_PALETTE[1]}
MAG_MARKERS = {0.0: "o", 0.5: "v", 1.0: "s", 2.0: "^", 3.0: "D"}


# --------------------------------------------------- Exp 2: link availability


def plot_link_availability(results_dir: Path) -> Path:
    """Render Exp 2: link availability vs beam divergence under vibration."""
    import matplotlib.pyplot as plt

    apply_journal_style()
    agg = pd.read_csv(results_dir / "results_agg.csv")
    if agg.empty:
        raise ValueError(f"results_agg.csv at {results_dir} is empty")

    fig, ax = plt.subplots(figsize=figsize_2col(aspect=0.45))
    for mag in sorted(agg["mag_mult"].unique()):
        rows = agg[agg["mag_mult"] == mag].sort_values("beta_deg")
        ax.errorbar(
            rows["beta_deg"], rows["mean_availability"] * 100,
            yerr=rows["std_availability"] * 100,
            marker=MAG_MARKERS.get(mag, "o"), capsize=4,
            linewidth=2, markersize=7, label=fr"Vibration $\times${mag:g}",
        )
    ax.set_xlabel(r"Beam divergence $\beta$ (deg)")
    ax.set_ylabel("Link availability (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    out = results_dir / "fig_link_availability_vs_fov"
    save_journal(fig, out)
    plt.close(fig)
    return out


# ------------------------------------------------ Exp 3: throughput laser vs VLC


def plot_throughput_comparison(results_dir: Path) -> Path:
    """Render Exp 3: laser vs VLC effective throughput across vibration mags."""
    import matplotlib.pyplot as plt

    apply_journal_style()
    agg = pd.read_csv(results_dir / "results_agg.csv")
    if agg.empty:
        raise ValueError(f"results_agg.csv at {results_dir} is empty")

    mags = sorted(agg["mag_mult"].unique())
    link_types = sorted(agg["link_type"].unique())
    x = np.arange(len(mags))
    width = 0.8 / max(1, len(link_types))

    fig, ax = plt.subplots(figsize=figsize_2col(aspect=0.45))
    for i, lt in enumerate(link_types):
        rows = agg[agg["link_type"] == lt].sort_values("mag_mult")
        offset = (i - len(link_types) / 2 + 0.5) * width
        ax.bar(
            x + offset, rows["mean_mbps"], width,
            yerr=rows["std_mbps"], capsize=3,
            label=lt.upper(), color=LINKTYPE_COLORS.get(lt, "#666"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([fr"$\times${m:g}" for m in mags])
    ax.set_xlabel("Vibration magnitude multiplier")
    ax.set_ylabel("Effective TCP throughput (Mbps)")
    ax.set_yscale("log")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, which="both", axis="y")
    out = results_dir / "fig_throughput_laser_vs_vlc"
    save_journal(fig, out)
    plt.close(fig)
    return out


# -------------------------------------------------- Exp 4: tracker comparison


def plot_tracker_outages(results_dir: Path) -> Path:
    """Render Exp 4 outages: mean outage count per trial across trackers/mags."""
    return _plot_tracker_metric(
        results_dir, metric="mean_outages",
        ylabel="Mean outage count per trial",
        title="Link outages by tracking algorithm",
        out_stem="fig_tracker_outages",
        ylim=None,
        as_pct=False,
    )


def plot_tracker_availability(results_dir: Path) -> Path:
    """Render Exp 4 availability: link availability % across trackers/mags."""
    return _plot_tracker_metric(
        results_dir, metric="mean_availability",
        ylabel="Link availability (%)",
        title="Link availability by tracking algorithm",
        out_stem="fig_tracker_availability",
        ylim=(0, 105),
        as_pct=True,
        yerr_col="std_availability",
    )


def _plot_tracker_metric(
    results_dir: Path, metric: str, ylabel: str, title: str,
    out_stem: str, ylim, as_pct: bool, yerr_col: str = None,
) -> Path:
    import matplotlib.pyplot as plt

    apply_journal_style()
    agg = pd.read_csv(results_dir / "results_agg.csv")
    if agg.empty:
        raise ValueError(f"results_agg.csv at {results_dir} is empty")

    mags = sorted(agg["mag_mult"].unique())
    order = ["none", "pid", "kalman", "rl"]
    trackers = [t for t in order if t in agg["tracker"].unique()]
    x = np.arange(len(mags))
    width = 0.8 / max(1, len(trackers))

    fig, ax = plt.subplots(figsize=figsize_2col(aspect=0.45))
    for i, tr in enumerate(trackers):
        rows = agg[agg["tracker"] == tr].sort_values("mag_mult")
        y = rows[metric].to_numpy()
        if as_pct:
            y = y * 100
        yerr = None
        if yerr_col and yerr_col in rows.columns:
            yerr = rows[yerr_col].to_numpy()
            if as_pct:
                yerr = yerr * 100
        offset = (i - len(trackers) / 2 + 0.5) * width
        ax.bar(
            x + offset, y, width,
            yerr=yerr, capsize=3,
            label=tr.upper(), color=TRACKER_COLORS.get(tr, "#666"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([fr"$\times${m:g}" for m in mags])
    ax.set_xlabel("Vibration magnitude multiplier")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    out = results_dir / out_stem
    save_journal(fig, out)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------- CLI


# =========================================================================
# Phase F: paper-load-bearing envelope figures
# =========================================================================


def plot_availability_envelope(results_dir: Path) -> Path:
    """Figure 3a: availability heatmap grid (one panel per tracker).

    Rows of each heatmap = beta_deg (ascending), cols = mag_mult (ascending).
    Cell color = mean_availability (%). Same colorbar across all panels for
    comparison. The "RL opportunity gap" is the red region in the
    tight-beam / high-jitter corner of the {none, pid, kalman} panels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    apply_journal_style()
    agg = pd.read_csv(results_dir / "results_agg.csv")
    if agg.empty:
        raise ValueError(f"results_agg.csv at {results_dir} is empty")

    betas = sorted(agg["beta_deg"].unique())
    mags = sorted(agg["mag_mult"].unique())
    tracker_order = ["none", "pid", "kalman", "rl"]
    trackers = [t for t in tracker_order if t in agg["tracker"].unique()]

    # Red-yellow-green colormap (low = red = bad, high = green = good).
    cmap = LinearSegmentedColormap.from_list(
        "availability",
        ["#b2182b", "#fee08b", "#1a9850"],
        N=256,
    )

    fig, axes = plt.subplots(1, len(trackers), figsize=figsize_2col(aspect=0.40),
                             sharey=True)
    if len(trackers) == 1:
        axes = [axes]

    for ax, tr in zip(axes, trackers):
        z = np.full((len(betas), len(mags)), np.nan)
        for i, b in enumerate(betas):
            for j, m in enumerate(mags):
                row = agg[(agg["tracker"] == tr)
                          & (agg["beta_deg"] == b)
                          & (agg["mag_mult"] == m)]
                if not row.empty:
                    z[i, j] = float(row["mean_availability"].iloc[0]) * 100.0
        im = ax.imshow(z, aspect="auto", origin="lower", cmap=cmap,
                       vmin=0.0, vmax=100.0)
        ax.set_xticks(range(len(mags)))
        ax.set_xticklabels([f"{m:g}×" for m in mags])
        ax.set_yticks(range(len(betas)))
        ax.set_yticklabels([f"{b:g}°" for b in betas])
        ax.set_xlabel("Vibration ×")
        ax.set_title(tr.upper(), pad=4)
        # Cell-level annotations.
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                v = z[i, j]
                if not np.isnan(v):
                    color = "white" if v < 40 or v > 80 else "black"
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            color=color, fontsize=6)
    axes[0].set_ylabel(r"Beam divergence $\beta$")

    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02,
                        orientation="vertical")
    cbar.set_label("Link availability (%)", rotation=270, labelpad=10)

    out = results_dir / "fig_availability_envelope"
    save_journal(fig, out)
    plt.close(fig)
    return out


def plot_throughput_frontier(results_dir: Path) -> Path:
    """Figure 3b: throughput vs availability Pareto scatter.

    Each point is one (beta, mag, tracker) operating cell. Color encodes the
    tracker; marker encodes the vibration magnitude. Pareto-optimal points
    sit on the upper-right frontier (high availability AND high throughput).
    """
    import matplotlib.pyplot as plt

    apply_journal_style()
    agg = pd.read_csv(results_dir / "results_agg.csv")
    if agg.empty:
        raise ValueError(f"results_agg.csv at {results_dir} is empty")

    tracker_order = ["none", "pid", "kalman", "rl"]
    trackers = [t for t in tracker_order if t in agg["tracker"].unique()]
    mags = sorted(agg["mag_mult"].unique())
    mag_markers = {0.5: "v", 1.0: "o", 2.0: "s", 3.0: "D"}

    fig, ax = plt.subplots(figsize=figsize_2col(aspect=0.5))
    for tr in trackers:
        for mag in mags:
            rows = agg[(agg["tracker"] == tr) & (agg["mag_mult"] == mag)]
            if rows.empty:
                continue
            ax.scatter(
                rows["mean_availability"] * 100, rows["mean_mbps"],
                marker=mag_markers.get(mag, "o"),
                color=TRACKER_COLORS.get(tr, "#666"),
                s=50, edgecolors="black", linewidths=0.5, alpha=0.85,
                label=f"{tr.upper()}, ×{mag:g}",
            )
    ax.set_xlabel("Link availability (%)")
    ax.set_ylabel("Effective TCP throughput (Mbps)")
    ax.set_xlim(-2, 105)
    ax.legend(loc="best", ncol=2, fontsize=6, frameon=True,
              framealpha=0.9, columnspacing=0.6, handletextpad=0.3)
    ax.grid(True, alpha=0.3)
    out = results_dir / "fig_throughput_frontier"
    save_journal(fig, out)
    plt.close(fig)
    return out


def plot_exemplar_traces(results_dir: Path) -> Path:
    """Figure 3c: representative TCP throughput vs time per tracker.

    Reads exemplars.pkl produced by `exp_link_maintenance_envelope.py
    --exemplar`. Each subplot shows the throughput trace with outage gaps
    visible (no points logged during PAUSED state). Annotations call out
    the number of outages and the first-link time so the qualitative
    behavior is legible at thumbnail size.
    """
    import matplotlib.pyplot as plt

    apply_journal_style()
    pkl = results_dir / "exemplars.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"{pkl} not found. Run --exemplar first.")
    with open(pkl, "rb") as f:
        exemplars = pickle.load(f)

    params = {}
    pjson = results_dir / "exemplar_params.json"
    if pjson.exists():
        with open(pjson) as f:
            params = json.load(f)
    duration = params.get("duration_s", None)

    fig, axes = plt.subplots(len(exemplars), 1,
                             figsize=figsize_2col(aspect=0.18 * len(exemplars) + 0.15),
                             sharex=True)
    if len(exemplars) == 1:
        axes = [axes]

    for ax, ex in zip(axes, exemplars):
        trace = ex["throughput_trace"]
        tr_name = ex["tracker"]
        color = TRACKER_COLORS.get(tr_name, "#666")
        if trace:
            t = np.asarray([p[0] for p in trace])
            bps = np.asarray([p[1] for p in trace])
            ax.plot(t, bps / 1e6, color=color, linewidth=0.7)
            # Shade outage gaps (any t-interval where consecutive samples are
            # more than 2x the median sample spacing apart).
            if len(t) > 5:
                dt = np.diff(t)
                med = float(np.median(dt))
                for i, gap in enumerate(dt):
                    if gap > 4 * med:
                        ax.axvspan(t[i], t[i + 1], color="red", alpha=0.12)
        ax.set_ylabel("Mbps", fontsize=7)
        ax.set_title(
            f"{tr_name.upper()}  —  outages: {ex['outage_count']}, "
            f"first link: {ex['first_link_time_s']*1000:.1f} ms",
            fontsize=7, loc="left", pad=2,
        )
        ax.grid(True, alpha=0.3)
        if duration:
            ax.set_xlim(0, duration)
    axes[-1].set_xlabel("Time (s)")

    if params:
        fig.suptitle(
            fr"Exemplar at $\beta$={params.get('beta_deg')}°, "
            fr"vibration ×{params.get('mag_mult')}, "
            f"seed={params.get('seed')}",
            fontsize=8, y=0.995,
        )

    out = results_dir / "fig_exemplar_traces"
    save_journal(fig, out)
    plt.close(fig)
    return out


# =========================================================================
# Discovery survival + quantile plots (for beta_speed / scanner_vs_beta).
# These read results_raw.csv directly (the agg CSV throws away the timing
# distribution, which is exactly what we want to plot here).
# =========================================================================

# Palette by beta value, sorted so larger beam = warmer color.
BETA_PALETTE = {
    3.0: "#08306b",
    5.0: "#2171b5",
    7.0: "#6baed6",
    10.0: "#fdae6b",
    15.0: "#fd8d3c",
    20.0: "#e6550d",
    25.0: "#a63603",
}


def plot_discovery_survival(results_dir: Path,
                            timeout_s: Optional[float] = None,
                            split_by: str = "beta_deg") -> Path:
    """Empirical survival curves: P(discovered by t) vs t, one curve per beta.

    For each unique value of `split_by` (default: beta_deg), this plots the
    Kaplan-Meier-ish CDF of discovery_time_s over the raw trial data.
    Failed trials (success=False) are right-censored at the timeout. Reveals
    the *shape* of the discovery-time distribution — especially the long-tail
    behavior at narrow beams where mean+/-std hides the censored fraction.

    Reads `results_raw.csv` (must have columns `discovery_time_s` and
    `success`, plus the split_by column).
    """
    import matplotlib.pyplot as plt

    apply_journal_style()
    raw = pd.read_csv(results_dir / "results_raw.csv")
    if "discovery_time_s" not in raw.columns or "success" not in raw.columns:
        raise ValueError(
            f"{results_dir}/results_raw.csv missing discovery_time_s/success "
            f"columns; available: {list(raw.columns)}"
        )

    if timeout_s is None:
        # Sniff the timeout from the max success-time across the data.
        timeout_s = float(raw["discovery_time_s"].max())

    fig, ax = plt.subplots(figsize=figsize_2col(aspect=0.5))
    splits = sorted(raw[split_by].unique())
    for v in splits:
        sub = raw[raw[split_by] == v]
        times = sub.loc[sub["success"], "discovery_time_s"].sort_values()
        n_total = len(sub)
        if n_total == 0:
            continue
        # KM-style step CDF over the SUCCESS times.
        x = np.concatenate([[0.0], times.to_numpy(), [timeout_s]])
        y = np.concatenate([[0.0],
                            (np.arange(1, len(times) + 1) / n_total),
                            [len(times) / n_total]])
        color = BETA_PALETTE.get(float(v), None) if split_by == "beta_deg" else None
        ax.step(x, y * 100, where="post", label=f"{split_by}={v:g}",
                color=color, linewidth=1.5)

    ax.axvline(timeout_s, color="grey", linestyle="--", alpha=0.5,
               label=f"timeout = {timeout_s:g} s")
    ax.set_xlabel("Discovery time t (s)")
    ax.set_ylabel("P(discovered by t)  (%)")
    ax.set_ylim(0, 105)
    ax.set_xscale("log")
    ax.legend(loc="lower right", ncol=2, fontsize=7)
    ax.grid(True, alpha=0.3, which="both")

    out = results_dir / f"fig_discovery_survival_by_{split_by}"
    save_journal(fig, out)
    plt.close(fig)
    return out


def plot_discovery_quantiles(results_dir: Path,
                             split_by: str = "beta_deg",
                             quantiles=(0.50, 0.95, 0.99)) -> Path:
    """Discovery-time quantiles (p50, p95, p99) vs the split axis.

    Much more honest than mean +/- std for heavy-tailed distributions like
    Beta-walk discovery time. Successes only; failures are flagged in the
    plot title as a censored-fraction annotation.
    """
    import matplotlib.pyplot as plt

    apply_journal_style()
    raw = pd.read_csv(results_dir / "results_raw.csv")
    if "discovery_time_s" not in raw.columns or "success" not in raw.columns:
        raise ValueError(f"{results_dir}/results_raw.csv missing required columns")

    fig, ax = plt.subplots(figsize=figsize_2col(aspect=0.5))

    qmarkers = {0.50: "o", 0.95: "s", 0.99: "^"}
    qcolors = [WONG_PALETTE[2], WONG_PALETTE[1], WONG_PALETTE[6]]

    splits = sorted(raw[split_by].unique())
    # Decide whether the split is numeric (line plot, x=values) or categorical
    # (bar-style plot, x=index, ticklabels=values).
    try:
        numeric_splits = [float(v) for v in splits]
        is_numeric = True
    except (TypeError, ValueError):
        numeric_splits = list(range(len(splits)))
        is_numeric = False

    censored = []
    for q, color in zip(quantiles, qcolors):
        xs, ys = [], []
        for v, xv in zip(splits, numeric_splits):
            sub = raw[raw[split_by] == v]
            succ = sub.loc[sub["success"], "discovery_time_s"]
            if len(succ) < 5:
                continue
            xs.append(xv)
            ys.append(float(np.quantile(succ, q)))
        ax.plot(xs, ys, marker=qmarkers.get(q, "o"), color=color,
                linewidth=1.5, markersize=6, label=f"p{int(q*100)}")

    # Censored fraction annotation per split.
    for v in splits:
        sub = raw[raw[split_by] == v]
        if len(sub):
            cf = 1.0 - sub["success"].mean()
            censored.append(f"{v}: {cf*100:.0f}% fail")

    if not is_numeric:
        ax.set_xticks(numeric_splits)
        ax.set_xticklabels([str(v) for v in splits])

    ax.set_xlabel({"beta_deg": r"Beam divergence $\beta$ (deg)",
                   "scanner": "Scanner type"}.get(split_by, split_by))
    ax.set_ylabel("Discovery time (s)")
    ax.set_yscale("log")
    if censored:
        ax.set_title("Discovery-time quantiles (successes only); "
                     "censored fraction in note below",
                     fontsize=8)
        ax.text(0.02, 0.02, "  ".join(censored), transform=ax.transAxes,
                fontsize=6, va="bottom", color="#444",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    out = results_dir / f"fig_discovery_quantiles_by_{split_by}"
    save_journal(fig, out)
    plt.close(fig)
    return out


def plot_discovery_survival_by_scanner(results_dir: Path,
                                       timeout_s: Optional[float] = None,
                                       beta_filter: Optional[float] = None) -> Path:
    """For scanner_vs_beta: a single-panel survival curve at fixed beta,
    comparing the three scanners. Falls back to multi-panel grid across all
    available betas if beta_filter is None."""
    import matplotlib.pyplot as plt

    apply_journal_style()
    raw = pd.read_csv(results_dir / "results_raw.csv")
    if "scanner" not in raw.columns:
        raise ValueError("scanner column missing; not a scanner_vs_beta CSV")

    if timeout_s is None:
        timeout_s = float(raw["discovery_time_s"].max())

    if beta_filter is not None:
        betas = [beta_filter]
    else:
        betas = sorted(raw["beta_deg"].unique())
    scanners = sorted(raw["scanner"].unique())
    scanner_colors = {"beta": WONG_PALETTE[2], "spiral": WONG_PALETTE[6],
                      "raster": WONG_PALETTE[3]}

    if len(betas) == 1:
        fig, axes = plt.subplots(1, 1, figsize=figsize_2col(aspect=0.5))
        axes = [axes]
    else:
        ncols = min(3, len(betas))
        nrows = (len(betas) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=figsize_2col(aspect=0.32 * nrows + 0.1),
                                 sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()

    for ax, beta in zip(axes, betas):
        for sc in scanners:
            sub = raw[(raw["scanner"] == sc) & (raw["beta_deg"] == beta)]
            times = sub.loc[sub["success"], "discovery_time_s"].sort_values()
            n_total = len(sub)
            if n_total == 0:
                continue
            x = np.concatenate([[0.0], times.to_numpy(), [timeout_s]])
            y = np.concatenate([[0.0],
                                (np.arange(1, len(times) + 1) / n_total),
                                [len(times) / n_total]])
            ax.step(x, y * 100, where="post", label=sc,
                    color=scanner_colors.get(sc), linewidth=1.5)
        ax.set_xscale("log")
        ax.axvline(timeout_s, color="grey", linestyle="--", alpha=0.4)
        ax.set_title(fr"$\beta$ = {beta:g}°", fontsize=8, pad=2)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_ylim(0, 105)
    # Hide any trailing empty axes.
    for ax in axes[len(betas):]:
        ax.axis("off")
    axes[0].set_ylabel("P(discovered by t)  (%)")
    for ax in axes[-min(3, len(betas)):]:
        ax.set_xlabel("Discovery time (s)")
    axes[0].legend(loc="lower right", fontsize=7)

    suffix = "all" if beta_filter is None else f"b{beta_filter:g}"
    out = results_dir / f"fig_scanner_survival_{suffix}"
    save_journal(fig, out)
    plt.close(fig)
    return out


def _detect_kind(results_dir: Path) -> str:
    """Sniff results_agg.csv columns to determine which experiment this is."""
    # Envelope experiment has both `tracker` and `beta_deg` (Exp 4 has only
    # tracker; Exp 2 has only beta_deg; envelope has both).
    if (results_dir / "axes.json").exists():
        return "envelope"
    if (results_dir / "exemplars.pkl").exists() and not (results_dir / "results_agg.csv").exists():
        return "envelope"
    agg_path = results_dir / "results_agg.csv"
    if not agg_path.exists():
        raise FileNotFoundError(f"no results_agg.csv in {results_dir}")
    agg = pd.read_csv(agg_path, nrows=1)
    cols = set(agg.columns)
    if "tracker" in cols and "beta_deg" in cols:
        return "envelope"
    if "tracker" in cols:
        return "4"
    if "link_type" in cols:
        return "3"
    if "beta_deg" in cols and "mean_availability" in cols:
        return "2"
    # Discovery-time experiments (beta_speed / scanner_vs_beta): if `agg` has
    # success_rate and mean_time_s but no availability column, treat as a
    # discovery-distribution experiment so survival/quantile plots fire.
    if {"success_rate", "mean_time_s"}.issubset(cols):
        return "scanner" if "scanner" in cols else "beta_speed"
    raise ValueError(f"Cannot determine experiment kind from columns: {sorted(cols)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results_dir", help="Directory containing results_agg.csv")
    ap.add_argument("--kind",
                    choices=("2", "3", "4", "envelope", "beta_speed", "scanner"),
                    default=None,
                    help="Which experiment to plot (auto-detected if omitted)")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    kind = args.kind or _detect_kind(rd)
    print(f"plotting kind={kind} for {rd}")

    if kind == "envelope":
        if (rd / "results_agg.csv").exists():
            print(f"  wrote {plot_availability_envelope(rd)}.pdf/.png")
            print(f"  wrote {plot_throughput_frontier(rd)}.pdf/.png")
        if (rd / "exemplars.pkl").exists():
            print(f"  wrote {plot_exemplar_traces(rd)}.pdf/.png")
        return

    if kind == "beta_speed":
        print(f"  wrote {plot_discovery_survival(rd)}.pdf/.png")
        print(f"  wrote {plot_discovery_quantiles(rd)}.pdf/.png")
        return

    if kind == "scanner":
        print(f"  wrote {plot_discovery_survival_by_scanner(rd)}.pdf/.png")
        print(f"  wrote {plot_discovery_quantiles(rd, split_by='scanner')}.pdf/.png")
        return

    if not (rd / "results_agg.csv").exists():
        sys.exit(f"results_agg.csv not found in {rd}")
    if kind == "2":
        print(f"  wrote {plot_link_availability(rd)}.pdf/.png")
    elif kind == "3":
        print(f"  wrote {plot_throughput_comparison(rd)}.pdf/.png")
    elif kind == "4":
        print(f"  wrote {plot_tracker_outages(rd)}.pdf/.png")
        print(f"  wrote {plot_tracker_availability(rd)}.pdf/.png")


if __name__ == "__main__":
    main()
