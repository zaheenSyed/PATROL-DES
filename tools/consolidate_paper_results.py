"""Consolidate the four paper experiments into a single results folder.

Usage:
    python tools/consolidate_paper_results.py \
        [--out paper_results] \
        [--envelope arcc_results/envelope_<latest>] \
        [--beta-speed arcc_results/beta_speed_<latest>] \
        [--scanner-vs-beta arcc_results/scanner_vs_beta_<latest>] \
        [--multi-node arcc_results/multi_node_<latest>] \
        [--regenerate-figures]

If a --<exp> path is not given, the script falls back to the LATEST matching
directory under arcc_results/ (sorted by name, which is timestamp-sortable).

For each experiment it:
  1. Creates paper_results/<short_name>/
  2. Copies the source CSVs (and exemplars.pkl, axes.json if present)
  3. Records the source path + row count + axis values in a per-experiment
     manifest.json
  4. Optionally regenerates figures via owdesmp.plotting.link_maintenance
     (works for envelope; other experiments retain their original PNGs)

Finally writes paper_results/README.md with a table that summarizes:
  - Experiment short name
  - Source directory
  - Trial count
  - Axis grid
  - Generation date
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENTS = {
    "envelope": {
        "src_glob": "envelope_*",
        "raw_csv": "results_raw.csv",
        "agg_csv": "results_agg.csv",
        "extras": ["axes.json", "exemplar_params.json", "exemplars.pkl",
                   "exemplar_trace_none.csv", "exemplar_trace_pid.csv",
                   "exemplar_trace_kalman.csv", "exemplar_trace_rl.csv"],
        "axis_cols": ["beta_deg", "mag_mult", "tracker"],
        "plot_kind": "envelope",
    },
    "beta_speed": {
        "src_glob": "beta_speed_*",
        "raw_csv": "results_raw.csv",
        "agg_csv": "results_agg.csv",
        "extras": ["discovery_time_vs_omega.png", "success_rate_vs_omega.png"],
        "axis_cols": ["beta_deg", "omega_mult"],
        "plot_kind": None,
    },
    "scanner_vs_beta": {
        "src_glob": "scanner_vs_beta_*",
        "raw_csv": "results_raw.csv",
        "agg_csv": "results_agg.csv",
        "extras": ["disc_time_vs_beta.png", "events_vs_beta.png",
                   "pdisc_vs_beta.png"],
        "axis_cols": ["scanner", "beta_deg"],
        "plot_kind": None,
    },
    "multi_node": {
        "src_glob": "multi_node_*",
        "raw_csv": "results_raw.csv",
        "agg_csv": "results_agg.csv",
        "extras": ["components_vs_n.png", "events_vs_n.png", "ttc_vs_n.png",
                   "wallclock_per_simsec.png"],
        "axis_cols": ["n_nodes"],
        "plot_kind": None,
    },
}


def latest_match(arcc_dir: Path, glob: str) -> Optional[Path]:
    matches = sorted(arcc_dir.glob(glob))
    return matches[-1] if matches else None


def summarize_axes(raw_csv: Path, axis_cols: list[str]) -> dict:
    df = pd.read_csv(raw_csv, usecols=lambda c: c in axis_cols)
    out: dict = {}
    for c in axis_cols:
        if c not in df.columns:
            continue
        vals = df[c].dropna().unique().tolist()
        # Sort; cast numeric to float for json-friendliness.
        try:
            out[c] = sorted(float(v) for v in vals)
        except (TypeError, ValueError):
            out[c] = sorted(str(v) for v in vals)
    return out


def consolidate_one(short: str, src: Path, out_root: Path,
                    regenerate_figures: bool) -> dict:
    cfg = EXPERIMENTS[short]
    dst = out_root / short
    dst.mkdir(parents=True, exist_ok=True)

    raw_src = src / cfg["raw_csv"]
    agg_src = src / cfg["agg_csv"]
    if not raw_src.exists():
        raise FileNotFoundError(f"{raw_src} not found")
    if not agg_src.exists():
        raise FileNotFoundError(f"{agg_src} not found")

    shutil.copy2(raw_src, dst / "results_raw.csv")
    shutil.copy2(agg_src, dst / "results_agg.csv")

    copied_extras = []
    for e in cfg["extras"]:
        src_e = src / e
        if src_e.exists():
            shutil.copy2(src_e, dst / e)
            copied_extras.append(e)

    n_trials = len(pd.read_csv(raw_src, usecols=[0]))
    axes = summarize_axes(raw_src, cfg["axis_cols"])

    manifest = {
        "short_name": short,
        "source_path": str(src),
        "consolidated_at": datetime.utcnow().isoformat() + "Z",
        "n_trials_raw": n_trials,
        "axes": axes,
        "extras_copied": copied_extras,
    }
    with open(dst / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if regenerate_figures and cfg["plot_kind"] is not None:
        # Only envelope has a plotter in owdesmp.plotting.link_maintenance.
        # Others retain their source PNGs.
        cmd = [sys.executable, "-m", "owdesmp.plotting.link_maintenance",
               str(dst), "--kind", cfg["plot_kind"]]
        try:
            subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: figure regeneration failed for {short}: {e}")

    return manifest


def write_readme(out_root: Path, manifests: dict[str, dict]) -> None:
    readme = out_root / "README.md"
    lines = []
    lines.append("# Paper results — consolidated")
    lines.append("")
    lines.append(f"_Generated {datetime.utcnow().isoformat()}Z by "
                 "`tools/consolidate_paper_results.py`._")
    lines.append("")
    lines.append("Each subdirectory contains a self-contained snapshot of one "
                 "experiment: `results_raw.csv` (one row per trial), "
                 "`results_agg.csv` (means/stds per cell), a `manifest.json` "
                 "with provenance, and the figures produced for that run.")
    lines.append("")

    lines.append("## Experiments")
    lines.append("")
    lines.append("| Experiment | Trials | Axis grid | Source |")
    lines.append("|---|---:|---|---|")
    for short, m in manifests.items():
        axes_str = "; ".join(
            f"`{k}` ∈ {v}" for k, v in m["axes"].items()
        )
        src_rel = Path(m["source_path"]).name
        lines.append(
            f"| `{short}` | {m['n_trials_raw']:,} | {axes_str} | `{src_rel}` |"
        )
    lines.append("")

    lines.append("## How to refresh figures from raw data")
    lines.append("")
    lines.append("Envelope (Figures 3a/3b/3c) is regenerated through "
                 "`owdesmp.plotting.link_maintenance`. The other three retain "
                 "their original-style PNGs but can be re-styled by editing "
                 "the notebook at `notebooks/edit_paper_figures.ipynb`.")
    lines.append("")
    lines.append("```bash")
    lines.append("# Envelope only — three figures")
    lines.append("python -m owdesmp.plotting.link_maintenance "
                 "paper_results/envelope --kind envelope")
    lines.append("```")
    lines.append("")

    lines.append("## Manifest schema")
    lines.append("")
    lines.append("Each `manifest.json` records:")
    lines.append("- `source_path` — original ARCC run directory")
    lines.append("- `consolidated_at` — UTC timestamp of this copy")
    lines.append("- `n_trials_raw` — row count of `results_raw.csv`")
    lines.append("- `axes` — sweep grid actually present in the data")
    lines.append("- `extras_copied` — auxiliary files copied alongside CSVs")

    with open(readme, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(REPO_ROOT / "paper_results"),
                    help="Root for consolidated results (default: paper_results/)")
    ap.add_argument("--arcc-dir", default=str(REPO_ROOT / "arcc_results"),
                    help="Directory holding the ARCC raw outputs to scan if "
                         "per-experiment paths aren't given")
    ap.add_argument("--envelope", default=None)
    ap.add_argument("--beta-speed", default=None)
    ap.add_argument("--scanner-vs-beta", default=None)
    ap.add_argument("--multi-node", default=None)
    ap.add_argument("--regenerate-figures", action="store_true",
                    help="Re-run the envelope figures through owdesmp.plotting")
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    arcc_dir = Path(args.arcc_dir).resolve()

    # Resolve each source path (CLI override > latest under arcc).
    cli_paths = {
        "envelope": args.envelope,
        "beta_speed": args.beta_speed,
        "scanner_vs_beta": args.scanner_vs_beta,
        "multi_node": args.multi_node,
    }

    manifests: dict[str, dict] = {}
    for short, cfg in EXPERIMENTS.items():
        cli_path = cli_paths[short]
        if cli_path:
            src = Path(cli_path).resolve()
        else:
            src = latest_match(arcc_dir, cfg["src_glob"])
        if src is None or not src.exists():
            print(f"[skip] {short}: no source found "
                  f"(--{short.replace('_', '-')} not given and no match for "
                  f"{cfg['src_glob']!r} under {arcc_dir})")
            continue
        print(f"[copy] {short} <- {src}")
        m = consolidate_one(short, src, out_root, args.regenerate_figures)
        manifests[short] = m
        print(f"       {m['n_trials_raw']:,} trials, axes={m['axes']}")

    if not manifests:
        sys.exit("No experiments consolidated. Check --arcc-dir or per-experiment paths.")

    write_readme(out_root, manifests)
    print(f"\nWrote {out_root / 'README.md'}")
    print(f"Consolidated {len(manifests)} experiments under {out_root}")


if __name__ == "__main__":
    main()
