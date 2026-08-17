"""Journal-grade plotting helpers for OWDES-MP experiments.

Targets two-column IEEE / Optics layouts:
  - Single-column figures: 3.5 inches wide (matches \\columnwidth)
  - Double-column figures: 7.16 inches wide (matches \\textwidth)

Provides:
  - apply_journal_style()  set serif fonts, sizes, palette
  - WONG_PALETTE           color-blind-safe categorical palette (Wong 2011)
  - figsize_1col(aspect)   width = 3.5", height = 3.5 * aspect
  - figsize_2col(aspect)   width = 7.16", height = 7.16 * aspect
  - save_journal(fig, p)   write <p>.pdf (vector) and <p>.png (600 dpi)
  - bootstrap_ci(...)      percentile-bootstrap confidence interval
  - power_law_fit(x, y)    log-log linear regression -> (exponent, prefactor, R^2)
  - ks_2samp_with_str(...) two-sample KS test + formatted annotation string
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence, Tuple

import numpy as np


# Wong 2011 color-blind-safe palette (Nature Methods, "Points of View: Color blindness").
# Order: black, orange, sky blue, bluish green, yellow, blue, vermilion, reddish purple.
WONG_PALETTE = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]

# IEEE Trans / Optics column widths (inches)
COL_1 = 3.5
COL_2 = 7.16


def apply_journal_style() -> None:
    """Apply rcParams suitable for two-column IEEE / Optics submissions."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    rc = mpl.rcParams
    # Use mathtext stixsans so math renders with serif body without
    # requiring a system LaTeX install.
    rc["font.family"] = "serif"
    rc["font.serif"] = ["DejaVu Serif", "Times New Roman", "STIX",
                        "Bitstream Vera Serif", "serif"]
    rc["mathtext.fontset"] = "stix"
    rc["font.size"] = 8.0
    rc["axes.titlesize"] = 9.0
    rc["axes.labelsize"] = 8.0
    rc["xtick.labelsize"] = 7.0
    rc["ytick.labelsize"] = 7.0
    rc["legend.fontsize"] = 7.0
    rc["legend.frameon"] = False
    rc["axes.spines.top"] = False
    rc["axes.spines.right"] = False
    rc["axes.linewidth"] = 0.7
    rc["xtick.major.width"] = 0.7
    rc["ytick.major.width"] = 0.7
    rc["lines.linewidth"] = 1.25
    rc["lines.markersize"] = 4.0
    rc["axes.grid"] = True
    rc["grid.alpha"] = 0.3
    rc["grid.linewidth"] = 0.4
    rc["savefig.dpi"] = 600
    rc["savefig.bbox"] = "tight"
    rc["pdf.fonttype"] = 42  # editable vector text in PDF
    rc["ps.fonttype"] = 42
    rc["axes.prop_cycle"] = plt.cycler(color=WONG_PALETTE)


def figsize_1col(aspect: float = 0.75) -> Tuple[float, float]:
    return (COL_1, COL_1 * aspect)


def figsize_2col(aspect: float = 0.45) -> Tuple[float, float]:
    return (COL_2, COL_2 * aspect)


def save_journal(fig, path_stem) -> Tuple[Path, Path]:
    """Save vector PDF + 600 dpi PNG side-by-side. Returns the two paths.

    `path_stem` may be a Path or str without extension. Existing extensions
    are stripped.
    """
    p = Path(path_stem)
    if p.suffix:
        p = p.with_suffix("")
    pdf_path = p.with_suffix(".pdf")
    png_path = p.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)
    return pdf_path, png_path


def bootstrap_ci(
    data: Sequence[float],
    ci_pct: float = 95.0,
    n_resamples: int = 2000,
    statistic: Callable[[np.ndarray], float] = np.mean,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, float]:
    """Return (point_estimate, ci_lo, ci_hi) for `statistic` over `data`."""
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(statistic(arr))
    if arr.size == 1:
        return (point, point, point)
    rng = rng or np.random.default_rng(0xC0FFEE)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    samples = arr[idx]
    stats = np.apply_along_axis(statistic, 1, samples)
    alpha = (100.0 - ci_pct) / 2.0
    lo, hi = np.percentile(stats, [alpha, 100.0 - alpha])
    return (point, float(lo), float(hi))


def power_law_fit(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, float]:
    """Fit y = a * x^k by log-log linear regression. Returns (k, a, R^2).

    Only positive (x, y) pairs are used; the rest are dropped.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = (x_arr > 0) & (y_arr > 0)
    if mask.sum() < 2:
        return (float("nan"), float("nan"), float("nan"))
    lx = np.log(x_arr[mask])
    ly = np.log(y_arr[mask])
    k, log_a = np.polyfit(lx, ly, 1)
    ly_hat = k * lx + log_a
    ss_res = float(np.sum((ly - ly_hat) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return (float(k), float(np.exp(log_a)), r2)


def ks_2samp_with_str(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, str]:
    """Two-sample KS test + a formatted annotation string.

    Falls back to a manual implementation if scipy is not installed so the
    experiment scripts have one fewer dependency.
    """
    a_arr = np.sort(np.asarray(a, dtype=float))
    b_arr = np.sort(np.asarray(b, dtype=float))
    try:
        from scipy.stats import ks_2samp
        res = ks_2samp(a_arr, b_arr)
        D, p = float(res.statistic), float(res.pvalue)
    except Exception:
        # Manual KS statistic; p-value uses the asymptotic Kolmogorov formula.
        all_v = np.concatenate([a_arr, b_arr])
        cdf_a = np.searchsorted(a_arr, all_v, side="right") / a_arr.size
        cdf_b = np.searchsorted(b_arr, all_v, side="right") / b_arr.size
        D = float(np.max(np.abs(cdf_a - cdf_b)))
        n = a_arr.size * b_arr.size / (a_arr.size + b_arr.size)
        lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * D
        terms = np.arange(1, 101)
        p = 2.0 * float(np.sum((-1) ** (terms - 1) * np.exp(-2.0 * (terms * lam) ** 2)))
        p = max(0.0, min(1.0, p))
    return D, p, f"KS $D$={D:.3f}, $p$={p:.3g}"
