"""Channel engine.

Computes received power for one (src -> dst) link at time t and decides whether
a transmitted packet survives the link. Loss model is ported from
legacy/fso_sim_modules.py:142-284 with three components:
  - Geometric spreading at distance d for a beam of full divergence beta
  - Atmospheric attenuation via Beer-Lambert at sigma_dBkm dB/km (Kim model
    available as an alternative when visibility_km is supplied directly)
  - Pointing loss with a Gaussian approximation in (Xab / half_div)

Optional turbulence (mode = "stochastic"): one Gamma-Gamma fading sample is
drawn per packet using the provided RNG. The sample is unit-mean so its
log-amplitude has zero offset on average.

Modes:
  - "disabled":   packet_succeeds always returns True. This is the default
                  for scenarios with no `channel` section, so slice-1 and
                  slice-2 tests continue to pass unchanged.
  - "threshold":  deterministic. P_rx_dBm >= receiver_sensitivity_dBm.
  - "stochastic": threshold check after adding a Gamma-Gamma turbulence
                  sample. Per-packet randomness.
"""

import math
from typing import Optional

import numpy as np

from owdesmp.engines.geometry import angle_between


def kim_attenuation_dBkm(visibility_km: float, wavelength_nm: float) -> float:
    """Kim model atmospheric attenuation coefficient, dB/km."""
    if visibility_km > 50:
        q = 1.6
    elif visibility_km > 6:
        q = 1.3
    else:
        q = 0.585 * visibility_km ** (1.0 / 3.0)
    sigma_per_km = (3.91 / visibility_km) * (wavelength_nm / 550.0) ** (-q)
    # The 3.91/V form is in units of 1/km (extinction). Convert to dB/km via 10/ln(10).
    return 10.0 * sigma_per_km / math.log(10.0)


def gamma_gamma_sample(rng, L_m: float, Cn2: float, wavelength_nm: float) -> float:
    """One Gamma-Gamma fading sample (Andrews & Phillips). Returns unit-mean RV.

    Matches legacy/fso_sim_modules.py:114-132 but uses the supplied stdlib
    random.Random for reproducibility.
    """
    k_wave = 2 * math.pi / (wavelength_nm * 1e-9)
    sigma_R2 = 1.23 * Cn2 * (k_wave ** (7 / 6)) * (L_m ** (11 / 6))
    if sigma_R2 < 1e-9:
        return 1.0
    alpha = (
        math.exp(0.49 * sigma_R2 / (1 + 1.11 * sigma_R2 ** (6 / 5)) ** (7 / 6)) - 1
    ) ** (-1)
    beta = (
        math.exp(0.51 * sigma_R2 / (1 + 0.69 * sigma_R2 ** (6 / 5)) ** (5 / 6)) - 1
    ) ** (-1)
    g1 = rng.gammavariate(alpha, 1.0 / alpha)
    g2 = rng.gammavariate(beta, 1.0 / beta)
    return g1 * g2


class ChannelModel:
    def __init__(
        self,
        mode: str = "disabled",
        wavelength_nm: float = 1550.0,
        sigma_dBkm: Optional[float] = None,
        visibility_km: Optional[float] = None,
        Cn2: float = 1e-15,
        margin_dB: float = 0.0,
    ) -> None:
        self.mode = mode
        self.wavelength_nm = float(wavelength_nm)
        # Atmospheric: either provided directly in dB/km, or derived from Kim/visibility.
        if sigma_dBkm is not None:
            self.sigma_dBkm = float(sigma_dBkm)
        elif visibility_km is not None:
            self.sigma_dBkm = kim_attenuation_dBkm(float(visibility_km), wavelength_nm)
        else:
            self.sigma_dBkm = 0.0
        self.Cn2 = float(Cn2)
        self.margin_dB = float(margin_dB)

    def received_power_dBm(self, src, dst, t: float, rng=None) -> float:
        pa = src.position_at(t)
        pb = dst.position_at(t)
        d = float(np.linalg.norm(pb - pa))

        tx_dBm = float(src.optical_terminal["tx_power_dBm"])
        if d == 0.0:
            return tx_dBm

        # Pointing error: angle between src.beam and the (src -> dst) direction.
        a_to_b = (pb - pa) / d
        Xab = angle_between(src.current_pointing, a_to_b) if src.current_pointing is not None else 0.0

        beam_div = float(src.optical_terminal["beam_divergence_rad"])
        rx_aperture = float(dst.optical_terminal.get("receiver_aperture_m", 0.08))

        # Geometric spreading. Beam radius at range d for a full-divergence beta:
        # w_d = d * tan(beta / 2). Receiver aperture pi*(r_rx)^2 / pi*(w_d)^2.
        w_d = d * math.tan(beam_div / 2.0)
        if w_d <= 0.0:
            geo_loss_dB = 0.0
        else:
            geo_loss_dB = 10.0 * math.log10((rx_aperture * rx_aperture) / (w_d * w_d))

        # Atmospheric (Beer-Lambert). Stored in dB/km so loss is linear in km.
        atm_loss_dB = -self.sigma_dBkm * (d / 1000.0)

        # Pointing loss (Gaussian approx). half_div is the beam half-angle.
        half_div = beam_div / 2.0
        if half_div > 0.0:
            ratio = Xab / half_div
            pointing_linear = math.exp(-2.0 * ratio * ratio)
            pointing_dB = 10.0 * math.log10(max(pointing_linear, 1e-20))
        else:
            pointing_dB = 0.0

        turb_dB = 0.0
        if self.mode == "stochastic" and rng is not None:
            h_turb = gamma_gamma_sample(rng, d, self.Cn2, self.wavelength_nm)
            turb_dB = 10.0 * math.log10(max(h_turb, 1e-20))

        return tx_dBm + geo_loss_dB + atm_loss_dB + pointing_dB + turb_dB

    def packet_succeeds(self, src, dst, t: float, rng=None) -> bool:
        if self.mode == "disabled":
            return True
        # Both endpoints need an optical_terminal with tx_power and sensitivity for
        # the channel to be evaluable. If missing, treat as transparent (slice 2 fallback).
        if src.optical_terminal is None or dst.optical_terminal is None:
            return True
        if "tx_power_dBm" not in src.optical_terminal:
            return True
        if "receiver_sensitivity_dBm" not in dst.optical_terminal:
            return True

        p_rx = self.received_power_dBm(src, dst, t, rng)
        sensitivity = float(dst.optical_terminal["receiver_sensitivity_dBm"])
        return p_rx >= (sensitivity + self.margin_dB)
