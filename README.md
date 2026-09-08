<div align="center">

# 🛰️ OWDES-MP

### Optical Wireless Discrete-Event Simulator for Mobile Platforms

*A modular, reproducible discrete-event simulator for **Pointing, Acquisition, Tracking, and Reacquisition (PAT)** in aerial free-space optical (FSO) mesh networks.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20MILCOM-red.svg)](#-citation)
[![Status](https://img.shields.io/badge/status-research-brightgreen.svg)](#)

</div>

---

## 📄 Featured Paper — PATROL @ IEEE MILCOM

This repository is the simulator behind our IEEE MILCOM paper:

> **PATROL: An Event-Driven Simulator for Intelligent PAT in Aerial Optical Wireless Networks**
> *Zaheen E Muktadi Syed and Murat Yuksel* — Electrical & Computer Engineering, University of Central Florida.

Optical wireless links promise **multi-gigabit, low-probability-of-intercept** connectivity across meshes of UAVs, high-altitude platforms, and LEO satellites — but real deployment is gated by **PAT under mobility, atmospheric turbulence, and beam-pointing jitter**. PATROL jointly models beam alignment, the FSO channel, asynchronous handshake protocols, mobility, scanning, and link maintenance for *N*-node aerial networks, and shows how beam divergence *β*, scanning strategy, and platform vibration drive **discovery latency, timeout probability, link availability, and outage behavior**.


📌 *See [How to cite](#-citation) for the BibTeX entry.*

---

## ✨ Features

- ⚡ **Event-driven core** — priority-queue event loop with fully deterministic, seed-reproducible traces.
- 🤝 **Pairwise 3-way handshake FSM** — packet-level optical neighbor discovery (BEACON / BEACON-ACK / ACK) validated against an independent reference simulator.
- 📡 **Coupled physics** — geometry, FSO channel (attenuation, turbulence, pointing loss), and protocol modeled as one system, not in isolation.
- 🚁 **Data-driven mobility** — fixed, waypoint, constant-velocity, circular-orbit, and **vibration profiles from real Pixhawk EKF telemetry**.
- 🔦 **Scanner strategies** — raster, spiral, uniform, and β-random beam search.
- 🎯 **Intelligent PAT** — beam-tracking layer + PPO reinforcement-learning environment for adaptive controllers.
- 🌐 **Scales to N = 32** aerial nodes with 496 concurrent pairwise state machines.
- 📊 **Reproducible by construction** — JSON scenarios, Monte-Carlo convergence at the theoretical 1/√K rate, journal-quality plotting.

---

## 🏗️ System Model

```
        Real-World Model                          Virtual Architecture
  ┌──────────────────────────┐         JSON Scenario ─► ScenarioLoader
  │  Node ●··· β ···► ● Node  │                              │
  │   (UAV)   FSO beam  (HAP) │                    ┌─────────▼─────────┐
  │   ⤢ search space, range   │                    │   Simulator Core  │
  └──────────────────────────┘                    │  event loop · RNG │
                                                   │  · termination    │
                                                   └─────────┬─────────┘
   Engines: Protocol FSM · Geometry · Channel ◄──────────────┤
            Beam Tracker · Pointing Jitter · TCP Flow · Data Transfer
                                                             │
   Models: Node · Packet · Network Graph                     ▼
   Mobility: Fixed · Waypoint · Vibration            Result / Metrics
   Scanners: Raster · Spiral · Random · β-random
```

---

## 🚀 Quickstart

```bash
pip install -r requirements.txt
```

There is no standalone CLI runner yet — scenarios are driven directly from Python:

```python
from owdesmp.core.scenario_loader import load_scenario
from owdesmp.core.simulator import Simulator

scenario = load_scenario("scenarios/two_node_validation.json")
for run_id in range(100):
    result = Simulator(scenario=scenario, run_id=run_id, seed=42 + run_id).run()
    print(run_id, result["success"])
```

For batch experiments (β sweeps, scanner comparisons, RL training) see the scripts in
[`experiments/`](experiments), each runnable standalone, e.g.:

```bash
python experiments/exp_scanner_compare_vs_beta.py --quick
```

**Try other scenarios:**

| Scenario | What it exercises |
|----------|-------------------|
| `two_node_validation.json`   | Handshake discovery parity |
| `two_node_beta_search.json`  | Beam-divergence β sweep |
| `two_node_channel.json`      | FSO channel + link budget |
| `two_node_vibration.json`    | Pixhawk-driven platform vibration |
| `two_node_maintenance.json`  | Link maintenance / reacquisition |
| `two_satellite_isl_1000km.json` | Satellite inter-satellite link |
| `three_node_validation.json` | Multi-node mesh |

---

## 📂 Repository Layout

```
owdesmp/            core package
  core/             event · event_queue · simulator · scenario_loader · random_manager · termination
  engines/          protocol FSM · geometry · channel · beam_tracker · pointing_jitter · tcp_flow · data_transfer
  mobility/         fixed · waypoint · constant_velocity · circular_orbit · vibration
  scanners/         raster · spiral · uniform · beta (β-random)
  models/           node · packet · network_graph
  rl/               Gymnasium tracker env + PPO training
  plotting/         journal-quality figures
scenarios/          JSON scenario configs
experiments/        experiment scripts (β sweeps, scaling, RL, cross-validation)
tests/              pytest suite (parity, channel, mobility, scanners, RL, multi-node)
legacy/             predecessor simulators kept as validation oracles
```

---

## ✅ Testing

```bash
python -m pytest tests/ -v
```

The suite covers handshake parity against the legacy oracle, channel and geometry engines, mobility and scanner behavior, the RL environment, and multi-node termination.

---

## 📝 Citation

If you use OWDES-MP / PATROL in your research, please cite:

```bibtex
@inproceedings{syed2026patrol,
  title     = {{PATROL}: An Event-Driven Simulator for Intelligent {PAT}
               in Aerial Optical Wireless Networks},
  author    = {Syed, Zaheen E Muktadi and Yuksel, Murat},
  booktitle = {Proc. IEEE Military Communications Conference (MILCOM)},
  year      = {2026},
  publisher = {IEEE}
}
```

---

## 📜 License

Released under the [MIT License](LICENSE) — © 2026 Zaheen E Muktadi Syed.

<div align="center">
<sub>Built at the Networking & Wireless Systems Lab, University of Central Florida.</sub>
</div>
