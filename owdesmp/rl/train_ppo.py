"""R4: PPO training for the ATP TrackerATPEnv.

Usage:
    # Smoke test (no training, no save):
    python -m owdesmp.rl.train_ppo --quick

    # Real training (~2-4 h on 8 CPUs):
    python -m owdesmp.rl.train_ppo \
        --output-dir runs/atp_ppo_$(date +%Y%m%d_%H%M%S) \
        --total-steps 2000000

    # Continue from a checkpoint:
    python -m owdesmp.rl.train_ppo \
        --output-dir runs/atp_ppo_resume \
        --resume runs/atp_ppo_<ts>/stage_2_mag_2.0.zip \
        --total-steps 500000 \
        --mag 3.0

Curriculum: vibration magnitude ramps 0.5 -> 1.0 -> 2.0 -> 3.0 across four
roughly equal stages (configurable). Each stage rebuilds the vec-env to
sample fresh trajectories at the new magnitude. The policy is preserved
across stages via PPO.set_env().

Vector env: SubprocVecEnv across `--num-envs` processes (default = cpu_count).
Each sub-process holds one TrackerATPEnv. PPO collects rollouts in parallel.

Reward shaping: see TrackerATPEnv docstring. The training target is to beat
switch_with_fallback's 22.0 MB / 5 s trial at mag=1.0 (the user's locked
contribution bar).
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "stable-baselines3 is required for R4 training. "
        "pip install stable-baselines3[extra]"
    ) from e

from experiments.exp_atp_baseline import build_scenario, sample_node_b_positions
from owdesmp.rl.tracker_env import TrackerATPEnv


# ----------------------------------------------------------------- factory


def _scenario_factory(mag_mult: float, dist_m: float, duration_s: float) -> Callable[[int], dict]:
    """Return a picklable scenario factory closed over (mag, dist, duration).
    Used by SubprocVecEnv where the factory has to round-trip through pickle.
    """
    def factory(seed: int) -> dict:
        pos_rng = Random(seed)
        pos = sample_node_b_positions(1, dist_m, jitter_radius_m=2.0, rng=pos_rng)[0]
        return build_scenario(
            policy="switch_with_fallback",
            mag_mult=mag_mult,
            dist_m=dist_m,
            pos_b=pos,
            duration_s=duration_s,
            seed=seed,
        )
    return factory


def make_env(
    rank: int,
    mag_mult: float,
    dist_m: float = 100.0,
    duration_s: float = 5.0,
    max_steps: int = 5000,
    base_seed: int = 0,
    monitor_path: Path | None = None,
) -> Callable[[], "TrackerATPEnv"]:
    """Returns a thunk that constructs one TrackerATPEnv. Wrapped in Monitor
    so SB3 logs episode reward / length to monitor.csv."""
    def _thunk():
        env = TrackerATPEnv(
            scenario_factory=_scenario_factory(mag_mult, dist_m, duration_s),
            max_steps=max_steps,
            seed=base_seed + rank,
        )
        if monitor_path is not None:
            env = Monitor(env, filename=str(monitor_path / f"env_{rank}"))
        return env
    return _thunk


def build_vec_env(
    num_envs: int,
    mag_mult: float,
    dist_m: float,
    duration_s: float,
    max_steps: int,
    base_seed: int,
    monitor_path: Path | None = None,
    use_subproc: bool = True,
):
    thunks = [
        make_env(rank=i, mag_mult=mag_mult, dist_m=dist_m,
                 duration_s=duration_s, max_steps=max_steps,
                 base_seed=base_seed, monitor_path=monitor_path)
        for i in range(num_envs)
    ]
    if use_subproc and num_envs > 1:
        return SubprocVecEnv(thunks, start_method="spawn")
    return DummyVecEnv(thunks)


# ---------------------------------------------------------------- training


def make_default_curriculum(total_steps: int):
    """Four-stage curriculum, roughly equal step budgets."""
    per_stage = total_steps // 4
    return [
        (0.5, per_stage),
        (1.0, per_stage),
        (2.0, per_stage),
        (3.0, total_steps - 3 * per_stage),
    ]


def train(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "tb"
    ckpt_dir = out_dir / "checkpoints"
    monitor_dir = out_dir / "monitor"
    log_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    monitor_dir.mkdir(exist_ok=True)

    num_envs = args.num_envs if args.num_envs > 0 else max(1, os.cpu_count() // 2)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] output dir: {out_dir}")
    print(f"  num_envs={num_envs}, total_steps={args.total_steps}, "
          f"duration_s={args.duration}, max_steps={args.max_steps}")

    # Curriculum.
    if args.mag is not None:
        curriculum = [(args.mag, args.total_steps)]
    else:
        curriculum = make_default_curriculum(args.total_steps)
    print(f"  curriculum: {curriculum}")

    # Vec env (initial stage).
    initial_mag, initial_steps = curriculum[0]
    vec_env = build_vec_env(
        num_envs=num_envs,
        mag_mult=initial_mag,
        dist_m=args.dist,
        duration_s=args.duration,
        max_steps=args.max_steps,
        base_seed=args.seed,
        monitor_path=monitor_dir,
        use_subproc=not args.dummy_vec,
    )

    # Build / resume the model.
    if args.resume is not None:
        print(f"  resuming from {args.resume}")
        model = PPO.load(args.resume, env=vec_env,
                         tensorboard_log=str(log_dir),
                         device=args.device)
    else:
        model = PPO(
            policy="MultiInputPolicy",
            env=vec_env,
            verbose=1,
            tensorboard_log=str(log_dir),
            learning_rate=3e-4,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            # log_std_init=-2 gives std~0.13 in normalized action space, so
            # stochastic samples stay well inside [-1,+1] instead of always
            # saturating against the clip boundary (the failure mode of the
            # first training run that produced a policy doing 0.4 MB / ep
            # instead of the 15.8 MB an all-zero action achieves).
            policy_kwargs=dict(log_std_init=-2.0),
            seed=args.seed,
            device=args.device,
        )

    # Checkpoint every save_freq env-steps (not policy-steps).
    save_freq_steps = max(1, args.save_freq // num_envs)
    ckpt_cb = CheckpointCallback(
        save_freq=save_freq_steps,
        save_path=str(ckpt_dir),
        name_prefix="ppo_atp",
    )

    # Walk the curriculum.
    cum_steps = 0
    for stage_idx, (mag, steps) in enumerate(curriculum):
        if stage_idx > 0:
            vec_env.close()
            vec_env = build_vec_env(
                num_envs=num_envs,
                mag_mult=mag,
                dist_m=args.dist,
                duration_s=args.duration,
                max_steps=args.max_steps,
                base_seed=args.seed + stage_idx * 100000,
                monitor_path=monitor_dir,
                use_subproc=not args.dummy_vec,
            )
            model.set_env(vec_env)
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] STAGE {stage_idx}: "
              f"mag={mag}, steps={steps}")
        t0 = time.time()
        model.learn(
            total_timesteps=steps,
            reset_num_timesteps=(stage_idx == 0 and args.resume is None),
            callback=ckpt_cb,
            log_interval=1,
            tb_log_name=f"stage_{stage_idx}_mag_{mag:g}",
            progress_bar=False,
        )
        wall = time.time() - t0
        cum_steps += steps
        stage_path = ckpt_dir / f"stage_{stage_idx}_mag_{mag:g}.zip"
        model.save(stage_path)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] stage {stage_idx} "
              f"done in {wall:.0f}s; saved {stage_path}")

    # Final.
    final_path = out_dir / "policy_final.zip"
    model.save(final_path)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] training complete: {final_path}")
    vec_env.close()


# --------------------------------------------------------------- smoke run


def smoke(args):
    """Build a tiny env, train for a few thousand steps, evaluate, exit.
    Verifies all wiring (env, policy, optimizer, vec-env, checkpoint).
    Should complete in well under a minute."""
    print("R4 PPO smoke test")
    num_envs = 2  # small for quick
    out_dir = Path(args.output_dir) if args.output_dir else Path("runs/smoke_tmp")
    out_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir = out_dir / "monitor"
    monitor_dir.mkdir(exist_ok=True)

    vec_env = build_vec_env(
        num_envs=num_envs,
        mag_mult=0.5,
        dist_m=100.0,
        duration_s=0.5,        # tiny episodes
        max_steps=300,         # safety
        base_seed=0,
        monitor_path=monitor_dir,
        use_subproc=False,     # DummyVecEnv avoids spawn cost
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=0,
        n_steps=128,
        batch_size=64,
        n_epochs=2,
        device="cpu",
        seed=0,
    )
    t0 = time.time()
    model.learn(total_timesteps=1024, log_interval=1, progress_bar=False)
    wall = time.time() - t0
    print(f"  learn(1024) wall = {wall:.2f}s on {num_envs} envs")

    # Quick eval rollout.
    eval_env = TrackerATPEnv(
        scenario_factory=_scenario_factory(0.5, 100.0, 0.5),
        max_steps=300,
        seed=99,
    )
    obs, _ = eval_env.reset(seed=99)
    total_reward = 0.0
    steps = 0
    done = False
    while not done and steps < 300:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
        steps += 1
    print(f"  eval: steps={steps}, total_reward={total_reward:.3f}, "
          f"bytes_acked={info.get('data_session', {}).get('bytes_acked', 0.0)/1e6:.2f} MB")
    eval_env.close()
    vec_env.close()
    print("smoke OK")


# ----------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: build env + model, train 1024 steps, "
                        "eval, exit. Fast verification of all wiring.")
    p.add_argument("--output-dir", default=None,
                   help="Output directory for checkpoints and logs.")
    p.add_argument("--total-steps", type=int, default=2_000_000,
                   help="Total training timesteps (across curriculum).")
    p.add_argument("--num-envs", type=int, default=0,
                   help="Parallel envs. 0 = cpu_count()/2.")
    p.add_argument("--mag", type=float, default=None,
                   help="Override curriculum: train at a single fixed mag.")
    p.add_argument("--dist", type=float, default=100.0)
    p.add_argument("--duration", type=float, default=5.0,
                   help="Per-trial sim duration (s).")
    p.add_argument("--max-steps", type=int, default=5000,
                   help="Per-episode truncation budget.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu",
                   help="torch device (cpu | cuda). Default cpu (small policy).")
    p.add_argument("--resume", default=None,
                   help="Path to a .zip to resume from.")
    p.add_argument("--save-freq", type=int, default=50_000,
                   help="Env-step interval between checkpoints.")
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--dummy-vec", action="store_true",
                   help="Use DummyVecEnv (in-process) instead of SubprocVecEnv.")
    args = p.parse_args()

    if args.quick:
        smoke(args)
        return

    if args.output_dir is None:
        args.output_dir = f"runs/atp_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    train(args)


if __name__ == "__main__":
    main()
