"""
evaluate.py — Evaluate a trained SAC model on PandaReachEnv
=============================================================
Usage:
    python scripts/evaluate.py --model results/models/sac_panda_xxx.zip
    python scripts/evaluate.py --model results/models/sac_panda_xxx.zip --render --episodes 20
"""

import argparse
import os
import time
import json
import numpy as np

from stable_baselines3 import SAC

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.envs.panda_env import PandaReachEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC model")
    parser.add_argument("--model",    type=str, required=True, help="Path to .zip model file")
    parser.add_argument("--episodes", type=int, default=50,    help="Number of test episodes")
    parser.add_argument("--render",   action="store_true",     help="Enable GUI rendering")
    parser.add_argument("--seed",     type=int, default=0,     help="Evaluation seed")
    parser.add_argument("--delay",    type=float, default=0.03, help="Step delay when rendering (s)")
    parser.add_argument("--out",      type=str, default=None,  help="Save results JSON path")
    return parser.parse_args()


def evaluate(model_path: str, n_episodes: int, render: bool, seed: int, delay: float):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"[Eval] Loading model: {model_path}")
    model = SAC.load(model_path)

    # Use training=False so reward shaping is off → clean success metric
    env = PandaReachEnv(render=render, training=False, seed=seed)

    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        step_i = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_i += 1
            if render:
                time.sleep(delay)

        results.append({
            "episode":   ep + 1,
            "distance":  info["distance"],
            "success":   info["success"],
            "collision": info["collision"],
            "steps":     step_i,
        })

        dist_mm = info["distance"] * 1000
        status  = "✅" if info["success"] else ("💥" if info["collision"] else "❌")
        print(f"  Ep {ep+1:>3}/{n_episodes}  {status}  dist={dist_mm:.2f}mm  steps={step_i}")

        if render:
            time.sleep(1.0)

    env.close()
    return results


def summarise(results: list) -> dict:
    dists     = [r["distance"] for r in results]
    successes = [r["success"]  for r in results]
    steps     = [r["steps"]    for r in results]

    summary = {
        "n_episodes":     len(results),
        "success_rate":   float(np.mean(successes)),
        "mean_dist_mm":   float(np.mean(dists) * 1000),
        "std_dist_mm":    float(np.std(dists)  * 1000),
        "median_dist_mm": float(np.median(dists) * 1000),
        "mean_steps":     float(np.mean(steps)),
        "within_5mm":     float(np.mean([d < 0.005 for d in dists])),
        "within_10mm":    float(np.mean([d < 0.010 for d in dists])),
    }
    return summary


def main():
    args    = parse_args()
    results = evaluate(args.model, args.episodes, args.render, args.seed, args.delay)
    summary = summarise(results)

    print("\n" + "─" * 50)
    print("📊  Evaluation Summary")
    print("─" * 50)
    print(f"  Episodes       : {summary['n_episodes']}")
    print(f"  Success rate   : {summary['success_rate']*100:.1f}%  (< 5 mm)")
    print(f"  Within 10 mm   : {summary['within_10mm']*100:.1f}%")
    print(f"  Mean distance  : {summary['mean_dist_mm']:.2f} ± {summary['std_dist_mm']:.2f} mm")
    print(f"  Median distance: {summary['median_dist_mm']:.2f} mm")
    print(f"  Mean steps     : {summary['mean_steps']:.1f}")
    print("─" * 50)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        payload = {"summary": summary, "episodes": results}
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[Eval] Results saved → {args.out}")


if __name__ == "__main__":
    main()
